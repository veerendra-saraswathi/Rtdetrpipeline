// Build a TensorRT engine from an exported ONNX model. Run it ON the Jetson that will use the
// engine (engines are tied to the GPU and TensorRT version).
//
//   ird_build_engine --onnx a_ir.onnx --engine a_ir_fp16.engine [--fp16] [--workspace-mb 1024]
//                    [--fp32 <layer-name-substring>]...  [--timing-cache file]
//   INT8 (post-training calibration on TRAINING frames; layers INT8 cannot run stay FP16):
//   ird_build_engine --onnx a_ir.onnx --engine a_ir_int8.engine --int8 --calib-list train_list.tsv
//                    --calib-root .. --calib-cfg a_ir.onnx.runtime.txt [--calib-n 300]
//                    [--calib-cache a_ir.int8.calib] [--no-int8 <layer-name-substring>]...
//
// --fp32 keeps matching layers in FP32 inside an FP16 engine (e.g. the box head, if the parity
// check shows FP16 box error). --no-int8 keeps matching layers in FP16 inside an INT8 engine
// (e.g. "decoder" for a mixed engine). Layer names are printed with --list-layers.
// Calibration frames go through the same C++ preprocessing as inference (ird/preproc.h), so the
// INT8 ranges are measured on exactly what the engine will see.
// The calibrator API is marked deprecated in TensorRT 10 (explicit Q/DQ is the newer route) but is
// still supported; silence the warning so real warnings stand out.
#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "ird/preproc.h"
#include "runtime_config.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TensorRT] %s\n", msg);
    }
} g_logger;

template <typename T>
struct Del {
    void operator()(T* p) const {
#if NV_TENSORRT_MAJOR >= 10
        delete p;
#else
        if (p) p->destroy();
#endif
    }
};
template <typename T>
using Ptr = std::unique_ptr<T, Del<T>>;

void usage() {
    fprintf(stderr, "usage: ird_build_engine --onnx M.onnx --engine M.engine [--fp16] [--workspace-mb N] "
                    "[--fp32 substr]... [--timing-cache F] [--list-layers]\n"
                    "       [--int8 --calib-list L.tsv --calib-root DIR --calib-cfg M.onnx.runtime.txt "
                    "[--calib-n 300] [--calib-cache F] [--no-int8 substr]...]\n");
}

bool ends_with(const std::string& s, const std::string& e) {
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

// Same reader as ird_infer: one band into uint16 HWC (want_c 1 = IR, 3 = visible).
bool read_band(const std::string& path, int w, int h, int want_c, std::vector<uint16_t>* out, int* bd) {
    if (ends_with(path, ".u16")) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        out->resize(static_cast<size_t>(w) * h * want_c);
        f.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(out->size() * 2));
        return static_cast<size_t>(f.gcount()) == out->size() * 2;
    }
    int iw = 0, ih = 0, ic = 0;
    const bool is16 = stbi_is_16_bit(path.c_str());
    void* px = is16 ? static_cast<void*>(stbi_load_16(path.c_str(), &iw, &ih, &ic, 0))
                    : static_cast<void*>(stbi_load(path.c_str(), &iw, &ih, &ic, 0));
    if (!px) return false;
    if (iw != w || ih != h) { stbi_image_free(px); return false; }
    out->resize(static_cast<size_t>(w) * h * want_c);
    for (size_t p = 0; p < static_cast<size_t>(w) * h; ++p)
        for (int c = 0; c < want_c; ++c) {
            const int src_c = ic == 1 ? 0 : std::min(c, ic - 1);
            const size_t si = p * ic + src_c;
            (*out)[p * want_c + c] = is16 ? static_cast<uint16_t*>(px)[si] : static_cast<uint8_t*>(px)[si];
        }
    if (!is16) *bd = std::min(*bd, 8);
    stbi_image_free(px);
    return true;
}

// Entropy calibrator (the standard TensorRT PTQ method), batch 1. Feeds preprocessed frames from a
// frame list (python -m ird.infer.make_list), evenly spaced, n of them. The calibration cache makes
// a rebuild skip the calibration pass; delete it after changing the model or the frames.
class Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Calibrator(const std::string& list, const std::string& root, const ird::RuntimeConfig& cfg, int n,
               const std::string& cache)
        : root_(root), cfg_(cfg), cache_path_(cache) {
        std::ifstream lf(list);
        std::string line;
        std::vector<std::vector<std::string>> rows;
        while (std::getline(lf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::vector<std::string> t(7);
            for (auto& x : t) std::getline(ss, x, '\t');
            const std::string& path = cfg_.modality == "vis" ? t[4] : t[3];
            if (path == "-" || path.empty()) continue;
            rows.push_back(t);
        }
        const size_t m = rows.size();
        const size_t k = std::min(static_cast<size_t>(std::max(n, 1)), m);
        for (size_t i = 0; i < k; ++i) rows_.push_back(rows[i * m / k]);   // evenly spaced
        plane_ = static_cast<size_t>(cfg_.out_h) * cfg_.out_w;
        host_.resize(3 * plane_);
        if (cudaMalloc(&dev_, host_.size() * sizeof(float)) != cudaSuccess) dev_ = nullptr;
        printf("calibration: %zu of %zu frames from %s (%s band)\n", rows_.size(), m, list.c_str(), cfg_.modality.c_str());
    }
    ~Calibrator() override { if (dev_) cudaFree(dev_); }
    int32_t getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], char const* /*names*/[], int32_t /*nb*/) noexcept override {
        if (!dev_) return false;
        while (next_ < rows_.size()) {
            const auto& t = rows_[next_++];
            const bool vis = cfg_.modality == "vis";
            const int w = std::stoi(t[1]), h = std::stoi(t[2]);
            int bd = (vis ? t[6] : t[5]) == "-" ? 8 : std::stoi(vis ? t[6] : t[5]);
            std::vector<uint16_t> px;
            if (!read_band(root_ + "/" + (vis ? t[4] : t[3]), w, h, vis ? 3 : 1, &px, &bd)) {
                fprintf(stderr, "calibration: cannot read %s, skipped\n", (vis ? t[4] : t[3]).c_str());
                continue;
            }
            ird::ImageU16View v;
            v.data = px.data(); v.width = w; v.height = h; v.channels = vis ? 3 : 1; v.bit_depth = bd;
            ird::PreprocInfo info;
            if (ird::preprocess(v, vis ? cfg_.vis : cfg_.ir, 3, host_.data(), &info) != ird::Status::Ok) continue;
            if (cudaMemcpy(dev_, host_.data(), host_.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return false;
            bindings[0] = dev_;
            if (next_ % 50 == 0) printf("  calibrated %zu / %zu\n", next_, rows_.size());
            return true;
        }
        return false;
    }
    void const* readCalibrationCache(std::size_t& length) noexcept override {
        cache_.clear();
        if (!cache_path_.empty()) {
            std::ifstream f(cache_path_, std::ios::binary);
            if (f) cache_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        length = cache_.size();
        if (length) printf("calibration: using cache %s\n", cache_path_.c_str());
        return length ? cache_.data() : nullptr;
    }
    void writeCalibrationCache(void const* ptr, std::size_t length) noexcept override {
        if (cache_path_.empty()) return;
        std::ofstream f(cache_path_, std::ios::binary);
        f.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(length));
    }

private:
    std::string root_;
    ird::RuntimeConfig cfg_;
    std::string cache_path_;
    std::vector<std::vector<std::string>> rows_;
    size_t next_ = 0, plane_ = 0;
    std::vector<float> host_;
    void* dev_ = nullptr;
    std::vector<char> cache_;
};
}  // namespace

int main(int argc, char** argv) {
    std::string onnx, engine, cache, calib_list, calib_root = ".", calib_cfg, calib_cache;
    bool fp16 = false, list = false, int8 = false;
    int calib_n = 300;
    size_t ws_mb = 1024;
    std::vector<std::string> fp32, no_int8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (a == "--onnx") onnx = next();
        else if (a == "--engine") engine = next();
        else if (a == "--fp16") fp16 = true;
        else if (a == "--workspace-mb") ws_mb = std::stoul(next());
        else if (a == "--fp32") fp32.push_back(next());
        else if (a == "--timing-cache") cache = next();
        else if (a == "--list-layers") list = true;
        else if (a == "--int8") int8 = true;
        else if (a == "--calib-list") calib_list = next();
        else if (a == "--calib-root") calib_root = next();
        else if (a == "--calib-cfg") calib_cfg = next();
        else if (a == "--calib-n") calib_n = std::stoi(next());
        else if (a == "--calib-cache") calib_cache = next();
        else if (a == "--no-int8") no_int8.push_back(next());
        else { usage(); return 2; }
    }
    if (onnx.empty() || (engine.empty() && !list)) { usage(); return 2; }
    if (int8 && (calib_list.empty() || calib_cfg.empty())) {
        fprintf(stderr, "--int8 needs --calib-list and --calib-cfg\n");
        return 2;
    }

    Ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_logger));
#if NV_TENSORRT_MAJOR >= 10
    Ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
#else
    Ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
#endif
    Ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, g_logger));
    if (!parser->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) fprintf(stderr, "ONNX parse error: %s\n", parser->getError(i)->desc());
        return 1;
    }
    if (list) {
        for (int i = 0; i < network->getNbLayers(); ++i) printf("%4d %s\n", i, network->getLayer(i)->getName());
        if (engine.empty()) return 0;
    }
    Ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, ws_mb << 20);
    if (fp16) {
        if (!builder->platformHasFastFp16()) fprintf(stderr, "warning: platform has no fast FP16\n");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    std::unique_ptr<Calibrator> calib;
    if (int8) {
        ird::RuntimeConfig rc;
        std::string err;
        if (!ird::load_runtime_config(calib_cfg, &rc, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (rc.in_channels != 3) { fprintf(stderr, "INT8 calibration supports 3-channel models only\n"); return 1; }
        if (!builder->platformHasFastInt8()) fprintf(stderr, "warning: platform has no fast INT8\n");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);   // layers INT8 cannot run fall back to FP16, not FP32
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        calib.reset(new Calibrator(calib_list, calib_root, rc, calib_n, calib_cache));
        config->setInt8Calibrator(calib.get());
        fp16 = true;
        if (!no_int8.empty()) {
            config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            int kept = 0;
            for (int i = 0; i < network->getNbLayers(); ++i) {
                nvinfer1::ILayer* l = network->getLayer(i);
                bool is_float = false;
                for (int o = 0; o < l->getNbOutputs(); ++o)
                    is_float |= l->getOutputType(o) == nvinfer1::DataType::kFLOAT || l->getOutputType(o) == nvinfer1::DataType::kHALF;
                if (!is_float) continue;
                for (const auto& sub : no_int8) {
                    if (std::strstr(l->getName(), sub.c_str())) {
                        l->setPrecision(nvinfer1::DataType::kHALF);
                        ++kept;
                        break;
                    }
                }
            }
            printf("kept %d layers out of INT8 (FP16)\n", kept);
            if (kept == 0) fprintf(stderr, "warning: no layer name matched --no-int8; see --list-layers\n");
        }
    }
    int pinned = 0;
    if (!fp32.empty()) {
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        for (int i = 0; i < network->getNbLayers(); ++i) {
            nvinfer1::ILayer* l = network->getLayer(i);
            for (const auto& s : fp32) {
                if (std::strstr(l->getName(), s.c_str())) {
                    l->setPrecision(nvinfer1::DataType::kFLOAT);
                    for (int o = 0; o < l->getNbOutputs(); ++o)
                        if (l->getOutputType(o) == nvinfer1::DataType::kFLOAT || l->getOutputType(o) == nvinfer1::DataType::kHALF)
                            l->setOutputType(o, nvinfer1::DataType::kFLOAT);
                    ++pinned;
                    break;
                }
            }
        }
        printf("kept %d layers in FP32\n", pinned);
    }
    std::vector<char> cache_blob;
    Ptr<nvinfer1::ITimingCache> tcache;
    if (!cache.empty()) {
        std::ifstream cf(cache, std::ios::binary);
        if (cf) cache_blob.assign(std::istreambuf_iterator<char>(cf), std::istreambuf_iterator<char>());
        tcache.reset(config->createTimingCache(cache_blob.data(), cache_blob.size()));
        config->setTimingCache(*tcache, false);
    }
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto* t = network->getInput(i);
        auto d = t->getDimensions();
        printf("input  %s [", t->getName());
        for (int k = 0; k < d.nbDims; ++k) printf("%s%ld", k ? "," : "", static_cast<long>(d.d[k]));
        printf("]\n");
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) printf("output %s\n", network->getOutput(i)->getName());

    Ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) { fprintf(stderr, "engine build failed\n"); return 1; }
    std::ofstream of(engine, std::ios::binary);
    of.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    if (!of) { fprintf(stderr, "cannot write %s\n", engine.c_str()); return 1; }
    if (tcache) {
        Ptr<nvinfer1::IHostMemory> ser(tcache->serialize());
        std::ofstream cf(cache, std::ios::binary);
        cf.write(static_cast<const char*>(ser->data()), static_cast<std::streamsize>(ser->size()));
    }
    printf("wrote %s (%.1f MB, TensorRT %d.%d.%d, %s)\n", engine.c_str(), plan->size() / 1048576.0,
           NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
           int8 ? (no_int8.empty() ? "INT8 (FP16 fallback)" : "INT8 mixed, listed layers FP16") : fp16 ? "FP16" : "FP32");
    return 0;
}
