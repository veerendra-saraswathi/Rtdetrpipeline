// IRD detector: TensorRT engine + shared preprocessing + decoder, behind the C API in ird_api.h.
// Works with TensorRT 8.5+ (JetPack 5.1) and 10.x (JetPack 6), using the named-tensor API.
#include "ird/ird_api.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iterator>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "decoder.h"
#include "ird/preproc.h"
#include "runtime_config.h"

namespace {

thread_local std::string g_last_error;

int fail(int code, const std::string& msg) {
    g_last_error = msg;
    return code;
}

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) {
            std::lock_guard<std::mutex> lk(mu_);
            fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }

   private:
    std::mutex mu_;
};
Logger g_logger;

template <typename T>
struct TrtDelete {
    void operator()(T* p) const {
#if NV_TENSORRT_MAJOR >= 10
        delete p;
#else
        if (p) p->destroy();
#endif
    }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

struct ird_detector {
    ird::RuntimeConfig cfg;
    ird_options opt{};
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    float *h_in = nullptr, *h_logits = nullptr, *h_boxes = nullptr;  // pinned host
    void *d_in = nullptr, *d_logits = nullptr, *d_boxes = nullptr;   // device
    size_t n_in = 0, n_logits = 0, n_boxes = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_ready = false, graph_failed = false;
    std::vector<ird::Det> dets;
    std::vector<float> scratch_s;
    std::vector<int> scratch_i;

    ~ird_detector() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamSynchronize(stream);
        cudaFreeHost(h_in); cudaFreeHost(h_logits); cudaFreeHost(h_boxes);
        cudaFree(d_in); cudaFree(d_logits); cudaFree(d_boxes);
        context.reset(); engine.reset(); runtime.reset();
        if (stream) cudaStreamDestroy(stream);
    }

    // Copy in, run, copy out. All on `stream`; caller synchronises.
    bool enqueue_all() {
        if (cudaMemcpyAsync(d_in, h_in, n_in * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        if (!context->enqueueV3(stream)) return false;
        if (cudaMemcpyAsync(h_logits, d_logits, n_logits * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        if (cudaMemcpyAsync(h_boxes, d_boxes, n_boxes * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        return true;
    }

    bool run() {
        if (opt.use_cuda_graph && !graph_failed) {
            if (!graph_ready) {
                // TensorRT needs one normal run before capture.
                if (!enqueue_all() || cudaStreamSynchronize(stream) != cudaSuccess) return false;
                if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                    bool ok = enqueue_all();
                    cudaError_t e = cudaStreamEndCapture(stream, &graph);
                    if (ok && e == cudaSuccess &&
#if CUDART_VERSION >= 12000
                        cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess
#else
                        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0) == cudaSuccess
#endif
                    ) {
                        graph_ready = true;
                    } else {
                        graph_failed = true;
                        cudaGetLastError();
                        fprintf(stderr, "[ird] CUDA graph capture failed; running without it\n");
                    }
                } else {
                    graph_failed = true;
                    cudaGetLastError();
                }
            }
            if (graph_ready) return cudaGraphLaunch(graph_exec, stream) == cudaSuccess;
        }
        return enqueue_all();
    }
};

extern "C" {

ird_options ird_default_options(void) {
    ird_options o{};
    o.score_threshold = 0.5f;
    for (int i = 0; i < IRD_MAX_CLASSES; ++i) o.class_thresholds[i] = -1.f;
    o.max_detections = 300;
    o.use_cuda_graph = 1;
    o.top_k = 300;
    return o;
}

const char* ird_last_error(void) { return g_last_error.c_str(); }

int ird_create(const char* engine_path, const char* runtime_cfg_path, const ird_options* options,
               ird_detector** out) {
    if (!engine_path || !runtime_cfg_path || !out) return fail(IRD_ERR_ARG, "null argument");
    *out = nullptr;
    auto det = std::make_unique<ird_detector>();
    det->opt = options ? *options : ird_default_options();
    std::string err;
    if (!ird::load_runtime_config(runtime_cfg_path, &det->cfg, &err)) return fail(IRD_ERR_FILE, err);
    if (det->cfg.num_classes > IRD_MAX_CLASSES) return fail(IRD_ERR_ARG, "too many classes");

    std::ifstream f(engine_path, std::ios::binary);
    if (!f) return fail(IRD_ERR_FILE, std::string("cannot open engine ") + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    det->runtime.reset(nvinfer1::createInferRuntime(g_logger));
    if (!det->runtime) return fail(IRD_ERR_ENGINE, "createInferRuntime failed");
    det->engine.reset(det->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!det->engine) return fail(IRD_ERR_ENGINE, "deserializeCudaEngine failed (engine built for another TensorRT version or GPU?)");
    det->context.reset(det->engine->createExecutionContext());
    if (!det->context) return fail(IRD_ERR_ENGINE, "createExecutionContext failed");
    if (cudaStreamCreateWithFlags(&det->stream, cudaStreamNonBlocking) != cudaSuccess) return fail(IRD_ERR_CUDA, "cudaStreamCreate failed");

    const auto& c = det->cfg;
    struct Want { const std::string* name; void** d; float** h; size_t* n; };
    Want wants[] = {{&c.input_name, &det->d_in, &det->h_in, &det->n_in},
                    {&c.logits_name, &det->d_logits, &det->h_logits, &det->n_logits},
                    {&c.boxes_name, &det->d_boxes, &det->h_boxes, &det->n_boxes}};
    for (auto& w : wants) {
        const char* name = w.name->c_str();
        nvinfer1::Dims d = det->engine->getTensorShape(name);
        if (d.nbDims <= 0) return fail(IRD_ERR_ENGINE, "engine has no tensor named " + *w.name);
        for (int i = 0; i < d.nbDims; ++i)
            if (d.d[i] < 0) return fail(IRD_ERR_ENGINE, "dynamic shapes are not supported; export with a fixed shape");
        if (det->engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
            return fail(IRD_ERR_ENGINE, "tensor " + *w.name + " is not float32");
        *w.n = volume(d);
        if (cudaMalloc(w.d, *w.n * sizeof(float)) != cudaSuccess) return fail(IRD_ERR_CUDA, "cudaMalloc failed");
        if (cudaMallocHost(reinterpret_cast<void**>(w.h), *w.n * sizeof(float)) != cudaSuccess) return fail(IRD_ERR_CUDA, "cudaMallocHost failed");
        if (!det->context->setTensorAddress(name, *w.d)) return fail(IRD_ERR_ENGINE, "setTensorAddress failed for " + *w.name);
    }
    const size_t want_in = static_cast<size_t>(c.in_channels) * c.out_h * c.out_w;
    if (det->n_in != want_in) return fail(IRD_ERR_ENGINE, "engine input size does not match runtime config");
    if (det->n_logits != static_cast<size_t>(c.num_queries) * c.num_classes || det->n_boxes != static_cast<size_t>(c.num_queries) * 4)
        return fail(IRD_ERR_ENGINE, "engine output sizes do not match runtime config");
    *out = det.release();
    return IRD_OK;
}

int ird_detect(ird_detector* det, const ird_image* ir, const ird_image* vis, ird_detection* out,
               int capacity, int* n_out, ird_timing* timing) {
    if (!det || !out || !n_out) return fail(IRD_ERR_ARG, "null argument");
    *n_out = 0;
    const auto& c = det->cfg;
    auto view = [](const ird_image* im) {
        ird::ImageU16View v;
        v.data = im->data; v.width = im->width; v.height = im->height;
        v.channels = im->channels; v.bit_depth = im->bit_depth;
        return v;
    };
    const auto t0 = std::chrono::steady_clock::now();
    ird::PreprocInfo info;
    const ird_image* ref = nullptr;
    const size_t plane = static_cast<size_t>(c.out_h) * c.out_w;
    if (c.modality == "ir") {
        if (!ir || !ir->data) return fail(IRD_ERR_ARG, "IR image required");
        if (ird::preprocess(view(ir), c.ir, 3, det->h_in, &info) != ird::Status::Ok) return fail(IRD_ERR_ARG, "bad IR image");
        ref = ir;
    } else if (c.modality == "vis") {
        if (!vis || !vis->data) return fail(IRD_ERR_ARG, "visible image required");
        if (ird::preprocess(view(vis), c.vis, 3, det->h_in, &info) != ird::Status::Ok) return fail(IRD_ERR_ARG, "bad visible image");
        ref = vis;
    } else {
        if (!ir || !vis || !ir->data || !vis->data) return fail(IRD_ERR_ARG, "fusion needs IR and visible images");
        if (ir->width != vis->width || ir->height != vis->height) return fail(IRD_ERR_ARG, "fusion images differ in size");
        ird::PreprocInfo info_ir;
        if (ird::preprocess(view(vis), c.vis, 3, det->h_in, &info) != ird::Status::Ok ||
            ird::preprocess(view(ir), c.ir, 1, det->h_in + 3 * plane, &info_ir) != ird::Status::Ok)
            return fail(IRD_ERR_ARG, "bad fusion images");
        ref = vis;
    }
    const double t_pre = ms_since(t0);

    const auto t1 = std::chrono::steady_clock::now();
    if (!det->run()) return fail(IRD_ERR_CUDA, "inference failed");
    if (cudaStreamSynchronize(det->stream) != cudaSuccess) return fail(IRD_ERR_CUDA, "cudaStreamSynchronize failed");
    const double t_inf = ms_since(t1);

    const auto t2 = std::chrono::steady_clock::now();
    float min_thr = det->opt.score_threshold;
    for (int k = 0; k < c.num_classes; ++k)
        if (det->opt.class_thresholds[k] >= 0.f) min_thr = std::min(min_thr, det->opt.class_thresholds[k]);
    ird::decode(det->h_logits, det->h_boxes, c.num_queries, c.num_classes, c.out_h, c.out_w,
                info.scale_x, info.scale_y, ref->width, ref->height, det->opt.top_k, min_thr,
                &det->dets, &det->scratch_s, &det->scratch_i);
    int n = 0;
    const int cap = std::min(capacity, det->opt.max_detections);
    for (const auto& d : det->dets) {
        const float thr = det->opt.class_thresholds[d.class_id] >= 0.f ? det->opt.class_thresholds[d.class_id]
                                                                       : det->opt.score_threshold;
        if (d.score < thr) continue;
        if (n >= cap) break;
        out[n++] = {d.class_id, static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.w),
                    static_cast<float>(d.h), d.score};
    }
    *n_out = n;
    const double t_dec = ms_since(t2);
    if (timing) *timing = {t_pre, t_inf, t_dec, ms_since(t0)};
    return IRD_OK;
}

void ird_destroy(ird_detector* det) { delete det; }

int ird_num_classes(const ird_detector* det) { return det ? det->cfg.num_classes : 0; }

const char* ird_class_name(const ird_detector* det, int class_id) {
    if (!det || class_id < 0 || class_id >= static_cast<int>(det->cfg.class_names.size())) return "";
    return det->cfg.class_names[class_id].c_str();
}

const char* ird_modality(const ird_detector* det) { return det ? det->cfg.modality.c_str() : ""; }

}  // extern "C"
