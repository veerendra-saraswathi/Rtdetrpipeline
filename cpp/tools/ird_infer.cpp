// Run a TensorRT engine over a split: writes a detections file (same format as
// python/ird/infer/predict.py, so the same evaluator scores it) and a latency report.
//
//   ird_infer --engine a_ir.engine --cfg a_ir.onnx.runtime.txt --list test_list.tsv --root <dataset root>
//             --out preds/a_ir_s0_trt.json --latency reports/a_ir_s0_latency.json
//             [--min-score 0.001] [--warmup 50] [--repeat 3] [--no-graph]
//
// The list comes from `python -m ird.infer.make_list` (one frame per line, tab-separated):
//   image_id  width  height  ir_path  vis_path  ir_bit_depth  vis_bit_depth   ('-' when absent)
// Image files: .png (8/16-bit, via stb_image, tools only) or .u16 raw (RCI convention).
//
// Latency is measured from "frame in memory" to "detection records ready" (ird_detect), after
// warm-up, over all frames x repeat. Files are decoded before timing starts.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "ird/ird_api.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

namespace {

struct Frame {
    long long id = 0;
    int w = 0, h = 0;
    std::string ir_path, vis_path;
    int ir_bd = 8, vis_bd = 8;
    std::vector<uint16_t> ir, vis;
    int ir_c = 1, vis_c = 3;
};

bool ends_with(const std::string& s, const std::string& e) {
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

// Reads a band into uint16 HWC. want_c: 1 for IR (first channel kept), 3 for visible.
bool read_band(const std::string& path, int w, int h, int want_c, std::vector<uint16_t>* out, int* bd) {
    if (ends_with(path, ".u16")) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        out->resize(static_cast<size_t>(w) * h * want_c);
        f.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(out->size() * 2));
        return static_cast<size_t>(f.gcount()) == out->size() * 2;   // little-endian host (Jetson, x86)
    }
    int iw = 0, ih = 0, ic = 0;
    const bool is16 = stbi_is_16_bit(path.c_str());
    void* px = is16 ? static_cast<void*>(stbi_load_16(path.c_str(), &iw, &ih, &ic, 0))
                    : static_cast<void*>(stbi_load(path.c_str(), &iw, &ih, &ic, 0));
    if (!px) return false;
    if (iw != w || ih != h) { stbi_image_free(px); return false; }
    out->resize(static_cast<size_t>(w) * h * want_c);
    for (size_t p = 0; p < static_cast<size_t>(w) * h; ++p) {
        for (int c = 0; c < want_c; ++c) {
            const int src_c = ic == 1 ? 0 : std::min(c, ic - 1);    // grey -> repeat; IR keeps channel 0
            const size_t si = p * ic + src_c;
            (*out)[p * want_c + c] = is16 ? static_cast<uint16_t*>(px)[si] : static_cast<uint8_t*>(px)[si];
        }
    }
    if (!is16) *bd = std::min(*bd, 8);
    stbi_image_free(px);
    return true;
}

struct Stats { double mean, p50, p95, p99, max; };
Stats stats(std::vector<double> v) {
    if (v.empty()) return {0, 0, 0, 0, 0};
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[std::min(v.size() - 1, static_cast<size_t>(std::ceil(p * v.size())) - (p > 0 ? 1 : 0))]; };
    return {std::accumulate(v.begin(), v.end(), 0.0) / v.size(), q(0.50), q(0.95), q(0.99), v.back()};
}
std::string stats_json(const Stats& s) {
    char b[256];
    snprintf(b, sizeof b, "{\"mean\": %.4f, \"p50\": %.4f, \"p95\": %.4f, \"p99\": %.4f, \"max\": %.4f}", s.mean, s.p50, s.p95, s.p99, s.max);
    return b;
}
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}
void usage() {
    fprintf(stderr, "usage: ird_infer --engine E --cfg runtime.txt --list list.tsv --root DIR --out preds.json "
                    "[--latency lat.json] [--min-score 0.001] [--warmup 50] [--repeat 1] [--no-graph]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string engine, cfg, list, root = ".", out, lat_out;
    float min_score = 0.001f;
    int warmup = 50, repeat = 1;
    bool graph = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (a == "--engine") engine = next();
        else if (a == "--cfg") cfg = next();
        else if (a == "--list") list = next();
        else if (a == "--root") root = next();
        else if (a == "--out") out = next();
        else if (a == "--latency") lat_out = next();
        else if (a == "--min-score") min_score = std::stof(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--repeat") repeat = std::max(1, std::stoi(next()));
        else if (a == "--no-graph") graph = false;
        else { usage(); return 2; }
    }
    if (engine.empty() || cfg.empty() || list.empty() || out.empty()) { usage(); return 2; }

    ird_options opt = ird_default_options();
    opt.score_threshold = min_score;
    opt.use_cuda_graph = graph ? 1 : 0;
    ird_detector* det = nullptr;
    if (ird_create(engine.c_str(), cfg.c_str(), &opt, &det) != IRD_OK) {
        fprintf(stderr, "ird_create: %s\n", ird_last_error());
        return 1;
    }
    const std::string modality = ird_modality(det);
    const bool need_ir = modality != "vis", need_vis = modality != "ir";

    // ---- load frames (not timed) ----
    std::vector<Frame> frames;
    std::ifstream lf(list);
    if (!lf) { fprintf(stderr, "cannot open %s\n", list.c_str()); return 1; }
    std::string line;
    while (std::getline(lf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tok[7];
        for (auto& t : tok) std::getline(ss, t, '\t');
        Frame f;
        f.id = std::stoll(tok[0]); f.w = std::stoi(tok[1]); f.h = std::stoi(tok[2]);
        f.ir_path = tok[3]; f.vis_path = tok[4];
        f.ir_bd = tok[5] == "-" ? 8 : std::stoi(tok[5]);
        f.vis_bd = tok[6] == "-" ? 8 : std::stoi(tok[6]);
        if (need_vis && f.vis_path == "-") continue;        // same rule as the Python side
        if (need_ir && !read_band(root + "/" + f.ir_path, f.w, f.h, 1, &f.ir, &f.ir_bd)) {
            fprintf(stderr, "cannot read %s\n", f.ir_path.c_str()); return 1;
        }
        if (need_vis && !read_band(root + "/" + f.vis_path, f.w, f.h, 3, &f.vis, &f.vis_bd)) {
            fprintf(stderr, "cannot read %s\n", f.vis_path.c_str()); return 1;
        }
        frames.push_back(std::move(f));
    }
    if (frames.empty()) { fprintf(stderr, "no frames\n"); return 1; }
    printf("loaded %zu frames, modality %s\n", frames.size(), modality.c_str());

    std::vector<ird_detection> buf(300);
    auto run = [&](const Frame& f, int* n, ird_timing* t) {
        ird_image ir{f.ir.data(), f.w, f.h, 1, f.ir_bd}, vis{f.vis.data(), f.w, f.h, 3, f.vis_bd};
        return ird_detect(det, need_ir ? &ir : nullptr, need_vis ? &vis : nullptr, buf.data(),
                          static_cast<int>(buf.size()), n, t);
    };
    int n = 0;
    ird_timing t{};
    for (int i = 0; i < warmup; ++i)
        if (run(frames[i % frames.size()], &n, &t) != IRD_OK) { fprintf(stderr, "ird_detect: %s\n", ird_last_error()); return 1; }

    std::vector<double> pre, inf, dec, tot;
    std::ofstream of(out);
    of << "{\"meta\": {\"backend\": \"tensorrt\", \"engine\": \"" << esc(engine) << "\", \"runtime_cfg\": \"" << esc(cfg)
       << "\", \"modality\": \"" << modality << "\", \"min_score\": " << min_score
       << ", \"cuda_graph\": " << (graph ? "true" : "false") << "},\n \"image_ids\": [";
    for (size_t i = 0; i < frames.size(); ++i) of << (i ? ", " : "") << frames[i].id;
    of << "],\n \"detections\": [";
    bool first = true;
    for (int r = 0; r < repeat; ++r) {
        for (const auto& f : frames) {
            if (run(f, &n, &t) != IRD_OK) { fprintf(stderr, "ird_detect: %s\n", ird_last_error()); return 1; }
            pre.push_back(t.preprocess_ms); inf.push_back(t.infer_ms); dec.push_back(t.decode_ms); tot.push_back(t.total_ms);
            if (r == 0) {
                for (int k = 0; k < n; ++k) {
                    const auto& d = buf[k];
                    char b[256];
                    snprintf(b, sizeof b, "%s\n  {\"image_id\": %lld, \"category_id\": %d, \"bbox\": [%.3f, %.3f, %.3f, %.3f], \"score\": %.6f}",
                             first ? "" : ",", f.id, d.class_id, d.x, d.y, d.w, d.h, d.score);
                    of << b;
                    first = false;
                }
            }
        }
    }
    of << "\n]}\n";
    of.close();

    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    const Stats st = stats(tot);
    printf("frames %zu x %d, total ms: mean %.2f p50 %.2f p95 %.2f p99 %.2f max %.2f (budget 40)\n",
           frames.size(), repeat, st.mean, st.p50, st.p95, st.p99, st.max);
    if (!lat_out.empty()) {
        std::ofstream lo(lat_out);
        lo << "{\"device\": \"" << esc(prop.name) << "\", \"engine\": \"" << esc(engine) << "\", \"modality\": \"" << modality
           << "\", \"frames\": " << frames.size() << ", \"repeat\": " << repeat << ", \"warmup\": " << warmup
           << ", \"cuda_graph\": " << (graph ? "true" : "false")
           << ", \"definition\": \"frame in memory -> detection records ready (ird_detect)\""
           << ",\n \"total_ms\": " << stats_json(st) << ",\n \"preprocess_ms\": " << stats_json(stats(pre))
           << ",\n \"infer_ms\": " << stats_json(stats(inf)) << ",\n \"decode_ms\": " << stats_json(stats(dec))
           << ",\n \"budget_ms\": 40, \"p99_within_budget\": " << (st.p99 <= 40.0 ? "true" : "false") << "}\n";
    }
    ird_destroy(det);
    return 0;
}
