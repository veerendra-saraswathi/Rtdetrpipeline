// See include/ird/preproc.h for the specification.
#include "ird/preproc.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ird {

namespace {

inline int row_stride(const ImageU16View& img) {
    return img.row_stride > 0 ? img.row_stride : img.width * img.channels;
}

bool valid(const ImageU16View& img, const PreprocParams& p, int out_channels) {
    if (!img.data || img.width <= 0 || img.height <= 0) return false;
    if (img.channels != 1 && img.channels != 3) return false;
    if (img.bit_depth < 1 || img.bit_depth > 16) return false;
    if (out_channels != 1 && out_channels != 3) return false;
    if (p.out_w <= 0 || p.out_h <= 0) return false;
    if (p.tone == Tone::Percentile && !(p.p_lo >= 0.0 && p.p_hi <= 100.0 && p.p_lo <= p.p_hi)) return false;
    return true;
}

// Nearest-rank index into the sorted samples.
inline long long rank_index(double pct, long long n) {
    return static_cast<long long>(std::floor(pct / 100.0 * static_cast<double>(n - 1) + 0.5));
}

}  // namespace

const char* version() { return "ird_preproc 0.1.0"; }

Status tone_range(const ImageU16View& img, const PreprocParams& p, int* lo, int* hi) {
    if (!valid(img, p, 1) || !lo || !hi) return Status::BadArgument;
    const int maxv = (1 << img.bit_depth) - 1;
    if (p.tone == Tone::Linear) {
        *lo = 0;
        *hi = maxv;
        return Status::Ok;
    }
    std::vector<uint32_t> hist(static_cast<size_t>(maxv) + 1, 0u);
    const int rs = row_stride(img);
    const int rowlen = img.width * img.channels;
    for (int y = 0; y < img.height; ++y) {
        const uint16_t* r = img.data + static_cast<size_t>(y) * rs;
        for (int i = 0; i < rowlen; ++i) ++hist[std::min<int>(r[i], maxv)];
    }
    const long long n = static_cast<long long>(img.width) * img.height * img.channels;
    const long long k_lo = rank_index(p.p_lo, n), k_hi = rank_index(p.p_hi, n);
    long long cum = 0;
    int v_lo = -1, v_hi = -1;
    for (int v = 0; v <= maxv; ++v) {
        cum += hist[v];
        if (v_lo < 0 && cum > k_lo) v_lo = v;
        if (v_hi < 0 && cum > k_hi) { v_hi = v; break; }
    }
    if (v_hi <= v_lo) v_hi = v_lo + 1;
    *lo = v_lo;
    *hi = v_hi;
    return Status::Ok;
}

Status preprocess(const ImageU16View& img, const PreprocParams& p, int out_channels,
                  float* dst, PreprocInfo* info) {
    int lo = 0, hi = 1;
    Status s = tone_range(img, p, &lo, &hi);
    if (s != Status::Ok) return s;
    return preprocess_with_range(img, p, out_channels, lo, hi, dst, info);
}

Status preprocess_with_range(const ImageU16View& img, const PreprocParams& p, int out_channels,
                             int lo, int hi, float* dst, PreprocInfo* info) {
    if (!valid(img, p, out_channels) || !dst) return Status::BadArgument;
    if (hi <= lo) hi = lo + 1;
    const int maxv = (1 << img.bit_depth) - 1;

    // Step 2 as a lookup table: exact same float ops for every sample.
    std::vector<float> lut(static_cast<size_t>(maxv) + 1);
    const float inv = 1.0f / static_cast<float>(hi - lo);
    for (int v = 0; v <= maxv; ++v) {
        float t = static_cast<float>(v - lo) * inv;
        lut[v] = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    }

    // Step 4 geometry.
    const int W = img.width, H = img.height, C = img.channels;
    int new_w = W, new_h = H;
    if (W > p.out_w || H > p.out_h) {
        const double s = std::min(static_cast<double>(p.out_w) / W, static_cast<double>(p.out_h) / H);
        new_w = std::max(1, std::min(p.out_w, static_cast<int>(std::floor(W * s + 0.5))));
        new_h = std::max(1, std::min(p.out_h, static_cast<int>(std::floor(H * s + 0.5))));
    }
    const size_t plane = static_cast<size_t>(p.out_h) * p.out_w;
    std::fill(dst, dst + plane * out_channels, p.pad_value);

    const int rs = row_stride(img);
    auto tm = [&](int y, int x, int c) -> float {
        return lut[std::min<int>(img.data[static_cast<size_t>(y) * rs + static_cast<size_t>(x) * C + c], maxv)];
    };
    // Value of output channel oc at source pixel (y, x) after tone mapping and channel mapping.
    auto pix = [&](int y, int x, int oc) -> float {
        if (C == out_channels) return tm(y, x, oc);
        if (C == 1) return tm(y, x, 0);
        return 0.299f * tm(y, x, 0) + 0.587f * tm(y, x, 1) + 0.114f * tm(y, x, 2);
    };

    if (new_w == W && new_h == H) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float* o = dst + plane * oc;
            for (int y = 0; y < H; ++y) {
                float* orow = o + static_cast<size_t>(y) * p.out_w;
                for (int x = 0; x < W; ++x) orow[x] = pix(y, x, oc);
            }
        }
    } else {
        const float rx = static_cast<float>(W) / static_cast<float>(new_w);
        const float ry = static_cast<float>(H) / static_cast<float>(new_h);
        for (int y = 0; y < new_h; ++y) {
            float sy = (static_cast<float>(y) + 0.5f) * ry - 0.5f;
            sy = std::min(std::max(sy, 0.f), static_cast<float>(H - 1));
            const int y0 = static_cast<int>(std::floor(sy));
            const int y1 = std::min(y0 + 1, H - 1);
            const float fy = sy - static_cast<float>(y0);
            for (int x = 0; x < new_w; ++x) {
                float sx = (static_cast<float>(x) + 0.5f) * rx - 0.5f;
                sx = std::min(std::max(sx, 0.f), static_cast<float>(W - 1));
                const int x0 = static_cast<int>(std::floor(sx));
                const int x1 = std::min(x0 + 1, W - 1);
                const float fx = sx - static_cast<float>(x0);
                for (int oc = 0; oc < out_channels; ++oc) {
                    const float a = pix(y0, x0, oc), b = pix(y0, x1, oc);
                    const float c = pix(y1, x0, oc), d = pix(y1, x1, oc);
                    const float top = (1.f - fx) * a + fx * b;
                    const float bot = (1.f - fx) * c + fx * d;
                    dst[plane * oc + static_cast<size_t>(y) * p.out_w + x] = (1.f - fy) * top + fy * bot;
                }
            }
        }
    }

    if (info) {
        info->scale_x = static_cast<float>(new_w) / static_cast<float>(W);
        info->scale_y = static_cast<float>(new_h) / static_cast<float>(H);
        info->new_w = new_w;
        info->new_h = new_h;
        info->lo = lo;
        info->hi = hi;
    }
    return Status::Ok;
}

}  // namespace ird
