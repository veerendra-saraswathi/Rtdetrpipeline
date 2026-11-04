// IRD preprocessing: raw uint16 frame -> float32 CHW network input.
//
// One implementation, used in two places:
//   - training, through the Python module `ird_preproc` (pybind11),
//   - the Jetson runtime (ird_runtime).
// So training and deployment see the same input. python/ird/data/preproc_ref.py is an
// independent NumPy copy of the same math; tests check they agree.
//
// Steps
//   1. Tone range [lo, hi]:
//        Linear:     lo = 0, hi = 2^bit_depth - 1
//        Percentile: lo/hi = p_lo/p_hi percentiles of all samples of the frame (all channels
//                    together), nearest rank: k = floor(p/100 * (N-1) + 0.5) in the sorted samples.
//      If hi <= lo then hi = lo + 1.
//   2. Tone map each sample: t(v) = clamp((float)(v - lo) * (1.0f / (float)(hi - lo)), 0, 1).
//      Values above 2^bit_depth - 1 are clamped to it first.
//   3. Channels: in 1 -> out 3: copy the grey plane 3 times. in 3 -> out 1:
//      0.299 R + 0.587 G + 0.114 B (after tone mapping). in == out: copy.
//   4. Geometry: the image is placed at the top-left of an out_h x out_w canvas filled with
//      pad_value. If it does not fit, it is first scaled down (keeping aspect) with bilinear
//      sampling at pixel centres. Boxes in network pixels map back with x / scale_x, y / scale_y.
//
// No mean/std normalisation: RT-DETR takes [0, 1] input.
#pragma once
#include <cstdint>
#include <cstddef>

namespace ird {

enum class Tone : int { Linear = 0, Percentile = 1 };

struct ImageU16View {
    const uint16_t* data = nullptr;  // interleaved HWC
    int width = 0, height = 0, channels = 1;
    int bit_depth = 14;              // 1..16
    int row_stride = 0;              // elements per row; 0 means width * channels
};

struct PreprocParams {
    int out_h = 512, out_w = 640;
    Tone tone = Tone::Percentile;
    double p_lo = 0.5, p_hi = 99.5;  // percent, used when tone == Percentile
    float pad_value = 0.0f;
};

struct PreprocInfo {
    float scale_x = 1.f, scale_y = 1.f;  // network px = original px * scale
    int new_w = 0, new_h = 0;            // size of the image area on the canvas
    int lo = 0, hi = 1;                  // tone range used
};

enum class Status : int { Ok = 0, BadArgument = 1 };

// Tone range of a frame (step 1).
Status tone_range(const ImageU16View& img, const PreprocParams& p, int* lo, int* hi);

// Full preprocessing (steps 1-4). dst must hold out_channels * out_h * out_w floats (CHW).
// out_channels must be 1 or 3.
Status preprocess(const ImageU16View& img, const PreprocParams& p, int out_channels,
                  float* dst, PreprocInfo* info);

// Same as preprocess, but uses a tone range given by the caller (e.g. computed once on a
// reference frame). Steps 2-4 only.
Status preprocess_with_range(const ImageU16View& img, const PreprocParams& p, int out_channels,
                             int lo, int hi, float* dst, PreprocInfo* info);

const char* version();

}  // namespace ird
