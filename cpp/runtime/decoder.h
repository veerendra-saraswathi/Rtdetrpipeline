// Decode raw RT-DETR outputs. Mirrors python/ird/infer/decode.py exactly:
//   scores = sigmoid(logits) over Q*C (index q*C + c), ordered by score descending, ties by lower
//   index; first top_k; stop below min_score; box cx,cy,w,h (canvas-normalised) -> canvas pixels
//   -> original pixels (divide by scale) -> clipped to the image; empty boxes dropped.
#pragma once
#include <vector>

namespace ird {

struct Det {
    int class_id;
    double x, y, w, h;
    float score;
};

void decode(const float* logits, const float* boxes, int num_queries, int num_classes, int out_h,
            int out_w, float scale_x, float scale_y, int img_w, int img_h, int top_k, float min_score,
            std::vector<Det>* out, std::vector<float>* scratch_scores, std::vector<int>* scratch_idx);

}  // namespace ird
