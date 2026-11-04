#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ird {

void decode(const float* logits, const float* boxes, int num_queries, int num_classes, int out_h,
            int out_w, float scale_x, float scale_y, int img_w, int img_h, int top_k, float min_score,
            std::vector<Det>* out, std::vector<float>* s, std::vector<int>* idx) {
    out->clear();
    const int n = num_queries * num_classes;
    s->resize(n);
    idx->resize(n);
    for (int i = 0; i < n; ++i) (*s)[i] = 1.0f / (1.0f + std::exp(-logits[i]));
    std::iota(idx->begin(), idx->end(), 0);
    const int k = std::min(top_k, n);
    auto cmp = [&](int a, int b) { return (*s)[a] > (*s)[b] || ((*s)[a] == (*s)[b] && a < b); };
    std::partial_sort(idx->begin(), idx->begin() + k, idx->end(), cmp);
    for (int r = 0; r < k; ++r) {
        const int i = (*idx)[r];
        const float sc = (*s)[i];
        if (sc < min_score) break;
        const int q = i / num_classes, c = i % num_classes;
        const double cx = boxes[q * 4 + 0], cy = boxes[q * 4 + 1], w = boxes[q * 4 + 2], h = boxes[q * 4 + 3];
        double x0 = (cx - 0.5 * w) * out_w / scale_x, y0 = (cy - 0.5 * h) * out_h / scale_y;
        double x1 = (cx + 0.5 * w) * out_w / scale_x, y1 = (cy + 0.5 * h) * out_h / scale_y;
        x0 = std::min(std::max(x0, 0.0), static_cast<double>(img_w));
        x1 = std::min(std::max(x1, 0.0), static_cast<double>(img_w));
        y0 = std::min(std::max(y0, 0.0), static_cast<double>(img_h));
        y1 = std::min(std::max(y1, 0.0), static_cast<double>(img_h));
        if (x1 <= x0 || y1 <= y0) continue;
        out->push_back({c, x0, y0, x1 - x0, y1 - y0, sc});
    }
}

}  // namespace ird
