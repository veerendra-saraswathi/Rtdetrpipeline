// Runtime config written by python/ird/export/export_onnx.py (<model>.onnx.runtime.txt).
#pragma once
#include <map>
#include <string>
#include <vector>

#include "ird/preproc.h"

namespace ird {

struct RuntimeConfig {
    std::string arm, modality = "ir";
    int in_channels = 3, out_h = 512, out_w = 640, num_classes = 4, num_queries = 300;
    std::vector<std::string> class_names;
    PreprocParams ir, vis;
    std::string input_name = "images", logits_name = "logits", boxes_name = "boxes";
};

// Returns false and sets err on failure.
bool load_runtime_config(const std::string& path, RuntimeConfig* cfg, std::string* err);

}  // namespace ird
