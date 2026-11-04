#include "runtime_config.h"

#include <fstream>
#include <sstream>

namespace ird {

namespace {
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
bool parse_tone(const std::string& v, Tone* t) {
    if (v == "linear") { *t = Tone::Linear; return true; }
    if (v == "percentile") { *t = Tone::Percentile; return true; }
    return false;
}
}  // namespace

bool load_runtime_config(const std::string& path, RuntimeConfig* cfg, std::string* err) {
    std::ifstream f(path);
    if (!f) { *err = "cannot open runtime config " + path; return false; }
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    try {
        auto get = [&](const char* k, const std::string& d) { auto it = kv.find(k); return it == kv.end() ? d : it->second; };
        cfg->arm = get("arm", "");
        cfg->modality = get("modality", "ir");
        cfg->in_channels = std::stoi(get("in_channels", "3"));
        cfg->out_h = std::stoi(get("out_h", "512"));
        cfg->out_w = std::stoi(get("out_w", "640"));
        cfg->num_classes = std::stoi(get("num_classes", "4"));
        cfg->num_queries = std::stoi(get("num_queries", "300"));
        cfg->input_name = get("input_name", "images");
        cfg->logits_name = get("logits_name", "logits");
        cfg->boxes_name = get("boxes_name", "boxes");
        cfg->class_names.clear();
        std::stringstream ss(get("class_names", ""));
        std::string item;
        while (std::getline(ss, item, ',')) cfg->class_names.push_back(trim(item));
        for (auto* band : {"ir", "vis"}) {
            PreprocParams& p = std::string(band) == "ir" ? cfg->ir : cfg->vis;
            std::string b(band);
            p.out_h = cfg->out_h;
            p.out_w = cfg->out_w;
            if (!parse_tone(get((b + "_tone").c_str(), b == "ir" ? "percentile" : "linear"), &p.tone)) {
                *err = "bad " + b + "_tone";
                return false;
            }
            p.p_lo = std::stod(get((b + "_p_lo").c_str(), "0.5"));
            p.p_hi = std::stod(get((b + "_p_hi").c_str(), "99.5"));
            p.pad_value = std::stof(get((b + "_pad_value").c_str(), "0"));
        }
    } catch (const std::exception& e) {
        *err = std::string("bad value in runtime config: ") + e.what();
        return false;
    }
    if (cfg->modality != "ir" && cfg->modality != "vis" && cfg->modality != "fusion") {
        *err = "unknown modality " + cfg->modality;
        return false;
    }
    if ((cfg->modality == "fusion") != (cfg->in_channels == 4)) {
        *err = "modality and in_channels disagree";
        return false;
    }
    return true;
}

}  // namespace ird
