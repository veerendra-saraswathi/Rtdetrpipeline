// Golden-file test: the C++ preprocessing on the target (e.g. Jetson, ARM) must reproduce the
// tensors written by tests/make_golden.py (NumPy reference) on the training machine.
//
//   python tests/make_golden.py --out golden/      (training machine)
//   ./build/test_preproc_golden golden/            (Jetson)
//
// golden/cases.txt: one case per line:
//   name  width  height  channels  bit_depth  out_channels  out_h  out_w  tone  p_lo  p_hi  pad  tol
// golden/<name>.u16: input (uint16 LE, HWC). golden/<name>.f32: expected output (float32 LE, CHW).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ird/preproc.h"

template <typename T>
bool read_all(const std::string& path, std::vector<T>* v, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    v->resize(n);
    f.read(reinterpret_cast<char*>(v->data()), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<size_t>(f.gcount()) == n * sizeof(T);
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "golden";
    std::ifstream cases(dir + "/cases.txt");
    if (!cases) { fprintf(stderr, "cannot open %s/cases.txt\n", dir.c_str()); return 2; }
    std::string line;
    int n_cases = 0, n_fail = 0;
    while (std::getline(cases, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string name, tone;
        int w, h, c, bd, oc, oh, ow;
        double plo, phi, tol;
        float pad;
        ss >> name >> w >> h >> c >> bd >> oc >> oh >> ow >> tone >> plo >> phi >> pad >> tol;
        std::vector<uint16_t> in;
        std::vector<float> exp, got(static_cast<size_t>(oc) * oh * ow);
        if (!read_all(dir + "/" + name + ".u16", &in, static_cast<size_t>(w) * h * c) ||
            !read_all(dir + "/" + name + ".f32", &exp, got.size())) {
            fprintf(stderr, "%s: cannot read files\n", name.c_str());
            ++n_fail; ++n_cases;
            continue;
        }
        ird::ImageU16View v;
        v.data = in.data(); v.width = w; v.height = h; v.channels = c; v.bit_depth = bd;
        ird::PreprocParams p;
        p.out_h = oh; p.out_w = ow; p.p_lo = plo; p.p_hi = phi; p.pad_value = pad;
        p.tone = tone == "linear" ? ird::Tone::Linear : ird::Tone::Percentile;
        ird::PreprocInfo info;
        if (ird::preprocess(v, p, oc, got.data(), &info) != ird::Status::Ok) {
            fprintf(stderr, "%s: preprocess failed\n", name.c_str());
            ++n_fail; ++n_cases;
            continue;
        }
        double md = 0;
        for (size_t i = 0; i < got.size(); ++i) md = std::max(md, static_cast<double>(std::fabs(got[i] - exp[i])));
        const bool ok = md <= tol;
        printf("%-28s max|diff| %.3g (tol %.3g) %s\n", name.c_str(), md, tol, ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
        ++n_cases;
    }
    printf("%d cases, %d failed\n", n_cases, n_fail);
    return n_fail == 0 && n_cases > 0 ? 0 : 1;
}
