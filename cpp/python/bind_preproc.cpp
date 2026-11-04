// Python module `ird_preproc`: exposes the C++ preprocessing used by the Jetson runtime,
// so training uses the same code.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "ird/preproc.h"

namespace py = pybind11;

namespace {

ird::Tone parse_tone(const std::string& t) {
    if (t == "linear") return ird::Tone::Linear;
    if (t == "percentile") return ird::Tone::Percentile;
    throw std::invalid_argument("tone must be 'linear' or 'percentile'");
}

ird::ImageU16View view_of(const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>& a,
                          int bit_depth) {
    if (a.ndim() != 2 && a.ndim() != 3) throw std::invalid_argument("image must be HxW or HxWxC");
    ird::ImageU16View v;
    v.data = a.data();
    v.height = static_cast<int>(a.shape(0));
    v.width = static_cast<int>(a.shape(1));
    v.channels = a.ndim() == 3 ? static_cast<int>(a.shape(2)) : 1;
    v.bit_depth = bit_depth;
    return v;
}

}  // namespace

PYBIND11_MODULE(ird_preproc, m) {
    m.doc() = "IRD preprocessing (C++), shared by training and the Jetson runtime";
    m.def("version", &ird::version);

    m.def(
        "tone_range",
        [](py::array_t<uint16_t, py::array::c_style | py::array::forcecast> img, int bit_depth,
           const std::string& tone, double p_lo, double p_hi) {
            ird::PreprocParams p;
            p.tone = parse_tone(tone);
            p.p_lo = p_lo;
            p.p_hi = p_hi;
            int lo = 0, hi = 0;
            if (ird::tone_range(view_of(img, bit_depth), p, &lo, &hi) != ird::Status::Ok)
                throw std::invalid_argument("tone_range: bad argument");
            return py::make_tuple(lo, hi);
        },
        py::arg("img"), py::arg("bit_depth"), py::arg("tone") = "percentile", py::arg("p_lo") = 0.5,
        py::arg("p_hi") = 99.5);

    m.def(
        "preprocess",
        [](py::array_t<uint16_t, py::array::c_style | py::array::forcecast> img, int bit_depth,
           int out_channels, int out_h, int out_w, const std::string& tone, double p_lo, double p_hi,
           float pad_value) {
            ird::PreprocParams p;
            p.out_h = out_h;
            p.out_w = out_w;
            p.tone = parse_tone(tone);
            p.p_lo = p_lo;
            p.p_hi = p_hi;
            p.pad_value = pad_value;
            py::array_t<float> out({out_channels, out_h, out_w});
            ird::PreprocInfo info;
            ird::ImageU16View v = view_of(img, bit_depth);
            ird::Status s;
            {
                py::gil_scoped_release release;
                s = ird::preprocess(v, p, out_channels, out.mutable_data(), &info);
            }
            if (s != ird::Status::Ok) throw std::invalid_argument("preprocess: bad argument");
            py::dict d;
            d["scale_x"] = info.scale_x;
            d["scale_y"] = info.scale_y;
            d["new_w"] = info.new_w;
            d["new_h"] = info.new_h;
            d["lo"] = info.lo;
            d["hi"] = info.hi;
            return py::make_tuple(out, d);
        },
        py::arg("img"), py::arg("bit_depth"), py::arg("out_channels"), py::arg("out_h") = 512,
        py::arg("out_w") = 640, py::arg("tone") = "percentile", py::arg("p_lo") = 0.5,
        py::arg("p_hi") = 99.5, py::arg("pad_value") = 0.0f);
}
