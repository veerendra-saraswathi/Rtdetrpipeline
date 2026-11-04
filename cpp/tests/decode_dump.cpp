// Helper for tests/test_decode_parity.py: runs ird::decode on raw outputs from a file and prints
// detections, so the Python and C++ decoders can be compared.
//   decode_dump in.bin Q C out_h out_w scale_x scale_y img_w img_h top_k min_score
// in.bin: float32 logits (Q*C) then float32 boxes (Q*4).
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "decoder.h"

int main(int argc, char** argv) {
    if (argc != 13) { fprintf(stderr, "bad args\n"); return 2; }
    const int Q = std::stoi(argv[2]), C = std::stoi(argv[3]), oh = std::stoi(argv[4]), ow = std::stoi(argv[5]);
    const float sx = std::stof(argv[6]), sy = std::stof(argv[7]);
    const int iw = std::stoi(argv[8]), ih = std::stoi(argv[9]), k = std::stoi(argv[10]);
    const float ms = std::stof(argv[11]);
    (void)argv[12];
    std::vector<float> buf(static_cast<size_t>(Q) * C + static_cast<size_t>(Q) * 4);
    std::ifstream f(argv[1], std::ios::binary);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * 4));
    std::vector<ird::Det> out;
    std::vector<float> s;
    std::vector<int> idx;
    ird::decode(buf.data(), buf.data() + static_cast<size_t>(Q) * C, Q, C, oh, ow, sx, sy, iw, ih, k, ms, &out, &s, &idx);
    for (const auto& d : out) printf("%d %.6f %.6f %.6f %.6f %.9g\n", d.class_id, d.x, d.y, d.w, d.h, d.score);
    return 0;
}
