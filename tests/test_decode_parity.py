"""C++ decoder (cpp/runtime/decoder.cpp) must give the same detections as python/ird/infer/decode.py.

    g++ -std=c++17 -O2 -ffp-contract=off -Icpp/runtime cpp/tests/decode_dump.cpp cpp/runtime/decoder.cpp -o /tmp/decode_dump
    PYTHONPATH=python python tests/test_decode_parity.py /tmp/decode_dump
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ird.infer.decode import decode  # noqa: E402


def run(exe, seed, scale=1.0, img=(640, 512), min_score=0.001):
    rng = np.random.default_rng(seed)
    Q, C = 300, 4
    logits = rng.normal(-4, 2, size=(Q, C)).astype(np.float32)
    logits[:5] = rng.normal(3, 1, size=(5, C)).astype(np.float32)
    logits[10, 1] = logits[11, 1]                      # a tie
    boxes = rng.uniform(0, 1, size=(Q, 4)).astype(np.float32)
    boxes[:, 2:] *= 0.2
    info = {"scale_x": scale, "scale_y": scale}
    ref = decode(logits, boxes, (512, 640), info, img[0], img[1], 300, min_score)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(logits.tobytes() + boxes.tobytes())
    out = subprocess.check_output([exe, fh.name, "300", "4", "512", "640", repr(np.float32(scale).item()),
                                   repr(np.float32(scale).item()), str(img[0]), str(img[1]), "300",
                                   repr(min_score), "x"], text=True)
    os.unlink(fh.name)
    got = [l.split() for l in out.strip().splitlines()]
    assert len(got) == len(ref), (len(got), len(ref))
    for g, r in zip(got, ref):
        assert int(g[0]) == r[0]
        assert np.allclose([float(v) for v in g[1:5]], r[1:5], atol=1e-4), (g, r)
        assert abs(float(g[5]) - r[5]) <= 1e-6, (g, r)
    return len(ref)


if __name__ == "__main__":
    exe = sys.argv[1]
    for seed in range(5):
        n = run(exe, seed)
        n2 = run(exe, seed + 10, scale=0.5, img=(1280, 800))
        print(f"seed {seed}: {n} and {n2} detections identical")
    print("decoder parity passed")
