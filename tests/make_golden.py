"""Write golden preprocessing cases for cpp/tests/test_preproc_golden.cpp.

    PYTHONPATH=python python tests/make_golden.py --out golden [--ann data/airst_v0.1s/airst_test.json --data-root ..]

Expected outputs come from the NumPy reference (python/ird/data/preproc_ref.py), so the C++
build on any machine is checked against an independent implementation. With --ann, the first
3 real frames of each band are added.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ird.data import preproc_ref  # noqa: E402
from ird.data.io import read_band  # noqa: E402


def synthetic(h, w, c, bd, seed):
    rng = np.random.default_rng(seed)
    maxv = (1 << bd) - 1
    a = rng.normal(maxv * 0.3, maxv * 0.05, size=(h, w, c))
    a[h // 4: h // 4 + 20, w // 3: w // 3 + 30] += maxv * 0.5
    return np.clip(a, 0, maxv).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="golden")
    ap.add_argument("--ann")
    ap.add_argument("--data-root", default="..")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = [  # name, img, bit_depth, out_c, tone, tol
        ("syn_ir14_512", synthetic(512, 640, 1, 14, 1), 14, 3, "percentile", 0.0),
        ("syn_ir14_480", synthetic(480, 640, 1, 14, 2), 14, 3, "percentile", 0.0),
        ("syn_ir14_plane", synthetic(512, 640, 1, 14, 3), 14, 1, "percentile", 0.0),
        ("syn_vis12_1280", synthetic(800, 1280, 3, 12, 4), 12, 3, "linear", 1e-6),
        ("syn_vis8", synthetic(512, 640, 3, 8, 5), 8, 3, "linear", 0.0),
    ]
    if args.ann:
        doc = json.load(open(args.ann))
        for k, e in enumerate(doc["images"][:3]):
            a, bd = read_band(args.data_root, e, "ir")
            cases.append((f"real_ir_{e['id']}", a[..., None], bd, 3, "percentile", 0.0))
            if e["files"].get("vis"):
                v, vbd = read_band(args.data_root, e, "vis")
                cases.append((f"real_vis_{e['id']}", v, vbd, 3, "linear", 0.0))
    lines = ["# name width height channels bit_depth out_channels out_h out_w tone p_lo p_hi pad tol"]
    for name, img, bd, oc, tone, tol in cases:
        img = img if img.ndim == 3 else img[..., None]
        h, w, c = img.shape
        out, _ = preproc_ref.preprocess(img, bd, oc, 512, 640, tone, 0.5, 99.5, 0.0)
        img.astype("<u2").tofile(os.path.join(args.out, name + ".u16"))
        out.astype("<f4").tofile(os.path.join(args.out, name + ".f32"))
        lines.append(f"{name} {w} {h} {c} {bd} {oc} 512 640 {tone} 0.5 99.5 0.0 {tol:g}")
    open(os.path.join(args.out, "cases.txt"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
