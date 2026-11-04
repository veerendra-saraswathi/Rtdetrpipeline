"""C++ preprocessing (ird_preproc) must match the NumPy reference.

Run: PYTHONPATH=python python -m pytest tests/test_preproc.py   (or python tests/test_preproc.py)
Exact equality without resizing; at most 1e-6 with resizing (float order of operations).
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ird.data import preproc_ref  # noqa: E402

import ird_preproc  # noqa: E402

rng = np.random.default_rng(0)


def frame(h, w, c, bd):
    maxv = (1 << bd) - 1
    base = rng.normal(maxv * 0.3, maxv * 0.05, size=(h, w, c))
    base[100:120, 200:230] += maxv * 0.5            # a hot target
    a = np.clip(base, 0, maxv).astype(np.uint16)
    return a[:, :, 0] if c == 1 else a


CASES = [
    # (h, w, in_c, bit_depth, out_c, tone, expect_exact)
    (512, 640, 1, 14, 3, "percentile", True),    # RCI IR
    (480, 640, 1, 14, 3, "percentile", True),    # RCI IR, 480 rows -> padded
    (512, 640, 1, 14, 1, "percentile", True),    # fusion IR plane
    (512, 640, 1, 8, 3, "percentile", True),     # AIRST IR
    (512, 640, 3, 8, 3, "linear", True),         # AIRST visible
    (512, 640, 3, 8, 1, "linear", True),         # visible to grey
    (800, 1280, 3, 12, 3, "linear", False),      # RCI visible, downscaled
    (800, 1280, 3, 10, 3, "percentile", False),
    (300, 900, 1, 14, 3, "percentile", False),   # wide frame
]


def check(h, w, c, bd, oc, tone, exact):
    img = frame(h, w, c, bd)
    x_cpp, i_cpp = ird_preproc.preprocess(img, bd, oc, 512, 640, tone, 0.5, 99.5, 0.0)
    x_ref, i_ref = preproc_ref.preprocess(img, bd, oc, 512, 640, tone, 0.5, 99.5, 0.0)
    assert x_cpp.shape == x_ref.shape == (oc, 512, 640)
    assert (i_cpp["lo"], i_cpp["hi"]) == (i_ref["lo"], i_ref["hi"]), (i_cpp, i_ref)
    assert (i_cpp["new_w"], i_cpp["new_h"]) == (i_ref["new_w"], i_ref["new_h"])
    d = float(np.abs(x_cpp - x_ref).max())
    if exact:
        assert d == 0.0, f"max diff {d}"
    else:
        assert d <= 1e-6, f"max diff {d}"
    assert x_cpp.min() >= 0.0 and x_cpp.max() <= 1.0
    return d


def test_cases():
    for case in CASES:
        check(*case)


def test_percentile_matches_numpy_sort():
    for bd, n in itertools.product([8, 14, 16], [1, 7, 1000, 327680]):
        a = rng.integers(0, 1 << bd, size=n).astype(np.uint16).reshape(1, n)
        lo, hi = ird_preproc.tone_range(a, bd, "percentile", 0.5, 99.5)
        s = np.sort(a.reshape(-1))
        k = lambda p: int(np.floor(p / 100 * (n - 1) + 0.5))
        exp_lo, exp_hi = int(s[k(0.5)]), int(s[k(99.5)])
        if exp_hi <= exp_lo:
            exp_hi = exp_lo + 1
        assert (lo, hi) == (exp_lo, exp_hi)


def test_constant_frame_and_overflow():
    a = np.full((512, 640), 1000, np.uint16)
    x, info = ird_preproc.preprocess(a, 14, 3)
    assert info["hi"] == info["lo"] + 1 and np.all(x[:, :512, :640] == 0.0)
    b = np.full((512, 640), 65535, np.uint16)            # above 14-bit max -> clamped
    x, _ = ird_preproc.preprocess(b, 14, 3, tone="linear")
    assert np.all(x == 1.0)


def test_padding_region():
    a = frame(480, 640, 1, 14)
    x, _ = ird_preproc.preprocess(a, 14, 3, pad_value=0.0)
    assert np.all(x[:, 480:, :] == 0.0)


if __name__ == "__main__":
    for case in CASES:
        print(case, "max diff", check(*case))
    test_percentile_matches_numpy_sort()
    test_constant_frame_and_overflow()
    test_padding_region()
    print("all preprocessing tests passed")
