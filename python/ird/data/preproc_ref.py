"""NumPy reference of cpp/src/preproc.cpp (same spec, written independently).

Used for two things: tests check that the C++ module gives the same output, and it is the
fallback when the C++ module is not built (set IRD_REQUIRE_CPP=1 to forbid the fallback).
"""
import math

import numpy as np


def tone_range(img, bit_depth, tone="percentile", p_lo=0.5, p_hi=99.5):
    maxv = (1 << bit_depth) - 1
    if tone == "linear":
        return 0, maxv
    x = np.minimum(np.asarray(img).reshape(-1).astype(np.int64), maxv)
    n = x.size
    k_lo = int(math.floor(p_lo / 100.0 * (n - 1) + 0.5))
    k_hi = int(math.floor(p_hi / 100.0 * (n - 1) + 0.5))
    s = np.partition(x, (k_lo, k_hi))
    lo, hi = int(s[k_lo]), int(s[k_hi])
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _lut(bit_depth, lo, hi):
    maxv = (1 << bit_depth) - 1
    inv = np.float32(1.0) / np.float32(hi - lo)
    v = (np.arange(maxv + 1, dtype=np.int64) - lo).astype(np.float32) * inv
    return np.clip(v, np.float32(0), np.float32(1)).astype(np.float32)


def preprocess(img, bit_depth, out_channels, out_h=512, out_w=640, tone="percentile",
               p_lo=0.5, p_hi=99.5, pad_value=0.0, lo_hi=None):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    if C not in (1, 3) or out_channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    maxv = (1 << bit_depth) - 1
    lo, hi = lo_hi if lo_hi is not None else tone_range(img, bit_depth, tone, p_lo, p_hi)
    if hi <= lo:
        hi = lo + 1
    t = _lut(bit_depth, lo, hi)[np.minimum(img.astype(np.int64), maxv)]  # H W C float32

    if C == out_channels:
        planes = [t[:, :, c] for c in range(C)]
    elif C == 1:
        planes = [t[:, :, 0]] * out_channels
    else:
        planes = [np.float32(0.299) * t[:, :, 0] + np.float32(0.587) * t[:, :, 1]
                  + np.float32(0.114) * t[:, :, 2]]

    new_w, new_h = W, H
    if W > out_w or H > out_h:
        s = min(out_w / W, out_h / H)
        new_w = max(1, min(out_w, int(math.floor(W * s + 0.5))))
        new_h = max(1, min(out_h, int(math.floor(H * s + 0.5))))

    out = np.full((out_channels, out_h, out_w), np.float32(pad_value), dtype=np.float32)
    if (new_w, new_h) == (W, H):
        for oc in range(out_channels):
            out[oc, :H, :W] = planes[oc]
    else:
        rx = np.float32(W) / np.float32(new_w)
        ry = np.float32(H) / np.float32(new_h)
        sy = np.clip((np.arange(new_h, dtype=np.float32) + np.float32(0.5)) * ry - np.float32(0.5),
                     np.float32(0), np.float32(H - 1)).astype(np.float32)
        sx = np.clip((np.arange(new_w, dtype=np.float32) + np.float32(0.5)) * rx - np.float32(0.5),
                     np.float32(0), np.float32(W - 1)).astype(np.float32)
        y0 = np.floor(sy).astype(np.int64); y1 = np.minimum(y0 + 1, H - 1)
        x0 = np.floor(sx).astype(np.int64); x1 = np.minimum(x0 + 1, W - 1)
        fy = (sy - y0.astype(np.float32))[:, None]
        fx = (sx - x0.astype(np.float32))[None, :]
        one = np.float32(1)
        for oc in range(out_channels):
            P = planes[oc]
            a, b = P[y0][:, x0], P[y0][:, x1]
            c, d = P[y1][:, x0], P[y1][:, x1]
            top = (one - fx) * a + fx * b
            bot = (one - fx) * c + fx * d
            out[oc, :new_h, :new_w] = (one - fy) * top + fy * bot
    info = {"scale_x": float(np.float32(new_w) / np.float32(W)),
            "scale_y": float(np.float32(new_h) / np.float32(H)),
            "new_w": new_w, "new_h": new_h, "lo": lo, "hi": hi}
    return out, info
