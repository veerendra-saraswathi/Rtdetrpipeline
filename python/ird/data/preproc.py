"""Preprocessing entry point for Python code (training, export checks, PyTorch prediction).

Uses the C++ module `ird_preproc` (the same code as the Jetson runtime). If it is not built,
falls back to the NumPy reference with a warning. Set IRD_REQUIRE_CPP=1 to make that an error
(recommended for real training runs).
"""
import os
import warnings
from dataclasses import dataclass, asdict

import numpy as np

from . import preproc_ref

try:
    import ird_preproc as _cpp  # built by `pip install ./` at the repo root
except ImportError:  # pragma: no cover
    _cpp = None

_warned = False


def backend():
    return "cpp" if _cpp is not None else "numpy"


@dataclass
class BandParams:
    """Preprocessing settings for one band. Stored in every config and in the runtime config."""
    tone: str = "percentile"      # 'percentile' (IR default) or 'linear' (visible default)
    p_lo: float = 0.5
    p_hi: float = 99.5
    pad_value: float = 0.0

    def to_dict(self):
        return asdict(self)


DEFAULT_IR = BandParams("percentile", 0.5, 99.5, 0.0)
DEFAULT_VIS = BandParams("linear", 0.5, 99.5, 0.0)


def preprocess(img, bit_depth, out_channels, out_hw=(512, 640), params=DEFAULT_IR):
    """img: HxW or HxWxC uint8/uint16 array. Returns (float32 CxHxW, info dict)."""
    global _warned
    img = np.ascontiguousarray(img, dtype=np.uint16)
    kw = dict(out_h=int(out_hw[0]), out_w=int(out_hw[1]), tone=params.tone,
              p_lo=float(params.p_lo), p_hi=float(params.p_hi), pad_value=float(params.pad_value))
    if _cpp is not None:
        return _cpp.preprocess(img, int(bit_depth), int(out_channels), **kw)
    if os.environ.get("IRD_REQUIRE_CPP") == "1":
        raise RuntimeError("ird_preproc C++ module not found; build it with `pip install ./` "
                           "at the repo root, or unset IRD_REQUIRE_CPP")
    if not _warned:
        warnings.warn("ird_preproc C++ module not found: using the NumPy reference (same math)")
        _warned = True
    return preproc_ref.preprocess(img, int(bit_depth), int(out_channels), **kw)


def build_input(images, modality, out_hw, ir_params=DEFAULT_IR, vis_params=DEFAULT_VIS):
    """Build the network input for one frame.

    images: dict with 'ir' and/or 'vis' -> (array, bit_depth)
    modality: 'ir'     -> 3 channels, grey IR copied 3 times
              'vis'    -> 3 channels RGB
              'fusion' -> 4 channels [R, G, B, IR]
    Returns (float32 CxHxW, info of the first band used).
    """
    if modality == "ir":
        a, bd = images["ir"]
        return preprocess(a, bd, 3, out_hw, ir_params)
    if modality == "vis":
        a, bd = images["vis"]
        return preprocess(a, bd, 3, out_hw, vis_params)
    if modality == "fusion":
        v, vbd = images["vis"]
        r, rbd = images["ir"]
        x_vis, info = preprocess(v, vbd, 3, out_hw, vis_params)
        x_ir, info_ir = preprocess(r, rbd, 1, out_hw, ir_params)
        if (info["new_w"], info["new_h"]) != (info_ir["new_w"], info_ir["new_h"]):
            raise ValueError("fusion needs IR and visible frames of the same size")
        return np.concatenate([x_vis, x_ir], axis=0), info
    raise ValueError(f"unknown modality {modality!r}")


def in_channels(modality):
    return 4 if modality == "fusion" else 3
