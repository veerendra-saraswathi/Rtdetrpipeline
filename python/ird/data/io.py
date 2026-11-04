"""Image reading for the canonical annotation format.

Image entries carry files = {"ir": path, "vis": path} relative to the dataset root and
bit_depth = {"ir": b, "vis": b}. Supported files:
  .png/.tif/.jpg  read with PIL (8- or 16-bit). An IR file stored as 3 equal channels is reduced to 1.
  .u16            raw uint16 little-endian, row-major, no header (RCI raw convention);
                  needs width, height in the image entry and channels (default 1 for IR, 3 for vis).
Returns (uint16 array HxW or HxWxC, bit_depth).
"""
import os

import numpy as np
from PIL import Image


def read_band(root, entry, band):
    rel = entry["files"].get(band)
    if not rel:
        raise FileNotFoundError(f"image {entry['id']} has no {band} file")
    path = os.path.join(root, rel)
    bd = (entry.get("bit_depth") or {}).get(band) or 8
    if path.lower().endswith(".u16"):
        ch = (entry.get("channels") or {}).get(band, 1 if band == "ir" else 3)
        a = np.fromfile(path, dtype="<u2")
        a = a.reshape(entry["height"], entry["width"], ch) if ch > 1 else a.reshape(entry["height"], entry["width"])
        return a, bd
    with Image.open(path) as im:
        a = np.asarray(im)
    if band == "ir" and a.ndim == 3:
        a = a[..., 0]                      # grey stored as R=G=B (checked by prepare_airst)
    if band == "vis" and a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if band == "vis" and a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.dtype == np.uint8:
        bd = min(bd, 8)
    return a.astype(np.uint16, copy=False), bd


def read_frame(root, entry, modality):
    """Bands needed for a modality: ir -> {ir}, vis -> {vis}, fusion -> {ir, vis}."""
    bands = {"ir": ["ir"], "vis": ["vis"], "fusion": ["ir", "vis"]}[modality]
    return {b: read_band(root, entry, b) for b in bands}
