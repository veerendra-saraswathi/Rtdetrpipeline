"""Decode raw RT-DETR outputs into detections. Mirrors cpp/runtime/decoder.cpp exactly.

Inputs for one image:
  logits (Q, C)  raw class logits      boxes (Q, 4)  cx, cy, w, h normalised to the canvas
Steps (same as RTDETRPostProcessor with focal loss, plus clipping and a score floor):
  1. scores = sigmoid(logits), flattened to Q*C (index = q*C + c).
  2. Order by score descending; ties by lower index (stable). Keep the first top_k.
  3. Keep scores >= min_score.
  4. Box: x0 = (cx - w/2) * out_w, ... ; then divide by scale_x / scale_y (back to original
     pixels; the image sits at the canvas top-left) and clip to the original image.
Output: list of (class_id, x, y, w, h, score), x/y/w/h in original pixels.
"""
import numpy as np


def sigmoid(x):
    x = x.astype(np.float32)
    return (np.float32(1) / (np.float32(1) + np.exp(-x))).astype(np.float32)


def decode(logits, boxes, out_hw, info, img_w, img_h, top_k=300, min_score=0.0):
    Q, C = logits.shape
    s = sigmoid(logits).reshape(-1)
    order = np.argsort(-s, kind="stable")[: min(top_k, Q * C)]
    out_h, out_w = out_hw
    dets = []
    for i in order:
        sc = float(s[i])
        if sc < min_score:
            break
        q, c = divmod(int(i), C)
        cx, cy, w, h = [float(v) for v in boxes[q]]
        x0 = (cx - 0.5 * w) * out_w / info["scale_x"]
        y0 = (cy - 0.5 * h) * out_h / info["scale_y"]
        x1 = (cx + 0.5 * w) * out_w / info["scale_x"]
        y1 = (cy + 0.5 * h) * out_h / info["scale_y"]
        x0, x1 = min(max(x0, 0.0), img_w), min(max(x1, 0.0), img_w)
        y0, y1 = min(max(y0, 0.0), img_h), min(max(y1, 0.0), img_h)
        if x1 <= x0 or y1 <= y0:
            continue
        dets.append((c, x0, y0, x1 - x0, y1 - y0, sc))
    return dets
