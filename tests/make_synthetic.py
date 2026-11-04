"""Make a tiny synthetic paired IR+visible dataset in the canonical format (for smoke tests only).

    python tests/make_synthetic.py --out /tmp/ird_syn

Frames are 640x512: IR 8-bit grey PNG (3 equal channels, like AIRST) with warm ship blobs, visible
RGB PNG of the same scene. Writes <out>/data/syn_{train,val,test}.json with paths relative to <out>.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

CATS = [{"id": 0, "name": "ship"}, {"id": 1, "name": "tank"}, {"id": 2, "name": "aircraft"},
        {"id": 3, "name": "civil_vehicle"}]


def frame(rng):
    H, W = 512, 640
    ir = rng.normal(70, 6, (H, W))
    vis = np.stack([rng.normal(40, 5, (H, W)), rng.normal(80, 5, (H, W)), rng.normal(110, 5, (H, W))], -1)
    boxes = []
    for _ in range(rng.integers(1, 6)):
        cls = 0 if rng.random() < 0.7 else 3
        w, h = (rng.integers(14, 70), rng.integers(8, 30)) if cls == 0 else (rng.integers(8, 20), rng.integers(8, 20))
        x, y = rng.integers(0, W - w), rng.integers(0, H - h)
        ir[y:y + h, x:x + w] += 90 if cls == 0 else 60
        vis[y:y + h, x:x + w] = [200, 200, 190] if cls == 0 else [180, 30, 30]
        boxes.append((cls, [float(x), float(y), float(w), float(h)]))
    ir = np.clip(ir, 0, 255).astype(np.uint8)
    return np.repeat(ir[..., None], 3, -1), np.clip(vis, 0, 255).astype(np.uint8), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs=3, default=[12, 4, 4], help="train val test")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(args.out, "img", "ir"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "img", "vis"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    fid, aid = 1, 1
    for split, n in zip(["train", "val", "test"], args.n):
        images, anns = [], []
        for _ in range(n):
            ir, vis, boxes = frame(rng)
            pir, pvis = f"img/ir/{fid:06d}.png", f"img/vis/{fid:06d}.png"
            Image.fromarray(ir).save(os.path.join(args.out, pir))
            Image.fromarray(vis).save(os.path.join(args.out, pvis))
            images.append({"id": fid, "file_name": pir, "files": {"ir": pir, "vis": pvis}, "width": 640,
                           "height": 512, "bit_depth": {"ir": 8, "vis": 8}, "split": split})
            for c, b in boxes:
                ign = int((b[2] * b[3]) ** 0.5 < 16)
                anns.append({"id": aid, "image_id": fid, "category_id": c, "bbox": b, "area": b[2] * b[3],
                             "iscrowd": ign, "ignore": ign})
                aid += 1
            fid += 1
        json.dump({"info": {"split": split, "synthetic": True}, "categories": CATS, "images": images,
                   "annotations": anns}, open(os.path.join(args.out, "data", f"syn_{split}.json"), "w"))
    print(f"wrote {fid - 1} synthetic frames to {args.out}")


if __name__ == "__main__":
    main()
