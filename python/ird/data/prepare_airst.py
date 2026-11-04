"""Build canonical annotation files for the AIRST-CS ship subset (dataset version DS-v0.1s).

Run from anywhere:
    python -m ird.data.prepare_airst --root "<dataset root>" --out data/airst_v0.1s

<dataset root> is the folder that contains AIRST-CS_ship_only/.

What it does
- Reads AIRST-CS_ship_only/manifest.csv and duplicate_pairs.txt.
- Keeps the official split. Resolves the 174 duplicate pairs (same IR pixels, different labels):
  keeps one copy per pair and puts it in the stricter split (test > val > train), so no copy
  leaks across splits. The dropped copy and both label sets go to duplicates_e10.json (E-10).
- Maps classes to the master list: source 1 (ship) -> 0 ship, source 0 (vehicle) -> 3 civil_vehicle.
- Converts YOLO boxes to pixel [x, y, w, h]. Boxes with sqrt(w*h) < min_size get ignore=1
  (and iscrowd=1, so COCO tools also ignore them). They stay in training targets.
- Writes airst_<split>.json (COCO style plus our fields), duplicates_e10.json, stats.json, stats.md.

Image entries carry both band files: files = {"ir": path, "vis": path or null}. IR and visible
labels are identical in AIRST-CS (checked here), so one annotation set serves both bands.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

CATEGORIES = [
    {"id": 0, "name": "ship"},
    {"id": 1, "name": "tank"},
    {"id": 2, "name": "aircraft"},
    {"id": 3, "name": "civil_vehicle"},
]
SRC_TO_MASTER = {1: 0, 0: 3}          # AIRST source id -> master class id
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
SUBSET = "AIRST-CS_ship_only"
SIZE_BUCKETS = [(0, 16), (16, 32), (32, 96), (96, 1e9)]


def rel(*parts):
    return "/".join(parts)


def read_yolo(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if len(p) != 5:
                raise ValueError(f"bad line in {path}: {line!r}")
            rows.append((int(float(p[0])), *map(float, p[1:])))
    return rows


def find_up_dir(root):
    for cand in [os.path.join(root, SUBSET, "up"), os.path.join(root, SUBSET)]:
        if os.path.isdir(os.path.join(cand, "ship_only")):
            return cand
    sys.exit(f"cannot find ship_only/ under {root}/{SUBSET}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root that contains AIRST-CS_ship_only/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=float, default=16.0, help="sqrt(w*h) below this -> ignore")
    ap.add_argument("--version", default="DS-v0.1s")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    up = find_up_dir(root)
    up_rel = os.path.relpath(up, root).replace(os.sep, "/")
    sub = os.path.join(root, SUBSET)

    manifest = {r["frame_id"]: r for r in csv.DictReader(open(os.path.join(sub, "manifest.csv")))}
    pairs = []
    for line in open(os.path.join(sub, "duplicate_pairs.txt")):
        if line.startswith("#") or "=" not in line:
            continue
        a, b = [s.strip() for s in line.split("=")]
        pairs.append((a.split(":")[1], b.split(":")[1]))

    # ---- load every frame -------------------------------------------------
    frames = {}
    label_mismatch = []
    for fid, m in sorted(manifest.items()):
        g = m["group"]
        ir_path = rel(up_rel, g, "images", "infrared", f"{fid}.png")
        vis_path = rel(up_rel, g, "images", "visible", f"{fid}.png")
        has_vis = os.path.exists(os.path.join(root, vis_path))
        lab_ir = os.path.join(up, g, "labels", "infrared", f"{fid}.txt")
        lab_vis = os.path.join(up, g, "labels", "visible", f"{fid}.txt")
        yolo = read_yolo(lab_ir)
        if os.path.exists(lab_vis) and open(lab_vis).read() != open(lab_ir).read():
            label_mismatch.append(fid)
        im = Image.open(os.path.join(root, ir_path))
        w, h = im.size
        px = np.asarray(im)
        if px.ndim == 3:
            if not (px[..., 0] == px[..., 1]).all() or not (px[..., 0] == px[..., 2]).all():
                sys.exit(f"IR frame {fid} is not grey (R != G != B)")
            px = px[..., 0]
        bit_depth = 16 if px.dtype == np.uint16 else 8
        frames[fid] = dict(
            fid=fid, group=g, src_split=m["original_split"], split=m["original_split"],
            files={"ir": ir_path, "vis": vis_path if has_vis else None},
            width=w, height=h, bit_depth={"ir": bit_depth, "vis": 8 if has_vis else None},
            content_hash=hashlib.md5(px.tobytes()).hexdigest(), yolo=yolo,
        )

    # ---- resolve duplicate pairs -----------------------------------------
    by_hash = defaultdict(list)
    for f in frames.values():
        by_hash[f["content_hash"]].append(f["fid"])
    hash_pairs = {tuple(sorted(v)) for v in by_hash.values() if len(v) > 1}
    listed = {tuple(sorted(p)) for p in pairs}
    if hash_pairs != listed:
        print(f"WARNING: duplicate_pairs.txt ({len(listed)}) differs from pixel-hash pairs ({len(hash_pairs)})")
    dropped, e10 = set(), []
    for a, b in sorted(hash_pairs | listed):
        fa, fb = frames[a], frames[b]

        def key(f):  # more boxes, then more classes, then lower id
            return (len(f["yolo"]), len({r[0] for r in f["yolo"]}), -int(f["fid"]))
        keep, drop = (fa, fb) if key(fa) >= key(fb) else (fb, fa)
        new_split = max(fa["split"], fb["split"], key=SPLIT_RANK.get)
        keep["split"] = new_split
        keep["duplicate_of"] = drop["fid"]
        keep["dup_resolution"] = "kept"
        dropped.add(drop["fid"])
        e10.append({
            "kept": keep["fid"], "dropped": drop["fid"], "split": new_split,
            "kept_src_split": keep["src_split"], "dropped_src_split": drop["src_split"],
            "width": keep["width"], "height": keep["height"],
            "boxes_kept": yolo_to_boxes(keep["yolo"], keep["width"], keep["height"]),
            "boxes_dropped": yolo_to_boxes(drop["yolo"], drop["width"], drop["height"]),
        })

    # ---- write split files -------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    stats = {"version": args.version, "root_relative_paths": True, "min_size": args.min_size,
             "label_mismatch_ir_vis": label_mismatch, "duplicates_dropped": len(dropped),
             "splits": {}}
    ann_id = 1
    for split in ["train", "val", "test"]:
        images, anns = [], []
        for fid in sorted(frames):
            f = frames[fid]
            if fid in dropped or f["split"] != split:
                continue
            images.append({
                "id": int(fid), "file_name": f["files"]["ir"], "files": f["files"],
                "width": f["width"], "height": f["height"], "bit_depth": f["bit_depth"],
                "split": split, "src_split": f["src_split"], "group": f["group"],
                "source": "AIRST-CS", "content_hash": f["content_hash"],
                "duplicate_of": f.get("duplicate_of"),
            })
            for b in yolo_to_boxes(f["yolo"], f["width"], f["height"]):
                ign = int(math.sqrt(b["bbox"][2] * b["bbox"][3]) < args.min_size)
                anns.append({"id": ann_id, "image_id": int(fid), "category_id": b["category_id"],
                             "bbox": b["bbox"], "area": round(b["bbox"][2] * b["bbox"][3], 3),
                             "iscrowd": ign, "ignore": ign})
                ann_id += 1
        doc = {"info": {"dataset": "AIRST-CS ship subset", "version": args.version,
                        "license": "CC BY 4.0", "doi": "10.17632/xv8t5byb5y.1",
                        "split": split, "min_size": args.min_size},
               "categories": CATEGORIES, "images": images, "annotations": anns}
        with open(os.path.join(args.out, f"airst_{split}.json"), "w") as fh:
            json.dump(doc, fh)
        stats["splits"][split] = split_stats(images, anns)

    with open(os.path.join(args.out, "duplicates_e10.json"), "w") as fh:
        json.dump({"min_size": args.min_size, "pairs": e10}, fh)
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    with open(os.path.join(args.out, "stats.md"), "w") as fh:
        fh.write(stats_md(stats))
    print(stats_md(stats))


def yolo_to_boxes(yolo, W, H):
    out = []
    for c, cx, cy, w, h in yolo:
        x0 = max(0.0, (cx - w / 2) * W); y0 = max(0.0, (cy - h / 2) * H)
        x1 = min(float(W), (cx + w / 2) * W); y1 = min(float(H), (cy + h / 2) * H)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append({"category_id": SRC_TO_MASTER[c],
                    "bbox": [round(x0, 3), round(y0, 3), round(x1 - x0, 3), round(y1 - y0, 3)]})
    return out


def split_stats(images, anns):
    s = {"frames": len(images), "frames_with_vis": sum(1 for i in images if i["files"]["vis"]),
         "boxes": Counter(), "ignored": Counter(), "size_buckets": {}}
    sizes = defaultdict(list)
    for a in anns:
        name = CATEGORIES[a["category_id"]]["name"]
        s["boxes"][name] += 1
        s["ignored"][name] += a["ignore"]
        sizes[name].append(math.sqrt(a["bbox"][2] * a["bbox"][3]))
    for name, v in sizes.items():
        v = np.array(v)
        s["size_buckets"][name] = {f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+": int(((v >= lo) & (v < hi)).sum())
                                   for lo, hi in SIZE_BUCKETS}
        s["size_buckets"][name]["median"] = round(float(np.median(v)), 1)
    s["boxes"], s["ignored"] = dict(s["boxes"]), dict(s["ignored"])
    return s


def stats_md(st):
    L = [f"# AIRST-CS ship subset, {st['version']}", "",
         f"Duplicates dropped: {st['duplicates_dropped']}. IR/visible label mismatches: {len(st['label_mismatch_ir_vis'])}.",
         f"Targets with sqrt(w*h) < {st['min_size']:g} px are marked ignore.", "",
         "| Split | Frames | With visible | Ship boxes (ignored) | civil_vehicle boxes (ignored) | Ship median px | Ship 16-32 / 32-96 / 96+ |",
         "|---|---|---|---|---|---|---|"]
    for sp, s in st["splits"].items():
        sb = s["size_buckets"].get("ship", {})
        L.append(f"| {sp} | {s['frames']} | {s['frames_with_vis']} | {s['boxes'].get('ship', 0)} ({s['ignored'].get('ship', 0)}) | "
                 f"{s['boxes'].get('civil_vehicle', 0)} ({s['ignored'].get('civil_vehicle', 0)}) | {sb.get('median', '-')} | "
                 f"{sb.get('16-32', 0)} / {sb.get('32-96', 0)} / {sb.get('96+', 0)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
