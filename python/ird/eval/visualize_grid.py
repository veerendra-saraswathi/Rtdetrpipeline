"""Side-by-side detection grids: one column per model, one row per frame.

    python -m ird.eval.visualize_grid --gt data/airst_v0.1s/airst_test.json --band ir \
        --col A-IR preds/a_ir_s0_trt.json --col B preds/b_joint_s0_trt.json \
        --col B-ft-IR preds/b_ft_ir_s0_trt.json --out reports/grid_ir_trt

Each panel shows the image as the network sees it (C++ preprocessing, ird_preproc) with:
  cyan    detection that matches a target (score shown)
  yellow  false alarm (detection above tau with no target)
  red     missed target
  grey    target below 16 px (ignored in scoring, as in evaluate.py)
Matching is the evaluator's rule: class-aware, one-to-one, IoU >= 0.5, at each model's tau
(<pred>.tau.json chosen on val, unless --tau). Writes <out>_p01.jpg, ... (rows per page) and <out>.md.
Frames: the frames where the models disagree most, then random ones (fixed seed), unless --ids.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.data.io import read_band  # noqa: E402
from ird.data.preproc import DEFAULT_IR, DEFAULT_VIS, preprocess  # noqa: E402
from ird.eval.evaluate import resolve_tau  # noqa: E402
from ird.eval.metrics import box_size, iou_matrix, one_to_one  # noqa: E402

COL = {"tp": (0, 230, 255), "fp": (255, 220, 0), "miss": (255, 40, 40), "ign": (150, 150, 150)}


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def match(gts, dts, tau, iou_thr=0.5, min_size=16.0):
    """gts: [(cls, xywh, ignore)], dts: [(cls, xywh, score)]. Returns per-box labels."""
    dts = [d for d in dts if d[2] >= tau]
    gl = [(c, b, ig or box_size(b) < min_size) for c, b, ig in gts]
    g_state = ["ign" if g[2] else "miss" for g in gl]
    d_state = ["fp"] * len(dts)
    for cls in sorted({g[0] for g in gl} | {d[0] for d in dts}):
        gi = [i for i, g in enumerate(gl) if g[0] == cls and not g[2]]
        ii = [i for i, g in enumerate(gl) if g[0] == cls and g[2]]
        di = [i for i, d in enumerate(dts) if d[0] == cls]
        iou = iou_matrix([dts[i][1] for i in di], [gl[i][1] for i in gi])
        for a, b in one_to_one(iou, iou_thr):
            d_state[di[a]] = "tp"
            g_state[gi[b]] = "hit"
        iou_ign = iou_matrix([dts[i][1] for i in di], [gl[i][1] for i in ii])
        for k, i in enumerate(di):   # detections on ignored targets are not false alarms
            if d_state[i] == "fp" and iou_ign.shape[1] and iou_ign[k].max() >= iou_thr:
                d_state[i] = "ign"
    return gl, g_state, dts, d_state


def load_preds(path):
    doc = json.load(open(path))
    by = {}
    for d in doc["detections"]:
        by.setdefault(d["image_id"], []).append((d["category_id"], d["bbox"], float(d["score"])))
    return set(doc.get("image_ids") or by.keys()), by


def panel(img_rgb, scale, gl, g_state, dts, d_state, title, names, W):
    im = Image.fromarray(img_rgb).resize((W, int(round(img_rgb.shape[0] * W / img_rgb.shape[1]))), Image.BILINEAR)
    k = W / img_rgb.shape[1] * scale
    dr = ImageDraw.Draw(im)
    f = font(12)

    def rect(b, color, width):
        x, y, w, h = [v * k for v in b]
        dr.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], outline=color, width=width)
    for (c, b, _), s in zip(gl, g_state):
        if s in ("miss", "ign"):
            rect(b, COL[s], 2 if s == "miss" else 1)
    for (c, b, sc), s in zip(dts, d_state):
        if s == "ign":
            continue
        rect(b, COL[s], 2)
        x, y = b[0] * k, b[1] * k
        dr.text((x, max(0, y - 14)), "%s %.2f" % (names.get(c, c)[:4], sc), fill=COL[s], font=f)
    hdr = Image.new("RGB", (W, 22), (25, 25, 25))
    ImageDraw.Draw(hdr).text((6, 3), title, fill=(255, 255, 255), font=font(14))
    out = Image.new("RGB", (W, im.height + 22))
    out.paste(hdr, (0, 0)); out.paste(im, (0, 22))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--col", nargs=2, action="append", required=True, metavar=("NAME", "PRED"))
    ap.add_argument("--band", choices=["ir", "vis"], default="ir")
    ap.add_argument("--out", required=True, help="output prefix, e.g. reports/grid_ir_trt")
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--rows", type=int, default=4, help="frames per page")
    ap.add_argument("--width", type=int, default=560, help="panel width in pixels")
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    entries = {e["id"]: e for e in gt["images"]}
    gts = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append((a["category_id"], a["bbox"], bool(a.get("ignore", 0))))
    cols = []
    for name, path in args.col:
        ids, by = load_preds(path)
        cols.append((name, path, resolve_tau(path, args.tau), ids, by))
    common = sorted(set.intersection(*[c[3] for c in cols]))

    stats = {}
    for i in common:
        per = []
        for name, path, tau, _, by in cols:
            gl, gs, dts, ds = match(gts.get(i, []), by.get(i, []), tau)
            per.append((gs.count("hit"), gs.count("miss"), ds.count("fp")))
        stats[i] = per
    if args.ids:
        pick = [i for i in args.ids if i in stats]
    else:
        # half: frames where the columns differ most (misses + false alarms), half: random frames
        spread = sorted(common, key=lambda i: -(max(m + f for _, m, f in stats[i]) - min(m + f for _, m, f in stats[i])
                                              + 0.01 * sum(m + f for _, m, f in stats[i])))
        pick = spread[: args.frames // 2]
        rest = [i for i in common if i not in pick]
        random.Random(0).shuffle(rest)
        pick += rest[: args.frames - len(pick)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pages, rows = [], []
    params = DEFAULT_IR if args.band == "ir" else DEFAULT_VIS
    for n, i in enumerate(pick):
        e = entries[i]
        img, bd = read_band(args.data_root, e, args.band)
        x, info = preprocess(img, bd, 3, (512, 640), params)            # what the network sees
        h, w = int(info["new_h"]), int(info["new_w"])
        rgb = (np.transpose(x, (1, 2, 0))[:h, :w] * 255 + 0.5).astype(np.uint8)
        row = []
        for (name, path, tau, _, by), (hit, miss, fp) in zip(cols, stats[i]):
            gl, gs, dts, ds = match(gts.get(i, []), by.get(i, []), tau)
            title = f"{name}  frame {i}  hit {hit}  miss {miss}  false alarm {fp}"
            row.append(panel(rgb, info["scale_x"], gl, gs, dts, ds, title, names, args.width))
        rows.append(row)
        if len(rows) == args.rows or n == len(pick) - 1:
            H = sum(r[0].height for r in rows) + 6 * (len(rows) - 1)
            sheet = Image.new("RGB", (len(cols) * args.width + 6 * (len(cols) - 1), H + 30), (255, 255, 255))
            d = ImageDraw.Draw(sheet)
            d.text((6, 6), "cyan = correct detection   yellow = false alarm   red = missed target   grey = target < 16 px (ignored)",
                   fill=(0, 0, 0), font=font(14))
            y = 30
            for r in rows:
                for c, p in enumerate(r):
                    sheet.paste(p, (c * (args.width + 6), y))
                y += r[0].height + 6
            path = f"{args.out}_p{len(pages) + 1:02d}.jpg"
            sheet.save(path, quality=90)
            pages.append(path)
            rows = []

    tot = [[sum(stats[i][k][j] for i in common) for j in range(3)] for k in range(len(cols))]
    md = [f"# Detection grids ({args.band})", "", f"GT: `{args.gt}`, {len(common)} frames in common.", "",
          "| Column | Predictions | tau | Hits | Misses | False alarms (all frames) |", "|---|---|---|---|---|---|"]
    for (name, path, tau, _, _), (h, m, f) in zip(cols, tot):
        md.append(f"| {name} | `{path}` | {tau:.2f} | {h} | {m} | {f} |")
    md += ["", "Frames shown: " + ", ".join(str(i) for i in pick), ""] + [f"![page]({os.path.basename(p)})" for p in pages]
    open(args.out + ".md", "w").write("\n".join(md) + "\n")
    print("wrote", ", ".join(pages), "and", args.out + ".md")


if __name__ == "__main__":
    main()
