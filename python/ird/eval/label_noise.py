"""E-10: how well do two human labelings of the same frame agree?

AIRST-CS has 174 pixel-identical frame pairs that were labeled twice (prepare_airst.py keeps one
copy and stores both label sets in duplicates_e10.json). Treating one labeling as "ground truth"
and the other as "detections" (score 1) gives the Pd, AO and mean IoU a perfect detector would get
against noisy labels. If human AO is below RCI's 90%, the AO target is stricter than the labels.

    python -m ird.eval.label_noise --pairs data/airst_v0.1s/duplicates_e10.json --out reports/e10_label_noise
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.eval.metrics import EvalData, rci_metrics  # noqa: E402

CATS = [{"id": 0, "name": "ship"}, {"id": 1, "name": "tank"}, {"id": 2, "name": "aircraft"},
        {"id": 3, "name": "civil_vehicle"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=float, default=16.0)
    args = ap.parse_args()
    pairs = json.load(open(args.pairs))["pairs"]
    images = [{"id": k, "width": p["width"], "height": p["height"]} for k, p in enumerate(pairs)]
    res = {}
    for name, a_key, b_key in [("kept_as_truth", "boxes_kept", "boxes_dropped"),
                               ("dropped_as_truth", "boxes_dropped", "boxes_kept")]:
        gt = {"categories": CATS, "images": images, "annotations": [
            {"id": n, "image_id": k, "category_id": b["category_id"], "bbox": b["bbox"]}
            for n, (k, b) in enumerate((k, b) for k, p in enumerate(pairs) for b in p[a_key])]}
        pr = {"image_ids": [i["id"] for i in images], "detections": [
            {"image_id": k, "category_id": b["category_id"], "bbox": b["bbox"], "score": 1.0}
            for k, p in enumerate(pairs) for b in p[b_key]]}
        for iou in (0.5, 0.3, 0.1):
            r, _ = rci_metrics(EvalData(gt, pr, args.min_size), tau=0.5, match_iou=iou, class_aware=True)
            res[f"{name}@iou{iou}"] = {k: r[k] for k in ("n_gt", "tp", "fp", "pd", "ao", "mean_iou_tp")}
            res[f"{name}@iou{iou}"]["by_size"] = r["by_size"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"pairs": len(pairs), "min_size": args.min_size, "results": res}, open(args.out + ".json", "w"), indent=1)
    L = [f"# E-10 label agreement on {len(pairs)} AIRST duplicate pairs (targets >= {args.min_size:g} px)", "",
         "One labeling scored against the other, as if it were a perfect detector.", "",
         "| Truth | Match IoU | Targets | Pd | AO | Mean IoU of matches | Extra boxes |", "|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        t, i = k.split("@iou")
        L.append(f"| {t} | {i} | {v['n_gt']} | {100 * v['pd']:.1f}% | {100 * v['ao']:.1f}% | "
                 f"{100 * v['mean_iou_tp']:.1f}% | {v['fp']} |")
    md = "\n".join(L) + "\n"
    open(args.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
