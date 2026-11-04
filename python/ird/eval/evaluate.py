"""Evaluate one detections file against the ground truth.

    python -m ird.eval.evaluate --gt data/airst_v0.1s/airst_test.json --pred preds/a_ir_s0_trt.json \
        --out reports/a_ir_s0_trt [--tau 0.5] [--match-iou 0.5] [--min-size 16] [--boot 1000]

Writes <out>.json (everything) and <out>.md (one-page summary). Same code for every arm and backend.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.eval.metrics import (BUCKETS, EvalData, bootstrap, coco_ap, operating_curve,  # noqa: E402
                              rci_metrics)

RCI_TARGETS = {"pd": 0.95, "ao": 0.90, "far_per_frame": 1e-5, "cs_mean_tp_score": 0.9}


def evaluate(gt_doc, pred_doc, tau=0.5, match_iou=0.5, min_size=16.0, n_boot=1000):
    data = EvalData(gt_doc, pred_doc, min_size)
    rci, per_img = rci_metrics(data, tau, match_iou, class_aware=True)
    rci_agn, per_img_agn = rci_metrics(data, tau, match_iou, class_aware=False)
    ap, records = coco_ap(data, per_image=True)
    ap_buckets = {k: coco_ap(data, rng=v) for k, v in BUCKETS.items()}
    ci = bootstrap(per_img, records, data.image_ids, n_boot) if n_boot else {}
    ci_agn = bootstrap(per_img_agn, records, data.image_ids, n_boot, with_ap=False) if n_boot else {}
    return {
        "params": {"tau": tau, "match_iou": match_iou, "min_size": min_size, "n_boot": n_boot,
                   "size_rule": "sqrt(w*h)", "matching": "one-to-one max IoU (Hungarian)"},
        "pred_meta": pred_doc.get("meta", {}),
        "frames": len(data.image_ids),
        "rci_class_aware": rci, "rci_class_agnostic": rci_agn,
        "ci95_class_aware": ci, "ci95_class_agnostic": ci_agn,
        "coco_ap": ap, "coco_ap_by_size": ap_buckets,
        "operating_curve_agnostic": operating_curve(data, class_aware=False),
        "rci_targets": RCI_TARGETS,
        "per_image_class_aware": {str(k): v for k, v in per_img.items()},
    }


def f(x, pct=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3g}"


def to_markdown(rep):
    m = rep["pred_meta"]
    a, g = rep["rci_class_aware"], rep["rci_class_agnostic"]
    ci, cig = rep["ci95_class_aware"], rep["ci95_class_agnostic"]
    p = rep["params"]

    def c(ci_, k, pct=True):
        return f"[{f(ci_[k][0], pct)}, {f(ci_[k][1], pct)}]" if k in ci_ else ""
    L = [f"# Evaluation: {m.get('arm')} / {m.get('backend')} / {m.get('modality')}", "",
         f"Model: `{m.get('model')}`  ",
         f"GT: `{m.get('ann_file')}`, {rep['frames']} frames. tau = {p['tau']}, match IoU = {p['match_iou']}, "
         f"targets below {p['min_size']:g} px ignored.", "",
         "## RCI metrics", "",
         "| Metric | Class-aware | 95% CI | Class-agnostic | 95% CI | RCI target |",
         "|---|---|---|---|---|---|",
         f"| Pd | {f(a['pd'])} | {c(ci, 'pd')} | {f(g['pd'])} | {c(cig, 'pd')} | > 95% |",
         f"| AO (misses = 0) | {f(a['ao'])} | {c(ci, 'ao')} | {f(g['ao'])} | {c(cig, 'ao')} | > 90% |",
         f"| FAR per frame | {f(a['far_per_frame'], False)} | {c(ci, 'far_per_frame', False)} | "
         f"{f(g['far_per_frame'], False)} | {c(cig, 'far_per_frame', False)} | 1e-5 |",
         f"| FAR upper 95% bound | {f(a['far_upper95'], False)} | | {f(g['far_upper95'], False)} | | 1e-5 |",
         f"| Mean IoU of hits | {f(a['mean_iou_tp'])} | | {f(g['mean_iou_tp'])} | | |",
         f"| CS (mean score of hits) | {f(a['cs_mean_tp_score'], False)} | | | | > 0.9 |",
         f"| Calibration error (ECE) | {f(a['ece'], False)} | | | | |", "",
         f"Counts (class-aware): {a['n_gt']} targets, {a['tp']} hits, {a['fp']} false alarms in {a['frames']} frames.", "",
         "## By target size (class-aware)", "", "| Size (px) | Targets | Pd | AO | AP50 |", "|---|---|---|---|---|"]
    for k, v in a["by_size"].items():
        L.append(f"| {k} | {v['n_gt']} | {f(v['pd'])} | {f(v['ao'])} | {f(rep['coco_ap_by_size'][k]['AP50'])} |")
    L += ["", "## By class (class-aware)", "", "| Class | Targets | Pd | AO | FAR/frame | AP | AP50 |",
          "|---|---|---|---|---|---|---|"]
    for k, v in a["per_class"].items():
        apc = rep["coco_ap"]["per_class"].get(k, {})
        L.append(f"| {k} | {v['n_gt']} | {f(v['pd'])} | {f(v['ao'])} | {f(v['far_per_frame'], False)} | "
                 f"{f(apc.get('AP'))} | {f(apc.get('AP50'))} |")
    ap = rep["coco_ap"]
    L += ["", f"COCO-style (s >= {p['min_size']:g} px, class mean): AP {f(ap['AP'])} {c(ci, 'AP')}, "
              f"AP50 {f(ap['AP50'])} {c(ci, 'AP50')}, AP75 {f(ap['AP75'])}.", "",
          "## Operating curve (class-agnostic)", "", "| tau | Pd | AO | FAR/frame |", "|---|---|---|---|"]
    for r in rep["operating_curve_agnostic"]:
        L.append(f"| {r['tau']:.2f} | {f(r['pd'])} | {f(r['ao'])} | {f(r['far_per_frame'], False)} |")
    return "\n".join(L) + "\n"


def resolve_tau(pred_path, tau):
    """Explicit tau wins; else the tau chosen on validation (<pred>.tau.json); else 0.5."""
    if tau is not None:
        return tau
    side = pred_path + ".tau.json"
    if os.path.exists(side):
        return float(json.load(open(side))["tau"])
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--tau", type=float, default=None,
                    help="operating threshold; default: <pred>.tau.json from choose_tau (chosen on val), else 0.5")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--min-size", type=float, default=16.0)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    tau = resolve_tau(args.pred, args.tau)
    rep = evaluate(json.load(open(args.gt)), json.load(open(args.pred)), tau, args.match_iou,
                   args.min_size, args.boot)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out + ".json", "w"), indent=1)
    md = to_markdown(rep)
    open(args.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
