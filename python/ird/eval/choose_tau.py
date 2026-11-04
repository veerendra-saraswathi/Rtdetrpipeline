"""Choose the operating threshold tau on the VALIDATION split, then use it on test.

    python -m ird.eval.choose_tau --gt data/airst_v0.1s/airst_val.json --pred preds/a_ir_s0_val_torch.json \
        -o reports/a_ir_s0_tau.json
    python -m ird.eval.evaluate ... --tau $(python -c "import json;print(json.load(open('reports/a_ir_s0_tau.json'))['tau'])")

Rule (fixed before looking at test): on val, class-aware, taus 0.01..0.99:
  1. the highest tau with Pd >= target_pd (default 0.95, RCI) -> fewest false alarms at RCI's Pd;
  2. if no tau reaches it: the tau with the highest F1.
Never choose tau on the test split.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.eval.metrics import EvalData, match_image  # noqa: E402


def curve(data, taus, match_iou=0.5, class_aware=True):
    rows = []
    for t in taus:
        n = tp = fp = 0
        for i in data.image_ids:
            r = match_image(data.gt[i], data.dt[i], t, match_iou, class_aware)
            n += r["n_gt"]; tp += r["tp"]; fp += r["fp"]
        pd = tp / n if n else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pd * prec / (pd + prec) if pd + prec else 0.0
        rows.append({"tau": round(float(t), 2), "pd": pd, "far_per_frame": fp / len(data.image_ids), "f1": f1})
    return rows


def choose(rows, target_pd):
    ok = [r for r in rows if r["pd"] >= target_pd]
    if ok:
        return max(ok, key=lambda r: r["tau"]), "highest tau with Pd >= target"
    return max(rows, key=lambda r: (r["f1"], r["tau"])), "Pd target not reached on val: best F1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="validation annotations")
    ap.add_argument("--pred", required=True, help="validation detections")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--target-pd", type=float, default=0.95)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--min-size", type=float, default=16.0)
    args = ap.parse_args()
    data = EvalData(json.load(open(args.gt)), json.load(open(args.pred)), args.min_size)
    rows = curve(data, np.arange(0.01, 0.995, 0.01), args.match_iou)
    best, rule = choose(rows, args.target_pd)
    out = {"tau": best["tau"], "rule": rule, "at_tau_on_val": best, "target_pd": args.target_pd,
           "val_gt": os.path.abspath(args.gt), "val_pred": os.path.abspath(args.pred), "curve": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print(f"tau = {best['tau']:.2f} ({rule}); on val: Pd {100 * best['pd']:.1f}%, "
          f"FAR/frame {best['far_per_frame']:.3g}")


if __name__ == "__main__":
    main()
