"""Compare training arms across seeds, with paired bootstrap CIs of the differences.

    python -m ird.eval.compare --gt data/airst_v0.1s/airst_test.json \
        --arm A-IR preds/a_ir_s0_trt.json preds/a_ir_s1_trt.json preds/a_ir_s2_trt.json \
        --arm B    preds/b_s0_trt.json    preds/b_s1_trt.json    preds/b_s2_trt.json \
        --baseline A-IR --out reports/compare_ir

For each arm: mean +- std over seeds of Pd, AO, FAR, AP, AP50 (class-aware, tau 0.5).
For each arm vs the baseline: difference of seed means with a 95% CI from a paired bootstrap over
test images (same resampled images for both arms). Decision rule (doc section 7.2): a difference
counts only if its CI excludes 0. Arms must be evaluated on the same images.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.eval.metrics import EvalData, ap_from_records, coco_ap, rci_metrics  # noqa: E402
from ird.eval.evaluate import resolve_tau  # noqa: E402

KEYS = ["pd", "ao", "far_per_frame", "AP", "AP50"]


def load_run(gt, pred_path, tau, match_iou, min_size):
    pred = json.load(open(pred_path))
    data = EvalData(gt, pred, min_size)
    tau = resolve_tau(pred_path, tau)
    rci, per_img = rci_metrics(data, tau, match_iou, class_aware=True)
    ap, records = coco_ap(data, per_image=True)
    return {"ids": data.image_ids, "per_img": per_img, "records": records,
            "tau": tau, "point": {"pd": rci["pd"], "ao": rci["ao"], "far_per_frame": rci["far_per_frame"],
                      "AP": ap["AP"], "AP50": ap["AP50"]}}


def metrics_on(run, ids):
    arr = np.array([run["per_img"][i] for i in ids], float)
    s = arr.sum(axis=0)
    ap, ap50 = ap_from_records(run["records"], ids)
    return {"pd": s[1] / s[0] if s[0] else np.nan, "ao": s[2] / s[0] if s[0] else np.nan,
            "far_per_frame": s[3] / len(ids), "AP": ap, "AP50": ap50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--arm", nargs="+", action="append", required=True, help="NAME pred1 [pred2 ...]")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=None, help="default: each run's <pred>.tau.json, else 0.5")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--min-size", type=float, default=16.0)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    arms = {a[0]: [load_run(gt, p, args.tau, args.match_iou, args.min_size) for p in a[1:]] for a in args.arm}
    ids_sets = {tuple(sorted(r["ids"])) for runs in arms.values() for r in runs}
    if len(ids_sets) != 1:
        sys.exit("arms were evaluated on different image sets; compare on the same frames")
    ids = list(next(iter(ids_sets)))

    summary = {}
    for name, runs in arms.items():
        pts = {k: np.array([r["point"][k] for r in runs], float) for k in KEYS}
        summary[name] = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                             "seeds": v.tolist()} for k, v in pts.items()}

    rng = np.random.default_rng(0)
    base = arms[args.baseline]
    diffs = {name: {k: [] for k in KEYS} for name in arms if name != args.baseline}
    for _ in range(args.boot):
        sample = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        b = [metrics_on(r, sample) for r in base]
        bm = {k: np.nanmean([x[k] for x in b]) for k in KEYS}
        for name in diffs:
            o = [metrics_on(r, sample) for r in arms[name]]
            for k in KEYS:
                diffs[name][k].append(np.nanmean([x[k] for x in o]) - bm[k])
    vs = {}
    for name, d in diffs.items():
        vs[name] = {}
        for k, v in d.items():
            lo, hi = float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
            point = summary[name][k]["mean"] - summary[args.baseline][k]["mean"]
            vs[name][k] = {"diff": point, "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}

    rep = {"params": vars(args), "frames": len(ids), "summary": summary, "vs_baseline": vs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out + ".json", "w"), indent=1)

    def pc(x):
        return f"{100 * x:.1f}"
    L = [f"# Arm comparison ({len(ids)} frames, class-aware, tau per run chosen on val unless --tau)", "",
         "| Arm | Seeds | Pd % | AO % | FAR/frame | AP % | AP50 % |", "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        L.append(f"| {name} | {len(arms[name])} | {pc(s['pd']['mean'])} ± {pc(s['pd']['std'])} | "
                 f"{pc(s['ao']['mean'])} ± {pc(s['ao']['std'])} | {s['far_per_frame']['mean']:.3g} | "
                 f"{pc(s['AP']['mean'])} ± {pc(s['AP']['std'])} | {pc(s['AP50']['mean'])} ± {pc(s['AP50']['std'])} |")
    L += ["", f"## Difference vs {args.baseline} (paired bootstrap 95% CI; * = CI excludes 0)", "",
          "| Arm | Pd | AO | FAR/frame | AP | AP50 |", "|---|---|---|---|---|---|"]
    for name, d in vs.items():
        cells = []
        for k in KEYS:
            x = d[k]
            if k == "far_per_frame":
                cells.append(f"{x['diff']:+.3g} [{x['ci95'][0]:+.3g}, {x['ci95'][1]:+.3g}]{'*' if x['significant'] else ''}")
            else:
                cells.append(f"{100 * x['diff']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"
                             f"{'*' if x['significant'] else ''}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(L) + "\n"
    open(args.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
