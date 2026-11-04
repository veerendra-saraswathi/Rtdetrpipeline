"""One table per seed: each model at each precision (PyTorch FP32, TensorRT FP16, INT8, INT8-mixed),
with engine size, desktop latency and the RCI metrics on test.

    python -m ird.eval.precision_report --seed 0 --out reports/precision_summary_s0

Reads what cpp_windows.ps1 writes:
  reports/<arm>_s<seed>_<band>_torch.json            PyTorch
  reports/<arm>_s<seed>_trt[_int8|_int8mix].json     TensorRT evaluation
  reports/<arm>_s<seed>[_int8|_int8mix]_latency_desktop.json
  export/<arm>_s<seed>[_int8|_int8mix].engine
Missing rows are skipped.
"""
import argparse
import json
import os

ARMS = [("A-IR", "a_ir", "ir"), ("B", "b_joint", "ir"), ("B-ft-IR", "b_ft_ir", "ir"),
        ("A-VIS", "a_vis", "vis"), ("B-ft-VIS", "b_ft_vis", "vis")]
PREC = [("PyTorch FP32", None), ("TensorRT FP16", ""), ("TensorRT INT8", "_int8"), ("TensorRT INT8-mix", "_int8mix")]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = args.seed
    rows = []
    for name, arm, band in ARMS:
        for pname, suf in PREC:
            if suf is None:
                rep = load(f"reports/{arm}_s{s}_{band}_torch.json")
                lat, eng = None, None
            else:
                rep = load(f"reports/{arm}_s{s}_trt{suf}.json")
                lat = load(f"reports/{arm}_s{s}{suf}_latency_desktop.json")
                e = f"export/{arm}_s{s}{suf}.engine"
                eng = os.path.getsize(e) / 2 ** 20 if os.path.exists(e) else None
            if rep is None:
                continue
            r, c = rep["rci_class_aware"], rep["coco_ap"]
            rows.append({"model": name, "band": band, "precision": pname, "tau": rep["params"]["tau"],
                         "pd": r["pd"], "ao": r["ao"], "far": r["far_per_frame"], "cs": r["cs_mean_tp_score"],
                         "ap": c["AP"], "ap50": c["AP50"], "engine_mb": eng,
                         "ms_mean": lat["total_ms"]["mean"] if lat else None,
                         "ms_p99": lat["total_ms"]["p99"] if lat else None,
                         "infer_ms": lat["infer_ms"]["mean"] if lat else None})
    pct = lambda v: f"{100 * v:.1f}%"
    num = lambda v, f="{:.1f}": "—" if v is None else f.format(v)
    L = [f"# Precision comparison, seed {s} (test set, class-aware)", "",
         "TensorRT rows: C++ pipeline on this PC. Each row uses the threshold chosen on validation for that backend.", "",
         "| Model | Band | Precision | Threshold | Pd | AO | FAR/frame | CS | AP | AP50 | Engine MB | Total ms mean | p99 | TensorRT ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {x['model']} | {x['band']} | {x['precision']} | {x['tau']:.2f} | {pct(x['pd'])} | {pct(x['ao'])} | "
                 f"{x['far']:.2f} | {x['cs']:.2f} | {pct(x['ap'])} | {pct(x['ap50'])} | {num(x['engine_mb'])} | "
                 f"{num(x['ms_mean'], '{:.2f}')} | {num(x['ms_p99'], '{:.2f}')} | {num(x['infer_ms'], '{:.2f}')} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out + ".md", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(args.out + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
