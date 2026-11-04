"""Time per frame for one model on every backend, measured the same way, in one table.

    python -m ird.eval.bench_backends --run a_ir_s0 --arm a_ir

Python backends measured here (batch 1, test frames already in memory, same as ird_infer):
  PyTorch FP32 (GPU)      the trained model as is ("normal")
  PyTorch FP16 (GPU)      same model under autocast, for reference
  ONNX Runtime (CPU)      the exported ONNX file
  ONNX Runtime (CUDA)     only if onnxruntime-gpu is installed
C++ TensorRT rows (FP16, INT8, INT8-mix) are read from reports/<run>[_int8|_int8mix]_latency_desktop.json,
written by ird_infer in cpp_windows.ps1.

Stages per frame: preprocess (the shared C++ preprocessing in every backend) | model (copy to GPU,
network, copy back; GPU synchronised) | decode (Python decoder here, C++ decoder in TensorRT rows).
Writes reports/latency_compare_<run>.md and .json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.data.io import read_frame  # noqa: E402
from ird.data.preproc import BandParams, build_input  # noqa: E402
from ird.infer.decode import decode  # noqa: E402


def stats(v):
    v = np.sort(np.asarray(v, float))
    q = lambda p: float(v[min(len(v) - 1, max(0, int(np.ceil(p * len(v))) - 1))])
    return {"mean": float(v.mean()), "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": float(v.max())}


def bench(name, run, frames, info, warmup):
    out_hw = tuple(info["out_hw"])
    irp, visp = BandParams(**info["ir_params"]), BandParams(**info["vis_params"])
    mod = info["modality"]
    pre, mdl, dec, tot = [], [], [], []
    for k in range(warmup + len(frames)):
        e, bands = frames[k % len(frames)] if k < warmup else frames[k - warmup]
        t0 = time.perf_counter()
        x, pinfo = build_input(bands, mod, out_hw, irp, visp)
        t1 = time.perf_counter()
        lg, bx = run(x)
        t2 = time.perf_counter()
        decode(lg, bx, out_hw, pinfo, e["width"], e["height"], 300, 0.001)
        t3 = time.perf_counter()
        if k >= warmup:
            pre.append((t1 - t0) * 1e3); mdl.append((t2 - t1) * 1e3); dec.append((t3 - t2) * 1e3); tot.append((t3 - t0) * 1e3)
    r = {"backend": name, "frames": len(frames), "total_ms": stats(tot), "preprocess_ms": stats(pre),
         "model_ms": stats(mdl), "decode_ms": stats(dec)}
    print(f"{name:24s} total mean {r['total_ms']['mean']:7.2f} ms  p99 {r['total_ms']['p99']:7.2f}  "
          f"(pre {r['preprocess_ms']['mean']:.2f}, model {r['model_ms']['mean']:.2f}, decode {r['decode_ms']['mean']:.2f})", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="export name, e.g. a_ir_s0")
    ap.add_argument("--arm", required=True, help="config name in configs/arms, e.g. a_ir")
    ap.add_argument("--ann", default="data/airst_v0.1s/airst_test.json")
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--skip", nargs="*", default=[], help="backends to skip: torch32 torch16 ort_cpu ort_cuda")
    args = ap.parse_args()

    onnx = f"export/{args.run}.onnx"
    info = json.load(open(onnx + ".json"))
    mod = info["modality"]
    doc = json.load(open(args.ann))
    ims = [e for e in doc["images"] if mod == "ir" or e["files"].get("vis")]
    frames = [(e, read_frame(args.data_root, e, mod)) for e in ims]     # loaded once, not timed
    rows = []

    import torch
    if torch.cuda.is_available() and not {"torch32", "torch16"} <= set(args.skip):
        from ird.train.build import build_model
        model, _ = build_model(f"configs/arms/{args.arm}.yml", f"runs/{args.arm}/{args.run.split('_')[-1]}/best.pth", device="cuda")

        def torch_run(fp16):
            def run(x):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    o = model(torch.from_numpy(x)[None].to("cuda", non_blocking=False))
                lg, bx = o["pred_logits"][0].float().cpu().numpy(), o["pred_boxes"][0].float().cpu().numpy()
                return lg, bx      # .cpu() waits for the GPU, so the time is complete
            return run
        if "torch32" not in args.skip:
            rows.append(bench("PyTorch FP32 (GPU)", torch_run(False), frames, info, args.warmup))
        if "torch16" not in args.skip:
            rows.append(bench("PyTorch FP16 autocast (GPU)", torch_run(True), frames, info, args.warmup))
        del model
        torch.cuda.empty_cache()

    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()          # lets onnxruntime-gpu use the CUDA/cuDNN DLLs that PyTorch ships
        except Exception:
            pass
    provs = ort.get_available_providers()
    for name, key, prov in [("ONNX Runtime (CPU)", "ort_cpu", "CPUExecutionProvider"),
                            ("ONNX Runtime (CUDA)", "ort_cuda", "CUDAExecutionProvider")]:
        if key in args.skip:
            continue
        if prov not in provs:
            print(f"{name}: not available (providers: {provs})")
            continue
        sess = ort.InferenceSession(onnx, providers=[prov])
        if prov not in sess.get_providers():
            print(f"{name}: could not start, skipped")
            continue
        rows.append(bench(name, lambda x, s=sess: tuple(t[0] for t in s.run(None, {"images": x[None]})), frames, info, args.warmup))

    for name, suf in [("TensorRT FP16 (C++)", ""), ("TensorRT INT8 (C++)", "_int8"), ("TensorRT INT8-mix (C++)", "_int8mix")]:
        p = f"reports/{args.run}{suf}_latency_desktop.json"
        if os.path.exists(p):
            d = json.load(open(p))
            rows.append({"backend": name, "frames": d["frames"], "total_ms": d["total_ms"], "preprocess_ms": d["preprocess_ms"],
                         "model_ms": d["infer_ms"], "decode_ms": d["decode_ms"]})

    base = next((r["total_ms"]["mean"] for r in rows if r["backend"].startswith("PyTorch FP32")), None)
    f = lambda v: f"{v:.2f}"
    L = [f"# Time per frame, {args.run} ({mod}), {len(frames)} test frames, batch 1", "",
         "Frame already in memory -> detections ready. Same preprocessing code everywhere (C++).", "",
         "| Backend | Total mean (ms) | p50 | p99 | Preprocess | Model (incl. copies) | Decode | Frames/s | Speed-up vs PyTorch FP32 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["total_ms"]
        sp = f"{base / t['mean']:.1f}x" if base else "—"
        L.append(f"| {r['backend']} | {f(t['mean'])} | {f(t['p50'])} | {f(t['p99'])} | {f(r['preprocess_ms']['mean'])} | "
                 f"{f(r['model_ms']['mean'])} | {f(r['decode_ms']['mean'])} | {1000 / t['mean']:.0f} | {sp} |")
    L += ["", "Python rows use the Python decoder; TensorRT rows use the C++ decoder. ONNX Runtime CPU runs on the laptop CPU, "
          "not the GPU. Times vary by about 0.5 ms between runs."]
    os.makedirs("reports", exist_ok=True)
    open(f"reports/latency_compare_{args.run}.md", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"reports/latency_compare_{args.run}.json", "w"), indent=1)
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
