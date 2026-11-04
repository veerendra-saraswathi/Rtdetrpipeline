"""Run a model on a split and write a detections file (the evaluator's input).

PyTorch:  python -m ird.infer.predict --backend torch -c configs/arms/a_ir.yml -r runs/a_ir/s0/best.pth \
              --ann data/airst_v0.1s/airst_test.json -o preds/a_ir_s0_torch.json
ONNX:     python -m ird.infer.predict --backend onnx --onnx export/a_ir_s0.onnx \
              --ann data/airst_v0.1s/airst_test.json -o preds/a_ir_s0_onnx.json
(TensorRT predictions come from the C++ tool ird_infer; same file format.)

Detections file:
  {"meta": {...}, "image_ids": [...], "detections": [{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]}
Boxes are in original image pixels. min_score defaults to 0.001 so AP can use the full ranking.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.data.io import read_frame  # noqa: E402
from ird.data.preproc import BandParams, backend, build_input  # noqa: E402
from ird.infer.decode import decode  # noqa: E402


def frames_for(doc, modality):
    if modality == "ir":
        return list(doc["images"])
    return [i for i in doc["images"] if i["files"].get("vis")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "onnx"], required=True)
    ap.add_argument("-c", "--config")
    ap.add_argument("-r", "--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--data-root", default="..")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--modality", help="override (default: the model's val modality)")
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu")
    ap.add_argument("--rtdetr")
    args = ap.parse_args()

    if args.backend == "torch":
        import torch
        dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        from ird.train.build import build_model
        model, info = build_model(args.config, args.checkpoint, args.rtdetr, device=dev)

        def run(x):
            with torch.no_grad():
                o = model(torch.from_numpy(x)[None].to(dev))
            return o["pred_logits"][0].float().cpu().numpy(), o["pred_boxes"][0].float().cpu().numpy()
    else:
        import onnxruntime as ort
        info = json.load(open(args.onnx + ".json"))
        sess = ort.InferenceSession(args.onnx, providers=ort.get_available_providers())

        def run(x):
            lg, bx = sess.run(None, {"images": x[None]})
            return lg[0], bx[0]

    mod = args.modality or info["modality"]
    out_hw = tuple(info["out_hw"])
    irp, visp = BandParams(**info["ir_params"]), BandParams(**info["vis_params"])
    doc = json.load(open(args.ann))
    ims = frames_for(doc, mod)
    dets, times = [], []
    for e in ims:
        x, pinfo = build_input(read_frame(args.data_root, e, mod), mod, out_hw, irp, visp)
        t0 = time.perf_counter()
        lg, bx = run(x)
        times.append((time.perf_counter() - t0) * 1000)
        for c, x0, y0, w, h, s in decode(lg, bx, out_hw, pinfo, e["width"], e["height"],
                                         args.top_k, args.min_score):
            dets.append({"image_id": e["id"], "category_id": int(c),
                         "bbox": [round(x0, 3), round(y0, 3), round(w, 3), round(h, 3)],
                         "score": round(s, 6)})
    meta = {"backend": args.backend, "arm": info.get("arm"), "modality": mod,
            "ann_file": os.path.abspath(args.ann), "split": doc.get("info", {}).get("split"),
            "model": info.get("onnx") or info.get("checkpoint"), "config": info.get("config"),
            "checkpoint_sha256": info.get("checkpoint_sha256"), "min_score": args.min_score,
            "top_k": args.top_k, "preproc_backend": backend(), "ir_params": info["ir_params"],
            "vis_params": info["vis_params"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_ms_mean": float(np.mean(times)) if times else None,
            "note": "model_ms is host-side model time only; latency acceptance uses ird_infer on Orin"}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump({"meta": meta, "image_ids": [e["id"] for e in ims], "detections": dets}, fh)
    print(f"{len(ims)} frames, {len(dets)} detections -> {args.output}")


if __name__ == "__main__":
    main()
