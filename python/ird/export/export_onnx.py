"""Export a trained arm to ONNX for TensorRT.

    python -m ird.export.export_onnx -c configs/arms/a_ir.yml -r runs/a_ir/s0/best.pth -o export/a_ir_s0.onnx

Graph: input  'images'  float32 [1, C, 512, 640]   (C = 3, or 4 for arm C)
       output 'logits'  float32 [1, 300, num_classes]  raw class logits
              'boxes'   float32 [1, 300, 4]            cx, cy, w, h normalised to the canvas
Decoding (sigmoid, top-k, scaling) is done outside the graph, identically in Python
(ird/infer/decode.py) and C++ (cpp/runtime/decoder.cpp). This keeps int64 top-k out of TensorRT.

Also writes <onnx>.runtime.txt (key=value) for the C++ runtime and <onnx>.json (full info), and
checks ONNX Runtime against PyTorch on a random input and, with --check-images, on real frames.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.train.build import build_model  # noqa: E402


class RawOutputs(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


def write_runtime_txt(path, info):
    ir, vis = info["ir_params"], info["vis_params"]
    lines = {
        "arm": info["arm"], "modality": info["modality"], "in_channels": info["in_channels"],
        "out_h": info["out_hw"][0], "out_w": info["out_hw"][1],
        "num_classes": info["num_classes"], "num_queries": info["num_queries"],
        "class_names": ",".join(info["class_names"]),
        "ir_tone": ir.get("tone", "percentile"), "ir_p_lo": ir.get("p_lo", 0.5),
        "ir_p_hi": ir.get("p_hi", 99.5), "ir_pad_value": ir.get("pad_value", 0.0),
        "vis_tone": vis.get("tone", "linear"), "vis_p_lo": vis.get("p_lo", 0.5),
        "vis_p_hi": vis.get("p_hi", 99.5), "vis_pad_value": vis.get("pad_value", 0.0),
        "input_name": "images", "logits_name": "logits", "boxes_name": "boxes",
    }
    with open(path, "w") as fh:
        fh.write("# IRD runtime config, written by export_onnx.py. key=value\n")
        for k, v in lines.items():
            fh.write(f"{k}={v}\n")


def compare_detections(ref, got, out_hw, min_score=0.3):
    """Decode both outputs and match each PyTorch detection to an ONNX one (same class, best IoU).
    Order-independent, unlike a raw tensor difference. Detections within 0.02 of min_score on one
    side only are not counted as unmatched (they can cross the floor by rounding)."""
    from ird.infer.decode import decode
    from ird.eval.metrics import iou_matrix
    H, W = out_hw
    unit = {"scale_x": 1.0, "scale_y": 1.0}
    a = decode(ref[0][0], ref[1][0], out_hw, unit, W, H, min_score=min_score - 0.02)
    b = decode(got[0][0], got[1][0], out_hw, unit, W, H, min_score=min_score - 0.02)
    res = {"n": 0, "unmatched": 0, "min_iou": 1.0, "max_score_diff": 0.0}
    for side, other in ((a, b), (b, a)):
        for c, x, y, w, h, sc in side:
            if sc < min_score:
                continue
            cand = [o for o in other if o[0] == c]
            iou = iou_matrix([[x, y, w, h]], [list(o[1:5]) for o in cand])
            if side is a:
                res["n"] += 1
            if not cand or iou.max() < 0.5:
                res["unmatched"] += 1
                continue
            j = int(iou.argmax())
            res["min_iou"] = min(res["min_iou"], float(iou[0, j]))
            res["max_score_diff"] = max(res["max_score_diff"], abs(sc - cand[j][5]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--checkpoint", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--rtdetr")
    ap.add_argument("--check-images", help="annotation JSON: also compare on the first 4 real frames")
    ap.add_argument("--data-root", default="..")
    args = ap.parse_args()

    model, info = build_model(args.config, args.checkpoint, args.rtdetr)
    net = RawOutputs(model).eval()
    C, (H, W) = info["in_channels"], info["out_hw"]
    x = torch.rand(1, C, H, W)
    with torch.no_grad():
        ref = [t.numpy() for t in net(x)]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    export_kw = dict(input_names=["images"], output_names=["logits", "boxes"],
                     opset_version=args.opset, do_constant_folding=True)
    try:
        torch.onnx.export(net, (x,), args.output, dynamo=False, **export_kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(net, (x,), args.output, **export_kw)

    import onnx
    onnx.checker.check_model(onnx.load(args.output))
    info.update({"onnx": os.path.abspath(args.output), "opset": args.opset,
                 "inputs": {"images": [1, C, H, W]},
                 "outputs": {"logits": list(ref[0].shape), "boxes": list(ref[1].shape)}})

    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": x.numpy()})
    diffs = {"random_logits_max_abs": float(np.abs(got[0] - ref[0]).max()),
             "random_boxes_max_abs": float(np.abs(got[1] - ref[1]).max())}

    det_check = [compare_detections(ref, got, (H, W))]
    if args.check_images:
        from ird.data.io import read_frame
        from ird.data.preproc import BandParams, build_input
        doc = json.load(open(args.check_images))
        mod = info["modality"]
        ims = [i for i in doc["images"] if mod == "ir" or i["files"].get("vis")][:4]
        for k, e in enumerate(ims):
            xin, _ = build_input(read_frame(args.data_root, e, mod), mod, (H, W),
                                 BandParams(**info["ir_params"]), BandParams(**info["vis_params"]))
            with torch.no_grad():
                r = [t.numpy() for t in net(torch.from_numpy(xin)[None])]
            g = sess.run(None, {"images": xin[None]})
            diffs[f"frame{k}_logits_max_abs"] = float(np.abs(g[0] - r[0]).max())
            diffs[f"frame{k}_boxes_max_abs"] = float(np.abs(g[1] - r[1]).max())
            det_check.append(compare_detections(r, g, (H, W)))
    # Raw tensors can differ a lot when a near-tie changes which 300 queries the encoder selects
    # (the same boxes come out in a different order). What matters is the decoded detections.
    dc = {"detections_compared": sum(d["n"] for d in det_check),
          "unmatched": sum(d["unmatched"] for d in det_check),
          "min_iou": min(d["min_iou"] for d in det_check),
          "max_score_diff": max(d["max_score_diff"] for d in det_check)}
    info["onnxruntime_vs_torch"] = {"raw": diffs, "detections": dc}
    with open(args.output + ".json", "w") as fh:
        json.dump(info, fh, indent=1)
    write_runtime_txt(args.output + ".runtime.txt", info)
    print("raw outputs, max abs difference (informational):", json.dumps(diffs))
    print("decoded detections (score >= 0.3), ONNX Runtime vs PyTorch:", json.dumps(dc))
    if dc["unmatched"] or dc["min_iou"] < 0.99 or dc["max_score_diff"] > 0.01:
        print("WARNING: ONNX Runtime detections differ from PyTorch (see above)")
    else:
        print("OK: ONNX Runtime and PyTorch give the same detections")
    print(f"wrote {args.output}, .json, .runtime.txt")


if __name__ == "__main__":
    main()
