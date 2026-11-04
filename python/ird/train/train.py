"""Train one arm with the official RT-DETRv2 solver.

    python -m ird.train.train -c configs/arms/a_ir.yml --tuning weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
        --seed 0 --output-dir runs/a_ir/s0 [--use-amp] [-u key=value ...]

Same as RT-DETRv2's tools/train.py, plus:
- registers IRDDetection / IRDPhotometric,
- loads the tuning checkpoint itself so the 4-channel stem (arm C) is initialised from the RGB filters,
- writes run_info.json (arm, seed, config, checkpoint, RT-DETRv2 commit, preprocessing backend).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ird.train.paths import add_rtdetr, rtdetr_commit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--tuning", help="checkpoint to start from (COCO or one of ours)")
    ap.add_argument("-r", "--resume", help="resume an interrupted run of this arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("-d", "--device")
    ap.add_argument("--rtdetr", help="path to RT-DETR/rtdetrv2_pytorch")
    ap.add_argument("-u", "--update", nargs="+", help="override config keys, e.g. epoches=10")
    args = ap.parse_args()

    root = add_rtdetr(args.rtdetr)
    from src.misc import dist_utils
    from src.core import YAMLConfig, yaml_utils
    from src.solver import TASKS
    import ird.train.rtdetr_ext as ext
    from ird.data.preproc import backend

    dist_utils.setup_distributed(0, "builtin", seed=args.seed)
    upd = yaml_utils.parse_cli(args.update) if args.update else {}
    upd.update({"seed": args.seed, "output_dir": args.output_dir})
    if args.use_amp:
        upd["use_amp"] = True
    if args.device:
        upd["device"] = args.device
    if args.resume:
        upd["resume"] = args.resume
    cfg = YAMLConfig(args.config, **upd)
    ird_cfg = cfg.yaml_cfg.get("ird", {})
    in_ch = int(ird_cfg.get("in_channels", 3))

    model = cfg.model
    report = None
    if args.tuning and not args.resume:
        report = ext.load_weights(model, args.tuning, in_ch=in_ch)
        print(f"loaded {report['loaded']} tensors from {args.tuning}; skipped (shape) "
              f"{len(report['unmatched'])}, not in checkpoint {len(report['missed'])}")
    elif in_ch != 3:
        ext.expand_stem(model, in_ch)
    cfg._model = model  # same object, changed in place; YAMLConfig.model has no setter

    os.makedirs(args.output_dir, exist_ok=True)
    info_name = "run_info_resume.json" if args.resume else "run_info.json"   # keep the original record
    with open(os.path.join(args.output_dir, info_name), "w") as fh:
        json.dump({"arm": ird_cfg.get("arm"), "seed": args.seed, "config": os.path.abspath(args.config),
                   "tuning": args.tuning, "resume": args.resume, "in_channels": in_ch,
                   "rtdetr_root": root, "rtdetr_commit": rtdetr_commit(root),
                   "preproc_backend": backend(), "updates": args.update,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "load_report": {k: (v if not isinstance(v, list) else len(v)) for k, v in (report or {}).items()}},
                  fh, indent=1)
    if backend() != "cpp":
        print("WARNING: training without the C++ preprocessing module (NumPy reference in use)")

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.fit()
    # Marker that the run reached its last epoch (best.pth alone also exists in stopped runs).
    with open(os.path.join(args.output_dir, "done.json"), "w") as fh:
        json.dump({"epochs": cfg.yaml_cfg.get("epoches"), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
