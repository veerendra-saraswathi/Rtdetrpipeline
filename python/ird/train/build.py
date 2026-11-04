"""Build a trained IRD model from (config, checkpoint) for prediction or export."""
import hashlib
import os

from ird.train.paths import add_rtdetr


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(config, checkpoint, rtdetr=None, device="cpu"):
    """Returns (model in deploy mode, info dict). info has in_channels, out_hw, modality,
    preprocessing params, class names, num_queries."""
    add_rtdetr(rtdetr)
    from src.core import YAMLConfig
    import ird.train.rtdetr_ext as ext

    cfg = YAMLConfig(config)
    ird_cfg = cfg.yaml_cfg.get("ird", {})
    in_ch = int(ird_cfg.get("in_channels", 3))
    model = cfg.model
    rep = ext.load_weights(model, checkpoint, in_ch=in_ch) if checkpoint else None
    if rep and (rep["unmatched"] or rep["missed"]):
        raise RuntimeError(f"checkpoint does not match config: unmatched {rep['unmatched'][:3]}, "
                           f"missed {rep['missed'][:3]}")
    if not checkpoint and in_ch != 3:
        ext.expand_stem(model, in_ch)
    model = model.deploy().to(device).eval()
    val = cfg.yaml_cfg["val_dataloader"]["dataset"]
    n_cls = cfg.yaml_cfg["num_classes"]
    names = ["ship", "tank", "aircraft", "civil_vehicle"] if n_cls == 4 else [f"class_{i}" for i in range(n_cls)]
    try:
        import json
        cats = sorted(json.load(open(val["ann_file"]))["categories"], key=lambda c: c["id"])
        if len(cats) == n_cls:
            names = [c["name"].replace(",", " ") for c in cats]
    except Exception:
        pass
    info = {
        "arm": ird_cfg.get("arm"),
        "in_channels": in_ch,
        "out_hw": list(val.get("out_hw", [512, 640])),
        "modality": val.get("modality", "ir"),
        "ir_params": val.get("ir_params", {}),
        "vis_params": val.get("vis_params", {"tone": "linear"}),
        "num_classes": cfg.yaml_cfg["num_classes"],
        "class_names": names,
        "num_queries": cfg.yaml_cfg["RTDETRTransformerv2"]["num_queries"],
        "config": os.path.abspath(config),
        "checkpoint": os.path.abspath(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
    }
    return model, info
