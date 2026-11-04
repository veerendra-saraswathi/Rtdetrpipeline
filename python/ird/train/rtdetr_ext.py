"""Extensions registered into the official RT-DETRv2 code (lyuwenyu/RT-DETR, rtdetrv2_pytorch).

Import this module after the RT-DETRv2 folder is on sys.path (see ird/train/paths.py).

- IRDDetection: COCO-style dataset that reads our canonical JSON, builds the network input with
  the shared C++ preprocessing, and supports the arms:
      modality 'ir'     IR copied to 3 channels                     (A-IR, B-ft-IR)
               'vis'    RGB                                          (A-VIS, B-ft-VIS)
               'joint'  every IR frame plus every visible frame      (B)
               'fusion' 4 channels [R, G, B, IR]                     (C)
- IRDPhotometric: brightness/contrast/noise on all channels together. Replaces
  RandomPhotometricDistort, which changes hue/saturation and permutes channels (wrong for IR
  and for the 4-channel input).
- expand_stem / load_weights: 4-channel first conv for arm C, initialised from 3-channel weights.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from src.core import register
from src.data.dataset.coco_dataset import CocoDetection
from src.data._misc import convert_to_tv_tensor, Image as TVImage

from ird.data.io import read_frame
from ird.data.preproc import BandParams, build_input

STEM_KEY = "backbone.conv1.conv1_1.conv.weight"


@register()
class IRDDetection(CocoDetection):
    __inject__ = ["transforms"]
    __share__ = []

    def __init__(self, ann_file, data_root, modality="ir", transforms=None, out_hw=(512, 640),
                 ir_params=None, vis_params=None, gray_p=0.0, max_items=None):
        super().__init__(img_folder=data_root, ann_file=ann_file, transforms=transforms,
                         return_masks=False, remap_mscoco_category=False)
        self.data_root = data_root
        self.modality = modality
        self.out_hw = tuple(out_hw)
        self.ir_params = BandParams(**(ir_params or {}))
        self.vis_params = BandParams(**(vis_params or {"tone": "linear"}))
        self.gray_p = float(gray_p)
        has_vis = [i for i in self.ids if self.coco.imgs[i]["files"].get("vis")]
        if modality == "ir":
            self.items = [(i, "ir") for i in self.ids]
        elif modality == "vis":
            self.items = [(i, "vis") for i in has_vis]
        elif modality == "fusion":
            self.items = [(i, "fusion") for i in has_vis]
        elif modality == "joint":
            self.items = [(i, "ir") for i in self.ids] + [(i, "vis") for i in has_vis]
        else:
            raise ValueError(f"unknown modality {modality!r}")
        if max_items:
            self.items = self.items[: int(max_items)]

    def __len__(self):
        return len(self.items)

    def load_item(self, idx):
        img_id, mod = self.items[idx]
        entry = self.coco.imgs[img_id]
        bands = read_frame(self.data_root, entry, mod)
        x, info = build_input(bands, mod, self.out_hw, self.ir_params, self.vis_params)
        if mod == "vis" and self.gray_p > 0 and random.random() < self.gray_p:
            g = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
            x = np.stack([g, g, g]).astype(np.float32)
        image = TVImage(torch.from_numpy(x))

        anns = [a for a in self.coco.imgToAnns[img_id]]
        boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] *= info["scale_x"]
        boxes[:, 1::2] *= info["scale_y"]
        H, W = self.out_hw
        boxes[:, 0::2].clamp_(0, W)
        boxes[:, 1::2].clamp_(0, H)
        labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        # Ignore-flagged (< 16 px) boxes stay in the training targets: they are real targets.
        target = {
            "image_id": torch.tensor([img_id]),
            "boxes": convert_to_tv_tensor(boxes[keep], key="boxes", spatial_size=(H, W)),
            "labels": labels[keep],
            "area": torch.tensor([a["area"] for a in anns], dtype=torch.float32)[keep],
            "iscrowd": torch.zeros(int(keep.sum()), dtype=torch.int64),
            # Postprocessor scales normalised boxes by this; dividing by the scale maps back to
            # original pixels because the image sits at the top-left of the canvas.
            "orig_size": torch.tensor([W / info["scale_x"], H / info["scale_y"]]).round().to(torch.int64),
            "idx": torch.tensor([idx]),
        }
        return image, target


@register()
class IRDPhotometric(T.Transform):
    """Random gain/offset/contrast and optional Gaussian noise, same for all channels.
    Inputs are float images in [0, 1]."""

    def __init__(self, p=0.5, brightness=0.1, contrast=(0.8, 1.2), noise_std=0.0):
        super().__init__()
        self.p, self.brightness, self.contrast, self.noise_std = p, brightness, tuple(contrast), noise_std

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        img, rest = sample[0], sample[1:]
        if torch.rand(1).item() < self.p:
            c = random.uniform(*self.contrast)
            b = random.uniform(-self.brightness, self.brightness)
            m = img.mean()
            out = (img - m) * c + m + b
            if self.noise_std > 0:
                out = out + torch.randn_like(out) * random.uniform(0, self.noise_std)
            img = TVImage(out.clamp(0, 1))
        return (img, *rest)


def stem_in_channels(model):
    return dict(model.named_parameters())[STEM_KEY].shape[1]


def expand_stem(model, in_ch):
    """Make the first conv take `in_ch` channels. New channels get the mean of the RGB filters;
    all filters are scaled by 3/in_ch so a grey input gives the same response as before."""
    conv = model.backbone.conv1.conv1_1.conv
    if conv.in_channels == in_ch:
        return model
    w = conv.weight.data
    extra = w.mean(dim=1, keepdim=True).repeat(1, in_ch - w.shape[1], 1, 1)
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    bias=conv.bias is not None)
    new.weight.data = torch.cat([w, extra], dim=1) * (w.shape[1] / in_ch)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    model.backbone.conv1.conv1_1.conv = new
    return model


def checkpoint_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in state:
        return state["ema"]["module"]
    return state.get("model", state)


def load_weights(model, path, in_ch=3, strict_shapes=False):
    """Load a checkpoint into `model` (built with 3 input channels), then give it `in_ch` inputs.
    Works for COCO checkpoints (80 classes: head skipped), our 3-channel checkpoints, and our
    4-channel checkpoints. Returns a report dict."""
    params = checkpoint_state(path)
    ck_in = params[STEM_KEY].shape[1] if STEM_KEY in params else 3
    if ck_in != 3:
        expand_stem(model, ck_in)
    own = model.state_dict()
    matched = {k: v for k, v in params.items() if k in own and own[k].shape == v.shape}
    unmatched = [k for k, v in params.items() if k in own and own[k].shape != v.shape]
    missed = [k for k in own if k not in params]
    if strict_shapes and (unmatched or missed):
        raise RuntimeError(f"checkpoint mismatch: unmatched={unmatched[:5]} missed={missed[:5]}")
    model.load_state_dict(matched, strict=False)
    if in_ch != stem_in_channels(model):
        expand_stem(model, in_ch)
    return {"loaded": len(matched), "unmatched": unmatched, "missed": missed, "ckpt_in_ch": ck_in}
