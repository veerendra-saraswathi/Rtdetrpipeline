# IRD pipeline: RT-DETRv2-R18 on AIRST-CS, three training approaches, C++/TensorRT on Jetson

One fixed pipeline for every experiment:

```
canonical data (JSON)  ->  C++ preprocessing  ->  training (RT-DETRv2, PyTorch)  ->  ONNX
        ->  TensorRT FP16 engine (C++)  ->  C++ inference + latency  ->  one evaluator (RCI metrics)
```

The same C++ preprocessing code runs in training (Python module `ird_preproc`) and on the Jetson
(`ird_runtime`). The same evaluator scores PyTorch, ONNX and TensorRT outputs for every arm.

## What is in this repository

Code, configs, scripts and tests only. Not included (large, generated, or licensed separately):

| Not in Git | How to get it |
|---|---|

| `weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` | Downloaded by `setup_windows.ps1` (RT-DETRv2-S COCO checkpoint from the RT-DETR README) |
| `data/airst_v0.1s/` | `python -m ird.data.prepare_airst --root .. --out data/airst_v0.1s` (needs the AIRST-CS ship subset next to this folder) |
| `runs/`, `preds/`, `reports/` | Written by `train_arms.ps1` / `train_arms.sh` |
| `export/` (ONNX, TensorRT engines, INT8 calibration) | Written by `export_all.ps1` and `cpp_windows.ps1`; engines must be rebuilt on each GPU (Orin included) |

TensorRT precisions: `cpp_windows.ps1 -Precisions fp16,int8,int8mix` builds FP16, INT8 and INT8 with an
FP16 decoder. INT8 is calibrated on 300 training frames and its threshold is chosen again on val.

## Windows quick start (VS Code, PowerShell terminal, from this folder)

```
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1                           # once
powershell -ExecutionPolicy Bypass -File scripts\train_arms.ps1 -Arms a_ir -Seeds 0 -Epochs 12 # quick check
powershell -ExecutionPolicy Bypass -File scripts\train_arms.ps1                                # all arms, 3 seeds
powershell -ExecutionPolicy Bypass -File scripts\export_all.ps1                                # ONNX
powershell -ExecutionPolicy Bypass -File scripts\cpp_windows.ps1 -TensorRT C:\TensorRT-10.x  # C++ TensorRT
```
Needs an NVIDIA GPU. `setup_windows.ps1` checks it, makes `.venv`, installs PyTorch for your CUDA
driver, builds the C++ preprocessing (needs Visual Studio 2022 Build Tools, "Desktop development
with C++"), clones RT-DETRv2, downloads the COCO checkpoint and runs the tests.
`cpp_windows.ps1` also needs CMake, the CUDA Toolkit and TensorRT for Windows. Engines built on
the PC are for checking accuracy only; build the final engines on the Orin (`jetson_run.sh`).
If GPU memory runs out, add `-Batch 4`.

## What is in here

| Path | What |
|---|---|
| `python/ird/data/prepare_airst.py` | AIRST-CS ship subset → split JSONs (official split, 174 duplicate pairs resolved, < 16 px marked ignore) |
| `cpp/include/ird/preproc.h`, `cpp/src/preproc.cpp` | Preprocessing spec + C++ code (raw uint16 → float CHW) |
| `cpp/python/bind_preproc.cpp`, `setup.py` | Python module `ird_preproc` (same C++ code, for training) |
| `python/ird/data/preproc_ref.py` | Independent NumPy copy of the spec (tests, fallback) |
| `python/ird/train/` | RT-DETRv2 integration: dataset, IR-safe augmentation, 4-channel stem, trainer |
| `configs/base_rtdetrv2_r18.yml`, `configs/arms/*.yml` | All hyperparameters; one file per arm |
| `python/ird/export/export_onnx.py` | ONNX export (raw logits + boxes), runtime config, ONNX vs PyTorch check |
| `python/ird/infer/` | Shared decoder, PyTorch/ONNX prediction, frame list for C++ |
| `cpp/include/ird/ird_api.h`, `cpp/runtime/` | C API + TensorRT runtime (FP16, CUDA graph, pinned memory) |
| `cpp/tools/ird_build_engine.cpp`, `cpp/tools/ird_infer.cpp` | ONNX → engine; inference + latency report |
| `python/ird/eval/` | Evaluator (Pd, AO, FAR, CS, AP, size buckets, CIs), arm comparison, E-10 label noise |
| `scripts/*.sh` | The runs in order |
| `tests/` | Preprocessing, decoder and metric tests; synthetic data; golden files |

## Data used now: AIRST-CS ship subset (DS-v0.1s)

Built with `prepare_airst.py` on 29 Oct 2026 (already done; files in `data/airst_v0.1s/`):

| Split | Frames | With visible | Ship boxes (ignored < 16 px) | civil_vehicle boxes (ignored) |
|---|---|---|---|---|
| train | 772 | 770 | 4,111 (1,385) | 2,334 (1,536) |
| val | 178 | 178 | 922 (376) | 470 (310) |
| test | 197 | 197 | 793 (208) | 566 (388) |

- Official AIRST-CS split kept. Of each duplicate pair, one copy is kept, in the stricter split
  (test > val > train), so no frame is in two splits. Both label sets go to `duplicates_e10.json`.
- Classes use the master list: 0 ship, 1 tank, 2 aircraft, 3 civil_vehicle. The model has 4 outputs.
- Boxes below 16 px (sqrt(w·h)) stay in training but are ignored in every metric.
- Limits: only frames that contain ships (no background-only frames). 8-bit AGC IR, not raw 14-bit.
  Re-download full AIRST-CS for final numbers and false-alarm runs.

To rebuild: `PYTHONPATH=python python -m ird.data.prepare_airst --root .. --out data/airst_v0.1s`

## Preprocessing (identical in training and on the Jetson)

Raw uint16 frame (any bit depth 8-16) → float32 CHW in [0, 1], 512×640:
1. Tone range: IR uses the 0.5 / 99.5 percentiles of the frame; visible uses the full range.
2. Tone map: `clamp((v - lo) / (hi - lo), 0, 1)`.
3. Channels: IR copied to 3 channels (arms A-IR, B); fusion = [R, G, B, IR] (arm C).
4. Image placed at the top-left of the 512×640 canvas (640×480 frames get 32 padded rows);
   larger frames are scaled down first. Boxes map back by dividing by the scale.

Full spec: `cpp/include/ird/preproc.h`. Tests: C++ = NumPy reference exactly (`tests/test_preproc.py`,
`cpp/tests/test_preproc_golden.cpp`).

## Training arms

All arms: RT-DETRv2-S (ResNet18-vd), start from the COCO checkpoint, 72 epochs, batch 8,
AdamW 1e-4 (backbone 1e-5), EMA, multi-scale 416-608 px high, same augmentation
(brightness/contrast/noise, zoom-out, IoU crop, flip; no hue/saturation changes), 3 seeds.

| Arm | Config | Train on | Validate on | Input |
|---|---|---|---|---|
| A-IR | `a_ir.yml` | IR frames | IR | 1×3×512×640 |
| A-VIS | `a_vis.yml` | visible frames | visible | 1×3×512×640 |
| B | `b_joint.yml` | every IR frame + every visible frame | IR | 1×3×512×640 |
| B-ft-IR / B-ft-VIS | `b_ft_ir.yml`, `b_ft_vis.yml` | B's best, then 24 epochs on one band | that band | 1×3×512×640 |
| C | `c_fusion.yml` | paired frames, 4 channels | fusion | 1×4×512×640 |

Arm C: the first conv gets a 4th input channel initialised from the mean of the RGB filters
(all filters scaled by 3/4). Note: RCI has no paired sensors and no calibration, so C cannot be
deployed; it is run for evidence only.

Latency is the same for A and B (same model and input). C adds one input channel only. Latency
is therefore measured per engine, and the arms are compared on accuracy.

## How to run

### 0. Set up (training machine with an NVIDIA GPU)
```

pip install torch torchvision faster-coco-eval pycocotools scipy pyyaml tensorboard onnx onnxruntime pybind11
pip install ./                       # builds ird_preproc (C++). Windows needs MSVC Build Tools.
# COCO checkpoint for RT-DETRv2-S (from the RT-DETR README, "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"):
mkdir weights && <download it into weights/>
```
Check: `PYTHONPATH=python python tests/test_preproc.py && PYTHONPATH=python python tests/test_metrics.py`

### 1. E-00 step zero (reproducibility)
`COCO=/path/to/coco COCO_CKPT=weights/... bash scripts/e00_coco.sh`
- Official eval must match the published AP.
- Our path: PyTorch vs ONNX here, TensorRT on the Jetson. Note: our preprocessing does not upscale
  small COCO images (official eval stretches to 640×640), so compare our three backends with
  each other, and the official number with the paper.

### 2. Train and score all arms (PyTorch)
`COCO_CKPT=weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth bash scripts/train_arms.sh`
Outputs: `runs/<arm>/s<seed>/` (best.pth, log.txt, run_info.json), `preds/`, `reports/*.md`,
`reports/compare_ir_torch.md`, `reports/compare_vis_torch.md`.

One run by hand:
```
PYTHONPATH=python python -m ird.train.train -c configs/arms/a_ir.yml --tuning weights/<coco>.pth --seed 0 --output-dir runs/a_ir/s0 --use-amp
```

### 3. Export (training machine) → TensorRT (Jetson)
```
bash scripts/export_all.sh                         # export/*.onnx, *.runtime.txt, test_list.tsv, golden/
# copy the repo (with export/) and the AIRST images to the Orin Nano, then:
DATA_ROOT=/path/containing/AIRST-CS_ship_only bash scripts/jetson_run.sh
```
`jetson_run.sh` builds the C++ code, checks preprocessing against golden files, builds FP16 engines
on the device, runs `ird_infer` (detections + latency p50/p95/p99, budget 40 ms) and evaluates.
Before reporting any TensorRT number, compare it with the PyTorch number of the same model
(`reports/<arm>_s<seed>_trt.md` vs `reports/<arm>_s<seed>_<band>_torch.md`). If boxes drift in FP16,
rebuild with `--fp32 <layer substring>` (see `ird_build_engine --list-layers`).

### 4. Compare arms on TensorRT results
```
PYTHONPATH=python python -m ird.eval.compare --gt data/airst_v0.1s/airst_test.json --baseline A-IR \
  --arm A-IR preds/a_ir_s0_trt.json preds/a_ir_s1_trt.json preds/a_ir_s2_trt.json \
  --arm B preds/b_joint_s0_trt.json ... --out reports/compare_ir_trt
```
(TensorRT predictions of B use the engine's validation band, IR. To score B on visible with
TensorRT, write a runtime file copy with `modality=vis`.)

## Metrics (doc v0.3 §10, RCI answers 29-10-2026)

| Metric | Definition | RCI target |
|---|---|---|
| Pd | matched targets / targets (one-to-one, IoU ≥ 0.5, class-aware) | > 95% |
| AO | sum of IoU of matched targets / targets (missed = 0) | > 90% |
| FAR | unmatched detections / frames; upper 95% bound reported too | 1e-5 per frame |
| CS | mean score of matched detections; ECE also reported | > 0.9 |
| AP, AP50, AP75 | COCO rules, targets ≥ 16 px; checked equal to pycocotools | - |

Operating threshold tau is chosen on the **validation** split with `ird.eval.choose_tau`
(highest tau with Pd >= 95% on val; if none, best F1) and then applied to test. `train_arms.sh`
does this for every run and stores it as `<pred>.tau.json`; `evaluate.py` and `compare.py` read it,
and `jetson_run.sh` reuses the same tau for TensorRT. Never pick tau on test.
Class-agnostic results are reported too.
Each report has 95% bootstrap CIs over test frames. `compare.py` gives mean ± std over seeds and
paired-bootstrap CIs of each arm minus the baseline; a difference counts only if its CI excludes 0.

E-10 (`python -m ird.eval.label_noise --pairs data/airst_v0.1s/duplicates_e10.json --out reports/e10`)
scores one labeling of each duplicate frame against the other.

## Tests done while building (29 Oct 2026, in a Linux workspace without GPU)

- C++ preprocessing = NumPy reference, bit for bit, on 9 cases (8/10/12/14-bit, grey/RGB, 480/512 rows, downscaling).
- C++ golden-file test passes on x86 (run it again on the Jetson).
- C++ decoder = Python decoder on 10 random cases.
- Evaluator AP/AP50/AP75 = pycocotools on 5 random datasets; hand-checked Pd/AO/FAR cases.
- TensorRT runtime, engine builder and inference tool compile cleanly against TensorRT 10.3
  (JetPack 6) and CUDA 12 headers. They have not run on a GPU yet: first run is on the Orin.
- End-to-end smoke test for all four arms, starting from the real COCO checkpoint
  (train → export → PyTorch and ONNX predictions → evaluate → compare): passes; PyTorch and ONNX
  scores agree within 1e-6 (`tests/smoke_test.sh`).
- Learning check on synthetic frames, A-IR from the COCO checkpoint, CPU: val AP50 0.64 after 1 epoch,
  0.77 after 2. Our evaluator gave the same AP50 (77.0%) as RT-DETR's COCO evaluator.
- Checkpoint loading: COCO → 4 classes (11 head tensors skipped), COCO → 4-channel stem,
  B → B-ft, and 4-channel checkpoints reload with 0 mismatches.
- Not yet run on real AIRST training or a GPU: the first full run is yours.

## Third-party code

| Item | Use | License |
|---|---|---| | Apache-2.0 |
| PyTorch, torchvision | training | BSD-3 |
| pybind11 | Python binding of the preprocessing | BSD-3 |
| stb_image.h (`third_party/stb`) | PNG reading in `ird_infer` only (tool, not runtime) | Public domain / MIT |
| CUDA, TensorRT | Jetson runtime | NVIDIA (JetPack) |
