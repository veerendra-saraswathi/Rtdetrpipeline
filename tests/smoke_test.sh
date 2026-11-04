#!/usr/bin/env bash
# End-to-end smoke test on tiny synthetic data, CPU is enough (a few minutes):
# every arm: train 2 epochs -> export ONNX -> predict (PyTorch and ONNX) -> evaluate; then compare.
# Checks that the pipeline runs and that PyTorch and ONNX agree. Accuracy is meaningless here.
#   RTDETR_ROOT=third_party/RT-DETR/rtdetrv2_pytorch bash tests/smoke_test.sh [COCO_CKPT]
set -euo pipefail
export PYTHONPATH=python
W=${SMOKE_DIR:-/tmp/ird_smoke}
CKPT=${1:-}
rm -rf $W && python tests/make_synthetic.py --out $W/root > /dev/null
U="train_dataloader.dataset.ann_file=$W/root/data/syn_train.json train_dataloader.dataset.data_root=$W/root \
val_dataloader.dataset.ann_file=$W/root/data/syn_val.json val_dataloader.dataset.data_root=$W/root \
epoches=2 print_freq=5 train_dataloader.total_batch_size=2 val_dataloader.total_batch_size=2 \
train_dataloader.num_workers=0 val_dataloader.num_workers=0 lr_warmup_scheduler.warmup_duration=2 ema.warmups=2 \
train_dataloader.dataset.transforms.policy.epoch=1 train_dataloader.collate_fn.stop_epoch=1"
T=$W/root/data/syn_test.json
for arm in a_ir a_vis b_joint c_fusion; do
  tune=""; [ -n "$CKPT" ] && tune="--tuning $CKPT"
  python -m ird.train.train -c configs/arms/$arm.yml $tune --seed 0 --output-dir $W/runs/$arm -d cpu -u $U > $W/$arm.train.log 2>&1 \
    || { tail -30 $W/$arm.train.log; exit 1; }
  echo "trained $arm"
  python -m ird.export.export_onnx -c configs/arms/$arm.yml -r $W/runs/$arm/last.pth -o $W/export/$arm.onnx \
    --check-images $T --data-root $W/root > $W/$arm.export.log 2>&1 || { tail -30 $W/$arm.export.log; exit 1; }
  for b in torch onnx; do
    a="-c configs/arms/$arm.yml -r $W/runs/$arm/last.pth"; [ $b = onnx ] && a="--onnx $W/export/$arm.onnx"
    python -m ird.infer.predict --backend $b $a --ann $T --data-root $W/root -o $W/preds/${arm}_$b.json --device cpu
    python -m ird.eval.evaluate --gt $T --pred $W/preds/${arm}_$b.json --out $W/reports/${arm}_$b --boot 50 > /dev/null
  done
done
python - <<EOF
import json
for arm in ["a_ir", "a_vis", "b_joint", "c_fusion"]:
    t = json.load(open("$W/preds/%s_torch.json" % arm))["detections"]
    o = json.load(open("$W/preds/%s_onnx.json" % arm))["detections"]
    top = lambda d: sorted(d, key=lambda x: -x["score"])[:50]
    dmax = max((abs(a["score"] - b["score"]) for a, b in zip(top(t), top(o))), default=0)
    print(f"{arm}: torch {len(t)} dets, onnx {len(o)} dets, top-50 max score diff {dmax:.2e}")
    assert dmax < 1e-3, arm
EOF
python -m ird.eval.compare --gt $T --baseline A-IR --boot 50 --out $W/reports/compare \
  --arm A-IR $W/preds/a_ir_torch.json --arm B $W/preds/b_joint_torch.json --arm C $W/preds/c_fusion_torch.json > /dev/null
echo "smoke test passed ($W)"
