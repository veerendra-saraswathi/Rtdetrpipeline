#!/usr/bin/env bash
# E-00 step zero. Needs COCO val2017 in $COCO (images in $COCO/val2017, annotations in $COCO/annotations).
#  1. Official PyTorch eval of the released checkpoint (must match the published AP).
#  2. Same weights through our path: PyTorch vs ONNX (here) and TensorRT (on the Jetson, jetson_run.sh).
set -euo pipefail
export PYTHONPATH=python
COCO=${COCO:?set COCO to the COCO folder}; CKPT=${COCO_CKPT:?set COCO_CKPT}
RT=${RTDETR_ROOT:-third_party/RT-DETR/rtdetrv2_pytorch}
( cd $RT && python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r "$CKPT" --test-only \
    -u val_dataloader.dataset.img_folder=$COCO/val2017 \
       val_dataloader.dataset.ann_file=$COCO/annotations/instances_val2017.json ) | tee reports/e00_official.txt
python -m ird.infer.coco_to_canonical --coco-ann $COCO/annotations/instances_val2017.json --img-dir val2017 \
    -o data/coco_val2017_canonical.json
python -m ird.export.export_onnx -c configs/e00_coco_r18.yml -r "$CKPT" -o export/e00_coco.onnx
for b in torch onnx; do
  args="-c configs/e00_coco_r18.yml -r $CKPT"; [ $b = onnx ] && args="--onnx export/e00_coco.onnx"
  python -m ird.infer.predict --backend $b $args --ann data/coco_val2017_canonical.json --data-root $COCO \
      -o preds/e00_$b.json
  python -m ird.eval.evaluate --gt data/coco_val2017_canonical.json --pred preds/e00_$b.json --min-size 0 \
      --boot 0 --out reports/e00_$b > /dev/null
done
python -m ird.infer.make_list --ann data/coco_val2017_canonical.json -o export/e00_list.tsv
grep -h "COCO-style" reports/e00_*.md
