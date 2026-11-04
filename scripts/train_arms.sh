#!/usr/bin/env bash
# Train every arm with 3 seeds, predict on test with PyTorch, evaluate, compare.
# Run from the repo root on the GPU training machine:
#   COCO_CKPT=weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth bash scripts/train_arms.sh
# Optional: SEEDS="0 1 2"  ARMS="a_ir a_vis b_joint c_fusion"  EXTRA="-u epoches=72"
set -euo pipefail
export PYTHONPATH=python IRD_REQUIRE_CPP=1
COCO_CKPT=${COCO_CKPT:?set COCO_CKPT to the RT-DETRv2-S COCO checkpoint}
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"a_ir a_vis b_joint c_fusion"}
TEST=${TEST:-data/airst_v0.1s/airst_test.json}
AMP=${AMP:---use-amp}

for arm in $ARMS; do
  for s in $SEEDS; do
    out=runs/$arm/s$s
    [ -f $out/best.pth ] || python -m ird.train.train -c configs/arms/$arm.yml --tuning "$COCO_CKPT" \
        --seed $s --output-dir $out $AMP ${EXTRA:-}
    if [ $arm = b_joint ]; then          # B then per-band fine-tunes, starting from B's best
      for ft in b_ft_ir b_ft_vis; do
        [ -f runs/$ft/s$s/best.pth ] || python -m ird.train.train -c configs/arms/$ft.yml \
            --tuning $out/best.pth --seed $s --output-dir runs/$ft/s$s $AMP
      done
    fi
  done
done

mkdir -p preds reports
# PyTorch predictions + evaluation on test (B is scored on IR and on visible).
VAL=${VAL:-data/airst_v0.1s/airst_val.json}
# pred ARM SEED BAND: predict val and test; choose tau on val; evaluate test at that tau.
pred() { local m="-c configs/arms/$1.yml -r runs/$1/s$2/best.pth --modality $3" p=preds/${1}_s$2_${3}
         python -m ird.infer.predict --backend torch $m --ann $VAL -o ${p}_val_torch.json
         python -m ird.infer.predict --backend torch $m --ann $TEST -o ${p}_torch.json
         python -m ird.eval.choose_tau --gt $VAL --pred ${p}_val_torch.json -o ${p}_torch.json.tau.json
         python -m ird.eval.evaluate --gt $TEST --pred ${p}_torch.json --out reports/${1}_s$2_${3}_torch > /dev/null; }
for s in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      a_ir) pred a_ir $s ir ;;
      a_vis) pred a_vis $s vis ;;
      c_fusion) pred c_fusion $s fusion ;;
      b_joint) pred b_joint $s ir; pred b_joint $s vis; pred b_ft_ir $s ir; pred b_ft_vis $s vis ;;
    esac
  done
done

# Comparisons. IR arms against A-IR on the IR test frames; visible arms against A-VIS.
list() { for s in $SEEDS; do printf "preds/%s_s%s_%s_torch.json " $1 $s $2; done; }
python -m ird.eval.compare --gt $TEST --baseline A-IR --out reports/compare_ir_torch \
  --arm A-IR $(list a_ir ir) --arm B $(list b_joint ir) --arm B-ft-IR $(list b_ft_ir ir) --arm C $(list c_fusion fusion)
python -m ird.eval.compare --gt $TEST --baseline A-VIS --out reports/compare_vis_torch \
  --arm A-VIS $(list a_vis vis) --arm B $(list b_joint vis) --arm B-ft-VIS $(list b_ft_vis vis) \
  --arm C $(list c_fusion fusion)
echo "done: see reports/*.md"
