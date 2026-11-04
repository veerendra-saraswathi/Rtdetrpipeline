#!/usr/bin/env bash
# Export every trained arm/seed to ONNX (+ .json, .runtime.txt) and write the C++ frame list.
# Run on the training machine; copy export/ to the Jetson.
set -euo pipefail
export PYTHONPATH=python IRD_REQUIRE_CPP=1
SEEDS=${SEEDS:-"0 1 2"}
TEST=${TEST:-data/airst_v0.1s/airst_test.json}
mkdir -p export
for d in runs/*/s*; do
  arm=$(basename $(dirname $d)); s=$(basename $d)
  [ -f $d/best.pth ] || continue
  python -m ird.export.export_onnx -c configs/arms/$arm.yml -r $d/best.pth -o export/${arm}_${s}.onnx \
      --check-images $TEST
  band=$(python -c "import json;print(json.load(open('export/${arm}_${s}.onnx.json'))['modality'])")
  tau=preds/${arm}_${s}_${band}_torch.json.tau.json          # tau chosen on val (train_arms.sh)
  [ -f $tau ] && cp $tau export/${arm}_${s}.tau.json
done
python -m ird.infer.make_list --ann $TEST -o export/test_list.tsv
python tests/make_golden.py --out export/golden --ann $TEST
echo "copy export/ (and the test images) to the Jetson, then run scripts/jetson_run.sh"
