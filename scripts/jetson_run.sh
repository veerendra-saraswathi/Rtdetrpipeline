#!/usr/bin/env bash
# On the Jetson (Orin Nano, JetPack 6): build C++, check preprocessing, build FP16 engines,
# run TensorRT inference + latency, evaluate with the same evaluator.
#   DATA_ROOT=/data/ird bash scripts/jetson_run.sh
# DATA_ROOT is the folder that contains AIRST-CS_ship_only/ (paths in the list are relative to it).
set -euo pipefail
export PYTHONPATH=python
DATA_ROOT=${DATA_ROOT:-..}
TEST=${TEST:-data/airst_v0.1s/airst_test.json}
REPEAT=${REPEAT:-3}

mkdir -p preds reports
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
./build/test_preproc_golden export/golden            # C++ on ARM must match the NumPy reference

for onnx in export/*.onnx; do
  name=$(basename $onnx .onnx)
  [ $name = e00_coco ] && continue                     # handled below
  [ -f export/$name.engine ] || ./build/ird_build_engine --onnx $onnx --engine export/$name.engine --fp16 \
      --timing-cache export/timing.cache
  ./build/ird_infer --engine export/$name.engine --cfg $onnx.runtime.txt --list export/test_list.tsv \
      --root "$DATA_ROOT" --out preds/${name}_trt.json --latency reports/${name}_latency.json --repeat $REPEAT
  [ -f export/$name.tau.json ] && cp export/$name.tau.json preds/${name}_trt.json.tau.json   # same tau as PyTorch
  python3 -m ird.eval.evaluate --gt $TEST --pred preds/${name}_trt.json --out reports/${name}_trt --boot 200 > /dev/null
done
if [ -f export/e00_coco.onnx ] && [ -n "${COCO:-}" ]; then   # E-00 TensorRT leg
  [ -f export/e00_coco.engine ] || ./build/ird_build_engine --onnx export/e00_coco.onnx --engine export/e00_coco.engine --fp16
  ./build/ird_infer --engine export/e00_coco.engine --cfg export/e00_coco.onnx.runtime.txt --list export/e00_list.tsv \
      --root "$COCO" --out preds/e00_trt.json --latency reports/e00_latency.json --warmup 20
  python3 -m ird.eval.evaluate --gt data/coco_val2017_canonical.json --pred preds/e00_trt.json --min-size 0 --boot 0 \
      --out reports/e00_trt > /dev/null
fi
echo "done: reports/*_trt.md and reports/*_latency.json"
