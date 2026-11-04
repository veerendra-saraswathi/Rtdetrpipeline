# C++ side on Windows: build, check preprocessing, ONNX -> TensorRT engines (FP16, and with
# -Precisions also INT8), C++ inference, evaluate, compare with PyTorch, draw detection grids.
# Gives TensorRT accuracy (and desktop latency) before moving to the Jetson.
# Needs: Visual Studio 2022 Build Tools (C++), CUDA Toolkit 12.x, TensorRT 10.x for Windows (zip),
# all matching your driver. Engines built here run only on this GPU; rebuild them on the Orin.
# Run scripts\export_all.ps1 first (writes export\*.onnx). Check the PC with scripts\check_trt_windows.ps1.
#   powershell -ExecutionPolicy Bypass -File scripts\cpp_windows.ps1 -TensorRT C:\TensorRT-10.x.x.x
#   ... -Precisions fp16,int8,int8mix      also INT8 engines (calibrated on 300 TRAINING frames)
# Precisions: fp16 = FP16 engine (threshold from PyTorch on val)
#             int8 = INT8 everywhere TensorRT can, FP16 elsewhere
#             int8mix = INT8 backbone + encoder, decoder kept FP16
# INT8 thresholds are chosen again on val with the INT8 engine itself (its scores can shift).
param([Parameter(Mandatory = $true)][string]$TensorRT, [int]$Repeat = 3, [string[]]$Seeds = @("0"),
      [string[]]$Precisions = @("fp16"), [int]$CalibFrames = 300)
# "Continue": Windows PowerShell 5.1 treats any stderr text from python/cmake as fatal under "Stop".
# Failures are caught with $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
# "powershell -File" passes "fp16,int8" as one string: split on commas.
$Precisions = @($Precisions | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Seeds = @($Seeds | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$bad = @($Precisions | Where-Object { @("fp16", "int8", "int8mix") -notcontains $_ })
if ($bad.Count -gt 0) { Write-Host "Unknown precision(s): $($bad -join ', '). Use fp16, int8, int8mix" -ForegroundColor Red; exit 1 }
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = (Resolve-Path .venv\Scripts\python.exe).Path
$env:PYTHONPATH = "python"
$env:PATH = "$TensorRT\lib;$TensorRT\bin;$env:PATH"
$TEST = "data/airst_v0.1s/airst_test.json"; $VAL = "data/airst_v0.1s/airst_val.json"; $TRAIN = "data/airst_v0.1s/airst_train.json"
New-Item -ItemType Directory -Force preds, reports | Out-Null
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "failed: $what" } }
if (-not (Get-ChildItem export\*.onnx -ErrorAction SilentlyContinue)) { throw "no export\*.onnx: run scripts\export_all.ps1 first" }

# cmake: the one on PATH, else a copy installed into .venv with pip (VS Build Tools keeps its own off PATH)
$CMAKE = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $CMAKE) {
    if (-not (Test-Path .venv\Scripts\cmake.exe)) { & $PY -m pip install cmake; Check "pip install cmake" }
    $CMAKE = (Resolve-Path .venv\Scripts\cmake.exe).Path
}
& $CMAKE -S cpp -B build -DTENSORRT_ROOT="$TensorRT"; Check "cmake configure"
& $CMAKE --build build --config Release -j; Check "cmake build"
$bin = "build\Release"
if (-not (Test-Path "$bin\ird_infer.exe")) { throw "ird_infer.exe not built: cmake did not find CUDA or TensorRT (see the configure output above)" }
& "$bin\test_preproc_golden.exe" export\golden; Check "golden preprocessing test"

$needInt8 = @($Precisions | Where-Object { $_ -like "int8*" }).Count -gt 0
if ($needInt8) {
    if (-not (Test-Path export\train_list.tsv)) { & $PY -m ird.infer.make_list --ann $TRAIN -o export\train_list.tsv; Check "train list" }
    if (-not (Test-Path export\val_list.tsv)) { & $PY -m ird.infer.make_list --ann $VAL -o export\val_list.tsv; Check "val list" }
}

Get-ChildItem export\*.onnx | Where-Object { $_.BaseName -ne "e00_coco" } | ForEach-Object {
    $name = $_.BaseName; $onnx = $_.FullName
    foreach ($prec in $Precisions) {
        $suf = if ($prec -eq "fp16") { "" } else { "_$prec" }
        $eng = "export\$name$suf.engine"; $pred = "preds\${name}_trt$suf.json"
        Write-Host "`n=== $name $prec" -ForegroundColor Cyan
        if (-not (Test-Path $eng)) {
            if ($prec -eq "fp16") {
                & "$bin\ird_build_engine.exe" --onnx $onnx --engine $eng --fp16 --timing-cache export\timing.cache
            } else {
                $extra = if ($prec -eq "int8mix") { @("--no-int8", "decoder") } else { @() }
                & "$bin\ird_build_engine.exe" --onnx $onnx --engine $eng --int8 --calib-list export\train_list.tsv --calib-root .. `
                    --calib-cfg "export\$name.onnx.runtime.txt" --calib-n $CalibFrames --calib-cache "export\$name.int8.calib" `
                    --timing-cache export\timing.cache @extra
            }
            Check "engine $name $prec"
        }
        & "$bin\ird_infer.exe" --engine $eng --cfg "export\$name.onnx.runtime.txt" --list export\test_list.tsv `
            --root .. --out $pred --latency "reports\${name}${suf}_latency_desktop.json" --repeat $Repeat
        Check "inference $name $prec"
        if ($prec -eq "fp16") {
            if (Test-Path "export\$name.tau.json") { Copy-Item "export\$name.tau.json" "$pred.tau.json" -Force }
        } else {   # INT8 scores can shift: choose its threshold on val with the INT8 engine itself
            & "$bin\ird_infer.exe" --engine $eng --cfg "export\$name.onnx.runtime.txt" --list export\val_list.tsv `
                --root .. --out "preds\${name}_val_trt$suf.json" --warmup 5 --repeat 1 | Out-Null
            Check "val inference $name $prec"
            & $PY -m ird.eval.choose_tau --gt $VAL --pred "preds\${name}_val_trt$suf.json" -o "$pred.tau.json"; Check "tau $name $prec"
        }
        & $PY -m ird.eval.evaluate --gt $TEST --pred $pred --out "reports\${name}_trt$suf" --boot 200 | Out-Null
        Check "evaluate $name $prec"
    }
}

# IR models side by side: TensorRT comparison, TensorRT vs PyTorch, and detection grids
foreach ($s in $Seeds) {
    $m = @(@(@("A-IR", "a_ir"), @("B", "b_joint"), @("B-ft-IR", "b_ft_ir")) | Where-Object { Test-Path "preds\$($_[1])_s${s}_trt.json" })
    if ($m.Count -lt 2) { continue }
    $trt = @(); $both = @(); $grid = @()
    foreach ($x in $m) {
        $t = "preds\$($x[1])_s${s}_trt.json"; $p = "preds\$($x[1])_s${s}_ir_torch.json"
        $trt += @("--arm", $x[0], $t); $grid += @("--col", $x[0], $t)
        $both += @("--arm", "$($x[0]) TRT", $t)
        if (Test-Path $p) { $both += @("--arm", "$($x[0]) torch", $p) }
    }
    & $PY -m ird.eval.compare --gt $TEST --baseline A-IR --out "reports\compare_ir_trt_s$s" @trt; Check "compare TensorRT"
    & $PY -m ird.eval.compare --gt $TEST --baseline "A-IR torch" --out "reports\parity_ir_s$s" @both; Check "compare TensorRT vs torch"
    & $PY -m ird.eval.visualize_grid --gt $TEST --band ir --out "reports\grid_ir_trt_s$s" @grid; Check "grids"
}
foreach ($s in $Seeds) {
    if ($Precisions.Count -gt 1 -or $Precisions[0] -ne "fp16") {
        & $PY -m ird.eval.precision_report --seed $s --out "reports\precision_summary_s$s"; Check "precision report"
        foreach ($x in @(@("A-IR", "a_ir"), @("B", "b_joint"), @("B-ft-IR", "b_ft_ir"))) {
            $arms = @(); $base = $null
            $p = "preds\$($x[1])_s${s}_ir_torch.json"; if (Test-Path $p) { $arms += @("--arm", "$($x[0]) torch", $p); $base = "$($x[0]) torch" }
            foreach ($prec in $Precisions) {
                $suf = if ($prec -eq "fp16") { "" } else { "_$prec" }
                $t = "preds\$($x[1])_s${s}_trt$suf.json"; if (Test-Path $t) { $arms += @("--arm", "$($x[0]) $prec", $t) }
            }
            if ($base -and ($arms | Where-Object { $_ -eq "--arm" }).Count -ge 2) {
                & $PY -m ird.eval.compare --gt $TEST --baseline $base --out "reports\precision_$($x[1])_s$s" @arms | Out-Null
                Check "precision compare $($x[1])"
            }
        }
    }
}
Write-Host "`nDone." -ForegroundColor Green
Write-Host "  reports\compare_ir_trt_s0.md   A-IR vs B vs B-ft-IR, TensorRT FP16"
Write-Host "  reports\parity_ir_s0.md        each model, TensorRT vs PyTorch (should agree within ~0.5 point)"
Write-Host "  reports\grid_ir_trt_s0_p*.jpg  detection grids, one column per model"
Write-Host "  reports\*_latency_desktop.json  C++ timing per frame on this GPU"
Write-Host "  reports\precision_summary_s0.md   every model at every precision: accuracy, size, speed (with -Precisions)"
Write-Host "  reports\precision_<arm>_s0.md     PyTorch vs FP16 vs INT8 for one model, with significance"
