# Export every trained run to ONNX (+ .json, .runtime.txt, val-chosen tau), write the C++ frame
# list and the golden preprocessing files (Windows). Python is needed only here: PyTorch writes the ONNX.
#   powershell -ExecutionPolicy Bypass -File scripts\export_all.ps1
# "Continue": Windows PowerShell 5.1 treats any stderr text from python/cmake as fatal under "Stop".
# Failures are caught with $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = (Resolve-Path .venv\Scripts\python.exe).Path
$env:PYTHONPATH = "python"; $env:IRD_REQUIRE_CPP = "1"
$TEST = "data/airst_v0.1s/airst_test.json"
New-Item -ItemType Directory -Force export | Out-Null
function Run { & $PY @args; if ($LASTEXITCODE -ne 0) { throw "failed: python $args" } }

Get-ChildItem runs -Directory | ForEach-Object {
    $arm = $_.Name
    Get-ChildItem $_.FullName -Directory | Where-Object { Test-Path (Join-Path $_.FullName best.pth) } | ForEach-Object {
        $s = $_.Name; $name = "${arm}_$s"
        Write-Host "=== export $name" -ForegroundColor Cyan
        Run -m ird.export.export_onnx -c "configs/arms/$arm.yml" -r "runs/$arm/$s/best.pth" -o "export/$name.onnx" --check-images $TEST
        $band = (Get-Content "export/$name.onnx.json" | ConvertFrom-Json).modality
        $tau = "preds/${arm}_${s}_${band}_torch.json.tau.json"
        if (Test-Path $tau) { Copy-Item $tau "export/$name.tau.json" -Force }
    }
}
Run -m ird.infer.make_list --ann $TEST -o export/test_list.tsv
Run tests/make_golden.py --out export/golden --ann $TEST
Write-Host "Done: export\*.onnx. Next: scripts\cpp_windows.ps1 (TensorRT on this PC) or copy export\ to the Jetson." -ForegroundColor Green
