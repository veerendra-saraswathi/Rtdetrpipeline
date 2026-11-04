# One-time setup on Windows (run from the ird_pipeline folder in a VS Code PowerShell terminal):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Creates .venv, installs PyTorch with CUDA, builds the C++ preprocessing module, gets RT-DETRv2
# and the COCO checkpoint, and runs the tests. Re-running is safe (finished steps are skipped).
# "Continue": Windows PowerShell 5.1 treats any stderr text from python/cmake as fatal under "Stop".
# Failures are caught with $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Step($m) { Write-Host "`n=== $m" -ForegroundColor Cyan }
function Must($what) { if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what" -ForegroundColor Red; exit 1 } }

Step "GPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Write-Host "No NVIDIA driver found (nvidia-smi missing). Training needs an NVIDIA GPU." -ForegroundColor Red; exit 1 }
$smiOut = (& nvidia-smi) -join "`n"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$cuda = [double]([regex]::Match($smiOut, "CUDA Version:\s*([0-9.]+)").Groups[1].Value)
$index = if ($cuda -ge 12.8) { "cu128" } elseif ($cuda -ge 12.6) { "cu126" } elseif ($cuda -ge 12.4) { "cu124" } else { "cu118" }
Write-Host "Driver supports CUDA $cuda -> PyTorch wheels $index"

Step "Python environment (.venv)"
if (-not (Test-Path .venv\Scripts\python.exe)) {
    $py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
    & $py -m venv .venv; Must "create .venv"
}
$PY = (Resolve-Path .venv\Scripts\python.exe).Path
& $PY -m pip install --upgrade pip wheel | Out-Null   # setuptools left alone: torch needs < 82
& $PY -c "import sys; print(sys.version)"

Step "PyTorch with CUDA"
$hasTorch = ((& $PY -c "import torch;print(torch.cuda.is_available())" 2>$null) | Select-Object -Last 1) -eq "True"
if (-not $hasTorch) {
    & $PY -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$index"; Must "install PyTorch"
}
& $PY -c "import torch;assert torch.cuda.is_available(), 'CUDA not available in PyTorch';print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"; Must "PyTorch sees the GPU"

Step "Other packages"
& $PY -m pip install faster-coco-eval pycocotools scipy pyyaml tensorboard onnx onnxruntime pybind11 pillow numpy; Must "install packages"

Step "C++ preprocessing module (ird_preproc)"
& $PY -m pip install --no-build-isolation .
if ($LASTEXITCODE -ne 0) {
    Write-Host "C++ build failed. Install 'Visual Studio 2022 Build Tools' with 'Desktop development with C++', then re-run." -ForegroundColor Red
    exit 1
}
& $PY -c "import ird_preproc; print(ird_preproc.version())"; Must "import ird_preproc"

Step "RT-DETRv2 code"
if (-not (Test-Path third_party\RT-DETR\rtdetrv2_pytorch)) {
    New-Item -ItemType Directory -Force third_party | Out-Null
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone https://github.com/lyuwenyu/RT-DETR third_party/RT-DETR; Must "git clone RT-DETR"
    } else {
        Write-Host "git not found: downloading the repository as a zip"
        $zip = "third_party\RT-DETR.zip"
        Invoke-WebRequest "https://github.com/lyuwenyu/RT-DETR/archive/refs/heads/main.zip" -OutFile $zip
        Expand-Archive $zip -DestinationPath third_party -Force
        Rename-Item third_party\RT-DETR-main RT-DETR
        Remove-Item $zip
        "zip download (no git): record the download date in the experiment log" | Out-File third_party\RT-DETR\SOURCE.txt
    }
}
if (-not (Test-Path third_party\RT-DETR\rtdetrv2_pytorch\src\zoo)) { Write-Host "FAILED: RT-DETRv2 code missing" -ForegroundColor Red; exit 1 }
if (Get-Command git -ErrorAction SilentlyContinue) { git -C third_party/RT-DETR rev-parse HEAD }
Write-Host "RT-DETRv2 code OK"

Step "COCO checkpoint (RT-DETRv2-S, 48.1 AP)"
New-Item -ItemType Directory -Force weights | Out-Null
$ck = "weights\rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
if (-not (Test-Path $ck)) {
    Invoke-WebRequest "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" -OutFile $ck
}
"{0:N1} MB" -f ((Get-Item $ck).Length / 1MB)

Step "Tests"
$env:PYTHONPATH = "python"
& $PY tests\test_preproc.py; Must "preprocessing tests"
& $PY tests\test_metrics.py; Must "metric tests"
Write-Host "`nSetup done. Next: powershell -ExecutionPolicy Bypass -File scripts\train_arms.ps1 -Arms a_ir -Seeds 0 -Epochs 72" -ForegroundColor Green
