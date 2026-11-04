# Check whether this Windows PC is ready to build and run the TensorRT C++ pipeline. Changes nothing.
#   powershell -ExecutionPolicy Bypass -File scripts\check_trt_windows.ps1 [-TensorRT C:\TensorRT-10.x.x.x]
param([string]$TensorRT = "")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Ok($m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Bad($m) { Write-Host "  MISSING  $m" -ForegroundColor Red; $script:missing++ }
$missing = 0

Write-Host "GPU and driver"
$smi = (& nvidia-smi 2>$null) -join "`n"
if ($smi) {
    $cap = (& nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>$null)
    Ok "$cap"
    Ok ("driver supports CUDA " + [regex]::Match($smi, "CUDA Version:\s*([0-9.]+)").Groups[1].Value)
} else { Bad "nvidia-smi (NVIDIA driver)" }

Write-Host "CUDA Toolkit (headers and cudart for the C++ build)"
if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\include\cuda_runtime.h")) {
    $v = (& "$env:CUDA_PATH\bin\nvcc.exe" --version 2>$null | Select-String "release") -join ""
    Ok "CUDA_PATH = $env:CUDA_PATH  $v"
} else { Bad "CUDA Toolkit 12.x (CUDA_PATH not set). Install from developer.nvidia.com/cuda-downloads" }

Write-Host "TensorRT"
if (-not $TensorRT) {
    $TensorRT = (Get-ChildItem C:\ -Directory -Filter "TensorRT-*" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1).FullName
}
if ($TensorRT -and (Test-Path "$TensorRT\include\NvInfer.h")) {
    $lib = Get-ChildItem "$TensorRT\lib" -Filter "nvinfer_10*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $lib) { $lib = Get-ChildItem "$TensorRT\bin" -Filter "nvinfer_10*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($lib) { Ok "$TensorRT ($($lib.Name))" } else { Bad "TensorRT 10 DLLs under $TensorRT\lib or \bin" }
} else { Bad "TensorRT 10.x for Windows (zip for CUDA 12.x), unzipped to e.g. C:\TensorRT-10.x.x.x" }

Write-Host "Build tools"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if ((Test-Path $vswhere) -and (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)) {
    Ok "MSVC (Visual Studio 2022 Build Tools)"
} else { Bad "MSVC C++ build tools" }
if (Get-Command cmake -ErrorAction SilentlyContinue) { Ok ("cmake " + ((cmake --version | Select-Object -First 1) -replace "cmake version ", "")) }
elseif (Test-Path .venv\Scripts\cmake.exe) { Ok "cmake in .venv" }
else { Write-Host "  note  cmake not on PATH: cpp_windows.ps1 will install it into .venv with pip" -ForegroundColor Yellow }

Write-Host "Exported models"
$onnx = Get-ChildItem export\*.onnx -ErrorAction SilentlyContinue
if ($onnx) { Ok ("" + $onnx.Count + " ONNX files in export\") } else { Write-Host "  note  no export\*.onnx yet: run scripts\export_all.ps1 first" -ForegroundColor Yellow }

if ($missing -eq 0) { Write-Host "`nReady. Next: scripts\cpp_windows.ps1 -TensorRT `"$TensorRT`"" -ForegroundColor Green }
else { Write-Host "`n$missing item(s) missing (above)." -ForegroundColor Red }
