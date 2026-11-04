# Train the arms, predict val + test, choose tau on val, evaluate test, compare arms (Windows).
#   powershell -ExecutionPolicy Bypass -File scripts\train_arms.ps1                       # everything, 3 seeds
#   powershell -ExecutionPolicy Bypass -File scripts\train_arms.ps1 -Arms a_ir -Seeds 0    # one quick check
#   powershell -ExecutionPolicy Bypass -File scripts\train_arms.ps1 -Schedule short -Seeds 0   # 36-epoch recipe
# Schedules (same for every arm, so the comparison stays fair):
#   full  : 72 epochs, constant LR after warmup (official RT-DETRv2 fine-tune recipe)
#   short : 36 epochs, LR x0.1 near epoch 28, clean last 4 epochs; B-ft 12 epochs
#           (chosen from the 12-epoch check on 29 Oct: val AP still rising; a late LR drop
#            settles a short schedule)
# Arms: a_ir a_vis b_joint (also trains b_ft_ir, b_ft_vis) c_fusion. Finished runs (done.json) are skipped;
# stopped runs (last.pth, no done.json) are resumed from their last epoch.
param(
    [string[]]$Arms = @("a_ir", "a_vis", "b_joint", "c_fusion"),
    [string[]]$Seeds = @("0", "1", "2"),
    [ValidateSet("full", "short")][string]$Schedule = "full",
    [int]$Epochs = 0,
    [int]$Batch = 8,
    [int]$Workers = 4,
    [string]$Ckpt = "weights\rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
# "Continue": Windows PowerShell 5.1 treats any stderr text from python/cmake as fatal under "Stop".
# Failures are caught with $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
# "powershell -File" passes "a_ir,b_joint" as one string, not a list: split on commas here.
$Arms = @($Arms | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Seeds = @($Seeds | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$known = @("a_ir", "a_vis", "b_joint", "c_fusion")
$bad = @($Arms | Where-Object { $known -notcontains $_ })
if ($bad.Count -gt 0) { Write-Host "Unknown arm(s): $($bad -join ', '). Use: $($known -join ', ')" -ForegroundColor Red; exit 1 }
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = (Resolve-Path .venv\Scripts\python.exe).Path
$env:PYTHONPATH = "python"; $env:IRD_REQUIRE_CPP = "1"
$VAL = "data/airst_v0.1s/airst_val.json"; $TEST = "data/airst_v0.1s/airst_test.json"
New-Item -ItemType Directory -Force runs, preds, reports | Out-Null
function Run { & $PY @args; if ($LASTEXITCODE -ne 0) { throw "failed: python $args" } }

if ($Epochs -eq 0) { $Epochs = if ($Schedule -eq "short") { 36 } else { 72 } }
$common = @("train_dataloader.total_batch_size=$Batch", "train_dataloader.num_workers=$Workers", "val_dataloader.num_workers=$Workers")
$stop = [math]::Max(1, [int]($Epochs * 66 / 72))      # last epochs without strong augmentation
if ($Schedule -eq "short") { $stop = $Epochs - 4 }
$upd = $common + @("epoches=$Epochs", "train_dataloader.dataset.transforms.policy.epoch=$stop", "train_dataloader.collate_fn.stop_epoch=$stop")
$updFt = $common
if ($Schedule -eq "short") {
    # LR scheduler steps once per epoch after the warmup (300 iterations, about 3 epochs at batch 8),
    # so milestone 25 drops the LR near epoch 28.
    $upd += @("lr_warmup_scheduler.warmup_duration=300", "ema.warmups=500", "lr_scheduler.milestones=[25]")
    $updFt += @("epoches=12", "train_dataloader.dataset.transforms.policy.epoch=10", "train_dataloader.collate_fn.stop_epoch=10",
                "lr_scheduler.milestones=[8]")
}
Write-Host "Schedule $Schedule : $Epochs epochs, batch $Batch, arms $($Arms -join ', '), seeds $($Seeds -join ', ')" -ForegroundColor Cyan

function Train($arm, $seed, $tuning) {
    $out = "runs/$arm/s$seed"
    if (Test-Path "$out/done.json") { Write-Host "skip $out (finished)"; return }
    $u = if ($arm -like "b_ft_*") { $updFt } else { $upd }
    if (Test-Path "$out/last.pth") {      # stopped earlier: continue from its last epoch
        Write-Host "`n=== resume $arm seed $seed" -ForegroundColor Cyan
        Run -m ird.train.train -c "configs/arms/$arm.yml" --resume "$out/last.pth" --seed $seed --output-dir $out --use-amp -u @u
    } else {
        Write-Host "`n=== train $arm seed $seed" -ForegroundColor Cyan
        Run -m ird.train.train -c "configs/arms/$arm.yml" --tuning $tuning --seed $seed --output-dir $out --use-amp -u @u
    }
}

function Score($arm, $seed, $band) {
    $p = "preds/${arm}_s${seed}_$band"
    $m = @("-c", "configs/arms/$arm.yml", "-r", "runs/$arm/s$seed/best.pth", "--modality", $band)
    Run -m ird.infer.predict --backend torch @m --ann $VAL -o "${p}_val_torch.json"
    Run -m ird.infer.predict --backend torch @m --ann $TEST -o "${p}_torch.json"
    Run -m ird.eval.choose_tau --gt $VAL --pred "${p}_val_torch.json" -o "${p}_torch.json.tau.json"
    Run -m ird.eval.evaluate --gt $TEST --pred "${p}_torch.json" --out "reports/${arm}_s${seed}_${band}_torch" | Out-Null
}

foreach ($arm in $Arms) {
    foreach ($s in $Seeds) {
        Train $arm $s $Ckpt
        if ($arm -eq "b_joint") { foreach ($ft in "b_ft_ir", "b_ft_vis") { Train $ft $s "runs/b_joint/s$s/best.pth" } }
    }
}
foreach ($s in $Seeds) {
    foreach ($arm in $Arms) {
        switch ($arm) {
            "a_ir" { Score a_ir $s ir }
            "a_vis" { Score a_vis $s vis }
            "c_fusion" { Score c_fusion $s fusion }
            "b_joint" { Score b_joint $s ir; Score b_joint $s vis; Score b_ft_ir $s ir; Score b_ft_vis $s vis }
        }
    }
}

function List($arm, $band) { $Seeds | ForEach-Object { "preds/${arm}_s${_}_${band}_torch.json" } }
$ir = @(); $vis = @()
if ($Arms -contains "a_ir") { $ir += @("--arm", "A-IR") + (List a_ir ir) }
if ($Arms -contains "b_joint") { $ir += @("--arm", "B") + (List b_joint ir) + @("--arm", "B-ft-IR") + (List b_ft_ir ir)
                                 $vis += @("--arm", "B") + (List b_joint vis) + @("--arm", "B-ft-VIS") + (List b_ft_vis vis) }
if ($Arms -contains "c_fusion") { $ir += @("--arm", "C") + (List c_fusion fusion); $vis += @("--arm", "C") + (List c_fusion fusion) }
if ($Arms -contains "a_vis") { $vis = @("--arm", "A-VIS") + (List a_vis vis) + $vis }
if ($Arms -contains "a_ir" -and ($ir | Where-Object { $_ -eq "--arm" }).Count -ge 2) {
    Run -m ird.eval.compare --gt $TEST --baseline A-IR --out reports/compare_ir_torch @ir
}
if ($Arms -contains "a_vis" -and ($vis | Where-Object { $_ -eq "--arm" }).Count -ge 2) {
    Run -m ird.eval.compare --gt $TEST --baseline A-VIS --out reports/compare_vis_torch @vis
}
Write-Host "`nDone. Per-run reports: reports\*_torch.md   Comparisons: reports\compare_*_torch.md" -ForegroundColor Green
