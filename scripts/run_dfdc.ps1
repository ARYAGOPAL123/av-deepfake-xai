# DFDC small-subset pipeline on CPU: status -> ingest -> preprocess -> train/evaluate 3 models -> RESULTS.md
#   powershell -ExecutionPolicy Bypass -File scripts\run_dfdc.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_dfdc.ps1 -StatusOnly     (just count REAL/FAKE and list missing REAL files)
# Frame count, epochs and batch size live in configs/dfdc_*.yaml (data.n_frames, train.*).
param(
    [string]$Root = 'data/raw/DFDC_sample',
    [switch]$StatusOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = '.venv\Scripts\python.exe'
$env:PYTHONPATH = 'src'
$env:HF_HUB_DISABLE_XET = '1'   # plain HTTPS downloads for WavLM / EfficientNet weights (Xet fails behind TLS proxies)
$log = 'results/RUN_LOG.md'

function Step([string]$what, [string[]]$cmd) {
    Write-Host "`n>> $what" -ForegroundColor Cyan
    $t = Get-Date
    & $python @cmd
    $ok = $LASTEXITCODE -eq 0
    $mins = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    Add-Content $log "| $(Get-Date -Format 'yyyy-MM-dd HH:mm') | ``python $($cmd -join ' ')`` | $(if ($ok) { "Passed ($mins min)" } else { "FAILED (exit $LASTEXITCODE)" }) |"
    if (-not $ok) { throw "$what failed (exit $LASTEXITCODE)" }
}

if (-not (Test-Path "$Root/metadata.json")) { throw "$Root/metadata.json not found. Put the DFDC videos and metadata.json in $Root." }
Add-Content $log "`n## DFDC small preliminary subset ($(Get-Date -Format 'yyyy-MM-dd'))`n`n| Local time | Command | Outcome |`n|---|---|---|"

Step 'DFDC status (REAL/FAKE on disk, missing REAL files)' @('-m', 'avdf.ingest', '--dataset', 'dfdc', '--root', $Root, '--status')
if ($StatusOnly) { return }

Step 'Ingest (labels from metadata.json, split grouped by original)' @('-m', 'avdf.ingest', '--dataset', 'dfdc', '--root', $Root, '--out', 'data/manifests/dfdc.csv', '--split')
# Training needs both classes in train and a non-empty val/test split
& $python -c @"
import pandas as pd, sys
df = pd.read_csv('data/manifests/dfdc.csv')
c = df.groupby(['split', 'label']).size().unstack(fill_value=0).reindex(index=['train', 'val', 'test'], columns=['real', 'fake'], fill_value=0)
bad = [s for s in ['train', 'val', 'test'] if c.loc[s].sum() == 0] + (['train needs both classes'] if (c.loc['train'] == 0).any() else [])
if bad:
    sys.exit('Split unusable (%s). Download more videos, especially REAL ones: see data/manifests/dfdc_missing_real.txt' % ', '.join(bad))
"@
if ($LASTEXITCODE -ne 0) { throw 'Not enough DFDC videos for a train/val/test split' }

$frames = & $python -c "import yaml; print(yaml.safe_load(open('configs/dfdc_fusion.yaml'))['data']['n_frames'])"
Step "Preprocess ($frames face crops + 16 kHz audio per clip, CPU)" @('-m', 'avdf.preprocess', '--manifest', 'data/manifests/dfdc.csv', '--cache', 'data/cache/dfdc', '--n_frames', $frames, '--device', 'cpu')

foreach ($m in 'video_only', 'audio_only', 'fusion') {
    Step "Train dfdc_$m" @('-m', 'avdf.train', '--config', "configs/dfdc_$m.yaml")
    Step "Evaluate dfdc_$m on test" @('-m', 'avdf.evaluate', '--config', "configs/dfdc_$m.yaml", '--split', 'test', '--method', 'softmax')
}
Step 'Evaluate dfdc_fusion with MC Dropout (used by LIVE MODE)' @('-m', 'avdf.evaluate', '--config', 'configs/dfdc_fusion.yaml', '--split', 'test', '--method', 'mc_dropout')
Step 'Write RESULTS.md' @('scripts/dfdc_results.py')

Write-Host "`nDone. Start the dashboard in LIVE MODE with the DFDC fusion model:" -ForegroundColor Green
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 -Config configs/dfdc_fusion.yaml'
