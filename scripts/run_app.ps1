# -Config picks the model shown in LIVE MODE, e.g. -Config configs/dfdc_fusion.yaml
param([string]$Config = $(if ($env:AVDF_CONFIG) { $env:AVDF_CONFIG } else { 'configs/default.yaml' }))
$ErrorActionPreference = 'Stop'
$env:AVDF_CONFIG = $Config
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host 'Creating .venv...'
    python -m venv .venv
    & $python -m pip install -r requirements.txt
    & $python -m pip install -e .
}
$env:PYTHONPATH = 'src'
$url = 'http://127.0.0.1:8000/'
Write-Host "AVDF forensic app: $url"
Start-Process $url
& $python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
