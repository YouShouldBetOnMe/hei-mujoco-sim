param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
Push-Location $ProjectRoot
try {
    & $Python -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,13), 'Use Python 3.10, 3.11 or 3.12'"
    if ($LASTEXITCODE -ne 0) { throw 'A compatible Python interpreter is required.' }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
    }
    & '.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.venv\Scripts\python.exe' -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
    & '.venv\Scripts\python.exe' -m hei_sim.portable --verify-files --check
    if ($LASTEXITCODE -ne 0) { throw 'Scene/physics/camera validation failed.' }
} finally { Pop-Location }
