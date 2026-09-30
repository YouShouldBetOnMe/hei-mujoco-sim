param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CollectorArgs)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
Push-Location $ProjectRoot
try {
    & (Join-Path $ProjectRoot '.venv\Scripts\python.exe') -m hei_sim.collect @CollectorArgs
    if ($LASTEXITCODE -ne 0) { throw 'Collector exited with an error.' }
} finally { Pop-Location }
