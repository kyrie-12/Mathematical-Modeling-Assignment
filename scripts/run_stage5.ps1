$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    & $Python "src\05_train_degradation_classifier.py"
}
finally {
    Pop-Location
}
