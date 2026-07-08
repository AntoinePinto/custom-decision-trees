$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
}
else {
    Write-Host "No Python interpreter found (.venv, python, or py)." -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] Running Ruff autofix..." -ForegroundColor Cyan
& $pythonCmd -m ruff check . --fix
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ruff check failed. Skipping tests." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[2/2] Running tests..." -ForegroundColor Cyan
& $pythonCmd -m pytest -vv -s --log-cli-level=INFO
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "All checks passed." -ForegroundColor Green
