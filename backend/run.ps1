$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$venvPython = ".\venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "[OK] Launching uvicorn with venv..." -ForegroundColor Green
    & $venvPython -m uvicorn main:app --reload --port 8000
} else {
    Write-Host "[ERROR] Virtual environment not found in backend\venv" -ForegroundColor Red
}
