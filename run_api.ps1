Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv_prod\Scripts\Activate.ps1"

Write-Host "Iniciando API..." -ForegroundColor Cyan
python -m uvicorn deployment.api.main:app --reload
