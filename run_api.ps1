Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
#& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
& "$PSScriptRoot\.venv_prod\Scripts\Activate.ps1"

Write-Host "Iniciando API..." -ForegroundColor Cyan
python -m uvicorn deployment.api.main:app --reload
#python -m evaluation.patrones