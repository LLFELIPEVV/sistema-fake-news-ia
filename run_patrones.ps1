Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

Write-Host "Iniciando Evaluacion..." -ForegroundColor Cyan
python -m evaluation.patrones