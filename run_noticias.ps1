Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

Write-Host "Iniciando generacion de noticias..." -ForegroundColor Cyan
python -m evaluation.noticias