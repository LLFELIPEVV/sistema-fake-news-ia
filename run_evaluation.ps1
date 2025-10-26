# ===============================
# Ejecuta evaluaci�n de modelos en entorno virtual
# ===============================

Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

Write-Host "Ejecutando evaluaci�n de modelos..." -ForegroundColor Cyan
python -m training.transformers.beto

Write-Host "Proceso completado correctamente." -ForegroundColor Green
Pause
