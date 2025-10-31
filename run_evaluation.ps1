# ===============================
# Ejecuta evaluacion de modelos en entorno virtual
# ===============================

Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

#Write-Host "Ejecutando entrenamiento de Naive Bayes..." -ForegroundColor Cyan
#python -m training.scikit.naive_bayes

#Write-Host "Ejecutando entrenamiento de Random Forest..." -ForegroundColor Cyan
#python -m training.scikit.random_forest

Write-Host "Ejecutando entrenamiento de CNN..." -ForegroundColor Cyan
python -m training.tensorflow.cnn

Write-Host "Ejecutando entrenamiento de hibrido..." -ForegroundColor Cyan
python -m training.tensorflow.hibrido

Write-Host "Proceso completado correctamente." -ForegroundColor Green
Pause
