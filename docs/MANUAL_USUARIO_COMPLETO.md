# Manual de Usuario Completo - Sistema de Detección de Fake News IA

## 📋 Índice
1. [Introducción y Arquitectura](#introducción-y-arquitectura)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Preprocesamiento de Datos](#preprocesamiento-de-datos)
6. [Entrenamiento de Modelos](#entrenamiento-de-modelos)
7. [API FastAPI](#api-fastapi)
8. [Frontend React](#frontend-react)
9. [Conexión API-Frontend](#conexión-api-frontend)
10. [Testing y Validación](#testing-y-validación)
11. [Despliegue en Producción](#despliegue-en-producción)
12. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)
13. [Solución de Problemas](#solución-de-problemas)
14. [Ejemplos Prácticos](#ejemplos-prácticos)

---

## 🎯 Introducción y Arquitectura

### Propósito del Sistema
Sistema autónomo para la detección de noticias falsas en español utilizando:
- **Machine Learning tradicional**: Naive Bayes, Random Forest, SVM, Decision Tree
- **Deep Learning**: CNN, LSTM, BiLSTM, GRU, modelos híbridos
- **Transformers**: BETO, mBERT
- **API REST**: FastAPI para inferencia en tiempo real
- **Frontend**: React con Vite para interfaz de usuario

### Arquitectura del Sistema
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API FastAPI   │    │   Modelos ML    │
│   (React)       │◄──►│   (Python)      │◄──►│   (Pickle/H5)   │
│   Port: 5173    │    │   Port: 8000    │    │   /models/      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vite Dev      │    │   Uvicorn       │    │   TensorFlow    │
│   Server        │    │   ASGI Server   │    │   PyTorch       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 💻 Requisitos del Sistema

### Software Requerido
- **Windows 10/11** con PowerShell 5.1+
- **Python 3.8-3.11** (recomendado 3.10)
- **Node.js 16+** y **npm 8+**
- **Git** para control de versiones

### Hardware Recomendado
- **RAM**: 8GB mínimo, 16GB recomendado
- **Almacenamiento**: 10GB libres
- **GPU**: Opcional (CUDA compatible para aceleración)

### Verificación de Requisitos
```powershell
# Verificar Python
python --version

# Verificar Node.js
node --version
npm --version

# Verificar PowerShell
$PSVersionTable.PSVersion
```

---

## 🔧 Instalación y Configuración

### Paso 1: Clonar el Repositorio
```powershell
git clone <repository-url>
cd sistema-fake-news-ia
```

### Paso 2: Configurar Política de Ejecución (si es necesario)
```powershell
# Temporal para la sesión actual
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# O ejecutar scripts con bypass
powershell -ExecutionPolicy Bypass -File .\run_api.ps1
```

### Paso 3: Crear Entornos Virtuales

#### Entorno de Entrenamiento
```powershell
# Crear entorno virtual
python -m venv .venv_train

# Activar entorno
.\.venv_train\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias (nota: el archivo puede tener problemas de encoding)
pip install -r requirements.txt

# Si hay problemas con requirements.txt, instalar manualmente:
pip install tensorflow==2.20.0 scikit-learn==1.7.1 pandas==2.3.2 numpy==1.26.4
pip install transformers==4.56.0 torch==2.4.1 spacy==3.7.5
pip install matplotlib==3.10.6 seaborn==0.13.2 joblib==1.5.2
python -m spacy download es_core_news_sm
```

#### Entorno de Producción (API)
```powershell
# Crear entorno virtual
python -m venv .venv_prod

# Activar entorno
.\.venv_prod\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias de producción
pip install fastapi==0.120.2 uvicorn==0.38.0 tensorflow==2.20.0
pip install scikit-learn==1.7.1 pandas==2.3.2 numpy==1.26.4
pip install spacy==3.7.5 joblib==1.5.2
python -m spacy download es_core_news_sm
```

### Paso 4: Configurar Frontend
```powershell
cd deployment\web
npm install
cd ..\..
```

---

## 📁 Estructura del Proyecto

```
sistema-fake-news-ia/
├── 📁 data/                          # Datos del proyecto
│   ├── 📁 artifacts/                 # Artefactos generados (vectorizadores, etc.)
│   ├── 📁 processed/                 # Datos procesados
│   ├── 📁 raw/                       # Datos crudos unificados
│   └── 📁 splits/                    # División train/valid/test
├── 📁 datasets/                      # Datasets originales (CSV)
│   ├── 📁 fake_news_corpus_spanish/
│   ├── 📁 FakeNewsSpanish_Kaggle2/
│   └── 📁 Spanish Fake and Real News/
├── 📁 preprocessing/                 # Pipeline de preprocesamiento
│   └── 📄 preprocessing_pipeline_fake_news.py
├── 📁 training/                      # Scripts de entrenamiento
│   ├── 📁 scikit/                    # Modelos tradicionales ML
│   │   ├── 📄 naive_bayes.py
│   │   ├── 📄 random_forest.py
│   │   ├── 📄 logistic_regression.py
│   │   ├── 📄 svc.py
│   │   └── 📄 desicion_tree.py
│   ├── 📁 tensorflow/                # Modelos Deep Learning
│   │   ├── 📄 cnn.py
│   │   ├── 📄 lstm.py
│   │   ├── 📄 bilstm.py
│   │   ├── 📄 gru.py
│   │   └── 📄 hibrido.py
│   └── 📁 transformers/              # Modelos Transformer
│       ├── 📄 beto.py
│       └── 📄 mbert.py
├── 📁 models/                        # Modelos entrenados
│   ├── 📄 naive_bayes_best_model.pkl
│   ├── 📄 cnn_best_model.keras
│   ├── 📄 text_vectorizer_keras.keras
│   └── 📄 ...
├── 📁 deployment/                    # Despliegue
│   ├── 📁 api/                       # API FastAPI
│   │   ├── 📄 main.py               # Aplicación principal
│   │   ├── 📄 routes.py             # Endpoints
│   │   ├── 📄 models.py             # Lógica de modelos
│   │   ├── 📄 config.py             # Configuración
│   │   └── 📄 schemas.py            # Esquemas Pydantic
│   └── 📁 web/                       # Frontend React
│       ├── 📁 src/
│       │   ├── 📁 api/              # Cliente API
│       │   ├── 📁 components/       # Componentes React
│       │   └── 📁 pages/            # Páginas
│       └── 📄 package.json
├── 📁 evaluation/                    # Scripts de evaluación
├── 📁 reports/                       # Reportes y métricas
├── 📁 figures/                       # Gráficos generados
├── 📁 tests/                         # Pruebas automatizadas
├── 📄 run_api.ps1                   # Script para ejecutar API
├── 📄 run_evaluation.ps1            # Script para entrenamiento
└── 📄 README.md
```

---

## 🔄 Preprocesamiento de Datos

### Descripción del Pipeline
El pipeline de preprocesamiento unifica múltiples datasets, normaliza el texto y crea divisiones para entrenamiento.

### Ejecutar Preprocesamiento
```powershell
# Activar entorno de entrenamiento
.\.venv_train\Scripts\Activate.ps1

# Ejecutar pipeline
python preprocessing/preprocessing_pipeline_fake_news.py
```

### Salidas Generadas
- `data/raw/fake_news_unificado.parquet` - Datos unificados
- `data/processed/fake_news_estandarizado.parquet` - Datos procesados
- `data/splits/fake_news_train.parquet` - Conjunto de entrenamiento
- `data/splits/fake_news_valid.parquet` - Conjunto de validación
- `data/splits/fake_news_test.parquet` - Conjunto de prueba

### Verificar Resultados
```powershell
python -c "
import pandas as pd
train = pd.read_parquet('data/splits/fake_news_train.parquet')
print(f'Entrenamiento: {len(train)} muestras')
print(f'Distribución: {train.clase.value_counts()}')
"
```

---

## 🤖 Entrenamiento de Modelos

### Entrenamiento Automático (Recomendado)
```powershell
# Activar entorno de entrenamiento
.\.venv_train\Scripts\Activate.ps1

# Ejecutar todos los entrenamientos
.\run_evaluation.ps1
```

### Entrenamiento Individual

#### Modelos de Machine Learning
```powershell
# Naive Bayes
python -m training.scikit.naive_bayes

# Random Forest
python -m training.scikit.random_forest

# Regresión Logística
python -m training.scikit.logistic_regression

# SVM
python -m training.scikit.svc

# Árbol de Decisión
python -m training.scikit.desicion_tree
```

#### Modelos de Deep Learning
```powershell
# CNN
python -m training.tensorflow.cnn

# LSTM
python -m training.tensorflow.lstm

# BiLSTM
python -m training.tensorflow.bilstm

# GRU
python -m training.tensorflow.gru

# Modelo Híbrido (CNN + BiLSTM + GRU + Attention)
python -m training.tensorflow.hibrido
```

#### Modelos Transformer
```powershell
# BETO (BERT en español)
python -m training.transformers.beto

# mBERT (Multilingual BERT)
python -m training.transformers.mbert
```

### Verificar Modelos Entrenados
```powershell
# Listar modelos generados
dir models\*.pkl
dir models\*.keras
dir models\*.pt

# Ver métricas
type reports\model_evaluation_results.csv
```

---

## 🚀 API FastAPI

### Configuración de la API
La API está configurada en `deployment/api/config.py`:

```python
DEFAULT_MODEL = "CNN"
MODELS_ENABLED = {
    "Naive Bayes": True,
    "Random Forest": True,
    "CNN": True,
    "Hibrido": True,
}
```

### Iniciar la API

#### Método 1: Script Automático (Recomendado)
```powershell
.\run_api.ps1
```

#### Método 2: Manual
```powershell
# Activar entorno de producción
.\.venv_prod\Scripts\Activate.ps1

# Iniciar servidor
python -m uvicorn deployment.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints Disponibles

#### 🔮 Predicción
```http
POST /api/v1/predict
Content-Type: application/json

{
    "text": "Texto de la noticia a analizar",
    "model": "CNN"
}
```

**Respuesta:**
```json
{
    "prediction": "Fake",
    "confidence": 0.92,
    "model_used": "CNN",
    "prediction_id": "pred_1234567890"
}
```

#### 📊 Obtener Confianza
```http
GET /api/v1/confidence/{prediction_id}
```

#### 📋 Listar Modelos
```http
GET /api/v1/models
```

#### 📈 Métricas del Modelo
```http
GET /api/v1/metrics?model=CNN
```

#### 🏥 Estado de Salud
```http
GET /api/v1/health
```

#### 📝 Logs del Sistema
```http
GET /api/v1/logs?limit=20
```

#### 🔄 Recargar Modelo
```http
POST /api/v1/reload
Content-Type: application/json

{
    "model": "CNN"
}
```

### Documentación Interactiva
Una vez iniciada la API, accede a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Probar la API con PowerShell
```powershell
# Predicción
$payload = @{
    text = "El presidente anunció nuevas medidas económicas"
    model = "CNN"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $payload -ContentType "application/json"
Write-Output $response

# Obtener modelos disponibles
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/models" -Method Get

# Estado de salud
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" -Method Get
```

---

## 🎨 Frontend React

### Estructura del Frontend
```
deployment/web/
├── src/
│   ├── api/
│   │   └── predict.api.js      # Cliente API
│   ├── components/             # Componentes reutilizables
│   ├── pages/
│   │   ├── index.jsx          # Página principal
│   │   └── resultado.jsx      # Página de resultados
│   ├── App.jsx                # Componente principal
│   └── main.jsx               # Punto de entrada
├── package.json
└── vite.config.js
```

### Iniciar el Frontend

#### Desarrollo
```powershell
cd deployment\web
npm run dev
```

El frontend estará disponible en: http://localhost:5173

#### Producción
```powershell
cd deployment\web
npm run build
npm run preview
```

### Configuración de la API
El frontend está configurado para conectarse a la API en `src/api/predict.api.js`:

```javascript
const api = axios.create({
    baseURL: "http://127.0.0.1:8000/api/v1",
});
```

### Funcionalidades del Frontend
1. **Formulario de entrada**: Para ingresar texto de noticias
2. **Selector de modelo**: Permite elegir el modelo de ML/DL
3. **Visualización de resultados**: Muestra predicción y confianza
4. **Historial**: Mantiene registro de predicciones anteriores

---

## 🔗 Conexión API-Frontend

### Flujo de Comunicación
```
1. Usuario ingresa texto en el frontend
2. Frontend envía POST a /api/v1/predict
3. API procesa con el modelo seleccionado
4. API retorna predicción y confianza
5. Frontend muestra resultados al usuario
```

### Configuración CORS
La API está configurada para permitir todas las conexiones:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Manejo de Errores
El frontend maneja errores de conexión y muestra mensajes apropiados al usuario.

---

## 🧪 Testing y Validación

### Pruebas de la API
```powershell
# Activar entorno de entrenamiento (incluye pytest)
.\.venv_train\Scripts\Activate.ps1

# Instalar pytest si no está disponible
pip install pytest pytest-asyncio

# Ejecutar todas las pruebas
pytest -v

# Ejecutar solo pruebas de API
pytest -v tests/test_api_endpoints.py

# Ejecutar con cobertura
pip install pytest-cov
pytest --cov=deployment.api tests/
```

### Pruebas Manuales

#### Verificar Modelos
```powershell
python -c "
import os
from deployment.api.config import MODELS_DIR, archivos
for name, file in archivos.items():
    path = os.path.join(MODELS_DIR, file)
    exists = os.path.exists(path)
    print(f'{name}: {\"✓\" if exists else \"✗\"} {file}')
"
```

#### Probar Predicción Local
```powershell
python -c "
from deployment.api.models import predict_text
result = predict_text('Esta es una noticia de prueba', 'CNN')
print(result)
"
```

---

## 🌐 Despliegue en Producción

### Preparación para Producción

#### 1. Configurar Variables de Entorno
```powershell
# Crear archivo .env
echo "DEBUG=False" > .env
echo "API_HOST=0.0.0.0" >> .env
echo "API_PORT=8000" >> .env
```

#### 2. Optimizar Configuración
Editar `deployment/api/config.py`:
```python
DEBUG = False  # Cambiar a False en producción
```

#### 3. Instalar Dependencias de Producción
```powershell
.\.venv_prod\Scripts\Activate.ps1
pip install gunicorn  # Para Linux/Mac
# En Windows usar uvicorn con workers
```

### Despliegue con Docker (Opcional)

#### Dockerfile para API
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_prod.txt .
RUN pip install -r requirements_prod.txt

COPY deployment/ deployment/
COPY models/ models/
COPY reports/ reports/

EXPOSE 8000
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./reports:/app/reports
  
  frontend:
    build: ./deployment/web
    ports:
      - "3000:3000"
    depends_on:
      - api
```

### Despliegue en Windows Server

#### Como Servicio de Windows
```powershell
# Instalar NSSM (Non-Sucking Service Manager)
# Crear servicio
nssm install FakeNewsAPI "C:\path\to\python.exe"
nssm set FakeNewsAPI Parameters "-m uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000"
nssm set FakeNewsAPI AppDirectory "C:\path\to\sistema-fake-news-ia"
nssm start FakeNewsAPI
```

---

## 📊 Monitoreo y Mantenimiento

### Logs del Sistema
```powershell
# Ver logs de la API
Get-Content models\logs.json | ConvertFrom-Json | Format-Table

# Monitorear en tiempo real (requiere PowerShell 7+)
Get-Content models\logs.json -Wait
```

### Métricas de Rendimiento
```powershell
# Estado de salud
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health"

# Métricas de modelos
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/metrics?model=CNN"
```

### Actualización de Modelos
```powershell
# 1. Entrenar nuevo modelo
python -m training.tensorflow.cnn

# 2. Recargar modelo sin reiniciar API
$payload = @{ model = "CNN" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/reload" -Method Post -Body $payload -ContentType "application/json"
```

### Backup de Modelos
```powershell
# Crear backup
$date = Get-Date -Format "yyyyMMdd_HHmmss"
Compress-Archive -Path models\* -DestinationPath "backup_models_$date.zip"
```

---

## 🔧 Solución de Problemas

### Problemas Comunes

#### 1. Error de Política de Ejecución de PowerShell
```powershell
# Solución temporal
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# O ejecutar con bypass
powershell -ExecutionPolicy Bypass -File .\run_api.ps1
```

#### 2. Modelo spaCy no encontrado
```powershell
# Instalar modelo de spaCy
python -m spacy download es_core_news_sm

# O instalar desde URL
pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0.tar.gz
```

#### 3. Error de encoding en requirements.txt
Los archivos requirements parecen tener problemas de encoding. Instalar manualmente:

```powershell
# Dependencias principales
pip install tensorflow==2.20.0 scikit-learn==1.7.1 pandas==2.3.2
pip install fastapi==0.120.2 uvicorn==0.38.0 transformers==4.56.0
pip install torch==2.4.1 spacy==3.7.5 matplotlib==3.10.6
pip install seaborn==0.13.2 joblib==1.5.2 numpy==1.26.4
```

#### 4. Puerto 8000 en uso
```powershell
# Verificar qué proceso usa el puerto
netstat -ano | findstr :8000

# Cambiar puerto en la API
python -m uvicorn deployment.api.main:app --port 8001
```

#### 5. Frontend no conecta con API
Verificar configuración en `deployment/web/src/api/predict.api.js`:
```javascript
const api = axios.create({
    baseURL: "http://127.0.0.1:8000/api/v1",  // Verificar IP y puerto
});
```

#### 6. Modelos no encontrados
```powershell
# Verificar existencia de modelos
python -c "
import os
from deployment.api.config import MODELS_DIR, archivos
print(f'Directorio de modelos: {MODELS_DIR}')
for name, file in archivos.items():
    path = os.path.join(MODELS_DIR, file)
    print(f'{name}: {os.path.exists(path)} - {file}')
"
```

### Logs de Depuración

#### Habilitar logs detallados
```python
# En deployment/api/main.py, agregar:
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Ver logs de uvicorn
```powershell
python -m uvicorn deployment.api.main:app --log-level debug
```

---

## 💡 Ejemplos Prácticos

### Ejemplo 1: Predicción Completa via API

```powershell
# 1. Iniciar API
.\run_api.ps1

# 2. Hacer predicción
$noticia = @{
    text = "El gobierno anuncia que los extraterrestres han invadido la capital y que todos los ciudadanos deben evacuar inmediatamente"
    model = "CNN"
} | ConvertTo-Json

$resultado = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $noticia -ContentType "application/json"

Write-Output "Predicción: $($resultado.prediction)"
Write-Output "Confianza: $($resultado.confidence)"
Write-Output "ID: $($resultado.prediction_id)"

# 3. Obtener detalles de confianza
$confianza = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/confidence/$($resultado.prediction_id)"
Write-Output $confianza
```

### Ejemplo 2: Comparar Múltiples Modelos

```powershell
$texto = "Nueva investigación revela que el cambio climático afecta la economía global"
$modelos = @("Naive Bayes", "Random Forest", "CNN", "Hibrido")

foreach ($modelo in $modelos) {
    $payload = @{
        text = $texto
        model = $modelo
    } | ConvertTo-Json
    
    try {
        $resultado = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $payload -ContentType "application/json"
        Write-Output "$modelo : $($resultado.prediction) ($($resultado.confidence))"
    }
    catch {
        Write-Output "$modelo : Error - $($_.Exception.Message)"
    }
}
```

### Ejemplo 3: Monitoreo Automático

```powershell
# Script de monitoreo
while ($true) {
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health"
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Output "[$timestamp] API Status: $($health.status) - Uptime: $($health.uptime)"
        
        # Verificar modelos cargados
        Write-Output "Modelos cargados: $($health.loaded_models -join ', ')"
        
        Start-Sleep -Seconds 30
    }
    catch {
        Write-Output "[$timestamp] API no disponible: $($_.Exception.Message)"
        Start-Sleep -Seconds 10
    }
}
```

### Ejemplo 4: Batch Processing

```powershell
# Procesar múltiples noticias desde archivo CSV
$noticias = Import-Csv "noticias_test.csv"
$resultados = @()

foreach ($noticia in $noticias) {
    $payload = @{
        text = $noticia.texto
        model = "CNN"
    } | ConvertTo-Json
    
    try {
        $resultado = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $payload -ContentType "application/json"
        
        $resultados += [PSCustomObject]@{
            ID = $noticia.id
            Texto = $noticia.texto.Substring(0, [Math]::Min(50, $noticia.texto.Length))
            Prediccion = $resultado.prediction
            Confianza = $resultado.confidence
            Modelo = $resultado.model_used
        }
    }
    catch {
        Write-Output "Error procesando noticia $($noticia.id): $($_.Exception.Message)"
    }
}

# Exportar resultados
$resultados | Export-Csv "resultados_predicciones.csv" -NoTypeInformation
$resultados | Format-Table
```

---

## 📚 Recursos Adicionales

### Documentación de Referencia
- **FastAPI**: https://fastapi.tiangolo.com/
- **React**: https://react.dev/
- **TensorFlow**: https://www.tensorflow.org/
- **scikit-learn**: https://scikit-learn.org/
- **Transformers**: https://huggingface.co/docs/transformers/

### Comandos Útiles de Desarrollo

```powershell
# Verificar estructura del proyecto
tree /F

# Ver uso de GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Verificar versiones de dependencias
pip list | findstr -i "tensorflow scikit fastapi"

# Limpiar cache de Python
python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"
Get-ChildItem -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force

# Verificar tamaño de modelos
Get-ChildItem models\*.* | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}} | Format-Table
```

---

## 🎯 Conclusión

Este manual proporciona una guía completa para usar, configurar y mantener el sistema de detección de fake news. Para soporte adicional:

1. Revisar logs en `models/logs.json`
2. Consultar documentación de la API en `/docs`
3. Verificar issues conocidos en el repositorio
4. Contactar al equipo de desarrollo

**¡El sistema está listo para detectar noticias falsas en español con alta precisión!** 🚀