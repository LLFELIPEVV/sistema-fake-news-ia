# Manual de Usuario — sistema-fake-news-ia

Bienvenido. Este documento explica cómo usar, entender y desplegar el proyecto `sistema-fake-news-ia`.
Contiene instrucciones paso a paso para preparar entornos, ejecutar preprocesamiento, entrenar/evaluar modelos, levantar la API y el frontend, ejecutar tests y resolver problemas comunes.

---

**Resumen del proyecto**

- Propósito: detección de noticias falsas en español mediante modelos ML/DL y Transformers.
- Componentes principales:
  - `data/`, `datasets/`: datos crudos y procesados.
  - `preprocessing/`: pipeline de limpieza y creación de splits.
  - `training/`: scripts para entrenar modelos (scikit-learn, Keras, PyTorch/Transformers).
  - `models/`: modelos guardados y artefactos (vectorizadores, pesos).
  - `deployment/api/`: aplicación FastAPI para inferencia.
  - `deployment/web/`: frontend (Node.js / React + Vite).
  - `evaluation/`: utilidades y scripts avanzados de evaluación.
  - `run_api.ps1`, `run_evaluation.ps1`: helpers PowerShell para entornos Windows.

---

**Requisitos (Windows)**

- Windows 10/11 con PowerShell (v5.1). Si PowerShell bloquea scripts, use `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` o ejecutar con `-ExecutionPolicy Bypass`.
- Python 3.8+ (recomendado 3.10/3.11).
- Node.js + npm (para el frontend).
- GPU opcional: CUDA o DirectML (hay soporte parcial con `torch_directml`).

**Dependencias**

- Entorno de entrenamiento: `requirements.txt` (bibliotecas para preprocesamiento, entrenamiento y evaluación).
- Entorno de producción (API): `requirements_prod.txt` (FastAPI, uvicorn, httpx, etc.).

---

**Preparar entornos virtuales (PowerShell)**

Recomendado crear dos entornos separados:

1) Entorno para entrenamiento / exploración

```powershell
python -m venv .venv_train
.\.venv_train\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Entorno para producción (API)

```powershell
python -m venv .venv_prod
.\.venv_prod\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_prod.txt
```

Si PowerShell bloquea activación, ejecutar temporalmente:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# o
powershell -ExecutionPolicy Bypass -File .\run_api.ps1
```

---

**Preprocesamiento de datos**

- Script principal: `preprocessing/preprocessing_pipeline_fake_news.py`.
- Qué hace:
  - Lee varios CSV de `datasets/`, unifica columnas, normaliza texto y mapea etiquetas a 0/1.
  - Guarda datasets en `data/raw/`, `data/processed/` y splits en `data/splits/` en formato Parquet.

Ejecutar (desde la raíz del repo):

```powershell
.\.venv_train\Scripts\Activate.ps1
python preprocessing/preprocessing_pipeline_fake_news.py
```

Salida esperada:
- `data/raw/fake_news_unificado.parquet`
- `data/processed/fake_news_estandarizado.parquet`
- `data/splits/fake_news_train.parquet`, `fake_news_valid.parquet`, `fake_news_test.parquet`

Nota: el script intenta detectar separador (`comma` vs `;`) y normaliza columnas (`texto`, `clase`).

---

**Entrenamiento y evaluación**

- Script helper: `run_evaluation.ps1` (ejecuta varios entrenamientos definidos en `training/`).
- Estructura de entrenamiento:
  - `training/scikit/` contiene scripts de modelos clásicos (Naive Bayes, Random Forest, etc.).
  - `training/tensorflow/` contiene scripts Keras (CNN, LSTM, híbridos).
  - `training/transformers/` contiene evaluadores/entrenadores para BERT/BETO.

Ejemplos de ejecución directa (PowerShell):

```powershell
.\.venv_train\Scripts\Activate.ps1
# Ejecutar todo (usa run_evaluation.ps1)
.\run_evaluation.ps1

# Ejecutar un entrenamiento específico (ejemplo Naive Bayes)
python -m training.scikit.naive_bayes

# Ejecutar CNN (Keras)
python -m training.tensorflow.cnn
```

Modelos resultantes se guardan en `models/` con nombres descriptivos, por ejemplo:
- `cnn_best_model.keras`
- `text_vectorizer_keras.keras` (vectorizador usado por modelos Keras)
- `mbert_pytorch_best_model.pt`, `beto_pytorch_best_model.pt` (modelos PyTorch)
- `naive_bayes_best_model.pkl`, `random_forest_best_model.pkl` (sklearn)

---

**API (FastAPI)**

- Carpeta: `deployment/api/`.
- Archivo principal: `deployment/api/main.py`.
- Configuración: `deployment/api/config.py` (valores: `DEFAULT_MODEL`, `MODELS_DIR`, `METRICS_DIR`, `archivos`, `MODELS_ENABLED`).

Arrancar la API (entorno producción):

```powershell
.\.venv_prod\Scripts\Activate.ps1
.\run_api.ps1
# o manualmente
python -m uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints principales (prefijo `/api/v1`):

- `POST /api/v1/predict` — payload JSON:

  ```json
  {"text": "Texto a clasificar", "model": "CNN"}
  ```

  Respuesta (ejemplo):

  ```json
  {"prediction": "Fake", "confidence": 0.92, "model_used": "CNN", "prediction_id": "pred_..."}
  ```

- `GET /api/v1/confidence/{prediction_id}` — devuelve la confianza de una predicción previa.
- `GET /api/v1/models` — lista modelos disponibles (nombre, tipo, estado, tamaño, última vez usado).
- `POST /api/v1/reload` — recarga un modelo desde disco (útil para actualizar sin reiniciar).
- `GET /api/v1/metrics?model={MODEL_NAME}` — devuelve métricas extraídas de `reports/model_evaluation_results.csv`.
- `GET /api/v1/health` — estado del servicio, uptime, versión TF, GPUs disponibles, modelos cargados.
- `GET /api/v1/logs` — historiales de logs de predicciones (parámetro `limit` opcional).

Pruebas automáticas de endpoints: `tests/test_api_endpoints.py`. Estas pruebas usan `TestClient` de FastAPI y pueden ejecutarse sin levantar la API externa.

Ejecutar tests:

```powershell
.\.venv_train\Scripts\Activate.ps1
pip install pytest
pytest -q tests/test_api_endpoints.py
```

---

**Consumir la API desde PowerShell (ejemplos)**

Predict (Invoke-RestMethod):

```powershell
$payload = @{ text = "Ejemplo de noticia"; model = "CNN" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $payload -ContentType "application/json"
```

Obtener métricas:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/metrics?model=CNN" -Method Get
```

---

**Frontend (desarrollo)**

- Carpeta: `deployment/web/`.
- Comandos (desde la raíz del proyecto):

```powershell
cd deployment\web
npm install
npm run dev
```

- Asegúrese de que la API esté corriendo y que el frontend apunte al `baseUrl` correcto (por defecto `http://localhost:8000/api/v1`). Si el frontend tiene un archivo de configuración o `.env`, actualice la URL de la API ahí. Si no, abra `deployment/web/src` y busque la referencia al endpoint.

---

**Modelos y artefactos**

- Ubicación: `models/`.
- Archivos clave (según `deployment/api/config.py`):
  - `naive_bayes_best_model.pkl`
  - `random_forest_best_model.pkl`
  - `cnn_best_model.keras`
  - `hybrid_cnn_bilstm_gru_attention_best_model.keras`
  - `text_vectorizer_keras.keras` (vectorizador de Keras)
  - `beto_pytorch_best_model.pt`, `mbert_pytorch_best_model.pt` (pytorch)

Para inferencia local sin API, cargue el modelo con el framework correspondiente (joblib / keras.load_model / torch.load) y siga el preprocesamiento (`preprocessing/`) antes de pasar texto al modelo.

---

**Despliegue en producción (sugerencias)**

- Cree un entorno virtual con `requirements_prod.txt` y desactive `DEBUG` en `deployment/api/config.py`.
- Use `uvicorn` con `--workers` o Gunicorn (con uvicorn workers) para producción:

```powershell
python -m uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

- Considerar contenedores Docker: crear `Dockerfile` que instale dependencias desde `requirements_prod.txt`, copie `models/` y `reports/` y lance `uvicorn`.
- En Windows puede usar NSSM/Windows Service o tareas programadas para ejecutar el script de arranque.

---

**Integración Frontend ↔ API**

- El frontend debe llamar a `http(s)://{host}:{port}/api/v1/...`.
- Si está detrás de un proxy o balanceador, asegúrese de configurar CORS apropiadamente en `deployment/api/main.py` (por defecto `allow_origins=["*"]`). En producción restrinja a dominios necesarios.

---

**Pruebas y validación**

- `pytest -q` para ejecutar tests. `tests/test_api_endpoints.py` realiza pruebas de integración rápida usando `TestClient`.
- Validar que `reports/model_evaluation_results.csv` contiene métricas si solicita `/metrics`.

---

**Problemas comunes y soluciones**

- Error al activar virtualenv: cambiar política de ejecución de PowerShell o ejecutar con `-ExecutionPolicy Bypass`.
- `es_core_news_sm` faltante (spaCy): instale manualmente si falla: `python -m spacy download es_core_news_sm` o instale desde la url indicada en `requirements*.txt`.
- Errores con `pyarrow` al guardar/parquet: instale `pyarrow` en el entorno (`pip install pyarrow`).
- Modelos no encontrados: verifique `deployment/api/config.py` → `MODELS_DIR` y que los archivos listados en `archivos` existan en `models/`.
- Si la API no responde en 8000, confirme si `uvicorn` arrancó y en qué host/puerto; revise logs en consola.

---

**Consejos prácticos**

- Separe dependencias entre entrenamiento y producción para reducir el tamaño del entorno desplegado.
- Mantenga copias de respaldo de `models/` y `reports/` antes de actualizar modelos en producción.
- Para actualización de modelos sin downtime use el endpoint `POST /api/v1/reload`.

---

**Dónde mirar en el código** (archivo/s relevantes)

- `README.md` (resumen rápido)
- `preprocessing/preprocessing_pipeline_fake_news.py` (preprocesamiento)
- `training/` (scripts de entrenamiento por framework)
- `evaluation/` (evaluadores y generación de reportes y figuras)
- `deployment/api/` (`main.py`, `routes.py`, `models.py`, `config.py`, `schemas.py`)
- `deployment/web/` (frontend, revisar `src/` para configuración)

---

Si desea, puedo:

- Generar un `Dockerfile` y `docker-compose.yml` mínimo para desplegar API + frontend.
- Añadir ejemplos de llamadas en Python (requests) y notebook de demostración.

Indíqueme qué prefiere a continuación.
