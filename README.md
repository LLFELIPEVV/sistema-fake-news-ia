# sistema-fake-news-ia
Sistema autónomo para la detección de noticias falsas en español usando inteligencia artificial, NLP y machine learning. Incluye análisis de datasets, entrenamiento de modelos tradicionales, profundos y Transformers, con API en FastAPI e interfaz web en React.

### Ejecutar los entrenamientos con la siguiente estructura:

**Sistema de detección de Fake News (español)**

Proyecto para entrenar modelos de detección de noticias falsas en español, exponerlos mediante una API con FastAPI y disponer de un frontend para interacción. Incluye pipelines de preprocesamiento, experimentos con múltiples modelos (tensorflow/keras, scikit-learn, transformers) y utilidades para evaluación.

**Descripción**
- **Propósito:** Entrenar y evaluar modelos para clasificación de noticias (falsas vs reales) en español, y desplegar una API para inferencia.
- **Componentes:** entrenamiento y preprocesamiento (`training/`, `preprocessing/`), modelos guardados (`models/`), API (`deployment/api/`), frontend (`deployment/web/`), datos (`data/`, `datasets/`) y evaluación (`evaluation/`).

**Estructura principal**
- **`data/`**: datos crudos, procesados y artefactos usados por los pipelines.
- **`datasets/`**: colecciones de datasets y variantes (archivos CSV).
- **`preprocessing/`**: pipeline(s) para limpieza y transformaciones de texto.
- **`training/`**: scripts y utilidades para entrenar modelos (subcarpetas por framework).
- **`evaluation/`**: scripts para evaluación y generación de métricas y reportes.
- **`models/`**: modelos entrenados, pesos y archivos de score/registro.
- **`deployment/api/`**: aplicación FastAPI para servir modelos en producción.
- **`deployment/web/`**: frontend (Node.js) — desarrollo con `npm run dev`.
- **`reports/`**: salidas de evaluación, reportes y resúmenes.
- **`tests/`**: pruebas automatizadas (ej. `tests/test_api_endpoints.py`).
- **`run_api.ps1`**: script PowerShell para levantar la API.
- **`run_evaluation.ps1`**: script PowerShell para ejecutar pipelines de entrenamiento/evaluación.

**Requisitos**
- Tener instalado Python 3.8+ (recomendado 3.10/3.11).
- Node.js + npm para el frontend.
- PowerShell en Windows (los scripts `.ps1` están preparados para PowerShell).

**Entornos virtuales (Windows / PowerShell)**
Este repositorio usa dos entornos virtuales separados:

- Entorno para entrenamiento y preprocesamiento (usa `requirements.txt`).
- Entorno de producción para la API (usa `requirements_prod.txt`).

Comandos recomendados (PowerShell):

```powershell
# 1) Entorno para entrenamiento/preprocesamiento
python -m venv .venv_train
.\.venv_train\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Entorno para producción (API)
python -m venv .venv_prod
.\.venv_prod\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_prod.txt
```

Notas:
- Si su PowerShell bloquea la ejecución de scripts, puede usar temporalmente:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

o ejecutar el script con `powershell -ExecutionPolicy Bypass -File .\run_api.ps1`.

**Ejecutar la API**
1. Activar el entorno de producción (`.venv_prod`).
2. Ejecutar el script:

```powershell
.\run_api.ps1
# o
powershell -ExecutionPolicy Bypass -File .\run_api.ps1
```

El script `run_api.ps1` contiene los pasos necesarios para lanzar la aplicación FastAPI (por ejemplo usando `uvicorn`); úselo preferentemente en lugar de comandos manuales salvo que necesite ajustes.

**Entrenamiento y evaluación**
- Para ejecutar pipelines de entrenamiento/evaluación (ejecución desde Windows PowerShell):

```powershell
# activar el entorno de entrenamiento
.\.venv_train\Scripts\Activate.ps1
# ejecutar el script preparado
.\run_evaluation.ps1
```

- También puede inspeccionar y ejecutar directamente los scripts dentro de `training/` o `training/scikit` / `training/tensorflow` si necesita control fino.

**Frontend (desarrollo)**
- Carpeta: `deployment/web`
- Para iniciar el frontend (modo desarrollo):

```powershell
cd deployment\web
npm install
npm run dev
```

Esto iniciará el servidor de desarrollo del frontend. Asegúrese de que la API esté corriendo si el frontend consume endpoints locales.

**Ejecutar tests**
- Asegúrese de activar el entorno (usualmente el de entrenamiento si contiene las dependencias de test) e instale `pytest` si no está en `requirements.txt`.

```powershell
.\.venv_train\Scripts\Activate.ps1
pip install pytest
pytest -q
```

Para ejecutar pruebas concretas sobre la API:

```powershell
pytest -q tests/test_api_endpoints.py
```

**Modelos y artefactos**
- Los modelos entrenados y sus métricas se encuentran en `models/`.
- Artefactos generados por los pipelines (vectores, tokenizers, archivos intermedios) pueden encontrarse en `data/artifacts/`.

**Despliegue (sugerencias)**
- Para producción, cree un entorno reproducible basado en `requirements_prod.txt`.
- Configure un servicio (systemd, Windows Service, contenedor Docker) que active el entorno virtual y ejecute el comando que lanza la app (ej. `uvicorn` o el contenido de `run_api.ps1`).

**Consejos rápidos**
- Mantenga versiones separadas para dependencias de entrenamiento y de producción.
- Revise los archivos en `models/` para identificar qué artefactos necesita incluir en el despliegue.

**Contribuir**
- Abra issues o pull requests con cambios en pipelines, modelos o la API.

**Licencia y contacto**
- Consulte `LICENSE` en el repositorio para detalles de licencia.
- Para dudas: revisar el archivo `README.md` o contactar al autor del repositorio.

