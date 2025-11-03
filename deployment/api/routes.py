import os
import time
import json

from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from deployment.api.config import DEFAULT_MODEL, MODELS_ENABLED, MODELS_DIR, archivos
from deployment.api.models import (
    predict_text,
    get_file_size_mb,
    read_last_used,
    ModeloBase,
)
from deployment.api.schemas import (
    PredictRequest,
    PredictResponse,
    ConfidenceResponse,
    ModelsResponse,
    ModelInfo,
    ReloadRequest,
    ReloadResponse,
    MetricsResponse,
    MetricsInfo,
    HealthResponse,
    LogsResponse,
    LogInfo,
)

router = APIRouter()

# Variables globales para tracking
LOGS_FILE = os.path.join(MODELS_DIR, "logs.json")
models_cache = {}
start_time = time.time()
prediction_history = {}
logs_storage = []


# ======================================================
# 🔮 1. Predict (POST)
# ======================================================
@router.post("/predict", response_model=PredictResponse, tags=["Predictions"])
def predict(payload: PredictRequest):
    """
    Realiza una predicción sobre si un texto es una noticia falsa o real.

    - **text**: Texto de la noticia a analizar
    - **model**: Modelo a utilizar (opcional, por defecto CNN)
    """
    try:
        model_name = payload.model or DEFAULT_MODEL

        if model_name not in MODELS_ENABLED:
            raise HTTPException(
                status_code=400, detail=f"Modelo '{model_name}' no existe."
            )

        if not MODELS_ENABLED[model_name]:
            raise HTTPException(
                status_code=400, detail=f"Modelo '{model_name}' está deshabilitado."
            )

        result = predict_text(texts=payload.text, model_name=model_name)

        # Guardar en historial
        prediction_id = f"pred_{int(time.time() * 1000)}"
        prediction_history[prediction_id] = {
            **result,
            "timestamp": datetime.now().isoformat(),
            "text": payload.text[:100] + "..."
            if len(payload.text) > 100
            else payload.text,
        }

        # Guardar en logs
        log_entry = LogInfo(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model=model_name,
            text=payload.text[:200] + "..."
            if len(payload.text) > 200
            else payload.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
        )
        logs_storage.append(log_entry)

        if len(logs_storage) > 1000:
            logs_storage.pop(0)
        save_logs_to_file()

        # Actualizar última fecha de uso
        save_last_used(model_name)

        result["prediction_id"] = prediction_id
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# 📊 2. Confidence (GET)
# ======================================================
@router.get(
    "/confidence/{prediction_id}",
    response_model=ConfidenceResponse,
    tags=["Predictions"],
)
def get_confidence(prediction_id: str):
    """
    Devuelve el nivel de confianza de una predicción previa.

    - **prediction_id**: ID de la predicción realizada
    """
    if prediction_id not in prediction_history:
        raise HTTPException(
            status_code=404, detail=f"Predicción '{prediction_id}' no encontrada."
        )

    data = prediction_history[prediction_id]
    return ConfidenceResponse(
        prediction_id=prediction_id,
        prediction=data["prediction"],
        confidence=data["confidence"],
        model_used=data["model_used"],
        timestamp=data["timestamp"],
    )


# ======================================================
# 📋 3. Models (GET)
# ======================================================
@router.get("/models", response_model=ModelsResponse, tags=["Models"])
def get_models():
    """
    Lista todos los modelos disponibles con su información básica.

    Devuelve información sobre:
    - Nombre del modelo
    - Descripción
    - Tipo (ML/DL)
    - Estado (disponible/no disponible)
    - Tamaño en MB
    - Última vez usado
    """
    modelos_info = []

    modelos_descripciones = {
        "Naive Bayes": {
            "description": "Clasificador probabilístico basado en teorema de Bayes con TF-IDF.",
            "type": "Machine Learning",
        },
        "Random Forest": {
            "description": "Modelo de ensamble de árboles de decisión con TF-IDF.",
            "type": "Machine Learning",
        },
        "CNN": {
            "description": "Red neuronal convolucional para clasificación de texto.",
            "type": "Deep Learning",
        },
        "Hibrido": {
            "description": "Modelo híbrido CNN + BiLSTM + GRU con mecanismo de atención.",
            "type": "Deep Learning",
        },
    }

    for name, info in modelos_descripciones.items():
        if name in archivos:
            model_path = os.path.join(MODELS_DIR, archivos[name])
            modelos_info.append(
                ModelInfo(
                    name=name,
                    description=info["description"],
                    type=info["type"],
                    status=os.path.exists(model_path)
                    and MODELS_ENABLED.get(name, False),
                    size_mb=get_file_size_mb(model_path),
                    last_used=read_last_used(name),
                )
            )

    return ModelsResponse(models=modelos_info)


# ======================================================
# 🔄 4. Reload (POST)
# ======================================================
@router.post("/reload", response_model=ReloadResponse, tags=["Models"])
def reload_model(request: ReloadRequest):
    """
    Recarga un modelo desde disco.

    Útil para actualizar modelos sin reiniciar el servidor.

    - **model**: Nombre del modelo a recargar
    """
    try:
        if request.model not in MODELS_ENABLED:
            raise HTTPException(
                status_code=400, detail=f"Modelo '{request.model}' no existe."
            )

        start = time.time()

        # Recargar modelo
        archivo = archivos.get(request.model)
        if not archivo:
            raise HTTPException(
                status_code=400,
                detail=f"No se encontró archivo para modelo '{request.model}'.",
            )

        modelo = ModeloBase(request.model, archivo).cargar_modelo()
        models_cache[request.model] = modelo

        reload_time = (time.time() - start) * 1000

        return ReloadResponse(
            status="success",
            message=f"Modelo '{request.model}' recargado correctamente.",
            reload_time_ms=round(reload_time, 3),
        )

    except FileNotFoundError as e:
        return ReloadResponse(
            status="error",
            message=f"Archivo del modelo no encontrado: {str(e)}",
            reload_time_ms=0.0,
        )
    except Exception as e:
        return ReloadResponse(
            status="error",
            message=f"Error al recargar modelo: {str(e)}",
            reload_time_ms=0.0,
        )


# ======================================================
# 📈 5. Metrics (GET)
# ======================================================
@router.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
def get_metrics(
    model: str = Query(
        DEFAULT_MODEL, description="Modelo del que se desean las métricas"
    ),
):
    if model not in MODELS_ENABLED:
        raise HTTPException(status_code=400, detail=f"Modelo '{model}' no existe.")

    metrics_path = os.path.join(MODELS_DIR, f"{model}_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404, detail=f"No se encontraron métricas para '{model}'."
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        return MetricsResponse(metrics=MetricsInfo(**metrics_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo métricas: {e}")


# ======================================================
# 🏥 6. Health (GET)
# ======================================================
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Verifica el estado de salud de la API.

    Retorna información sobre:
    - Estado del servicio
    - Tiempo de actividad
    - Versión de TensorFlow
    - Disponibilidad de GPU
    - Modelos cargados
    """
    try:
        import tensorflow as tf

        tf_version = tf.__version__
        gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        tf_version = "No disponible"
        gpu_available = False

    uptime_seconds = time.time() - start_time
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    loaded_models = [
        name
        for name, enabled in MODELS_ENABLED.items()
        if enabled and os.path.exists(os.path.join(MODELS_DIR, archivos.get(name, "")))
    ]

    return HealthResponse(
        status="ok",
        uptime=uptime_str,
        framework="FastAPI",
        tensorflow_version=tf_version,
        gpu_available=gpu_available,
        loaded_models=loaded_models,
    )


# ======================================================
# 📝 7. Logs (GET)
# ======================================================
def save_logs_to_file():
    """Guarda logs en archivo JSON persistente."""
    try:
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [log.dict() for log in logs_storage], f, ensure_ascii=False, indent=2
            )
    except Exception as e:
        print(f"⚠️ No se pudieron guardar los logs: {e}")


def load_logs_from_file():
    """Carga logs previos desde archivo JSON si existen."""
    global logs_storage
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                logs_storage.extend([LogInfo(**entry) for entry in data])
        except Exception as e:
            print(f"⚠️ Error al cargar logs previos: {e}")


# Cargar logs al iniciar
load_logs_from_file()


@router.get("/logs", response_model=LogsResponse, tags=["System"])
def get_logs(
    limit: int = Query(
        20, ge=1, le=100, description="Número máximo de logs a devolver"
    ),
):
    limit = max(1, min(limit, 100))
    recent_logs = logs_storage[-limit:] if logs_storage else []
    recent_logs.reverse()
    return LogsResponse(logs=recent_logs)


# ======================================================
# 🛠️ Funciones auxiliares
# ======================================================
def save_last_used(model_name: str):
    """Guarda la fecha y hora del último uso de un modelo."""
    try:
        path = os.path.join(MODELS_DIR, f"{model_name}_last_used.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        print(f"⚠️ No se pudo guardar la fecha de último uso de '{model_name}'.")
