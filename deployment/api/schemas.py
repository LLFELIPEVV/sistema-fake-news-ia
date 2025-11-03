from pydantic import BaseModel, Field
from typing import Optional, Literal, List


# ======================================================
# 🔮 1. Predict (POST)
# ======================================================
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=10000,
        description="Texto de la noticia a analizar.",
        example="Breaking news: Scientists discover new method for detecting misinformation.",
    )
    model: Optional[str] = Field(
        None,
        description="Modelo a usar para la predicción. Si no se indica, se usa el predeterminado (CNN).",
        example="CNN",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Scientists have made a breakthrough discovery in artificial intelligence.",
                "model": "CNN",
            }
        }


class PredictResponse(BaseModel):
    prediction: Literal["real", "fake"] = Field(
        description="Clasificación de la noticia"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Nivel de confianza de la predicción (0-1)"
    )
    model_used: str = Field(description="Modelo utilizado para la predicción")
    inference_time_ms: float = Field(description="Tiempo de inferencia en milisegundos")
    tokens_count: Optional[int] = Field(
        None, description="Número de tokens/palabras en el texto"
    )
    prediction_id: Optional[str] = Field(None, description="ID único de la predicción")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "real",
                "confidence": 0.923,
                "model_used": "CNN",
                "inference_time_ms": 45.678,
                "tokens_count": 12,
                "prediction_id": "pred_1730563200000",
            }
        }


# ======================================================
# 📊 2. Confidence (GET)
# ======================================================
class ConfidenceResponse(BaseModel):
    prediction_id: str = Field(description="ID de la predicción")
    prediction: Literal["real", "fake"] = Field(
        description="Clasificación de la noticia"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Nivel de confianza")
    model_used: str = Field(description="Modelo utilizado")
    timestamp: str = Field(description="Fecha y hora de la predicción")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_1730563200000",
                "prediction": "real",
                "confidence": 0.923,
                "model_used": "CNN",
                "timestamp": "2025-11-02T14:30:45",
            }
        }


# ======================================================
# 📋 3. Models (GET)
# ======================================================
class ModelInfo(BaseModel):
    name: str = Field(description="Nombre del modelo")
    description: Optional[str] = Field(None, description="Descripción del modelo")
    type: Literal["Deep Learning", "Machine Learning"] = Field(
        description="Tipo de modelo"
    )
    status: bool = Field(description="Estado del modelo (disponible/no disponible)")
    size_mb: float = Field(description="Tamaño del archivo del modelo en MB")
    last_used: Optional[str] = Field(
        None, description="Última vez que se usó el modelo"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "CNN",
                "description": "Red neuronal convolucional para clasificación de texto.",
                "type": "Deep Learning",
                "status": True,
                "size_mb": 45.23,
                "last_used": "2025-11-02 14:30:45",
            }
        }


class ModelsResponse(BaseModel):
    models: List[ModelInfo] = Field(description="Lista de modelos disponibles")


# ======================================================
# 🔄 4. Reload (POST)
# ======================================================
class ReloadRequest(BaseModel):
    model: str = Field(description="Nombre del modelo a recargar", example="CNN")

    class Config:
        json_schema_extra = {"example": {"model": "CNN"}}


class ReloadResponse(BaseModel):
    status: Literal["success", "error"] = Field(description="Estado de la operación")
    message: str = Field(description="Mensaje descriptivo")
    reload_time_ms: float = Field(description="Tiempo de recarga en milisegundos")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Modelo 'CNN' recargado correctamente.",
                "reload_time_ms": 523.456,
            }
        }


# ======================================================
# 📈 5. Metrics (GET)
# ======================================================
class MetricsRequest(BaseModel):
    model: Optional[str] = Field(
        None,
        description="Modelo del que se desean las métricas (por defecto CNN).",
        example="CNN",
    )


class MetricsInfo(BaseModel):
    model: str = Field(description="Nombre del modelo")
    accuracy: float = Field(ge=0.0, le=1.0, description="Exactitud del modelo")
    precision: float = Field(ge=0.0, le=1.0, description="Precisión del modelo")
    recall: float = Field(ge=0.0, le=1.0, description="Recall del modelo")
    f1_score: float = Field(ge=0.0, le=1.0, description="F1-Score del modelo")
    auc: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Área bajo la curva ROC"
    )
    updated_at: Optional[str] = Field(
        None, description="Fecha de última actualización de métricas"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "CNN",
                "accuracy": 0.9534,
                "precision": 0.9487,
                "recall": 0.9567,
                "f1_score": 0.9527,
                "auc": 0.9789,
                "updated_at": "2025-01-20",
            }
        }


class MetricsResponse(BaseModel):
    metrics: MetricsInfo


# ======================================================
# 🏥 6. Health (GET)
# ======================================================
class HealthResponse(BaseModel):
    status: Literal["ok", "down"] = Field(description="Estado del servicio")
    uptime: str = Field(description="Tiempo de actividad del servidor")
    framework: str = Field(default="FastAPI", description="Framework utilizado")
    tensorflow_version: Optional[str] = Field(None, description="Versión de TensorFlow")
    gpu_available: Optional[bool] = Field(None, description="Disponibilidad de GPU")
    loaded_models: List[str] = Field(description="Modelos cargados en memoria")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "uptime": "2h 15m 30s",
                "framework": "FastAPI",
                "tensorflow_version": "2.15.0",
                "gpu_available": True,
                "loaded_models": ["CNN", "Naive Bayes", "Random Forest", "Hibrido"],
            }
        }


# ======================================================
# 📝 7. Logs (GET)
# ======================================================
class LogsRequest(BaseModel):
    limit: Optional[int] = Field(
        20, ge=1, le=100, description="Número máximo de logs a devolver.", example=20
    )


class LogInfo(BaseModel):
    timestamp: str = Field(description="Fecha y hora del log")
    model: str = Field(description="Modelo utilizado")
    text: str = Field(description="Fragmento del texto analizado")
    prediction: Literal["real", "fake"] = Field(description="Predicción realizada")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza de la predicción")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-11-02 14:30:45",
                "model": "CNN",
                "text": "Scientists have made a breakthrough discovery...",
                "prediction": "real",
                "confidence": 0.923,
            }
        }


class LogsResponse(BaseModel):
    logs: List[LogInfo] = Field(description="Lista de logs de predicciones")
