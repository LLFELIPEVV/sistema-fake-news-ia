from config import DEFAULT_MODEL
from pydantic import BaseModel, Field
from typing import Optional, Literal, List


# ======================================================
# 📍 1. Predict (POST)
# ======================================================
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Texto a analizar.")
    model: Optional[str] = Field(
        DEFAULT_MODEL,
        description="Modelo a usar para la predicción. Si no se indica, se usa el predeterminado.",
    )


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str
    inference_time_ms: float
    tokens_count: Optional[int] = None


# ======================================================
# 📍 2. Confidence (GET)
# ======================================================
class ConfidenceResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    model_used: str
    timestamp: str


# ======================================================
# 📍 3. Models (GET)
# ======================================================
class ModelInfo(BaseModel):
    name: str
    description: Optional[str] = None
    type: Literal["Deep Learning", "Machine Learning"]
    status: bool
    size_mb: float
    last_used: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


# ======================================================
# 📍 4. Reload (POST)
# ======================================================
class ReloadRequest(BaseModel):
    model: str


class ReloadResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    reload_time_ms: float


# ======================================================
# 📍 5. Metrics (GET)
# ======================================================
class MetricsRequest(BaseModel):
    model: Optional[str] = Field(
        DEFAULT_MODEL, description="Modelo del que se desean las métricas."
    )


class MetricsInfo(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    updated_at: Optional[str] = None


class MetricsResponse(BaseModel):
    metrics: MetricsInfo


# ======================================================
# 📍 6. Health (GET)
# ======================================================
class HealthResponse(BaseModel):
    status: Literal["ok", "down"]
    uptime: str
    framework: str = "FastAPI"
    tensorflow_version: Optional[str] = None
    gpu_available: Optional[bool] = None
    loaded_models: List[str]


# ======================================================
# 📍 7. Logs (GET)
# ======================================================
class LogsRequest(BaseModel):
    limit: Optional[int] = Field(20, description="Número máximo de logs a devolver.")


class LogInfo(BaseModel):
    timestamp: str
    model: str
    text: str
    prediction: Literal["real", "fake"]
    confidence: float


class LogsResponse(BaseModel):
    logs: List[LogInfo]
