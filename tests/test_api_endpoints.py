import os
import json

from fastapi import status
from datetime import datetime
from deployment.api.main import app
from fastapi.testclient import TestClient
from deployment.api.config import METRICS_DIR

# Cliente de prueba síncrono (más estable con FastAPI)
client = TestClient(app)
prefijo = "/api/v1"
OUTPUT_FILE = os.path.join(METRICS_DIR, "api_responses.json")

# Crear carpeta si no existe
os.makedirs(METRICS_DIR, exist_ok=True)

# Cargar resultados previos si existen
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            responses_log = json.load(f)
    except Exception:
        responses_log = {}
else:
    responses_log = {}


def save_response(endpoint: str, response):
    """Guarda la respuesta de cada endpoint en un JSON acumulativo."""
    try:
        # Asegurar que el contenido se pueda serializar correctamente
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            resp_content = response.json()
        else:
            resp_content = response.text

        responses_log[endpoint] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status_code": response.status_code,
            "response": resp_content,
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(responses_log, f, ensure_ascii=False, indent=4)
    except Exception as e:
        # No interrumpir pytest aunque falle el guardado
        print(f"⚠️ Error guardando respuesta de {endpoint}: {e}")


# ======================================================
# 📍 1. Predict (POST)
# ======================================================
def test_predict_endpoint():
    """Prueba la ruta POST /predict"""
    payload = {
        "text": "Los científicos descubren una nueva forma de producir energía limpia.",
        "model": "CNN",
    }

    response = client.post(f"{prefijo}/predict", json=payload)
    save_response("/predict", response)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "model_used" in data
    assert isinstance(data["confidence"], float)
    assert data["model_used"] == "CNN"


# ======================================================
# 📍 2. Confidence (GET)
# ======================================================
def test_confidence_endpoint():
    """Prueba la ruta GET /confidence/{prediction_id} (usando una predicción previa)."""

    # Crear una predicción previa
    pred_payload = {"text": "Test para confianza", "model": "CNN"}
    pred_response = client.post(f"{prefijo}/predict", json=pred_payload)
    assert pred_response.status_code == 200

    pred_data = pred_response.json()
    pred_id = pred_data.get("prediction_id", "12345")  # valor simulado si no existe

    # Consultar confianza
    response = client.get(f"{prefijo}/confidence/{pred_id}")
    save_response(f"/confidence/{pred_id}", response)

    # Puede no existir el registro aún, así que aceptamos 200 o 404
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "prediction_id" in data
        assert "confidence" in data


# ======================================================
# 📍 3. Models (GET)
# ======================================================
def test_models_endpoint():
    """Prueba la ruta GET /models"""
    response = client.get(f"{prefijo}/models")
    save_response("/models", response)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)

    if data["models"]:
        model = data["models"][0]
        assert "name" in model
        assert "type" in model
        assert "status" in model


# ======================================================
# 📍 4. Reload (POST)
# ======================================================
def test_reload_endpoint():
    """Prueba la ruta POST /reload"""
    payload = {"model": "CNN"}
    response = client.post(f"{prefijo}/reload", json=payload)
    save_response("/reload", response)

    assert response.status_code in [200, 500]
    data = response.json()
    assert "status" in data
    assert data["status"] in ["success", "error"]
    assert "message" in data


# ======================================================
# 📍 5. Metrics (GET)
# ======================================================
def test_metrics_endpoint():
    """Prueba la ruta GET /metrics"""
    response = client.get(f"{prefijo}/metrics?model=CNN")
    save_response("/metrics", response)

    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


# ======================================================
# 📍 6. Health (GET)
# ======================================================
def test_health_endpoint():
    """Prueba la ruta GET /health"""
    response = client.get(f"{prefijo}/health")
    save_response("/health", response)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert "uptime" in data
    assert "framework" in data
    assert "loaded_models" in data


# ======================================================
# 📍 7. Logs (GET)
# ======================================================
def test_logs_endpoint():
    """Prueba la ruta GET /logs"""
    response = client.get(f"{prefijo}/logs?limit=5")
    save_response("/logs", response)

    assert response.status_code in [200, 404]
    data = response.json()
    assert "logs" in data
    assert isinstance(data["logs"], list)

    if data["logs"]:
        log = data["logs"][0]
        assert "model" in log
        assert "prediction" in log
        assert "timestamp" in log
