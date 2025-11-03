import os
import time
import joblib
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from deployment.api.config import MODELS_DIR, archivos
from preprocessing.preprocessing_pipeline_fake_news import (
    normalizar_dataframe,
    estandarizar_texto,
)

# ======================================================
# 🧩 Paths de vectorizadores
# ======================================================
VECTORIZER_PATHS = {
    "Scikit-Learn": os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    "Keras": os.path.join(MODELS_DIR, "text_vectorizer_keras"),
}


# ======================================================
# 🧠 Clase ModeloBase
# ======================================================
class ModeloBase:
    """Clase base para manejo unificado de modelos ML/DL."""

    def __init__(self, tipo: str, archivo: str):
        self.tipo = tipo
        self.archivo = archivo
        self.framework = self._determinar_framework(archivo)
        self.ruta = os.path.join(MODELS_DIR, archivo)
        self.modelo = None
        self.vectorizer = None

    def _determinar_framework(self, archivo: str) -> str:
        if archivo.endswith(".pkl"):
            return "Scikit-Learn"
        elif archivo.endswith(".keras") or archivo.endswith(".h5"):
            return "Keras"
        return "Desconocido"

    def cargar_modelo(self):
        """Carga modelo y su vectorizador correspondiente."""
        if not os.path.exists(self.ruta):
            raise FileNotFoundError(f"❌ No se encontró el modelo: {self.ruta}")

        # --- Cargar modelo ---
        if self.framework == "Scikit-Learn":
            self.modelo = joblib.load(self.ruta)
        elif self.framework == "Keras":
            self.modelo = load_model(self.ruta)
        else:
            raise ValueError(f"Framework no soportado: {self.framework}")

        # --- Cargar vectorizador ---
        vec_path = VECTORIZER_PATHS[self.framework]
        if self.framework == "Scikit-Learn":
            self.vectorizer = joblib.load(vec_path)
        elif self.framework == "Keras":
            # ⚠️ Importante: TextVectorization se carga con tf.saved_model.load()
            self.vectorizer = tf.saved_model.load(vec_path)

        print(f"✅ Modelo '{self.tipo}' ({self.framework}) cargado correctamente.")
        return self


# ======================================================
# 🧹 Funciones de preprocesamiento
# ======================================================
def preprocess_for_keras(texts, vectorizer):
    df = pd.DataFrame({"texto": [texts]})
    df = normalizar_dataframe(df)
    df = estandarizar_texto(df)
    # La capa TextVectorization es callable (como una función)
    return vectorizer(df["texto"].values)


def preprocess_for_sklearn(texts, vectorizer):
    df = pd.DataFrame({"texto": [texts]})
    df = normalizar_dataframe(df)
    df = estandarizar_texto(df)
    return vectorizer.transform(df["texto"])


# ======================================================
# 🔮 Predicción unificada
# ======================================================
def predict_text(texts, model_name):
    start_time = time.time()

    archivo = archivos.get(model_name)
    if not archivo:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")

    modelo = ModeloBase(model_name, archivo).cargar_modelo()

    if modelo.framework == "Keras":
        X = preprocess_for_keras(texts, modelo.vectorizer)
        y_pred = modelo.modelo.predict(X)
        confidence = (
            float(y_pred[0][0]) if y_pred.shape[-1] == 1 else float(max(y_pred[0]))
        )
    else:
        X = preprocess_for_sklearn(texts, modelo.vectorizer)
        y_pred = modelo.modelo.predict_proba(X)
        confidence = float(max(y_pred[0]))

    pred_label = "real" if confidence >= 0.5 else "fake"
    inference_time = time.time() - start_time

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "model_used": model_name,
        "inference_time_ms": round(inference_time * 1000, 3),
        "tokens_count": len(texts.split()),
    }


def get_file_size_mb(filepath: str) -> float:
    """Devuelve el tamaño del archivo en MB."""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    return 0.0


def read_last_used(model_name: str) -> str:
    """Lee la última fecha de uso de un modelo (si existe el registro)."""
    path = os.path.join(MODELS_DIR, f"{model_name}_last_used.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "N/A"
