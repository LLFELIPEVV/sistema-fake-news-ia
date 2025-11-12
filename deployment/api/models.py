import os
import time
import spacy
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from deployment.api.config import MODELS_DIR, archivos
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.preprocessing_pipeline_fake_news import (
    normalizar_dataframe,
    estandarizar_texto,
)

# ======================================================
# 🧩 Paths de vectorizadores
# ======================================================
VECTORIZER_PATHS = {
    "Scikit-Learn": os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    "Keras": os.path.join(MODELS_DIR, "text_vectorizer_keras.keras"),
}


# ======================================================
# 🧠 Clase ModeloBase
# ======================================================
class ModeloBase:
    """Clase base para manejo unificado de modelos ML/DL."""

    def __init__(self, name: str, archivo: str):
        self.name = name
        self.archivo = archivo
        self.framework = self._determinar_framework(archivo)
        self.ruta = os.path.join(MODELS_DIR, archivo)
        self.tipo = self._determinar_tipo(self.framework)
        self.modelo = None
        self.vectorizer = None

    def _determinar_framework(self, archivo: str) -> str:
        if archivo.endswith(".pkl"):
            return "Scikit-Learn"
        elif archivo.endswith(".keras") or archivo.endswith(".h5"):
            return "Keras"
        return "Desconocido"

    def _determinar_tipo(self, framework: str) -> str:
        if framework == "Scikit-Learn":
            return "Machine Learning"
        elif framework == "Keras":
            return "Deep Learning"

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
            self.vectorizer = load_model(vec_path)

        print(f"✅ Modelo '{self.tipo}' ({self.framework}) cargado correctamente.")
        return self


# ======================================================
# 🧹 Funciones de preprocesamiento
# ======================================================
class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join([token.lemma_ for token in self.nlp(text)]) for text in X]


def preprocess_for_keras(texts, modelo):
    """Preprocesa texto para modelos Keras."""
    try:
        df = pd.DataFrame({"texto": [texts]})

        # Solo aplicar lematización para CNN
        if modelo.name == "CNN":
            lemmatizer = Lemmatizer()
            df["texto"] = lemmatizer.transform(df["texto"])

        df = normalizar_dataframe(df)
        df = estandarizar_texto(df)

        # Convertir a tensor de TensorFlow
        texto_procesado = df["texto"].values[0]

        # La capa TextVectorization espera un tensor de strings
        texto_tensor = tf.constant([texto_procesado], dtype=tf.string)

        # Aplicar vectorización
        vectorized = modelo.vectorizer(texto_tensor)

        return vectorized

    except Exception as e:
        print(f"❌ Error en preprocesamiento Keras: {e}")
        raise


def preprocess_for_sklearn(texts, vectorizer):
    """Preprocesa texto para modelos Scikit-Learn."""
    try:
        df = pd.DataFrame({"texto": [texts]})
        lemmatizer = Lemmatizer()
        df["texto"] = lemmatizer.transform(df["texto"])
        df = normalizar_dataframe(df)
        df = estandarizar_texto(df)
        return vectorizer.transform(df["texto"])
    except Exception as e:
        print(f"❌ Error en preprocesamiento Sklearn: {e}")
        raise


# ======================================================
# 🔮 Predicción unificada
# ======================================================
def predict_text(texts, model_name):
    """
    Realiza predicción de texto usando el modelo especificado.

    Args:
        texts: Texto a clasificar
        model_name: Nombre del modelo a usar

    Returns:
        dict: Resultado de la predicción con confianza, etiqueta y métricas
    """
    start_time = time.time()

    try:
        archivo = archivos.get(model_name)
        if not archivo:
            raise ValueError(f"Modelo '{model_name}' no reconocido.")

        modelo = ModeloBase(model_name, archivo).cargar_modelo()

        if modelo.framework == "Keras":
            # Preprocesar y predecir con Keras
            X = preprocess_for_keras(texts, modelo)
            y_pred = modelo.modelo.predict(X, verbose=0)

            # Extraer confianza según arquitectura del modelo
            if y_pred.shape[-1] == 1:
                # Salida binaria (sigmoid)
                confidence = float(y_pred[0][0])
            else:
                # Salida multi-clase (softmax)
                # Asumiendo que clase 0 = fake, clase 1 = real
                confidence = (
                    float(y_pred[0][1])
                    if y_pred.shape[-1] == 2
                    else float(np.max(y_pred[0]))
                )

        else:
            # Preprocesar mínimamente el texto (sin re-vectorizar)
            df = pd.DataFrame({"texto": [texts]})
            df = normalizar_dataframe(df)
            df = estandarizar_texto(df)
            texto_limpio = df["texto"].iloc[0]

            # El pipeline interno se encarga de lematizar + vectorizar
            y_pred = modelo.modelo.predict_proba([texto_limpio])

            confidence = (
                float(y_pred[0][1])
                if y_pred.shape[1] == 2
                else float(np.max(y_pred[0]))
            )

        # Determinar etiqueta según confianza
        # Si confianza >= 0.5, clasificamos como "real", sino "fake"
        pred_label = "real" if confidence >= 0.5 else "fake"

        inference_time = time.time() - start_time

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "model_used": model_name,
            "inference_time_ms": round(inference_time * 1000, 3),
            "tokens_count": len(texts.split()),
        }

    except Exception as e:
        print(f"❌ Error en predict_text: {e}")
        import traceback

        traceback.print_exc()
        raise


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
