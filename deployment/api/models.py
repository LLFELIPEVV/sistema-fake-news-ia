import os
import joblib

from config import MODELS_DIR
from keras.models import load_model


class ModeloBase:
    """
    Clase base para manejar modelos de ML/DL.
    Permite identificar el tipo de modelo y cargarlo desde la carpeta /models.
    """

    def __init__(self, tipo: str, archivo: str):
        self.tipo = tipo
        self.archivo = archivo
        self.nombre = self._determinar_nombre(tipo)
        self.framework = self._determinar_framework(archivo)
        self.ruta = os.path.join(MODELS_DIR, archivo)
        self.modelo = None

    def _determinar_nombre(self, tipo: str) -> str:
        """Mapea el tipo de desempeño a un nombre de modelo concreto."""
        mapping = {
            "Mejor en detección equilibrada de ambos tipos de noticias.": "CNN",
            "Mejor modelo general.": "CNN",
            "Mejor en consistencia general.": "CNN",
            "Mejor en detección de noticias reales.": "Random Forest",
            "Modelo con menor sobreajuste.": "Random Forest",
            "Mejor en rapidez y eficiencia.": "Naive Bayes",
            "Mejor en estabilidad y reproducibilidad.": "Naive Bayes",
            "Mejor en detección de noticias falsas.": "Hibrido",
        }
        return mapping.get(tipo, "Desconocido")

    def _determinar_framework(self, archivo: str) -> str:
        """Detecta framework según la extensión del archivo."""
        if archivo.endswith(".pkl"):
            return "Scikit-Learn"
        elif archivo.endswith(".keras") or archivo.endswith(".h5"):
            return "Keras"
        return "Desconocido"

    def cargar_modelo(self):
        """Carga el modelo desde disco según su framework."""
        if not os.path.exists(self.ruta):
            raise FileNotFoundError(f"❌ No se encontró el modelo: {self.ruta}")

        if self.framework == "Scikit-Learn":
            self.modelo = joblib.load(self.ruta)
        elif self.framework == "Keras":
            self.modelo = load_model(self.ruta)
        else:
            raise ValueError(f"Framework no soportado: {self.framework}")

        print(
            f"✅ Modelo '{self.nombre}' ({self.framework}) cargado desde: {self.ruta}"
        )
        return self.modelo


def cargar_modelo(archivo: str, tipo: str):
    """
    Función auxiliar que instancia y carga automáticamente
    un modelo basado en su extensión y tipo de desempeño.
    """
    modelo = ModeloBase(tipo, archivo)
    modelo.cargar_modelo()
    return modelo


if __name__ == "__main__":
    # Ejemplo 1: modelo Keras
    modelo_keras = cargar_modelo(
        "hybrid_cnn_bilstm_gru_attention_best_model.keras",
        "Mejor en detección de noticias falsas.",
    )

    # Ejemplo 2: modelo Scikit-Learn
    modelo_rf = cargar_modelo(
        "random_forest_best_model.pkl", "Modelo con menor sobreajuste."
    )
