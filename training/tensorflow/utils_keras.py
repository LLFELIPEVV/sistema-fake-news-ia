import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from training.utils_common import save_figure

# Embedding GloVe path
GLOVE_PATH = os.path.join("glove.840B.300d", "glove.840B.300d.txt")


def save_best_model_keras(model, score, BEST_SCORE_PATH, BEST_MODEL_PATH):
    """Guarda el mejor modelo de Keras si mejora el anterior."""
    if os.path.exists(BEST_SCORE_PATH):
        with open(BEST_SCORE_PATH, "r") as f:
            best_score = float(f.read().strip())
    else:
        best_score = -1

    if score > best_score:
        # Guardar en formato HDF5
        model.save(BEST_MODEL_PATH)
        with open(BEST_SCORE_PATH, "w") as f:
            f.write(str(score))
        print(f"[INFO] Nuevo mejor modelo Keras guardado con score {score:.4f}")
    else:
        print(
            f"[INFO] El modelo Keras ({score:.4f}) no supera al mejor ({best_score:.4f})."
        )


def load_best_model_keras(BEST_MODEL_PATH):
    """Carga el mejor modelo Keras guardado (.h5 o SavedModel)."""
    if os.path.exists(BEST_MODEL_PATH):
        print(f"[INFO] Modelo Keras cargado desde {BEST_MODEL_PATH}")
        return load_model(BEST_MODEL_PATH)
    else:
        raise FileNotFoundError(f"No se encontró el modelo en {BEST_MODEL_PATH}")


def plot_training_history(
    history, filename="training_history.png", metrics=("accuracy", "loss")
):
    """
    Grafica la evolución de métricas durante el entrenamiento de un modelo Keras.

    Args:
        history: objeto keras.callbacks.History devuelto por model.fit()
        filename: nombre del archivo donde guardar la figura
        metrics: tupla/lista de métricas a graficar (ej: ("accuracy", "loss"))
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]  # Asegura lista iterable

    for ax, metric in zip(axes, metrics):
        if metric not in history.history:
            print(f"⚠️ Métrica '{metric}' no encontrada en history.history")
            continue

        ax.plot(history.history[metric], label=f"Train {metric}")
        if f"val_{metric}" in history.history:
            ax.plot(history.history[f"val_{metric}"], label=f"Val {metric}")

        ax.set_title(f"Evolución de {metric}")
        ax.set_xlabel("Épocas")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


# ==============================
# 🧠 Construcción Embedding
# ==============================
def build_embedding_matrix(vectorizer, embedding_dim=300):
    vocab = vectorizer.get_vocabulary()
    word_index = dict(zip(vocab, range(len(vocab))))
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    skipped = 0
    loaded = 0

    with open(GLOVE_PATH, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            if len(values) != embedding_dim + 1:
                # Línea corrupta o con tokens de más/menos
                skipped += 1
                continue

            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
                coefs /= np.linalg.norm(coefs) + 1e-8  # Normalización
            except ValueError:
                skipped += 1
                continue

            if word in word_index:
                embedding_matrix[word_index[word]] = coefs
                loaded += 1

    print(f"[INFO] Embeddings cargados correctamente: {loaded}")
    print(f"[WARNING] Líneas omitidas por formato incorrecto: {skipped}")
    print(f"[INFO] Tamaño final de embedding_matrix: {embedding_matrix.shape}")
    return embedding_matrix
