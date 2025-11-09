import os
import json
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from keras.models import load_model
from training.utils_common import (
    save_figure,
    load_previous_results,
    _params_to_frozenset,
)

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


# =============================
# FUNCIONES DE REGISTRO
# =============================
def save_results(params, metrics, history_file):
    """Guarda hiperparámetros y métricas (Keras)."""
    history = load_previous_results(history_file)
    existing_combos = {_params_to_frozenset(h["params"]) for h in history}
    combo = _params_to_frozenset(params)

    if combo not in existing_combos:
        record = {
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "metrics": metrics,
        }
        history.append(record)

        with open(history_file, "w", encoding="utf8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Nueva combinación registrada en {history_file}")
    else:
        print("[INFO] Combinación de hiperparámetros ya registrada.")


def get_random_hyperparams(
    param_space, history_file, max_attempts=None, random_state=42
):
    """
    Genera combinaciones únicas de hiperparámetros para Keras/TensorFlow.

    Esta función primero intenta generar TODAS las combinaciones posibles del espacio
    de búsqueda. Si hay demasiadas (>10000), usa muestreo aleatorio.

    Args:
        param_space: dict con listas de valores posibles para cada hiperparámetro
                    Ejemplo: {'learning_rate': [0.001, 0.01], 'units': [64, 128]}
        history_file: archivo JSON con historial de combinaciones probadas
        max_attempts: número máximo de intentos para muestreo aleatorio (None = automático)
        random_state: semilla para reproducibilidad

    Returns:
        dict: combinación de hiperparámetros única, o None si no hay más disponibles
    """
    history = load_previous_results(history_file)

    # Convertir historial a set de combinaciones
    tried_combinations = {_params_to_frozenset(h["params"]) for h in history}

    print(f"[INFO] {len(tried_combinations)} combinaciones únicas ya probadas.")

    # Calcular número total de combinaciones posibles
    total_combinations = 1
    for values in param_space.values():
        total_combinations *= len(values)

    print(f"[INFO] Espacio de búsqueda: {total_combinations:,} combinaciones posibles")

    # Calcular saturación del espacio
    if total_combinations > 0:
        saturation_ratio = len(tried_combinations) / total_combinations
        print(f"[INFO] Saturación del espacio: {saturation_ratio * 100:.2f}%")
    else:
        saturation_ratio = 0

    # Estrategia 1: Si hay pocas combinaciones (<= 10000), usar búsqueda exhaustiva
    if total_combinations <= 10000:
        print("[INFO] Usando búsqueda EXHAUSTIVA (todas las combinaciones)")

        # Generar todas las combinaciones posibles
        keys = list(param_space.keys())
        values = [param_space[key] for key in keys]
        all_combinations = [
            dict(zip(keys, combo)) for combo in itertools.product(*values)
        ]

        print(f"[INFO] {len(all_combinations)} combinaciones generadas")

        # Filtrar las ya probadas
        available_combinations = []
        for params in all_combinations:
            combo = _params_to_frozenset(params)
            if combo not in tried_combinations:
                available_combinations.append(params)

        print(
            f"[INFO] {len(available_combinations)} combinaciones no probadas encontradas"
        )

        if available_combinations:
            # Establecer semilla para reproducibilidad
            random.seed(random_state)
            selected = random.choice(available_combinations)
            print("[INFO] ✅ Combinación única seleccionada")
            return selected
        else:
            print("[WARNING] ⚠️  Espacio de búsqueda completamente explorado")
            print(
                f"[INFO] Todas las {total_combinations} combinaciones ya fueron probadas"
            )
            return None

    # Estrategia 2: Si hay muchas combinaciones, usar muestreo aleatorio inteligente
    else:
        print("[INFO] Espacio muy grande, usando MUESTREO ALEATORIO")

        # Calcular max_attempts automáticamente si no se especificó
        if max_attempts is None:
            if saturation_ratio > 0.5:
                max_attempts = 1000  # Espacio muy saturado
            elif saturation_ratio > 0.1:
                max_attempts = 500  # Moderadamente saturado
            else:
                max_attempts = 100  # Poco explorado

        print(
            f"[INFO] Intentando generar combinación única (máx {max_attempts} intentos)"
        )

        # Establecer semilla para reproducibilidad
        random.seed(random_state)

        for attempt in range(max_attempts):
            params = {key: random.choice(values) for key, values in param_space.items()}
            combo = _params_to_frozenset(params)

            if combo not in tried_combinations:
                print(
                    f"[INFO] ✅ Nueva combinación encontrada en intento {attempt + 1}"
                )
                return params

        # Si no se encontró combinación única después de max_attempts
        print(
            f"[WARNING] ⚠️  No se encontró combinación única en {max_attempts} intentos"
        )

        # Calcular combinaciones restantes estimadas
        remaining = total_combinations - len(tried_combinations)
        print(f"[INFO] Combinaciones restantes estimadas: {remaining:,}")

        if saturation_ratio > 0.9:
            print(
                f"[WARNING] Espacio altamente saturado (>{saturation_ratio * 100:.1f}%)"
            )
            print("[INFO] Considera:")
            print("       - Expandir el espacio de búsqueda")
            print("       - Eliminar historial: rm {history_file}")
            return None
        else:
            # Si la saturación no es crítica, devolver una combinación aleatoria
            # (puede estar repetida, pero es poco probable)
            params = {key: random.choice(values) for key, values in param_space.items()}
            print("[INFO] ⚠️  Devolviendo combinación aleatoria (puede estar repetida)")
            return params
