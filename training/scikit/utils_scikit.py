import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from training.utils_common import save_figure


def save_best_model_sklearn(model, score, BEST_SCORE_PATH, BEST_MODEL_PATH):
    """Guarda el mejor modelo si mejora el anterior."""
    if os.path.exists(BEST_SCORE_PATH):
        with open(BEST_SCORE_PATH, "r") as f:
            best_score = float(f.read().strip())
    else:
        best_score = -1

    if score > best_score:
        joblib.dump(model, BEST_MODEL_PATH)
        with open(BEST_SCORE_PATH, "w") as f:
            f.write(str(score))
        print(f"[INFO] Nuevo mejor modelo guardado con score {score:.4f}")
    else:
        print(
            f"[INFO] El modelo actual ({score:.4f}) no supera al mejor ({best_score:.4f})."
        )


def plot_grid_search_results(grid, filename="grid_results.png", param_x="param_clf__C"):
    """
    Grafica resultados de RandomizedSearch para un parámetro dado.

    Args:
        grid: objeto RandomizedSearchCV ya entrenado
        filename: nombre del archivo donde guardar la figura
        param_x: nombre del hiperparámetro en grid.cv_results_ para el eje X
    """
    results = pd.DataFrame(grid.cv_results_)

    if param_x not in results.columns:
        raise ValueError(f"El parámetro {param_x} no está en los resultados.")

    fig, ax = plt.subplots(figsize=(10, 6))

    if "param_tfidf__ngram_range" in results.columns:
        for ngram in results["param_tfidf__ngram_range"].unique():
            subset = results[results["param_tfidf__ngram_range"] == ngram]
            subset = subset.sort_values(param_x)
            ax.plot(
                subset[param_x],
                subset["mean_test_score"],
                marker="o",
                label=f"N-gram {ngram}",
            )
    else:
        # Si no se probó con n-gramas, grafica solo contra param_x
        results = results.sort_values(param_x)
        ax.plot(results[param_x], results["mean_test_score"], marker="o")

    # Si param_x es 'C' o 'gamma', conviene logscale
    if any(k in param_x.lower() for k in ["c", "gamma"]):
        ax.set_xscale("log")

    ax.set_title(f"Resultados de RandomizedSearch ({param_x})")
    ax.set_xlabel(param_x)
    ax.set_ylabel("F1 ponderado (media CV)")
    ax.legend()
    save_figure(fig, filename)
    plt.close(fig)


# =============================
# FUNCIONES DE REGISTRO
# =============================
def load_previous_results(history_file):
    """Carga combinaciones ya probadas de hiperparámetros."""
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf8") as f:
            return json.load(f)
    return []


def save_results(cv_results, history_file, scoring_name="f1_macro"):
    """Guarda combinaciones probadas y sus resultados."""
    history = load_previous_results(history_file)
    new_entries = []

    for params, score in zip(cv_results["params"], cv_results["mean_test_score"]):
        record = {
            "timestamp": datetime.now().isoformat(),
            "scoring": scoring_name,
            "params": params,
            "mean_test_score": float(score),
        }
        if record["params"] not in [h["params"] for h in history]:
            new_entries.append(record)

    if new_entries:
        history.extend(new_entries)
        with open(history_file, "w", encoding="utf8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(
            f"[INFO] {len(new_entries)} nuevas combinaciones registradas en {history_file}"
        )
    else:
        print("[INFO] No se registraron nuevas combinaciones (todas ya probadas).")


def filter_used_combinations(param_distributions, history_file):
    """Elimina combinaciones ya probadas de los valores discretos."""
    history = load_previous_results(history_file)
    if not history:
        return param_distributions

    tried_params = [h["params"] for h in history]
    new_params = {}

    for key, values in param_distributions.items():
        if isinstance(values, list):
            tried_values = set()
            for p in tried_params:
                if key in p:
                    v = p[key]
                    # Convertir listas/tuplas a tupla hashable
                    if isinstance(v, (list, tuple)):
                        v = tuple(v)
                    tried_values.add(v)

            filtered = []
            for v in values:
                val_hash = tuple(v) if isinstance(v, (list, tuple)) else v
                if val_hash not in tried_values:
                    filtered.append(v)

            if filtered:
                new_params[key] = filtered
            else:
                new_params[key] = values  # si se agotaron, no filtrar
        else:
            new_params[key] = values  # distribuciones continuas no se filtran

    print(
        f"[INFO] Se filtraron valores ya probados. Nuevas combinaciones posibles: {len(new_params)} parámetros."
    )
    return new_params
