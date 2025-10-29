import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from training.utils_common import (
    save_figure,
    load_previous_results,
    _params_to_frozenset,
)
from sklearn.model_selection import ParameterSampler


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
def save_results(cv_results, history_file, scoring_name="f1_macro"):
    """Guarda combinaciones probadas y sus resultados."""
    history = load_previous_results(history_file)

    # Crear set de combinaciones existentes para búsqueda rápida
    existing_combos = {_params_to_frozenset(h["params"]) for h in history}

    new_entries = []
    for params, score in zip(cv_results["params"], cv_results["mean_test_score"]):
        combo = _params_to_frozenset(params)

        if combo not in existing_combos:
            record = {
                "timestamp": datetime.now().isoformat(),
                "scoring": scoring_name,
                "params": params,
                "mean_test_score": float(score),
            }
            new_entries.append(record)
            existing_combos.add(combo)  # Evitar duplicados en el mismo batch

    if new_entries:
        history.extend(new_entries)
        with open(history_file, "w", encoding="utf8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(
            f"[INFO] {len(new_entries)} nuevas combinaciones registradas en {history_file}"
        )
    else:
        print("[INFO] No se registraron nuevas combinaciones (todas ya probadas).")


def get_unique_param_samples(param_distributions, history_file, n_iter, random_state):
    """Genera n_iter muestras que no hayan sido probadas antes."""
    history = load_previous_results(history_file)

    # Convertir historial a set de combinaciones
    tried_combinations = {_params_to_frozenset(h["params"]) for h in history}

    print(f"[INFO] {len(tried_combinations)} combinaciones únicas ya probadas.")

    # Generar más muestras de las necesarias para filtrar
    sampler = ParameterSampler(
        param_distributions,
        n_iter=n_iter * 3,  # Generar 3x para compensar filtrado
        random_state=random_state,
    )

    unique_samples = []
    for params in sampler:
        combo = _params_to_frozenset(params)

        if combo not in tried_combinations:
            grid_params = {k: [v] for k, v in params.items()}
            unique_samples.append(grid_params)
            tried_combinations.add(combo)

        if len(unique_samples) >= n_iter:
            break

    if len(unique_samples) < n_iter:
        print(
            f"[WARNING] Solo se pudieron generar {len(unique_samples)} combinaciones únicas de {n_iter} solicitadas"
        )
    else:
        print(f"[INFO] {len(unique_samples)} combinaciones únicas generadas.")

    return unique_samples
