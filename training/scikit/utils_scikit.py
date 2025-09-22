import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
