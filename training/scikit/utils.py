import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Carpetas de salida
MODEL_DIR = "models"
FIG_DIR = "figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_datasets():
    """Carga los datasets preprocesados en formato parquet."""
    train_df = pd.read_parquet("data/splits/fake_news_train.parquet")
    valid_df = pd.read_parquet("data/splits/fake_news_valid.parquet")
    test_df = pd.read_parquet("data/splits/fake_news_test.parquet")
    return train_df, valid_df, test_df


def save_figure(fig, filename):
    """Guarda una figura en la carpeta FIG_DIR."""
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"[INFO] Gráfico guardado en {path}")


def plot_confusion_matrix(y_true, y_pred, title="Matriz de confusión", filename=None):
    """Muestra y guarda matriz de confusión normalizada con Seaborn."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["No Fake", "Fake"],
        yticklabels=["No Fake", "Fake"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Predicción")
    save_figure(fig, filename if filename else "confusion_matrix.png")
    plt.close(fig)


def plot_metrics(y_true, y_pred, dataset_name="Validación", filename=None):
    """Grafica métricas de clasificación en barras y guarda la imagen."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1": f1_score(y_true, y_pred, average="macro"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=list(metrics.keys()), y=list(metrics.values()), palette="viridis", ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Métricas en {dataset_name}")
    ax.set_ylabel("Valor")

    save_figure(fig, filename if filename else f"metrics_{dataset_name}.png")
    plt.close(fig)

    return metrics


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


def save_best_model(model, score, BEST_SCORE_PATH, BEST_MODEL_PATH):
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
