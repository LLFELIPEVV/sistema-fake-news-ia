import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import stopwordsiso as stopwords

from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Stopwords en español
SPANISH_STOPWORDS = list(stopwords.stopwords("es"))

# Carpetas de salida
MODEL_DIR = "models"
FIG_DIR = "figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_best_model.pkl")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "logistic_regression_best_score.txt")


def load_datasets():
    """Carga los datasets preprocesados en formato parquet."""
    train_df = pd.read_parquet("data/splits/fake_news_train.parquet")
    valid_df = pd.read_parquet("data/splits/fake_news_valid.parquet")
    test_df = pd.read_parquet("data/splits/fake_news_test.parquet")
    return train_df, valid_df, test_df


def build_pipeline():
    """Construye un pipeline con TF-IDF y Regresión Logística."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words=SPANISH_STOPWORDS)),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    tol=1e-4,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ],
        memory="__cache__",
    )


def optimize_logreg(X_train, y_train):
    """Optimiza hiperparámetros de Regresión Logística con RandomizedSearchCV."""
    pipeline = build_pipeline()

    param_distributions = {
        "clf__penalty": ["l2", "elasticnet"],
        "clf__l1_ratio": uniform(0, 1),
        "clf__C": loguniform(1e-3, 1e1),
        "clf__solver": ["saga"],
        "tfidf__max_features": [20000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=3,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)
    return random_search


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


def plot_grid_search_results(grid, filename="grid_results.png"):
    """Grafica resultados de RandomizedSearch con líneas en lugar de puntos."""
    results = pd.DataFrame(grid.cv_results_)

    fig, ax = plt.subplots(figsize=(10, 6))
    for ngram in results["param_tfidf__ngram_range"].unique():
        subset = results[results["param_tfidf__ngram_range"] == ngram]
        subset = subset.sort_values("param_clf__C")
        ax.plot(
            subset["param_clf__C"],
            subset["mean_test_score"],
            marker="o",
            label=f"N-gram {ngram}",
        )

    ax.set_xscale("log")
    ax.set_title("Resultados de RandomizedSearch (rendimiento por parámetros)")
    ax.set_xlabel("Valor de C (log scale)")
    ax.set_ylabel("F1 ponderado (media CV)")
    ax.legend()
    save_figure(fig, filename)
    plt.close(fig)


def save_best_model(model, score):
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


if __name__ == "__main__":
    # Cargar datasets
    train_df, valid_df, test_df = load_datasets()
    X_train, y_train = train_df["texto"], train_df["clase"]
    X_valid, y_valid = valid_df["texto"], valid_df["clase"]
    X_test, y_test = test_df["texto"], test_df["clase"]

    # Optimizar modelo
    print("=== Buscando mejores hiperparámetros... ===")
    grid = optimize_logreg(X_train, y_train)

    print("\n=== Mejores parámetros encontrados ===")
    print(grid.best_params_)
    print(f"\n=== Mejor F1 ponderado en CV: {grid.best_score_:.4f} ===")

    best_model = grid.best_estimator_

    # Guardar modelo si es mejor
    save_best_model(best_model, grid.best_score_)

    # Evaluación en validación
    y_valid_pred = best_model.predict(X_valid)
    print("\n=== Reporte de Validación ===")
    print(classification_report(y_valid, y_valid_pred))
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        "Matriz de confusión - Validación",
        "logreg_confusion_valid.png",
    )
    valid_metrics = plot_metrics(
        y_valid, y_valid_pred, "Validación", "logreg_metrics_valid.png"
    )

    # Evaluación en prueba
    y_test_pred = best_model.predict(X_test)
    print("\n=== Reporte de Prueba ===")
    print(classification_report(y_test, y_test_pred))
    plot_confusion_matrix(
        y_test, y_test_pred, "Matriz de confusión - Prueba", "logreg_confusion_test.png"
    )
    test_metrics = plot_metrics(
        y_test, y_test_pred, "Prueba", "logreg_metrics_test.png"
    )

    # Comparación de métricas entre validación y prueba
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)
    ax.set_title("Comparación de métricas entre Validación y Prueba")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    save_figure(fig, "logreg_comparison_valid_test.png")
    plt.close(fig)

    # Resultados de RandomizedSearch
    plot_grid_search_results(grid, "logreg_random_search_results.png")
