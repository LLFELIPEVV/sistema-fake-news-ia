import os
import pandas as pd
import matplotlib.pyplot as plt
import stopwordsiso as stopwords

from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from training.utils_common import (
    MODEL_DIR,
    load_datasets,
    plot_confusion_matrix,
    plot_metrics,
)
from training.scikit.utils_scikit import (
    save_figure,
    plot_grid_search_results,
    save_best_model_sklearn,
)

# Stopwords en español
SPANISH_STOPWORDS = list(stopwords.stopwords("es"))

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree_best_model.pkl")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "decision_tree_best_score.txt")


def build_pipeline():
    """Construye un pipeline con TF-IDF y Decision Tree."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words=SPANISH_STOPWORDS)),
            (
                "clf",
                DecisionTreeClassifier(
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ],
        memory="__cache__",
    )


def optimize_decision_tree(X_train, y_train):
    """Optimiza hiperparámetros de Decision Tree con RandomizedSearchCV."""
    pipeline = build_pipeline()

    param_distributions = {
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_depth": randint(5, 50),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 20),
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


if __name__ == "__main__":
    # Cargar datasets
    train_df, valid_df, test_df = load_datasets()
    X_train, y_train = train_df["texto"], train_df["clase"]
    X_valid, y_valid = valid_df["texto"], valid_df["clase"]
    X_test, y_test = test_df["texto"], test_df["clase"]

    # Optimizar modelo
    print("=== Buscando mejores hiperparámetros (Decision Tree)... ===")
    grid = optimize_decision_tree(X_train, y_train)

    print("\n=== Mejores parámetros encontrados ===")
    print(grid.best_params_)
    print(f"\n=== Mejor F1 ponderado en CV: {grid.best_score_:.4f} ===")

    best_model = grid.best_estimator_

    # Guardar modelo si es mejor
    save_best_model_sklearn(
        best_model, grid.best_score_, BEST_SCORE_PATH, BEST_MODEL_PATH
    )

    # Evaluación en validación
    y_valid_pred = best_model.predict(X_valid)
    print("\n=== Reporte de Validación ===")
    print(classification_report(y_valid, y_valid_pred))
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        "Matriz de confusión - Validación",
        "decision_tree_confusion_valid.png",
    )
    valid_metrics = plot_metrics(
        y_valid, y_valid_pred, "Validación", "decision_tree_metrics_valid.png"
    )

    # Evaluación en prueba
    y_test_pred = best_model.predict(X_test)
    print("\n=== Reporte de Prueba ===")
    print(classification_report(y_test, y_test_pred))
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        "Matriz de confusión - Prueba",
        "decision_tree_confusion_test.png",
    )
    test_metrics = plot_metrics(
        y_test, y_test_pred, "Prueba", "decision_tree_metrics_test.png"
    )

    # Comparación de métricas entre validación y prueba
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)

    ax.set_title("Comparación de métricas entre Validación y Prueba (Decision Tree)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)

    # === Etiquetas encima de cada barra ===
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)

    save_figure(fig, "decision_tree_comparison_valid_test.png")
    plt.close(fig)

    # Resultados de RandomizedSearch
    plot_grid_search_results(
        grid, "decision_tree_random_search_results.png", "param_clf__max_depth"
    )
