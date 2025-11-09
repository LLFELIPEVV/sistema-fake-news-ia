import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import stopwordsiso as stopwords

from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from training.scikit.utils_scikit import (
    save_figure,
    plot_grid_search_results,
    save_best_model_sklearn,
    get_unique_param_samples,
    save_results,
)
from training.utils_common import (
    MODEL_DIR,
    load_datasets,
    plot_confusion_matrix,
    plot_metrics,
    Lemmatizer,
)


# Stopwords en español
SPANISH_STOPWORDS = list(stopwords.stopwords("es"))
RANDOM_STATE = 42
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_best_model.pkl")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "random_forest_best_score.txt")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
HISTORY_FILE = os.path.join(MODEL_DIR, "rf_hyperparam_history.json")


def build_pipeline():
    """Construye un pipeline con lematización, TF-IDF y Random Forest optimizado."""
    return Pipeline(
        steps=[
            ("lemmatizer", Lemmatizer()),  # Aplica lematización antes del vectorizado
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=SPANISH_STOPWORDS,
                    sublinear_tf=True,
                    norm="l2",
                    max_df=0.85,
                    min_df=3,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced_subsample",
                    criterion="entropy",
                    n_jobs=-1,
                ),
            ),
        ],
        memory="__cache__",
    )


def optimize_random_forest(X_train, y_train):
    """Optimiza hiperparámetros de Random Forest con RandomizedSearchCV."""
    pipeline = build_pipeline()

    base_param_distributions = {
        "clf__n_estimators": randint(300, 800),
        "clf__max_depth": randint(10, 60),
        "clf__min_samples_split": randint(3, 12),
        "clf__min_samples_leaf": randint(2, 8),
        "clf__max_features": ["sqrt", "log2"],
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    }

    # Generar combinaciones únicas
    unique_params = get_unique_param_samples(
        base_param_distributions, HISTORY_FILE, n_iter=12, random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid=unique_params,
        cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)
    save_results(grid_search.cv_results_, HISTORY_FILE, "f1_weighted")
    return grid_search


if __name__ == "__main__":
    # Cargar datasets
    train_df, valid_df, test_df = load_datasets()
    X_train, y_train = train_df["texto"], train_df["clase"]
    X_valid, y_valid = valid_df["texto"], valid_df["clase"]
    X_test, y_test = test_df["texto"], test_df["clase"]

    # Optimizar modelo
    print("=== Buscando mejores hiperparámetros (Random Forest)... ===")
    grid = optimize_random_forest(X_train, y_train)

    print("\n=== Mejores parámetros encontrados ===")
    print(grid.best_params_)
    print(f"\n=== Mejor F1 ponderado en CV: {grid.best_score_:.4f} ===")

    best_model = grid.best_estimator_

    # Guardar modelo si es mejor
    save_best_model_sklearn(
        best_model, grid.best_score_, BEST_SCORE_PATH, BEST_MODEL_PATH
    )
    print("\n🔄 Cargando modelo guardado desde disco...")
    loaded_best_model = joblib.load(BEST_MODEL_PATH)

    # Guardar vectorizador TF-IDF
    VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    tfidf_vectorizer = best_model.named_steps["tfidf"]
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    print(f"✅ Vectorizador TF-IDF guardado en: {VECTORIZER_PATH}")

    # Evaluación en validación
    y_valid_pred = loaded_best_model.predict(X_valid)
    print("\n=== Reporte de Validación ===")
    print(classification_report(y_valid, y_valid_pred))
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        "Matriz de confusión - Validación",
        "random_forest_confusion_valid.png",
    )
    valid_metrics = plot_metrics(
        y_valid, y_valid_pred, "Validación", "random_forest_metrics_valid.png"
    )

    # Evaluación en prueba
    y_test_pred = loaded_best_model.predict(X_test)
    print("\n=== Reporte de Prueba ===")
    print(classification_report(y_test, y_test_pred))
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        "Matriz de confusión - Prueba",
        "random_forest_confusion_test.png",
    )
    test_metrics = plot_metrics(
        y_test, y_test_pred, "Prueba", "random_forest_metrics_test.png"
    )

    # Comparación de métricas entre validación y prueba
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)

    ax.set_title("Comparación de métricas entre Validación y Prueba (Random Forest)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)

    # === Etiquetas encima de cada barra ===
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)

    save_figure(fig, "random_forest_comparison_valid_test.png")
    plt.close(fig)

    # Resultados de RandomizedSearch
    plot_grid_search_results(
        grid, "random_forest_random_search_results.png", "param_clf__n_estimators"
    )
