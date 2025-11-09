import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import stopwordsiso as stopwords

from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from training.utils_common import (
    MODEL_DIR,
    load_datasets,
    plot_confusion_matrix,
    plot_metrics,
    Lemmatizer,
)
from training.scikit.utils_scikit import (
    save_figure,
    plot_grid_search_results,
    save_best_model_sklearn,
    get_unique_param_samples,
    save_results,
)

RANDOM_STATE = 42
SPANISH_STOPWORDS = list(stopwords.stopwords("es"))
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_best_model.pkl")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "naive_bayes_best_score.txt")
HISTORY_FILE = os.path.join(MODEL_DIR, "nb_hyperparam_history.json")


def build_pipeline():
    """Construye pipeline con lematización, TF-IDF y ComplementNB."""
    return Pipeline(
        steps=[
            ("lemmatizer", Lemmatizer()),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=SPANISH_STOPWORDS,
                    ngram_range=(1, 2),  # unigrama + bigrama
                    max_df=0.85,  # ignora palabras muy frecuentes
                    min_df=3,  # ignora palabras raras
                    sublinear_tf=True,  # suavizado logarítmico
                    norm="l2",  # regularización L2
                    lowercase=True,
                    max_features=30000,
                    smooth_idf=True,
                    use_idf=True,
                ),
            ),
            ("clf", ComplementNB(alpha=0.3, norm=True)),  # robusto al desbalance
        ],
        memory="__cache__",  # cache para eficiencia
    )


def optimize_nb(X_train, y_train):
    """Optimiza hiperparámetros de ComplementNB con RandomizedSearchCV."""
    pipeline = build_pipeline()

    base_param_distributions = {
        "clf__alpha": loguniform(1e-3, 5),
        "clf__norm": [True, False],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [10000, 20000, 30000],
        "tfidf__sublinear_tf": [True, False],
        "tfidf__smooth_idf": [True, False],
        "tfidf__min_df": [3, 5, 10],
    }

    # Generar combinaciones únicas
    unique_params = get_unique_param_samples(
        base_param_distributions, HISTORY_FILE, n_iter=12, random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid=unique_params,  # Lista de diccionarios
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)
    save_results(grid_search.cv_results_, HISTORY_FILE, "f1_macro")
    return grid_search


if __name__ == "__main__":
    # Cargar datasets
    train_df, valid_df, test_df = load_datasets()
    X_train, y_train = train_df["texto"], train_df["clase"]
    X_valid, y_valid = valid_df["texto"], valid_df["clase"]
    X_test, y_test = test_df["texto"], test_df["clase"]

    # Optimizar modelo
    print("=== Buscando mejores hiperparámetros (Naive Bayes)... ===")
    grid = optimize_nb(X_train, y_train)

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

    # Evaluación en validación
    y_valid_pred = loaded_best_model.predict(X_valid)
    print("\n=== Reporte de Validación ===")
    print(classification_report(y_valid, y_valid_pred))
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        "Matriz de confusión - Validación",
        "nb_confusion_valid.png",
    )
    valid_metrics = plot_metrics(
        y_valid, y_valid_pred, "Validación", "nb_metrics_valid.png"
    )

    # Evaluación en prueba
    y_test_pred = loaded_best_model.predict(X_test)
    print("\n=== Reporte de Prueba ===")
    print(classification_report(y_test, y_test_pred))
    plot_confusion_matrix(
        y_test, y_test_pred, "Matriz de confusión - Prueba", "nb_confusion_test.png"
    )
    test_metrics = plot_metrics(y_test, y_test_pred, "Prueba", "nb_metrics_test.png")

    # Comparación de métricas entre validación y prueba
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)

    ax.set_title("Comparación de métricas entre Validación y Prueba (Naive Bayes)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)

    # Etiquetas encima de cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)

    save_figure(fig, "nb_comparison_valid_test.png")
    plt.close(fig)

    # Resultados de RandomizedSearch
    plot_grid_search_results(grid, "nb_random_search_results.png", "param_clf__alpha")
