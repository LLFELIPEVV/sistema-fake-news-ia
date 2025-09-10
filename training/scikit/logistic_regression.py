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
    """Optimiza hiperparámetros de Regresión Logística con GridSearchCV."""
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
        n_iter=25,  # número de combinaciones a probar
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=3,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)
    return random_search


def plot_confusion_matrix(y_true, y_pred, title="Matriz de confusión"):
    """Muestra matriz de confusión normalizada con Seaborn."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["No Fake", "Fake"],
        yticklabels=["No Fake", "Fake"],
    )
    plt.title(title)
    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción")
    plt.show()


def plot_metrics(y_true, y_pred, dataset_name="Validación"):
    """Grafica métricas de clasificación en barras."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1": f1_score(y_true, y_pred, average="macro"),
    }

    plt.figure(figsize=(7, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.ylim(0, 1)
    plt.title(f"Métricas en {dataset_name}")
    plt.ylabel("Valor")
    plt.show()

    return metrics


def plot_grid_search_results(grid):
    """Grafica el rendimiento de GridSearch con cada combinación de parámetros."""
    results = pd.DataFrame(grid.cv_results_)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results,
        x="param_clf__C",
        y="mean_test_score",
        hue="param_tfidf__ngram_range",
        style="param_clf__penalty",
        size="param_tfidf__max_features",
        palette="deep",
        sizes=(40, 200),
    )
    plt.xscale("log")
    plt.title("Resultados de GridSearch (rendimiento por parámetros)")
    plt.xlabel("Valor de C (log scale)")
    plt.ylabel("F1 ponderado (media CV)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


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

    # Evaluación en validación
    y_valid_pred = best_model.predict(X_valid)
    print("\n=== Reporte de Validación ===")
    print(classification_report(y_valid, y_valid_pred))
    plot_confusion_matrix(y_valid, y_valid_pred, "Matriz de confusión - Validación")
    valid_metrics = plot_metrics(y_valid, y_valid_pred, "Validación")

    # Evaluación en prueba
    y_test_pred = best_model.predict(X_test)
    print("\n=== Reporte de Prueba ===")
    print(classification_report(y_test, y_test_pred))
    plot_confusion_matrix(y_test, y_test_pred, "Matriz de confusión - Prueba")
    test_metrics = plot_metrics(y_test, y_test_pred, "Prueba")

    # Comparación de métricas entre validación y prueba
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )
    comp_df.plot(kind="bar", figsize=(8, 6), colormap="viridis")
    plt.title("Comparación de métricas entre Validación y Prueba")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.show()

    # Resultados de GridSearch
    plot_grid_search_results(grid)
