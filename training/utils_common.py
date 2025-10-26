import os
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Fake", "No Fake"],
        yticklabels=["Fake", "No Fake"],
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


class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join([token.lemma_ for token in self.nlp(text)]) for text in X]
