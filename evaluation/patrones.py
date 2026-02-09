"""
Análisis de Patrones de Detección de Noticias Falsas
==================================================
IMPORTANTE: 0 = FAKE, 1 = REAL
"""

import os
import warnings
from tqdm import tqdm
from pathlib import Path
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import load_model, Model

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

N_JOBS = cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(N_JOBS)
tf.config.threading.set_inter_op_parallelism_threads(N_JOBS)

MODELS_DIR = Path("models")
OUTPUT_DIR = Path("pattern_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 300

BATCH_SIZE = 64


# ===============================
# UTILIDADES
# ===============================
def save_plot(fig, name):
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def save_dataframe(df, name):
    df.to_csv(OUTPUT_DIR / name, index=False)


# ===============================
# CARGA DE MODELOS
# ===============================
class ModelLoader:
    def __init__(self):
        self.models = {}

    def load_all(self):
        self.models["naive_bayes"] = joblib.load(
            MODELS_DIR / "naive_bayes_best_model.pkl"
        )
        self.models["random_forest"] = joblib.load(
            MODELS_DIR / "random_forest_best_model.pkl"
        )
        self.models["cnn"] = load_model(MODELS_DIR / "cnn_best_model.keras")
        self.models["hybrid"] = load_model(
            MODELS_DIR / "hybrid_cnn_bilstm_gru_attention_best_model.keras"
        )
        vec_model = load_model(MODELS_DIR / "text_vectorizer_keras.keras")
        self.models["vectorizer"] = vec_model.layers[0]
        return self.models


# ===============================
# NAIVE BAYES
# ===============================
class NaiveBayesAnalyzer:
    def __init__(self, pipeline, top_n=20):
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.model = pipeline.named_steps["clf"]
        self.top_n = top_n

    def analyze(self):
        features = np.array(self.vectorizer.get_feature_names_out())
        fake, real = self.model.feature_log_prob_

        odds = fake - real

        top_fake = np.argsort(odds)[-self.top_n :][::-1]
        top_real = np.argsort(odds)[: self.top_n]

        df_fake = pd.DataFrame(
            {"token": features[top_fake], "log_odds": odds[top_fake]}
        )
        df_real = pd.DataFrame(
            {"token": features[top_real], "log_odds": odds[top_real]}
        )

        save_dataframe(df_fake, "nb_fake_odds.csv")
        save_dataframe(df_real, "nb_real_odds.csv")

        for df, name, pal in [
            (df_fake, "nb_fake.png", "Reds_r"),
            (df_real, "nb_real.png", "Blues"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=df.sort_values("log_odds"),
                x="log_odds",
                y="token",
                palette=pal,
                ax=ax,
            )
            ax.axvline(0, color="black", ls="--")
            save_plot(fig, name)


# ===============================
# RANDOM FOREST
# ===============================
class RandomForestAnalyzer:
    def __init__(self, pipeline, top_n=20):
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.model = pipeline.named_steps["clf"]
        self.top_n = top_n

    def analyze(self, texts, labels):
        features = np.array(self.vectorizer.get_feature_names_out())
        X = self.vectorizer.transform(texts)

        norm_importance = self.model.feature_importances_ / (X.mean(axis=0).A1 + 1e-9)
        idx = np.argsort(norm_importance)[-self.top_n :][::-1]

        df = pd.DataFrame(
            {
                "token": features[idx],
                "importance": norm_importance[idx],
            }
        )

        save_dataframe(df, "rf_importance.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df.sort_values("importance"),
            x="importance",
            y="token",
            palette="viridis",
            ax=ax,
        )
        save_plot(fig, "rf_importance.png")


# ===============================
# CNN — SALIENCY REAL
# ===============================
class CNNAnalyzer:
    def __init__(self, model, vectorizer, top_n=20):
        self.model = model
        self.vectorizer = vectorizer
        self.top_n = top_n

        emb_layer = model.get_layer("embedding")
        self.emb_model = Model(model.input, emb_layer.output)

        vocab = vectorizer.get_vocabulary()
        self.id2token = dict(enumerate(vocab))

    def analyze(self, texts, labels):
        print("🧠 Analizando CNN (Activación de embeddings)...")

        activation_sum = np.zeros(len(self.id2token))
        activation_count = np.zeros(len(self.id2token))

        for text, lbl in tqdm(
            zip(texts, labels), total=len(texts), desc="CNN saliency (FAKE)", unit="doc"
        ):
            if lbl != 0:  # SOLO FAKE
                continue

            seq = self.vectorizer([text])
            emb = self.emb_model(seq)[0].numpy()  # (seq_len, emb_dim)

            norms = np.linalg.norm(emb, axis=1)
            tokens = seq.numpy()[0]

            for t, n in zip(tokens, norms):
                if t != 0:
                    activation_sum[t] += n
                    activation_count[t] += 1

        avg_activation = np.divide(
            activation_sum,
            activation_count,
            out=np.zeros_like(activation_sum),
            where=activation_count >= 5,
        )

        idx = np.argsort(avg_activation)[-self.top_n :][::-1]

        df = pd.DataFrame(
            {
                "token": [self.id2token[i] for i in idx],
                "avg_embedding_activation": avg_activation[idx],
            }
        )

        save_dataframe(df, "cnn_embedding_activation_fake.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df.sort_values("avg_embedding_activation"),
            x="avg_embedding_activation",
            y="token",
            palette="magma",
            ax=ax,
        )
        ax.set_title("CNN – Tokens más activados en noticias FAKE")
        save_plot(fig, "cnn_embedding_activation.png")


# ===============================
# HÍBRIDO — ATENCIÓN LIMPIA
# ===============================
class HybridAnalyzer:
    def __init__(self, model, vectorizer, top_n=20):
        self.model = model
        self.vectorizer = vectorizer
        self.top_n = top_n

        vocab = vectorizer.get_vocabulary()
        self.id2token = dict(enumerate(vocab))

        att_layer = next(i for i in model.layers if "attention" in i.name.lower())
        self.att_model = Model(model.input, att_layer.output)

    def analyze(self, texts, labels):
        sum_f = np.zeros(len(self.id2token))
        sum_r = np.zeros(len(self.id2token))
        cnt_f = np.zeros(len(self.id2token))
        cnt_r = np.zeros(len(self.id2token))

        seqs = self.vectorizer(texts).numpy()

        for seq, lbl in tqdm(
            zip(seqs, labels), total=len(seqs), desc="Hybrid attention", unit="doc"
        ):
            att = np.linalg.norm(self.att_model(seq[None, :])[0], axis=1)
            tokens = seq[seq != 0]
            att = att[: len(tokens)]

            if lbl == 0:
                np.add.at(sum_f, tokens, att)
                np.add.at(cnt_f, tokens, 1)
            else:
                np.add.at(sum_r, tokens, att)
                np.add.at(cnt_r, tokens, 1)

        avg_f = sum_f / np.maximum(cnt_f, 1)
        avg_r = sum_r / np.maximum(cnt_r, 1)
        delta = avg_f - avg_r

        idx = np.argsort(np.abs(delta))[-self.top_n :][::-1]

        df = pd.DataFrame(
            {
                "token": [self.id2token[i] for i in idx],
                "delta_attention": delta[idx],
            }
        )

        save_dataframe(df, "hybrid_attention.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df.sort_values("delta_attention"),
            x="delta_attention",
            y="token",
            palette="rocket",
            ax=ax,
        )
        ax.axvline(0, color="black", ls="--")
        save_plot(fig, "hybrid_attention.png")


# ===============================
# MAIN
# ===============================
def main():
    from training.utils_common import load_datasets

    loader = ModelLoader()
    models = loader.load_all()

    train, val, test = load_datasets()
    df = pd.concat([train, val, test])

    texts = df["texto"].astype(str).values
    labels = df["clase"].values

    print("🚀 Iniciando análisis de patrones...\n")

    print("🔹 Naive Bayes")
    NaiveBayesAnalyzer(models["naive_bayes"]).analyze()

    print("🔹 Random Forest")
    RandomForestAnalyzer(models["random_forest"]).analyze(texts, labels)

    print("🔹 CNN")
    CNNAnalyzer(models["cnn"], models["vectorizer"]).analyze(texts, labels)

    print("🔹 Modelo Híbrido")
    HybridAnalyzer(models["hybrid"], models["vectorizer"]).analyze(texts, labels)

    print("\n✅ Análisis completado")


if __name__ == "__main__":
    main()
