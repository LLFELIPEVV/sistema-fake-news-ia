"""
Análisis de Patrones de Detección de Noticias Falsas
==================================================
IMPORTANTE: 0 = FAKE, 1 = REAL
"""

import os
import psutil
import joblib
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from keras.models import load_model, Model

warnings.filterwarnings("ignore")

PHYSICAL_CORES = psutil.cpu_count(logical=False) or 1
AVAILABLE_RAM_GB = psutil.virtual_memory().available / (1024**3)
N_JOBS = max(1, PHYSICAL_CORES)
BATCH_SIZE = max(256, 256 * max(1, int(AVAILABLE_RAM_GB / 3)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = str(PHYSICAL_CORES)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(PHYSICAL_CORES)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["MKL_NUM_THREADS"] = str(PHYSICAL_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(PHYSICAL_CORES)

tf.config.threading.set_intra_op_parallelism_threads(PHYSICAL_CORES)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.optimizer.set_jit(True)
tf.keras.backend.set_floatx("float32")

MODELS_DIR = Path("models")
OUTPUT_DIR = Path("pattern_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SEQ_LEN = 200

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300


def save_dataframe(df, name):
    df.to_csv(OUTPUT_DIR / name, index=False)


def save_plot(fig, name):
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ===============================
# CARGA MODELOS
# ===============================
class ModelLoader:
    def load_all(self):
        models = {}
        for key, path in [
            ("naive_bayes", MODELS_DIR / "naive_bayes_best_model.pkl"),
            ("random_forest", MODELS_DIR / "random_forest_best_model.pkl"),
        ]:
            try:
                models[key] = joblib.load(path)
            except Exception:
                models[key] = None

        for key, path in [
            ("cnn", MODELS_DIR / "cnn_best_model.keras"),
            ("hybrid", MODELS_DIR / "hybrid_cnn_bilstm_gru_attention_best_model.keras"),
        ]:
            try:
                models[key] = load_model(path)
            except Exception:
                models[key] = None

        try:
            vec = load_model(MODELS_DIR / "text_vectorizer_keras.keras")
            models["vectorizer"] = vec.layers[0]
        except Exception:
            models["vectorizer"] = None

        return models


# ===============================
# NAIVE BAYES
# ===============================
class NaiveBayesAnalyzer:
    def __init__(self, pipeline, top_n=30):
        self.top_n = top_n
        self.vectorizer = (
            pipeline.named_steps.get("tfidf")
            if hasattr(pipeline, "named_steps")
            else None
        )
        self.model = (
            pipeline.named_steps.get("clf")
            if hasattr(pipeline, "named_steps")
            else None
        )

    def analyze(self):
        if self.vectorizer is None or self.model is None:
            return
        if not hasattr(self.model, "feature_log_prob_"):
            return

        tokens = np.array(self.vectorizer.get_feature_names_out())
        log_p = self.model.feature_log_prob_
        if log_p.shape[0] < 2:
            return

        log_p_fake, log_p_real = log_p[0], log_p[1]
        log_odds = log_p_fake - log_p_real

        df = pd.DataFrame(
            {
                "token": tokens,
                "log_p_fake": log_p_fake,
                "log_p_real": log_p_real,
                "log_odds": log_odds,
            }
        ).sort_values("log_odds", ascending=False)

        save_dataframe(df, "nb_full_token_scores.csv")

        for subset, name, pal in [
            (df.head(self.top_n), "nb_top_fake.png", "Reds_r"),
            (df.tail(self.top_n), "nb_top_real.png", "Blues"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(subset, x="log_odds", y="token", palette=pal, ax=ax)
            ax.axvline(0, color="black", ls="--")
            save_plot(fig, name)


# ===============================
# RANDOM FOREST
# ===============================
class RandomForestAnalyzer:
    def __init__(self, pipeline, top_n=30):
        self.top_n = top_n
        self.vectorizer = (
            pipeline.named_steps.get("tfidf")
            if hasattr(pipeline, "named_steps")
            else None
        )
        self.model = (
            pipeline.named_steps.get("clf")
            if hasattr(pipeline, "named_steps")
            else None
        )

    def analyze(self, texts):
        if self.vectorizer is None or self.model is None:
            return
        if not hasattr(self.model, "feature_importances_"):
            return

        X = self.vectorizer.transform(texts)
        features = np.array(self.vectorizer.get_feature_names_out())
        freq = np.asarray(X.mean(axis=0)).ravel()
        importance = self.model.feature_importances_
        norm_imp = importance / (freq + 1e-9)

        df = pd.DataFrame(
            {
                "token": features,
                "importance": importance,
                "mean_frequency": freq,
                "normalized_importance": norm_imp,
            }
        ).sort_values("normalized_importance", ascending=False)

        save_dataframe(df, "rf_feature_analysis.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            df.head(self.top_n),
            x="normalized_importance",
            y="token",
            palette="viridis",
            ax=ax,
        )
        save_plot(fig, "rf_feature_analysis.png")


# ===============================
# CNN – SALIENCY CORREGIDO
# ===============================
class CNNAnalyzer:
    def __init__(self, model, vectorizer, top_n=30, max_seq_len=200):
        self.model = model
        self.vectorizer = vectorizer
        self.top_n = top_n
        self.max_seq_len = max_seq_len

        vocab = vectorizer.get_vocabulary()
        self.id2token = dict(enumerate(vocab))
        self.vocab_size = len(vocab)

        try:
            self.embedding_layer = model.get_layer("embedding")
        except Exception:
            self.embedding_layer = None

        # Construir submodelo: embedding_output (float) → predicción
        self._post_emb_model = None
        if self.embedding_layer is not None:
            self._post_emb_model = self._build_post_embedding_model()

    def _build_post_embedding_model(self):
        """
        Reconstruye el grafo desde la salida del embedding hacia adelante.
        El input es ahora float32 (los vectores de embedding), lo cual
        permite calcular gradientes reales con GradientTape.
        """
        try:
            emb_out_shape = (
                self.embedding_layer.output_shape
            )  # (None, seq_len, emb_dim)
            emb_input = tf.keras.Input(
                shape=emb_out_shape[1:], dtype=tf.float32, name="emb_input"
            )

            layer_outputs = {self.embedding_layer.name: emb_input}

            for layer in self.model.layers:
                # Saltar la InputLayer original y la capa embedding ya procesada
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if layer.name == self.embedding_layer.name:
                    continue

                try:
                    inbound_nodes = layer._inbound_nodes
                    if not inbound_nodes:
                        continue

                    node = inbound_nodes[0]
                    input_tensors = node.input_tensors
                    if not isinstance(input_tensors, (list, tuple)):
                        input_tensors = [input_tensors]

                    resolved = []
                    all_ok = True
                    for t in input_tensors:
                        src_name = t._keras_history.layer.name
                        if src_name in layer_outputs:
                            resolved.append(layer_outputs[src_name])
                        else:
                            all_ok = False
                            break

                    if not all_ok:
                        continue

                    out = layer(resolved[0]) if len(resolved) == 1 else layer(resolved)
                    layer_outputs[layer.name] = out

                except Exception:
                    continue

            # Buscar la capa de salida (última capa densa o la que produce la predicción)
            output_layer = self.model.layers[-1]
            if output_layer.name in layer_outputs:
                return tf.keras.Model(
                    inputs=emb_input,
                    outputs=layer_outputs[output_layer.name],
                    name="post_emb_model",
                )
            return None

        except Exception as e:
            print(f"[CNNAnalyzer] No se pudo construir post_emb_model: {e}")
            return None

    def analyze(self, texts, labels):
        if self.embedding_layer is None:
            print("[CNNAnalyzer] Capa 'embedding' no encontrada. Saltando.")
            return

        # Vectorizar y filtrar FAKE
        seqs = self.vectorizer(texts)
        seqs = seqs[:, : self.max_seq_len]
        labels_np = np.array(labels)
        fake_mask = labels_np == 0
        fake_seqs = tf.boolean_mask(seqs, fake_mask)

        if self._post_emb_model is not None:
            print(
                "🧠 CNN – saliency via gradiente sobre embeddings (modo diferenciable)"
            )
            self._saliency_gradient(fake_seqs)
        else:
            print("⚠️  CNN – saliency via confianza ponderada (fallback)")
            self._saliency_confidence(fake_seqs)

    def _saliency_gradient(self, fake_seqs):
        """
        Calcula saliency real: gradiente del score FAKE respecto a los embeddings.
        Usa embedding lookup manual con tf.Variable para que GradientTape pueda watchear.
        """
        sal_sum = np.zeros(self.vocab_size, dtype=np.float64)
        sal_cnt = np.zeros(self.vocab_size, dtype=np.float64)

        # Pesos del embedding como constante float32
        emb_weights = tf.cast(
            self.embedding_layer.embeddings, tf.float32
        )  # (vocab, emb_dim)

        dataset = (
            tf.data.Dataset.from_tensor_slices(fake_seqs)
            .batch(
                min(BATCH_SIZE, 128)
            )  # batch pequeño para no saturar memoria con Variables
            .prefetch(tf.data.AUTOTUNE)
        )

        for batch in tqdm(dataset, desc="CNN saliency (grad)"):
            try:
                token_ids = tf.cast(batch, tf.int32)  # (B, seq_len)

                # Lookup manual: construir tensor de embeddings como Variable watcheable
                emb_init = tf.gather(emb_weights, token_ids)  # (B, seq_len, emb_dim)
                emb_var = tf.Variable(emb_init, trainable=True, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    # preds: (B, 1) con P(clase=1=REAL)
                    preds = self._post_emb_model(emb_var, training=False)
                    if len(preds.shape) > 1:
                        preds = preds[:, 0]
                    # Score FAKE = 1 - P(REAL)
                    # Usamos la suma para agregar gradientes sobre todo el batch
                    fake_score = tf.reduce_sum(1.0 - preds)

                grads = tape.gradient(fake_score, emb_var)
                # grads: (B, seq_len, emb_dim)

                if grads is None:
                    continue

                # Saliency por posición: norma L2 del gradiente
                token_saliency = tf.norm(
                    tf.cast(grads, tf.float32), axis=2
                ).numpy()  # (B, seq_len)
                ids_np = token_ids.numpy()  # (B, seq_len)

                # Acumular
                ids_flat = ids_np.ravel()
                sal_flat = token_saliency.ravel()
                valid = ids_flat != 0  # ignorar padding
                np.add.at(sal_sum, ids_flat[valid], sal_flat[valid])
                np.add.at(sal_cnt, ids_flat[valid], 1.0)

            except Exception:
                continue

        self._save_results(sal_sum, sal_cnt, "grad")

    def _saliency_confidence(self, fake_seqs):
        """
        Fallback: pondera la frecuencia de cada token por la confianza FAKE del modelo.
        No requiere diferenciación, funciona con cualquier arquitectura.
        """
        sal_sum = np.zeros(self.vocab_size, dtype=np.float64)
        sal_cnt = np.zeros(self.vocab_size, dtype=np.float64)

        dataset = (
            tf.data.Dataset.from_tensor_slices(fake_seqs)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        for batch in tqdm(dataset, desc="CNN saliency (conf)"):
            try:
                preds = self.model(batch, training=False)
                if len(preds.shape) > 1:
                    preds = preds[:, 0]
                # P(FAKE) = 1 - P(REAL)
                conf_fake = (1.0 - preds).numpy()  # (B,)

                ids_np = batch.numpy()  # (B, seq_len)

                for i, (seq, conf) in enumerate(zip(ids_np, conf_fake)):
                    valid = seq != 0
                    np.add.at(sal_sum, seq[valid], conf)
                    np.add.at(sal_cnt, seq[valid], 1.0)

            except Exception:
                continue

        self._save_results(sal_sum, sal_cnt, "conf")

    def _save_results(self, sal_sum, sal_cnt, mode):
        sal_avg = np.where(sal_cnt > 0, sal_sum / sal_cnt, 0.0)

        # Solo tokens que aparecieron al menos una vez
        appeared = sal_cnt > 0
        sal_filtered = np.where(appeared, sal_avg, -1.0)
        top_idx = np.argsort(sal_filtered)[::-1][: self.top_n]

        df = pd.DataFrame(
            {
                "token": [self.id2token.get(int(i), "") for i in top_idx],
                "saliency_fake": sal_avg[top_idx],
                "apariciones": sal_cnt[top_idx].astype(int),
                "modo": mode,
            }
        )
        df = df[df["saliency_fake"] > 0].reset_index(drop=True)

        save_dataframe(df, "cnn_saliency_fake.csv")
        print(f"✅ CNN saliency guardado: {len(df)} tokens (modo={mode})")
        print(df.head(10).to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(df.head(30), x="saliency_fake", y="token", palette="magma", ax=ax)
        ax.set_title(f"CNN – Top 30 tokens saliency FAKE (modo={mode})")
        save_plot(fig, "cnn_saliency_fake.png")


# ===============================
# HÍBRIDO
# ===============================
class HybridAnalyzer:
    def __init__(self, model, vectorizer, top_n=30):
        self.top_n = top_n
        self.model = model
        self.vectorizer = vectorizer

        if vectorizer:
            vocab = vectorizer.get_vocabulary()
            self.id2token = dict(enumerate(vocab))
        else:
            self.id2token = {}

        try:
            att_layer = next(i for i in model.layers if "attention" in i.name.lower())
            self.att_model = Model(model.input, att_layer.output)
        except Exception:
            self.att_model = None

    def analyze(self, texts, labels):
        if self.att_model is None or self.vectorizer is None:
            return

        seqs = self.vectorizer(texts)
        labels = tf.convert_to_tensor(labels)
        vocab_size = len(self.id2token)

        contrib_fake = np.zeros(vocab_size, dtype=np.float64)
        contrib_real = np.zeros(vocab_size, dtype=np.float64)

        dataset = (
            tf.data.Dataset.from_tensor_slices((seqs, labels))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        for batch, y in tqdm(dataset, desc="Hybrid attention"):
            try:
                att = tf.norm(self.att_model(batch, training=False), axis=2)
                tokens = tf.reshape(batch, [-1])
                values = tf.reshape(att, [-1])
                labs = tf.repeat(y, tf.shape(batch)[1])
                valid = tf.not_equal(tokens, 0)
                tokens = tf.boolean_mask(tokens, valid)
                values = tf.boolean_mask(values, valid)
                labs = tf.boolean_mask(labs, valid)

                contrib_fake += tf.math.unsorted_segment_sum(
                    tf.boolean_mask(values, labs == 0),
                    tf.boolean_mask(tokens, labs == 0),
                    vocab_size,
                ).numpy()
                contrib_real += tf.math.unsorted_segment_sum(
                    tf.boolean_mask(values, labs == 1),
                    tf.boolean_mask(tokens, labs == 1),
                    vocab_size,
                ).numpy()
            except Exception:
                continue

        delta = contrib_fake - contrib_real
        idx = np.argsort(np.abs(delta))[::-1][: self.top_n]

        df = pd.DataFrame(
            {
                "token": [self.id2token.get(int(i), "") for i in idx],
                "fake_contribution": contrib_fake[idx],
                "real_contribution": contrib_real[idx],
                "delta_contribution": delta[idx],
            }
        )

        save_dataframe(df, "hybrid_attention_contribution.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(df, x="delta_contribution", y="token", palette="rocket", ax=ax)
        ax.axvline(0, color="black", ls="--")
        save_plot(fig, "hybrid_attention_contribution.png")


# ===============================
# EXTRACTOR DE PATRONES CSV MAESTRO
# ===============================
class PatternExtractor:
    def __init__(self, texts, labels, models):
        self.texts = np.array(texts)
        self.labels = np.array(labels)
        self.models = models
        self.fake_texts = self.texts[self.labels == 0]
        self.real_texts = self.texts[self.labels == 1]

    def extract_all_patterns(self):
        patterns = []
        if self.models.get("naive_bayes"):
            patterns.extend(self._from_nb())
        if self.models.get("random_forest"):
            patterns.extend(self._from_rf())
        if self.models.get("cnn") and self.models.get("vectorizer"):
            patterns.extend(self._from_cnn_csv())
        patterns.extend(self._from_text())
        patterns.extend(self._from_structure())
        patterns.extend(self._from_emotion())

        df = pd.DataFrame(patterns)
        save_dataframe(df, "fake_news_pattern_schema.csv")
        print(f"✅ fake_news_pattern_schema.csv: {len(df)} filas")
        return df

    def _row(
        self,
        fuente,
        tipo,
        desc,
        kw,
        peso,
        longitud="",
        titular="",
        emocional="",
        puntuacion="",
        narrativa="",
    ):
        return {
            "modelo_fuente": fuente,
            "tipo_patron": tipo,
            "descripcion_patron": desc,
            "palabras_clave": kw,
            "peso_o_confianza": peso,
            "longitud_tipica_texto": longitud,
            "tipo_titular": titular,
            "intensidad_emocional": emocional,
            "uso_puntuacion": puntuacion,
            "estructura_narrativa": narrativa,
            "etiqueta": "FAKE",
        }

    def _from_nb(self):
        rows = []
        try:
            nb = self.models["naive_bayes"]
            vec = nb.named_steps.get("tfidf")
            clf = nb.named_steps.get("clf")
            if vec is None or clf is None or not hasattr(clf, "feature_log_prob_"):
                return rows
            tokens = np.array(vec.get_feature_names_out())
            lp = clf.feature_log_prob_
            if lp.shape[0] < 2:
                return rows
            log_odds = lp[0] - lp[1]
            for idx in np.argsort(log_odds)[::-1][:100]:
                rows.append(
                    self._row(
                        "naive_bayes",
                        "lexico",
                        "palabra_alta_prob_fake",
                        tokens[idx],
                        float(log_odds[idx]),
                    )
                )
        except Exception:
            pass
        return rows

    def _from_rf(self):
        rows = []
        try:
            rf = self.models["random_forest"]
            vec = rf.named_steps.get("tfidf")
            clf = rf.named_steps.get("clf")
            if vec is None or clf is None or not hasattr(clf, "feature_importances_"):
                return rows
            tokens = np.array(vec.get_feature_names_out())
            X = vec.transform(self.texts)
            freq = np.asarray(X.mean(axis=0)).ravel()
            norm_imp = clf.feature_importances_ / (freq + 1e-9)
            for idx in np.argsort(norm_imp)[::-1][:100]:
                rows.append(
                    self._row(
                        "random_forest",
                        "lexico",
                        "feature_importance_alto",
                        tokens[idx],
                        float(norm_imp[idx]),
                    )
                )
        except Exception:
            pass
        return rows

    def _from_cnn_csv(self):
        """Lee el CSV generado por CNNAnalyzer para unificarlo en el maestro."""
        rows = []
        try:
            cnn_csv = OUTPUT_DIR / "cnn_saliency_fake.csv"
            if not cnn_csv.exists():
                return rows
            df = pd.read_csv(cnn_csv)
            for _, r in df.iterrows():
                if r.get("saliency_fake", 0) > 0:
                    rows.append(
                        self._row(
                            "cnn",
                            "lexico",
                            "saliency_gradiente_alto",
                            r["token"],
                            float(r["saliency_fake"]),
                        )
                    )
        except Exception:
            pass
        return rows

    def _from_text(self):
        rows = []
        fake_w = Counter()
        real_w = Counter()
        for t in self.fake_texts[:5000]:
            fake_w.update(t.lower().split())
        for t in self.real_texts[:5000]:
            real_w.update(t.lower().split())
        tf_total = sum(fake_w.values()) or 1
        tr_total = sum(real_w.values()) or 1

        for word, fc in fake_w.most_common(200):
            if len(word) <= 2 or fc < 10:
                continue
            ratio = (fc / tf_total) / (real_w.get(word, 0) / tr_total + 1e-9)
            if ratio > 1.5:
                rows.append(
                    self._row(
                        "text_analysis",
                        "lexico",
                        "palabra_distintiva_fake",
                        word,
                        float(ratio),
                    )
                )

        bgrams = Counter()
        for t in self.fake_texts[:2000]:
            w = t.lower().split()
            for i in range(len(w) - 1):
                bgrams[f"{w[i]} {w[i + 1]}"] += 1
        for bg, cnt in bgrams.most_common(50):
            if cnt >= 5:
                rows.append(
                    self._row(
                        "text_analysis",
                        "lexico",
                        "bigrama_frecuente_fake",
                        bg,
                        float(cnt),
                    )
                )
        return rows

    def _from_structure(self):
        rows = []
        lengths = np.array([len(t.split()) for t in self.fake_texts])
        p25, p75 = int(np.percentile(lengths, 25)), int(np.percentile(lengths, 75))
        rng = f"{p25}-{p75}"

        rows.append(
            self._row(
                "structural_analysis",
                "estadistico",
                "longitud_texto_promedio",
                "",
                float(np.mean(lengths)),
                longitud=rng,
            )
        )
        rows.append(
            self._row(
                "structural_analysis",
                "estadistico",
                "longitud_texto_mediana",
                "",
                float(np.median(lengths)),
                longitud=rng,
            )
        )

        exc = np.array([t.count("!") for t in self.fake_texts[:2000]])
        que = np.array([t.count("?") for t in self.fake_texts[:2000]])
        mayus = np.array(
            [
                sum(c.isupper() for c in t) / max(len(t), 1) * 100
                for t in self.fake_texts[:2000]
            ]
        )

        rows.append(
            self._row(
                "structural_analysis",
                "estadistico",
                "uso_exclamaciones",
                "",
                float(np.mean(exc)),
                puntuacion=f"!={np.mean(exc):.2f}",
            )
        )
        rows.append(
            self._row(
                "structural_analysis",
                "estadistico",
                "uso_interrogaciones",
                "",
                float(np.mean(que)),
                puntuacion=f"?={np.mean(que):.2f}",
            )
        )
        rows.append(
            self._row(
                "structural_analysis",
                "estadistico",
                "porcentaje_mayusculas",
                "",
                float(np.mean(mayus)),
                puntuacion=f"MAYUS={np.mean(mayus):.2f}%",
            )
        )

        begins = Counter(
            " ".join(t.split()[:5])
            for t in self.fake_texts[:1000]
            if len(t.split()) >= 5
        )
        for beg, cnt in begins.most_common(10):
            rows.append(
                self._row(
                    "structural_analysis",
                    "estructural",
                    "patron_inicio_frecuente",
                    beg,
                    float(cnt),
                    narrativa="inicio",
                )
            )
        return rows

    def _from_emotion(self):
        high = [
            "urgente",
            "alerta",
            "peligro",
            "grave",
            "terrible",
            "escandaloso",
            "impactante",
            "increíble",
            "devastador",
        ]
        med = ["importante", "serio", "preocupante", "crítico", "significativo"]
        sample = self.fake_texts[:2000]
        n = len(sample) or 1
        hc = sum(sum(t.lower().count(w) for w in high) for t in sample)
        mc = sum(sum(t.lower().count(w) for w in med) for t in sample)
        return [
            self._row(
                "emotional_analysis",
                "emocional",
                "intensidad_emocional_alta",
                ", ".join(high),
                float(hc / n),
                titular="alarmista-sensacionalista",
                emocional="alta",
            ),
            self._row(
                "emotional_analysis",
                "emocional",
                "intensidad_emocional_media",
                ", ".join(med),
                float(mc / n),
                titular="clickbait",
                emocional="media",
            ),
        ]


# ===============================
# MAIN
# ===============================
def main():
    from training.utils_common import load_datasets

    models = ModelLoader().load_all()
    train, val, test = load_datasets()
    df = pd.concat([train, val, test])

    texts = df["texto"].astype(str).values
    labels = df["clase"].values

    print(
        f"🚀 Análisis iniciado — {len(texts)} textos "
        f"({(labels == 0).sum()} FAKE / {(labels == 1).sum()} REAL)"
    )

    if models["naive_bayes"]:
        NaiveBayesAnalyzer(models["naive_bayes"]).analyze()

    if models["random_forest"]:
        RandomForestAnalyzer(models["random_forest"]).analyze(texts)

    if models["cnn"] and models["vectorizer"]:
        CNNAnalyzer(
            models["cnn"], models["vectorizer"], max_seq_len=MAX_SEQ_LEN
        ).analyze(texts, labels)

    if models["hybrid"] and models["vectorizer"]:
        HybridAnalyzer(models["hybrid"], models["vectorizer"]).analyze(texts, labels)

    PatternExtractor(texts, labels, models).extract_all_patterns()

    print("✅ Análisis completo")


if __name__ == "__main__":
    main()
