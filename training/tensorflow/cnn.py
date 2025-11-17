import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    SpatialDropout1D,
    Conv1D,
    GlobalMaxPooling1D,
    Dropout,
    BatchNormalization,
    Input,
    Concatenate,
    LayerNormalization,
    LeakyReLU,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from training.tensorflow.utils_keras import (
    save_best_model_keras,
    plot_training_history,
    load_best_model_keras,
    build_embedding_matrix,
    save_results,
    get_random_hyperparams,
)
from training.utils_common import (
    load_datasets,
    plot_confusion_matrix,
    plot_metrics,
    MODEL_DIR,
    save_figure,
    Lemmatizer,
)

# Configuración de hardware
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[INFO] Se detectó GPU: {gpus}")
    device = "GPU"
    # Configurar crecimiento dinámico de memoria GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[WARNING] No se pudo configurar crecimiento de memoria GPU: {e}")
else:
    print("[INFO] No se detectó GPU, usando CPU")
    device = "CPU"

num_threads = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Tamaño de batch dinámico según hardware
if device == "GPU":
    BATCH_SIZE = 128 if num_threads >= 8 else 64
else:
    BATCH_SIZE = 64 if num_threads > 4 else 32

print(f"[INFO] Usando batch_size = {BATCH_SIZE} con {device} y {num_threads} hilos")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Configuración general
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_best_model.keras")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "cnn_best_score.txt")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer_keras.keras")
HISTORY_FILE = os.path.join(MODEL_DIR, "cnn_hyperparam_history.json")


# ==============================
# 🔤 Text Vectorization
# ==============================
def prepare_vectorizer(texts, max_tokens=30000, output_seq_len=200):
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(texts)
    return vectorizer


# ==============================
# 🏗️ Modelo CNN
# ==============================
def build_cnn_model(
    vocab_size,
    embedding_dim=300,
    sequence_length=200,
    embedding_matrix=None,
    num_filters=128,
    kernel_sizes=(3, 4, 5),
    dropout_rate=0.4 if device == "GPU" else 0.3,  # ✅ Dropout adaptativo
    l2_reg=1e-4,
):
    # Input layer
    input_layer = Input(shape=(sequence_length,))

    # Embedding
    if embedding_matrix is not None:
        x = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
            name="embedding",
        )(input_layer)
    else:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding")(
            input_layer
        )

    x = SpatialDropout1D(dropout_rate)(x)

    # ✅ CONVOLUCIONES PARALELAS (no secuenciales)
    conv_blocks = []
    for ks in kernel_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=ks,
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )(x)
        conv = LeakyReLU(alpha=0.1)(conv)
        conv = BatchNormalization()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    # Concatenar todas las convoluciones
    x = Concatenate()(conv_blocks)
    x = LayerNormalization()(x)

    # Dense layers
    x = Dense(128, kernel_regularizer=l2(l2_reg))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, kernel_regularizer=l2(l2_reg))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate * 0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def to_tf_dataset(texts, labels, vectorizer, batch_size=64, shuffle=True):
    """Convierte arrays en tf.data.Dataset optimizado con vectorización."""
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(texts), 10000), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(y):
    """Calcula pesos de clase para datos desbalanceados."""
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights_dict = {int(c): w for c, w in zip(classes, cw)}
    print(f"[INFO] Distribución de clases: {np.bincount(y)}")
    print(f"[INFO] Pesos de clase: {class_weights_dict}")
    return class_weights_dict


def evaluate_model(model, dataset, y_true, dataset_name="Dataset"):
    """Evalúa el modelo y muestra métricas."""
    print(f"\n=== Evaluación sobre {dataset_name} ===")
    y_proba = model.predict(dataset, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, digits=4))
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    return y_pred, y_proba, f1_macro


# ==============================
# 🚀 Main
# ==============================
if __name__ == "__main__":
    print("[INFO] Cargando datos...")
    train_df, valid_df, test_df = load_datasets()

    X_train, y_train = (
        train_df["texto"].astype(str).values,
        train_df["clase"].astype(int).values,
    )
    X_valid, y_valid = (
        valid_df["texto"].astype(str).values,
        valid_df["clase"].astype(int).values,
    )
    X_test, y_test = (
        test_df["texto"].astype(str).values,
        test_df["clase"].astype(int).values,
    )

    print(
        f"[INFO] Tamaños - Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}"
    )

    # Lematización Opcional, en algunos casos empeora el rendimiento.
    lemmatizer = Lemmatizer()
    X_train, X_valid, X_test = map(lemmatizer.transform, [X_train, X_valid, X_test])

    print("[INFO] Preparando vectorizador TextVectorization...")
    vectorizer = prepare_vectorizer(X_train)
    vocab_size = len(vectorizer.get_vocabulary())

    print(f"[INFO] Vocab size real: {vocab_size}")
    embedding_matrix = build_embedding_matrix(vectorizer)

    train_ds = (
        to_tf_dataset(X_train, y_train, vectorizer, BATCH_SIZE, True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = (
        to_tf_dataset(X_valid, y_valid, vectorizer, BATCH_SIZE, False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        to_tf_dataset(X_test, y_test, vectorizer, BATCH_SIZE, False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    # === Selección de hiperparámetros ===
    all_combinations = {
        "num_filters": [64, 96, 128, 160],
        "kernel_sizes": [(3, 4, 5), (2, 3, 4), (3, 5, 7)],
        "dropout_rate": [0.3, 0.4, 0.5],
        "l2_reg": [1e-3, 1e-4, 1e-5],
        "embedding_dim": [300],
    }
    params = get_random_hyperparams(all_combinations, HISTORY_FILE, max_attempts=50)
    print(f"[INFO] Hiperparámetros seleccionados: {params}")

    print("[INFO] Construyendo modelo CNN...")
    model = build_cnn_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        num_filters=params["num_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout_rate=params["dropout_rate"],
        l2_reg=params["l2_reg"],
        embedding_dim=params["embedding_dim"],
    )

    sample_batch = next(iter(train_ds.take(1)))
    model(sample_batch[0])
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    class_weights = compute_class_weights(y_train)

    print("[INFO] Entrenando modelo CNN (fase 1: embeddings congelados)...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # ✅ Fine-tuning: descongelar embeddings
    print("[INFO] Iniciando fine-tuning de embeddings...")
    for layer in model.layers:
        if "embedding" in layer.name:
            layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    history_ft = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=5,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    plot_training_history(
        history,
        filename="cnn_training_history.png",
        metrics=("accuracy", "loss", "precision", "recall"),
    )

    y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        model, valid_ds, y_valid, "Validación"
    )
    y_test_pred, y_test_proba, f1_test = evaluate_model(
        model, test_ds, y_test, "Prueba"
    )

    save_best_model_keras(model, f1_valid, BEST_SCORE_PATH, BEST_MODEL_PATH)
    # ==============================
    # 💾 Guardar vectorizador
    # ==============================
    print("[INFO] Guardando vectorizador TextVectorization...")

    # 1. Empaquetar el vectorizador en un modelo funcional
    vectorizer_model = Sequential([vectorizer])

    # 2. Guardar con formato TensorFlow SavedModel
    vectorizer_model.save(VECTORIZER_PATH)

    print(f"✅ Vectorizador guardado en: {VECTORIZER_PATH}")
    best_model = load_best_model_keras(BEST_MODEL_PATH)
    y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        best_model, valid_ds, y_valid, "Validación"
    )
    y_test_pred, y_test_proba, f1_test = evaluate_model(
        best_model, test_ds, y_test, "Prueba"
    )

    print("[INFO] Generando visualizaciones...")
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        title="Matriz de confusión - Validación (CNN)",
        filename="cnn_confusion_valid.png",
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        title="Matriz de confusión - Prueba (CNN)",
        filename="cnn_confusion_test.png",
    )

    plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (CNN)",
        filename="cnn_metrics_valid.png",
    )
    plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (CNN)",
        filename="cnn_metrics_test.png",
    )

    valid_metrics = plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (CNN)",
        filename="cnn_metrics_valid.png",
    )
    test_metrics = plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (CNN)",
        filename="cnn_metrics_test.png",
    )

    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)
    ax.set_title("Comparación de métricas entre Validación y Prueba (CNN)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)
    save_figure(fig, "cnn_comparison_valid_test.png")
    plt.close(fig)

    print("[INFO] Proceso terminado.")
    print(f"[INFO] F1 Score Final - Validación: {f1_valid:.4f}, Prueba: {f1_test:.4f}")
    metrics = {
        "f1_valid": float(f1_valid),
        "f1_test": float(f1_test),
        "device": device,
        "batch_size": BATCH_SIZE,
    }
    save_results(params, metrics, HISTORY_FILE)
