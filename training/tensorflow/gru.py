import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    SpatialDropout1D,
    GRU,
    Dropout,
    BatchNormalization,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from training.tensorflow.utils_keras import (
    save_best_model_keras,
    plot_training_history,
    load_best_model_keras,
)
from training.utils_common import (
    load_datasets,
    plot_confusion_matrix,
    plot_metrics,
    MODEL_DIR,
    save_figure,
)


# Configuración de hardware
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[INFO] Se detectó GPU: {gpus}")
    device = "GPU"
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

# Tamaño de batch dinámico
if device == "GPU":
    BATCH_SIZE = 128 if num_threads >= 8 else 64
else:
    BATCH_SIZE = 32 if num_threads <= 4 else 64

print(f"[INFO] Usando batch_size = {BATCH_SIZE} con {device} y {num_threads} hilos")


# Configuración general
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "gru_best_model.keras")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "gru_best_score.txt")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


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


def build_gru_model_sequential(
    vocab_size,
    embedding_dim=128,
    sequence_length=200,
    gru_units=128,
    dropout_rate=0.4,
):
    """Modelo Sequential con Embedding + GRU + Dense."""
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=sequence_length,
                mask_zero=True,
                name="embedding",
            ),
            SpatialDropout1D(0.4, name="spatial_dropout"),
            GRU(
                gru_units,
                dropout=dropout_rate,
                recurrent_dropout=0.4,
                return_sequences=False,
                name="gru_main",
            ),
            BatchNormalization(name="batch_norm"),
            Dense(64, activation="relu", name="dense_hidden"),
            Dropout(0.5, name="dropout_hidden"),
            Dense(1, activation="sigmoid", name="output"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def to_tf_dataset(texts, labels, vectorizer, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(texts), 10000), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(y):
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights_dict = {int(c): w for c, w in zip(classes, cw)}
    print(f"[INFO] Distribución de clases: {np.bincount(y)}")
    print(f"[INFO] Class weights: {class_weights_dict}")
    return class_weights_dict


def evaluate_model(model, dataset, y_true, dataset_name="Dataset"):
    print(f"\n=== Evaluación sobre {dataset_name} ===")
    y_proba = model.predict(dataset, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, digits=4))
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    return y_pred, y_proba, f1_macro


if __name__ == "__main__":
    # Cargar datos
    print("[INFO] Cargando datos...")
    train_df, valid_df, test_df = load_datasets()

    X_train_texts = train_df["texto"].astype(str).values
    y_train = train_df["clase"].astype(int).values
    X_valid_texts = valid_df["texto"].astype(str).values
    y_valid = valid_df["clase"].astype(int).values
    X_test_texts = test_df["texto"].astype(str).values
    y_test = test_df["clase"].astype(int).values

    print(
        f"[INFO] Tamaños - Train: {len(X_train_texts)}, Valid: {len(X_valid_texts)}, Test: {len(X_test_texts)}"
    )

    # Vectorización
    print("[INFO] Preparando vectorizador...")
    MAX_TOKENS, SEQ_LEN = 30000, 200
    vectorizer = prepare_vectorizer(
        X_train_texts, max_tokens=MAX_TOKENS, output_seq_len=SEQ_LEN
    )
    vocab_size = len(vectorizer.get_vocabulary())
    print(f"[INFO] Vocab size real: {vocab_size}")

    # Datasets
    train_ds = to_tf_dataset(
        X_train_texts, y_train, vectorizer, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_ds = to_tf_dataset(
        X_valid_texts, y_valid, vectorizer, batch_size=BATCH_SIZE, shuffle=False
    )
    test_ds = to_tf_dataset(
        X_test_texts, y_test, vectorizer, batch_size=BATCH_SIZE, shuffle=False
    )

    # Modelo GRU
    print("[INFO] Construyendo modelo GRU Sequential...")
    model = build_gru_model_sequential(
        vocab_size=vocab_size,
        embedding_dim=128,
        sequence_length=SEQ_LEN,
        gru_units=128,
        dropout_rate=0.5,
    )

    sample_batch = next(iter(train_ds.take(1)))
    model(sample_batch[0])
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=14,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode="min",
        ),
    ]

    class_weights = compute_class_weights(y_train)

    print("[INFO] Iniciando entrenamiento...")
    EPOCHS = 15
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    plot_training_history(
        history,
        filename="gru_training_history.png",
        metrics=("accuracy", "loss", "precision", "recall"),
    )

    # Evaluación
    y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        model, valid_ds, y_valid, "Validación"
    )
    y_test_pred, y_test_proba, f1_test = evaluate_model(
        model, test_ds, y_test, "Prueba"
    )

    save_best_model_keras(model, f1_valid, BEST_SCORE_PATH, BEST_MODEL_PATH)
    best_model = load_best_model_keras(BEST_MODEL_PATH)
    y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        best_model, valid_ds, y_valid, "Validación"
    )
    y_test_pred, y_test_proba, f1_test = evaluate_model(
        best_model, test_ds, y_test, "Prueba"
    )

    # Confusion matrices
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        title="Matriz de confusión - Validación (GRU)",
        filename="gru_confusion_valid.png",
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        title="Matriz de confusión - Prueba (GRU)",
        filename="gru_confusion_test.png",
    )

    # Métricas
    valid_metrics = plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (GRU)",
        filename="gru_metrics_valid.png",
    )
    test_metrics = plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (GRU)",
        filename="gru_metrics_test.png",
    )

    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)
    ax.set_title("Comparación de métricas entre Validación y Prueba (GRU)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)
    save_figure(fig, "gru_comparison_valid_test.png")
    plt.close(fig)

    print("[INFO] Proceso terminado.")
    print(f"[INFO] F1 Score Final - Validación: {f1_valid:.4f}, Prueba: {f1_test:.4f}")
