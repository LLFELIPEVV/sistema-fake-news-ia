import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    SpatialDropout1D,
    LSTM,
    Bidirectional,
    GRU,
    Conv1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Dropout,
    BatchNormalization,
    MultiHeadAttention,
    Input,
    Concatenate,
    LayerNormalization,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from training.tensorflow.utils_keras import (
    save_best_model_keras,
    plot_training_history,
    load_best_model_keras,
    build_embedding_matrix,
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
    if os.cpu_count() and os.cpu_count() >= 8:
        BATCH_SIZE = 128
    else:
        BATCH_SIZE = 64
else:  # CPU
    if os.cpu_count() and os.cpu_count() <= 4:
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64

print(f"[INFO] Usando batch_size = {BATCH_SIZE} con {device} y {os.cpu_count()} hilos")

# Configuración general
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "hybrid_cnn_bilstm_gru_attention_best_model.keras"
)
BEST_SCORE_PATH = os.path.join(
    MODEL_DIR, "hybrid_cnn_bilstm_gru_attention_best_score.txt"
)
VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer.pkl")

# Embedding GloVe path
GLOVE_PATH = os.path.join("glove.840B.300d", "glove.840B.300d.txt")
EMBEDDING_DIM = 300

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ==============================================
# 🔤 Vectorización y Embedding
# ==============================================
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


# ==============================================
# 🧠 Modelo híbrido optimizado
# ==============================================
def build_hybrid_model(
    vocab_size, embedding_matrix, sequence_length=200, device_type=device
):
    inp = Input(shape=(sequence_length,), dtype="int32", name="input_text")
    # start frozen
    emb = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        input_length=sequence_length,
        weights=[embedding_matrix],
        trainable=False,  # Fase 1: congelado
        name="embedding",
    )(inp)

    x = SpatialDropout1D(0.25)(emb)

    # CNN parallel branches
    convs = []
    for k in (3, 4, 5):
        c = Conv1D(
            filters=128,
            kernel_size=k,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(1e-4),
        )(x)
        c = BatchNormalization()(c)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    cnn_out = Concatenate()(convs)

    # Recurrent path (BiLSTM -> GRU)
    # Note: recurrent_dropout lowers CuDNN usage; keep moderate for CPU, set 0 on GPU for speed if desired
    recurrent_dropout = 0.25 if device_type == "CPU" else 0.0
    lstm = Bidirectional(
        LSTM(
            64, return_sequences=True, dropout=0.3, recurrent_dropout=recurrent_dropout
        )
    )(x)
    gru = GRU(
        64, return_sequences=True, dropout=0.3, recurrent_dropout=recurrent_dropout
    )(lstm)

    # Attention
    attn = MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.1)(gru, gru)
    attn = LayerNormalization()(attn)
    seq_feat = Concatenate()(
        [GlobalAveragePooling1D()(attn), GlobalMaxPooling1D()(attn)]
    )

    # Merge features
    merged = Concatenate()([cnn_out, seq_feat])
    merged = Dropout(0.4)(merged)

    # Dense head with L2 regularization and BatchNorm
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)

    out = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="Hybrid_CNN_BiLSTM_GRU_Att")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def to_tf_dataset(X, y, vectorizer, batch_size=BATCH_SIZE, shuffle=True):
    """Convierte arrays en tf.data.Dataset optimizado con vectorización."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10000), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(y):
    """Calcula pesos de clase para datos desbalanceados."""
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights_dict = {int(c): w for c, w in zip(classes, cw)}
    print(f"[INFO] Distribución de clases: {np.bincount(y)}")
    print(f"[INFO] Class weights: {class_weights_dict}")
    return class_weights_dict


def evaluate_model(model, dataset, y_true, dataset_name="Dataset"):
    """Evalúa el modelo y muestra métricas."""
    print(f"\n=== Evaluación sobre {dataset_name} ===")
    # Predicciones
    y_proba = model.predict(dataset, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    # Métricas
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

    lemmatizer = Lemmatizer()
    X_train, X_valid, X_test = map(lemmatizer.transform, [X_train, X_valid, X_test])

    # Vectorización
    print("[INFO] Preparando TextVectorization...")
    vectorizer = prepare_vectorizer(X_train, max_tokens=30000, output_seq_len=200)
    # Guardar vectorizer para producción
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer guardado en {VECTORIZER_PATH}")

    embedding_matrix = build_embedding_matrix(vectorizer)

    # Tamaño de vocabulario real
    vocab_size = len(vectorizer.get_vocabulary())
    print(f"[INFO] Vocab size real: {vocab_size}")

    # sample weights
    cw_dict = compute_class_weights(y_train)
    print(f"[INFO] Class weights dict: {cw_dict}")

    # Crear datasets
    print("[INFO] Creando datasets de TensorFlow...")
    train_ds = to_tf_dataset(X_train, y_train, vectorizer)
    valid_ds = to_tf_dataset(X_valid, y_valid, vectorizer)
    test_ds = to_tf_dataset(X_test, y_test, vectorizer)

    # Construir modelo
    print("[INFO] Construyendo modelo híbrido CNN + BiLSTM + GRU + Atención...")
    model = build_hybrid_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        sequence_length=200,
        device_type=device,
    )

    # Compilar modelo con datos de ejemplo para mostrar arquitectura completa
    sample_batch = next(iter(train_ds.take(1)))
    model(sample_batch[0])  # Build the model
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # Class weights
    class_weights = compute_class_weights(y_train)

    # Entrenamiento
    print("[INFO] Entrenamiento fase 1 (embeddings congelados)...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning: unfreeze embedding and continue with lower LR
    print("[INFO] Descongelando embeddings para fine-tuning (fase 2)...")
    for layer in model.layers:
        if layer.name == "embedding":
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
        verbose=1,
    )

    # Guardar historial de entrenamiento
    plot_training_history(
        history,
        filename="hybrid_cnn_bilstm_gru_attention_training_history.png",
        metrics=("accuracy", "loss", "precision", "recall"),
    )

    # Evaluaciones
    y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        model, valid_ds, y_valid, "Validación"
    )
    y_test_pred, y_test_proba, f1_test = evaluate_model(
        model, test_ds, y_test, "Prueba"
    )

    # Guardar mejor modelo
    save_best_model_keras(model, f1_valid, BEST_SCORE_PATH, BEST_MODEL_PATH)

    # Generar gráficos
    print("[INFO] Generando visualizaciones...")
    best_model = load_best_model_keras(BEST_MODEL_PATH)
    y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        best_model, valid_ds, y_valid, "Validación"
    )
    y_test_pred, y_test_proba, f1_test = evaluate_model(
        best_model, test_ds, y_test, "Prueba"
    )

    # Matrices de confusión
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        title="Matriz de confusión - Validación (Híbrido CNN+BiLSTM+GRU+Atención)",
        filename="hybrid_cnn_bilstm_gru_attention_confusion_valid.png",
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        title="Matriz de confusión - Prueba (Híbrido CNN+BiLSTM+GRU+Atención)",
        filename="hybrid_cnn_bilstm_gru_attention_confusion_test.png",
    )

    # Métricas detalladas
    plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (Híbrido CNN+BiLSTM+GRU+Atención)",
        filename="hybrid_cnn_bilstm_gru_attention_metrics_valid.png",
    )
    plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (Híbrido CNN+BiLSTM+GRU+Atención)",
        filename="hybrid_cnn_bilstm_gru_attention_metrics_test.png",
    )

    # Comparación de métricas entre Validación y Prueba
    print("[INFO] Comparando métricas entre Validación y Prueba...")
    valid_metrics = plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (Híbrido CNN+BiLSTM+GRU+Atención)",
        filename="hybrid_cnn_bilstm_gru_attention_metrics_valid.png",
    )
    test_metrics = plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (Híbrido CNN+BiLSTM+GRU+Atención)",
        filename="hybrid_cnn_bilstm_gru_attention_metrics_test.png",
    )

    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)
    ax.set_title(
        "Comparación de métricas entre Validación y Prueba (Híbrido CNN+BiLSTM+GRU+Atención)"
    )
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)

    # Etiquetas encima de cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)

    save_figure(fig, "hybrid_cnn_bilstm_gru_attention_comparison_valid_test.png")
    plt.close(fig)

    print("[INFO] Proceso terminado.")
    print(f"[INFO] F1 Score Final - Validación: {f1_valid:.4f}, Prueba: {f1_test:.4f}")
    print(f"[INFO] Modelo híbrido guardado en: {BEST_MODEL_PATH}")
