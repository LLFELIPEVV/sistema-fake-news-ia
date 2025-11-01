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
    Dropout,
    BatchNormalization,
    LayerNormalization,
    MultiHeadAttention,
    Input,
    Concatenate,
    LeakyReLU,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from training.tensorflow.utils_keras import (
    save_best_model_keras,
    plot_training_history,
    load_best_model_keras,
    build_embedding_matrix,
    get_random_hyperparams,
    save_results,
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
    BATCH_SIZE = 128 if (os.cpu_count() or 0) >= 8 else 64
else:
    BATCH_SIZE = 32 if (os.cpu_count() or 0) <= 4 else 64

print(f"[INFO] Usando batch_size = {BATCH_SIZE} con {device} y {os.cpu_count()} hilos")

# Configuración general
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_optimized_best_model.keras")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "hybrid_optimized_best_score.txt")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer_hybrid.pkl")
HISTORY_FILE = os.path.join(MODEL_DIR, "hybrid_optimized_history.json")

SEQ_LEN = 200
MAX_TOKENS = 30000
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def prepare_vectorizer(texts, max_tokens=MAX_TOKENS, output_seq_len=SEQ_LEN):
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(texts)
    return vectorizer


def build_hybrid_model_optimized(vocab_size, embedding_matrix, params):
    """
    Modelo híbrido SIMPLIFICADO y BALANCEADO inspirado en el CNN exitoso.

    Cambios clave:
    1. CNN simplificado (solo MaxPooling, sin duplicar features)
    2. Capas recurrentes más ligeras
    3. Capas densas reducidas (128 → 64 como en CNN)
    4. Regularización consistente (SpatialDropout + LayerNorm + BatchNorm)
    5. Dropout adaptativo según device
    """
    cnn_filters = params.get("filters", 128)
    rnn_units = params.get("rnn_units", 64)  # ✅ Mismo para LSTM y GRU
    dropout_rate = params.get("dropout_rate", 0.4 if device == "GPU" else 0.3)
    l2_reg = params.get("l2_reg", 1e-4)
    kernel_sizes = params.get("kernel_sizes", (3, 4, 5))

    inp = Input(shape=(SEQ_LEN,), name="input_text")

    # Embedding (igual que CNN)
    emb = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False,
        name="embedding",
    )(inp)

    # ✅ SpatialDropout como en CNN
    x = SpatialDropout1D(dropout_rate)(emb)

    # =====================================
    # RAMA CNN (SIMPLIFICADA - solo MaxPool)
    # =====================================
    cnn_outputs = []
    for ks in kernel_sizes:
        conv = Conv1D(
            filters=cnn_filters,
            kernel_size=ks,
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )(x)
        conv = LeakyReLU(alpha=0.1)(conv)  # ✅ LeakyReLU como en CNN
        conv = BatchNormalization()(conv)
        # ✅ SOLO MaxPooling (eliminar AvgPooling duplicado)
        conv = GlobalMaxPooling1D()(conv)
        cnn_outputs.append(conv)

    cnn_features = Concatenate()(cnn_outputs)

    # =====================================
    # RAMA RNN (SIMPLIFICADA)
    # =====================================
    # ✅ Reducir recurrent_dropout (0.3 → 0.2)
    rnn_out = Bidirectional(
        LSTM(
            rnn_units,
            return_sequences=True,
            dropout=dropout_rate * 0.5,  # ✅ Menos dropout
            recurrent_dropout=0.2,  # ✅ Reducido de 0.3
            kernel_regularizer=l2(l2_reg),
        )
    )(x)
    rnn_out = BatchNormalization()(rnn_out)

    # ✅ GRU más ligero
    rnn_out = GRU(
        rnn_units,
        return_sequences=True,
        dropout=dropout_rate * 0.5,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(l2_reg),
    )(rnn_out)
    rnn_out = BatchNormalization()(rnn_out)

    # =====================================
    # ATENCIÓN (MEJORADA)
    # =====================================
    # ✅ Más heads y key_dim ajustado
    attn_out = MultiHeadAttention(
        num_heads=4,  # ✅ Aumentado de 2 a 4
        key_dim=rnn_units // 2,  # ✅ Más eficiente
        dropout=0.1,
    )(rnn_out, rnn_out)

    # ✅ Solo MaxPooling (consistente con CNN)
    attn_features = GlobalMaxPooling1D()(attn_out)

    # =====================================
    # FUSIÓN (SIMPLIFICADA)
    # =====================================
    all_features = Concatenate()([cnn_features, attn_features])

    # ✅ LayerNormalization como en CNN
    all_features = LayerNormalization()(all_features)

    # =====================================
    # CAPAS DENSAS (IGUAL QUE CNN: 128 → 64)
    # =====================================
    x = Dense(128, kernel_regularizer=l2(l2_reg))(all_features)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, kernel_regularizer=l2(l2_reg))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate * 0.5)(x)

    # Output
    output = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=output, name="hybrid_optimized")

    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    return model


def to_tf_dataset(texts, labels, vectorizer, batch_size=BATCH_SIZE, shuffle=True):
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
    print("[INFO] Cargando datos...")
    train_df, valid_df, test_df = load_datasets()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("[ERROR] Uno o más datasets están vacíos")

    X_train = train_df["texto"].astype(str).values
    y_train = train_df["clase"].astype(int).values
    X_valid = valid_df["texto"].astype(str).values
    y_valid = valid_df["clase"].astype(int).values
    X_test = test_df["texto"].astype(str).values
    y_test = test_df["clase"].astype(int).values

    print(
        f"[INFO] Tamaños - Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}"
    )

    # Lematización
    lemmatizer = Lemmatizer()
    X_train, X_valid, X_test = map(lemmatizer.transform, [X_train, X_valid, X_test])

    # Vectorización
    print("[INFO] Preparando vectorizador...")
    vectorizer = prepare_vectorizer(X_train)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    embedding_matrix = build_embedding_matrix(vectorizer)
    vocab_size = len(vectorizer.get_vocabulary())
    print(f"[INFO] Vocab size real: {vocab_size}")

    # Datasets
    train_ds = (
        to_tf_dataset(X_train, y_train, vectorizer).cache().prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = (
        to_tf_dataset(X_valid, y_valid, vectorizer).cache().prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        to_tf_dataset(X_test, y_test, vectorizer).cache().prefetch(tf.data.AUTOTUNE)
    )

    # ✅ Hiperparámetros simplificados (alineados con CNN)
    all_combinations = {
        "filters": [96, 128, 160],  # Igual que CNN
        "rnn_units": [64, 80],  # Simplificado
        "dropout_rate": [0.3, 0.4, 0.5],  # Igual que CNN
        "l2_reg": [1e-4, 1e-5],  # Igual que CNN
        "kernel_sizes": [(3, 4, 5), (2, 3, 4), (3, 5, 7)],  # Igual que CNN
    }

    params = get_random_hyperparams(all_combinations, HISTORY_FILE, max_attempts=50)
    print(f"[INFO] Hiperparámetros seleccionados: {params}")

    # Construir modelo
    print("[INFO] Construyendo modelo híbrido optimizado...")
    model = build_hybrid_model_optimized(vocab_size, embedding_matrix, params)

    sample_batch = next(iter(train_ds.take(1)))
    model(sample_batch[0])
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

    class_weights_dict = compute_class_weights(y_train)

    # Fase 1: Embeddings congelados
    print("[INFO] Entrenamiento fase 1 (embeddings congelados)...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1,
    )

    # Fase 2: Fine-tuning
    print("[INFO] Iniciando fine-tuning de embeddings...")
    model.get_layer("embedding").trainable = True
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
        class_weight=class_weights_dict,
        verbose=1,
    )

    # Visualizaciones
    plot_training_history(
        history,
        filename="hybrid_optimized_training_history.png",
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
        title="Matriz de confusión - Validación (Híbrido Optimizado)",
        filename="hybrid_optimized_confusion_valid.png",
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        title="Matriz de confusión - Prueba (Híbrido Optimizado)",
        filename="hybrid_optimized_confusion_test.png",
    )

    # Métricas
    valid_metrics = plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (Híbrido Optimizado)",
        filename="hybrid_optimized_metrics_valid.png",
    )
    test_metrics = plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (Híbrido Optimizado)",
        filename="hybrid_optimized_metrics_test.png",
    )

    # Comparación
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)
    ax.set_title("Comparación de métricas - Híbrido Optimizado")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)
    save_figure(fig, "hybrid_optimized_comparison_valid_test.png")
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
