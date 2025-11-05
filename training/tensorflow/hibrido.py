import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
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
HISTORY_FILE = os.path.join(MODEL_DIR, "hybrid_history.json")
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def prepare_vectorizer(texts, max_tokens=30000, output_seq_len=200):
    """Crea y adapta un TextVectorization layer sobre los textos de entrenamiento."""
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(texts)
    return vectorizer


def build_hybrid_model(
    vocab_size,
    embedding_dim=128,
    embedding_matrix=None,
    sequence_length=200,
    cnn_filters=128,
    cnn_kernel_sizes=[3, 4, 5],
    bilstm_units=64,
    gru_units=64,
    dropout_rate=0.3,
):
    """
    Construye modelo híbrido CNN + BiLSTM + GRU + Atención.

    Arquitectura:
    1. Embedding + SpatialDropout
    2. Rama CNN paralela con múltiples kernel sizes
    3. BiLSTM para capturar dependencias bidireccionales
    4. GRU para modelado secuencial adicional
    5. Mecanismo de atención de Keras (self-attention)
    6. Concatenación de todas las representaciones
    7. Capas densas finales con regularización
    """

    # Input layer
    inputs = Input(shape=(sequence_length,), name="input_text")

    # Embedding layer
    if embedding_matrix is not None:
        embedded = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
            name="embedding",
        )(inputs)
    else:
        embedded = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
            mask_zero=False,
            name="embedding",
        )(inputs)

    # Spatial dropout para regularización en embeddings
    embedded_dropped = SpatialDropout1D(0.2, name="spatial_dropout")(embedded)

    cnn_outputs = []
    for i, kernel_size in enumerate(cnn_kernel_sizes):
        # Convolución 1D
        conv = Conv1D(
            filters=cnn_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="valid",
            name=f"conv1d_{kernel_size}",
        )(embedded_dropped)

        # Batch normalization
        conv = BatchNormalization(name=f"bn_conv_{kernel_size}")(conv)

        # Global max pooling
        pool_max = GlobalMaxPooling1D(name=f"global_max_pool_{kernel_size}")(conv)

        # Global average pooling
        pool_avg = GlobalAveragePooling1D(name=f"global_avg_pool_{kernel_size}")(conv)

        # Concatenar ambos poolings
        cnn_feature = Concatenate(name=f"cnn_concat_{kernel_size}")(
            [pool_max, pool_avg]
        )
        cnn_outputs.append(cnn_feature)

    # Concatenar todas las features CNN
    cnn_features = Concatenate(name="cnn_features_concat")(cnn_outputs)

    bilstm_out = Bidirectional(
        LSTM(
            bilstm_units,
            dropout=dropout_rate,
            recurrent_dropout=0.3,
            return_sequences=True,
            name="bilstm",
        ),
        name="bidirectional_lstm",
    )(embedded_dropped)

    # Batch normalization después de BiLSTM
    bilstm_out = BatchNormalization(name="bn_bilstm")(bilstm_out)

    gru_out = GRU(
        gru_units,
        dropout=dropout_rate,
        recurrent_dropout=0.3,
        return_sequences=True,
        name="gru",
    )(bilstm_out)

    # Batch normalization después de GRU
    gru_out = BatchNormalization(name="bn_gru")(gru_out)

    # La salida del GRU se usa como query, key y value
    attention_out = MultiHeadAttention(
        num_heads=4, key_dim=gru_units, dropout=0.1, name="self_attention"
    )(query=gru_out, value=gru_out, key=gru_out)

    # Pooling global para resumir
    attention_pooled = GlobalAveragePooling1D(name="attention_pooled")(attention_out)

    # También obtener representación mediante pooling global del GRU
    gru_max_pool = GlobalMaxPooling1D(name="gru_max_pool")(gru_out)
    gru_avg_pool = GlobalAveragePooling1D(name="gru_avg_pool")(gru_out)

    all_features = Concatenate(name="all_features_concat")(
        [
            cnn_features,  # Features CNN
            attention_pooled,  # Features con atención de Keras
            gru_max_pool,  # GRU max pooling
            gru_avg_pool,  # GRU average pooling
        ]
    )

    # Primera capa densa con regularización
    dense1 = Dense(256, activation="relu", name="dense_1")(all_features)
    dense1 = BatchNormalization(name="bn_dense1")(dense1)
    dense1 = Dropout(0.5, name="dropout_1")(dense1)

    # Segunda capa densa
    dense2 = Dense(128, activation="relu", name="dense_2")(dense1)
    dense2 = BatchNormalization(name="bn_dense2")(dense2)
    dense2 = Dropout(0.4, name="dropout_2")(dense2)

    # Tercera capa densa
    dense3 = Dense(64, activation="relu", name="dense_3")(dense2)
    dense3 = Dropout(0.3, name="dropout_3")(dense3)

    # Output layer
    outputs = Dense(1, activation="sigmoid", name="output")(dense3)

    # Crear modelo
    model = Model(
        inputs=inputs, outputs=outputs, name="hybrid_cnn_bilstm_gru_attention"
    )

    # Compilar con métricas adicionales
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),  # Gradient clipping
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

    X_train = train_df["texto"].astype(str).values
    y_train = train_df["clase"].astype(int).values
    X_valid = valid_df["texto"].astype(str).values
    y_valid = valid_df["clase"].astype(int).values
    X_test = test_df["texto"].astype(str).values
    y_test = test_df["clase"].astype(int).values

    print(
        f"[INFO] Tamaños - Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}"
    )

    # Vectorización
    print("[INFO] Preparando vectorizador TextVectorization...")
    MAX_TOKENS = 30000
    SEQ_LEN = 200
    vectorizer = prepare_vectorizer(
        X_train, max_tokens=MAX_TOKENS, output_seq_len=SEQ_LEN
    )

    # Tamaño de vocabulario real
    vocab_size = len(vectorizer.get_vocabulary())
    print(f"[INFO] Vocab size real: {vocab_size}")
    embedding_matrix = build_embedding_matrix(vectorizer)

    # Crear datasets
    print("[INFO] Creando datasets de TensorFlow...")
    train_ds = to_tf_dataset(
        X_train, y_train, vectorizer, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_ds = to_tf_dataset(
        X_valid, y_valid, vectorizer, batch_size=BATCH_SIZE, shuffle=False
    )
    test_ds = to_tf_dataset(
        X_test, y_test, vectorizer, batch_size=BATCH_SIZE, shuffle=False
    )

    # Hiperparámetros
    all_combinations = {
        "filters": [96, 128, 160],
        "lstm_units": [64, 80, 96],
        "gru_units": [64, 80, 96],
        "dropout_rate": [0.3, 0.4, 0.5],
        "l2_reg": [1e-4, 1e-5],
        "kernel_sizes": [[3, 4, 5], [2, 3, 4], [3, 5, 7]],
    }

    params = get_random_hyperparams(all_combinations, HISTORY_FILE, max_attempts=50)
    print(f"[INFO] Hiperparámetros seleccionados: {params}")

    cnn_filters = params.get("filters", 128)
    bilstm_units = params.get("lstm_units", 64)
    gru_units = params.get("gru_units", 64)
    dropout_rate = params.get("dropout_rate", 0.4)
    l2_reg = params.get("l2_reg", 1e-4)
    cnn_kernel_sizes = params.get("kernel_sizes", [3, 4, 5])
    
    # Construir modelo
    print("[INFO] Construyendo modelo híbrido CNN + BiLSTM + GRU + Atención...")
    model = build_hybrid_model(
        vocab_size=vocab_size,
        embedding_dim=300,
        embedding_matrix=embedding_matrix,
        sequence_length=SEQ_LEN,
        cnn_filters=cnn_filters,
        cnn_kernel_sizes=cnn_kernel_sizes,
        bilstm_units=bilstm_units,
        gru_units=gru_units,
        dropout_rate=dropout_rate,
    )

    # Compilar modelo con datos de ejemplo para mostrar arquitectura completa
    sample_batch = next(iter(train_ds.take(1)))
    model(sample_batch[0])  # Build the model
    model.summary()

    # Callbacks
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

    # Class weights
    class_weights = compute_class_weights(y_train)

    # Entrenamiento
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
        class_weight=class_weights,
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
    
    metrics = {
        "f1_valid": float(f1_valid),
        "f1_test": float(f1_test),
        "device": device,
        "batch_size": BATCH_SIZE,
    }
    
    save_results(params, metrics, HISTORY_FILE)