import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TORCH"] = "1"

from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from training.utils_common import (
    load_datasets,
    plot_confusion_matrix,
    plot_metrics,
    MODEL_DIR,
    save_figure,
)

# Configuración de hardware
try:
    import torch_directml

    dml_available = True
except ImportError:
    dml_available = False

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] GPU CUDA detectada: {torch.cuda.get_device_name(0)}")
    BATCH_SIZE = 16
elif hasattr(torch, "hip") and torch.version.hip:  # ROCm backend
    device = torch.device("cuda")  # En ROCm, el device se llama igual
    print("[INFO] GPU AMD (ROCm) detectada.")
    BATCH_SIZE = 16
elif dml_available:
    device = torch_directml.device()
    print("[INFO] GPU AMD detectada mediante DirectML.")
    BATCH_SIZE = 16
else:
    device = torch.device("cpu")
    print("[INFO] No se detectó GPU, usando CPU.")
    BATCH_SIZE = 8

print(f"[INFO] Usando dispositivo: {device}")
print(f"[INFO] Usando batch_size = {BATCH_SIZE}")


# Configuración general
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "mbert_pytorch_best_model.pt")
BEST_SCORE_PATH = os.path.join(MODEL_DIR, "mbert_pytorch_best_score.txt")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuración del modelo
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128


class TextDataset(Dataset):
    """Dataset personalizado para textos tokenizados."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }


class mBERTClassifier(nn.Module):
    """Modelo de clasificación basado en mBERT."""

    def __init__(self, model_name, dropout_rate=0.3):
        super(mBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)

        # Congelar las primeras capas (opcional)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < 10:  # Congelar las primeras 10 capas de 12
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Usar el [CLS] token (primera posición)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def compute_class_weights(y):
    """Calcula pesos de clase para datos desbalanceados."""
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights_dict = {int(c): w for c, w in zip(classes, cw)}
    print(f"[INFO] Distribución de clases: {np.bincount(y)}")
    print(f"[INFO] Class weights: {class_weights_dict}")
    return torch.tensor([cw[0]], dtype=torch.float).to(device)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
        correct += (predictions == labels.long()).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({"loss": loss.item(), "acc": correct / total})

    return total_loss / len(dataloader), correct / total


def evaluate_model(model, dataloader, criterion, device, dataset_name="Dataset"):
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probas = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()
            probas = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            predictions = (probas >= 0.5).astype(int)

            all_predictions.extend(
                predictions if isinstance(predictions, np.ndarray) else [predictions]
            )
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas if isinstance(probas, np.ndarray) else [probas])

    y_true = np.array(all_labels).astype(int)
    y_pred = np.array(all_predictions).astype(int)
    y_proba = np.array(all_probas)

    print(f"\n=== Evaluación sobre {dataset_name} ===")
    print(classification_report(y_true, y_pred, digits=4))
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    return total_loss / len(dataloader), y_pred, y_proba, f1_macro


def save_model(model, f1_score_val, best_score_path, best_model_path):
    """Guarda el modelo si mejora el F1 score."""
    if os.path.exists(best_score_path):
        with open(best_score_path, "r") as f:
            best_f1 = float(f.read().strip())
    else:
        best_f1 = 0.0

    if f1_score_val > best_f1:
        print(f"[INFO] Nuevo mejor F1: {f1_score_val:.4f} (anterior: {best_f1:.4f})")
        torch.save(model.state_dict(), best_model_path)
        with open(best_score_path, "w") as f:
            f.write(f"{f1_score_val:.6f}")
        return True
    return False


def plot_training_history(history, filename="mbert_training_history.png"):
    """Grafica el historial de entrenamiento."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", marker="o")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation", marker="s")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(epochs, history["train_acc"], label="Train", marker="o")
    axes[0, 1].plot(epochs, history["val_acc"], label="Validation", marker="s")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(
        epochs, history["val_f1"], label="Validation F1", marker="s", color="green"
    )
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Eliminar el subplot vacío
    fig.delaxes(axes[1, 1])

    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


if __name__ == "__main__":
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

    # Cargar tokenizer
    print(f"[INFO] Cargando tokenizer de {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Crear datasets
    print("[INFO] Creando datasets...")
    train_dataset = TextDataset(X_train_texts, y_train, tokenizer, MAX_LENGTH)
    valid_dataset = TextDataset(X_valid_texts, y_valid, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(X_test_texts, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Construir modelo
    print("[INFO] Construyendo modelo mBERT...")
    model = mBERTClassifier(MODEL_NAME, dropout_rate=0.3).to(device)

    # Calcular class weights
    pos_weight = compute_class_weights(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer y scheduler
    EPOCHS = 15
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print(f"[INFO] Total steps: {total_steps}, Warmup steps: {int(0.1 * total_steps)}")

    # Entrenamiento
    print("[INFO] Iniciando entrenamiento...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 50}")

        # Entrenar
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validar
        val_loss, y_val_pred, y_val_proba, val_f1 = evaluate_model(
            model, valid_loader, criterion, device, "Validación"
        )

        # Guardar historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append((y_val_pred == y_valid).mean())
        history["val_f1"].append(val_f1)

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {history['val_acc'][-1]:.4f}, Val F1: {val_f1:.4f}"
        )

        # Guardar mejor modelo
        if save_model(model, val_f1, BEST_SCORE_PATH, BEST_MODEL_PATH):
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[INFO] Paciencia: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"[INFO] Early stopping activado en epoch {epoch + 1}")
            break

    # Graficar historial
    plot_training_history(history, "mbert_training_history.png")

    # Cargar mejor modelo
    print("\n[INFO] Cargando mejor modelo...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))

    # Evaluación final
    _, y_valid_pred, y_valid_proba, f1_valid = evaluate_model(
        model, valid_loader, criterion, device, "Validación (Mejor Modelo)"
    )
    _, y_test_pred, y_test_proba, f1_test = evaluate_model(
        model, test_loader, criterion, device, "Prueba (Mejor Modelo)"
    )

    # Visualizaciones
    print("[INFO] Generando visualizaciones...")
    plot_confusion_matrix(
        y_valid,
        y_valid_pred,
        title="Matriz de confusión - Validación (mBERT)",
        filename="mbert_confusion_valid.png",
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        title="Matriz de confusión - Prueba (mBERT)",
        filename="mbert_confusion_test.png",
    )

    valid_metrics = plot_metrics(
        y_valid,
        y_valid_pred,
        dataset_name="Validación (mBERT)",
        filename="mbert_metrics_valid.png",
    )
    test_metrics = plot_metrics(
        y_test,
        y_test_pred,
        dataset_name="Prueba (mBERT)",
        filename="mbert_metrics_test.png",
    )

    # Comparación de métricas
    comp_df = pd.DataFrame(
        [valid_metrics, test_metrics], index=["Validación", "Prueba"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="bar", colormap="viridis", ax=ax)
    ax.set_title("Comparación de métricas entre Validación y Prueba (mBERT)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)
    save_figure(fig, "mbert_comparison_valid_test.png")
    plt.close(fig)

    print("\n[INFO] Proceso terminado.")
    print(f"[INFO] F1 Score Final - Validación: {f1_valid:.4f}, Prueba: {f1_test:.4f}")
