import os
import gc
import time
import torch
import psutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TORCH"] = "1"

from scipy import stats
from transformers import AutoTokenizer
from training.utils_common import load_datasets
from training.transformers.beto import BETOClassifier
from training.transformers.mbert import mBERTClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 600


def available_memory_gb():
    """Estima RAM disponible en GB"""
    return psutil.virtual_memory().available / (1024**3)


def clear_memory(full=True):
    """Libera memoria entre evaluaciones"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if full:
        time.sleep(0.5)
    gc.collect()


def to_text_list(batch):
    """Convierte batch a lista de strings"""
    if isinstance(batch, (pd.Series, np.ndarray)):
        batch = batch.tolist()
    return [str(x) for x in batch]


class AdvancedModelEvaluator:
    """Sistema avanzado de evaluación y análisis de modelos de Machine Learning"""

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.results = []
        self.confusion_matrices = {}
        self.predictions = {}

    @staticmethod
    def get_best_device(run_benchmark=True):
        """
        Detecta CUDA, DirectML o CPU y selecciona el mejor.
        Retorna: (device_string, device_object, memoria_disponible_gb, num_workers)
        """
        devices_info = []

        # ====== CPU CONFIG ======
        cpu_threads = psutil.cpu_count(logical=True)
        print(f"💻 CPU detectado: {cpu_threads} hilos lógicos")
        torch.set_num_threads(cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)

        # ====== 1. CUDA (NVIDIA) ======
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_mem = (props.total_memory - torch.cuda.memory_allocated(i)) / (
                    1024**3
                )
                tflops_estimate = props.multi_processor_count * props.major / 10

                devices_info.append(
                    {
                        "type": "cuda",
                        "id": i,
                        "name": props.name,
                        "device_str": f"cuda:{i}",
                        "device_obj": torch.device(f"cuda:{i}"),
                        "total_memory_gb": total_mem,
                        "free_memory_gb": free_mem,
                        "tflops_estimate": tflops_estimate,
                        "score": 0,
                    }
                )
                print(
                    f"🎮 GPU {i}: {props.name} ({free_mem:.2f}/{total_mem:.2f} GB libres)"
                )

        # ====== 2. DirectML (AMD / Intel / Microsoft GPU) ======
        try:
            import torch_directml

            dml_device = torch_directml.device()
            print("🎮 DirectML detectado correctamente (torch_directml)")

            devices_info.append(
                {
                    "type": "dml",
                    "id": 0,
                    "name": "DirectML (Intel/AMD GPU)",
                    "device_str": "dml",
                    "device_obj": dml_device,
                    "total_memory_gb": 4.0,
                    "free_memory_gb": 3.0,
                    "tflops_estimate": 2.0,
                    "score": 0,
                }
            )
        except ImportError:
            print("⚠️ DirectML no está instalado o no se detectó.")
        except Exception as e:
            print(f"⚠️ Error al inicializar DirectML: {e}")

        # ====== 3. CPU siempre disponible ======
        cpu_ram = psutil.virtual_memory().available / (1024**3)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_ghz = cpu_freq.max / 1000 if cpu_freq else 3.0
        cpu_gflops = cpu_threads * cpu_freq_ghz * 32

        devices_info.append(
            {
                "type": "cpu",
                "id": 0,
                "name": f"CPU ({cpu_threads} hilos @ {cpu_freq_ghz:.1f}GHz)",
                "device_str": "cpu",
                "device_obj": torch.device("cpu"),
                "total_memory_gb": cpu_ram,
                "free_memory_gb": cpu_ram,
                "tflops_estimate": cpu_gflops / 1000,
                "score": 0,
                "num_workers": max(2, cpu_threads // 2),
            }
        )
        print(
            f"💻 CPU disponible con {cpu_ram:.1f}GB RAM libre (~{cpu_gflops:.0f} GFLOPS teóricos)"
        )

        # ====== 4. Benchmark real ======
        if run_benchmark and len(devices_info) > 1:
            print("\n🏃 Ejecutando benchmark rápido (matmul 1024x1024)...")
            for dev in devices_info:
                try:
                    device = dev["device_obj"]
                    size = 1024
                    a = torch.randn(size, size).to(device)
                    b = torch.randn(size, size).to(device)

                    # warmup
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize() if dev["type"] == "cuda" else None

                    start = time.perf_counter()
                    for _ in range(5):
                        _ = torch.matmul(a, b)
                    torch.cuda.synchronize() if dev["type"] == "cuda" else None

                    elapsed = time.perf_counter() - start
                    gflops = (5 * 2 * size**3) / elapsed / 1e9
                    dev["benchmark_gflops"] = gflops
                    print(f"   {dev['device_str']}: {gflops:.1f} GFLOPS (real)")

                except Exception as e:
                    print(f"   {dev['device_str']}: Benchmark falló ({e})")
                    dev["benchmark_gflops"] = 0

        # ====== 4. Benchmark real ======
        if run_benchmark and len(devices_info) > 1:
            print("\n🏃 Ejecutando benchmark rápido (matmul 1024x1024)...")
            for dev in devices_info:
                try:
                    device = dev["device_obj"]
                    size = 1024
                    a = torch.randn(size, size)
                    b = torch.randn(size, size)
                    a_dev = a.to(device)
                    b_dev = b.to(device)

                    # warmup
                    _ = torch.matmul(a_dev, b_dev)
                    if dev["type"] == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    for _ in range(5):
                        _ = torch.matmul(a_dev, b_dev)
                    if dev["type"] == "cuda":
                        torch.cuda.synchronize()

                    elapsed = time.perf_counter() - start
                    gflops = (5 * 2 * size**3) / elapsed / 1e9
                    dev["benchmark_gflops"] = gflops
                    print(f"   {dev['device_str']}: {gflops:.1f} GFLOPS (real)")

                except Exception as e:
                    print(f"   {dev['device_str']}: Benchmark falló ({e})")
                    dev["benchmark_gflops"] = 0

        # ====== 5. Calcular score ======
        print("\n📊 Calculando scores...")
        for dev in devices_info:
            mem_score = dev["free_memory_gb"] * 100
            compute_score = (
                dev.get("benchmark_gflops", dev["tflops_estimate"] * 1000) * 3
            )
            type_bonus = {"cuda": 500, "dml": 200, "cpu": 0}
            dev["score"] = mem_score + compute_score + type_bonus[dev["type"]]
            print(f"   {dev['device_str']}: Score = {dev['score']:.0f}")

        # ====== 6. Seleccionar mejor ======
        best = max(devices_info, key=lambda d: d["score"])
        print(f"\n✅ Mejor dispositivo: {best['name']} ({best['device_str']})")
        return (
            best["device_str"],
            best["device_obj"],
            best["free_memory_gb"],
            best.get("num_workers", 0),
        )

    def calculate_optimal_batch_size(
        self, device_str, available_memory_gb, model_type="bert"
    ):
        """Calcula el batch size óptimo según dispositivo y memoria"""
        memory_per_sample_gb = (
            0.05 if model_type.lower() in ["mbert", "bert", "beto"] else 0.03
        )
        safety_factors = {"cuda": 0.7, "cpu": 0.8}
        device_type = device_str.split(":")[0] if ":" in device_str else device_str
        safety = safety_factors.get(device_type, 0.5)
        max_batch = int((available_memory_gb * safety) / memory_per_sample_gb)

        if device_type == "cuda":
            batch_size = max(4, min(max_batch, 64))
        elif device_type == "dml":
            batch_size = max(2, min(max_batch, 16))
        else:
            batch_size = max(8, min(max_batch, 32))

        print(f"📊 Batch size calculado: {batch_size}")
        return batch_size

    def evaluate_torch_model_complete(
        self,
        name,
        model_path,
        tokenizer_name,
        max_len=128,
        batch_size=None,
        device=None,
    ):
        """Evalúa modelos Transformer con gestión adaptativa de memoria y rendimiento"""
        print(f"\n{'=' * 60}\n⏳ Evaluando {name}...\n{'=' * 60}")

        # --- Selección de dispositivo ---
        try:
            import torch_directml

            dml_device = torch_directml.device()
            dml_available = True
        except ImportError:
            dml_device = None
            dml_available = False

        if device:
            device_str = device
            device_obj = torch.device(device)
            available_mem = 4.0
        elif torch.cuda.is_available():
            device_str = "cuda"
            device_obj = torch.device("cuda")
            available_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif dml_available:
            device_str = "dml"
            device_obj = dml_device
            available_mem = 4.0
        else:
            device_str = "cpu"
            device_obj = torch.device("cpu")
            available_mem = psutil.virtual_memory().available / (1024**3)

        print(
            f"🖥️ Usando dispositivo: {device_str} ({available_mem:.1f} GB disponibles)"
        )

        # --- Calcular batch size óptimo ---
        if batch_size is None:
            batch_size = max(8, int(available_mem * 32))
        print(f"⚙️ Batch size ajustado a: {batch_size}")

        # --- Cargar tokenizer y modelo ---
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if "beto" in name.lower():
            model = BETOClassifier(tokenizer_name)
        else:
            model = mBERTClassifier(tokenizer_name)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device_obj).eval()

        if device_str == "cpu" and available_mem < 4:
            max_len = min(max_len, 64)
            print(f"⚠️ max_len reducido a {max_len} por memoria limitada")

        # --- Datasets ---
        datasets = {
            "Train": (self.x_train, self.y_train),
            "Validation": (self.x_val, self.y_val),
            "Test": (self.x_test, self.y_test),
        }

        model_results = {"Modelo": name, "Tipo": "torch", "Dispositivo": device_str}
        cm_dict = {}

        # --- Evaluación por dataset ---
        for dataset_name, (X, y_true) in datasets.items():
            total_samples = len(X)
            print(
                f"\n📂 Procesando conjunto: {dataset_name} ({total_samples} muestras)"
            )

            y_pred_list, y_proba_list = [], []
            start_time = time.time()

            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                texts_batch = to_text_list(X[start:end])

                if not texts_batch:
                    continue

                inputs = tokenizer(
                    texts_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )

                # Mover tensores al dispositivo
                if device_str == "dml":
                    inputs = {k: v.to(dml_device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(device_obj) for k, v in inputs.items()}

                with torch.no_grad():
                    try:
                        outputs = model(**inputs)
                    except TypeError:
                        # Algunos modelos no aceptan token_type_ids
                        inputs.pop("token_type_ids", None)
                        outputs = model(**inputs)

                    logits = (
                        outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    )

                    if logits.ndim == 1 or logits.shape[-1] == 1:
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    else:
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                    preds = (probs > 0.5).astype(int)

                y_pred_list.extend(preds.tolist())
                y_proba_list.extend(probs.tolist())

            # --- Validación de tamaño ---
            if len(y_pred_list) != len(y_true):
                print(
                    f"⚠️ Ajustando tamaño de predicciones: {len(y_pred_list)} -> {len(y_true)}"
                )
                y_pred_list = y_pred_list[: len(y_true)]
                y_proba_list = y_proba_list[: len(y_true)]

            print(f"   ✅ {dataset_name} completado en {time.time() - start_time:.2f}s")

            metrics = self._calculate_extended_metrics(
                np.array(y_true),
                np.array(y_pred_list),
                np.array(y_proba_list),
                time.time() - start_time,
            )
            for metric, value in metrics.items():
                model_results[f"{dataset_name}_{metric}"] = value
            cm_dict[dataset_name] = confusion_matrix(y_true, y_pred_list)

            if dataset_name == "Test":
                self.predictions[name] = {
                    "y_true": y_true,
                    "y_pred": y_pred_list,
                    "y_proba": y_proba_list,
                }

            clear_memory()

        # --- Resultados finales ---
        del model
        gc.collect()
        clear_memory()

        self.confusion_matrices[name] = cm_dict
        self.results.append(model_results)

        print(f"\n✅ {name} evaluado correctamente en {device_str}\n{'=' * 60}\n")

    def _calculate_extended_metrics(self, y_true, y_pred, y_proba, prediction_time):
        """Calcula conjunto extendido de métricas"""
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_proba),
            "Log_Loss": log_loss(y_true, y_proba),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
            "Time": prediction_time,
        }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["NPV"] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        return metrics

    def generate_comprehensive_report(self):
        """Genera reporte completo de evaluación"""
        if not self.results:
            print("⚠️ No hay resultados para generar reporte")
            return None

        df = pd.DataFrame(self.results)
        os.makedirs("reports", exist_ok=True)
        df.to_csv("reports/model_evaluation_results_transformers.csv", index=False)
        print(
            "💾 Resultados guardados en: reports/model_evaluation_results_transformers.csv"
        )
        return df

    # ==================== VISUALIZACIONES ====================

    def plot_comprehensive_comparison(self, save_fig=False):
        """Comparación exhaustiva de todos los modelos"""
        df = pd.DataFrame(self.results)
        fig, axes = plt.subplots(4, 3, figsize=(20, 14), tight_layout=True)

        metrics = [
            "Accuracy",
            "F1",
            "Precision",
            "Recall",
            "ROC_AUC",
            "MCC",
            "Balanced_Accuracy",
            "Cohen_Kappa",
            "Specificity",
        ]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            for dataset in ["Train", "Validation", "Test"]:
                ax.bar(
                    df["Modelo"], df[f"{dataset}_{metric}"], label=dataset, alpha=0.8
                )
            ax.set_title(f"{metric.replace('_', ' ')}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Modelos", fontsize=9)
            ax.set_ylabel(metric.replace("_", " "), fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Comparación Exhaustiva de Métricas - Train/Validation/Test",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        if save_fig:
            plt.savefig(
                os.path.join("figures", "comprehensive_comparison_transformers.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_advanced_overfitting_analysis(self, save_fig=False):
        """Análisis avanzado de overfitting con más indicadores"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))

        # 1. Gap de Overfitting (Accuracy)
        ax1 = axes[0, 0]
        ax1.bar(
            df["Modelo"],
            df["Overfitting_Accuracy"],
            color=[
                "red" if x > 0.1 else "orange" if x > 0.05 else "green"
                for x in df["Overfitting_Accuracy"]
            ],
        )
        ax1.set_title("Gap Overfitting (Train - Val Accuracy)", fontweight="bold")
        ax1.set_ylabel("Diferencia")
        ax1.axhline(y=0.1, color="r", linestyle="--", alpha=0.7, label="Crítico (0.1)")
        ax1.axhline(
            y=0.05, color="orange", linestyle="--", alpha=0.7, label="Moderado (0.05)"
        )
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Gap de Overfitting (F1)
        ax2 = axes[0, 1]
        ax2.bar(
            df["Modelo"],
            df["Overfitting_F1"],
            color=[
                "red" if x > 0.1 else "orange" if x > 0.05 else "green"
                for x in df["Overfitting_F1"]
            ],
        )
        ax2.set_title("Gap Overfitting (Train - Val F1)", fontweight="bold")
        ax2.axhline(y=0.1, color="r", linestyle="--", alpha=0.7)
        ax2.axhline(y=0.05, color="orange", linestyle="--", alpha=0.7)
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. Gap de Overfitting (AUC)
        ax3 = axes[0, 2]
        ax3.bar(
            df["Modelo"],
            df["Overfitting_AUC"],
            color=[
                "red" if x > 0.08 else "orange" if x > 0.04 else "green"
                for x in df["Overfitting_AUC"]
            ],
        )
        ax3.set_title("Gap Overfitting (Train - Val AUC)", fontweight="bold")
        ax3.axhline(y=0.08, color="r", linestyle="--", alpha=0.7)
        ax3.axhline(y=0.04, color="orange", linestyle="--", alpha=0.7)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Rendimiento Promedio
        ax4 = axes[1, 0]
        ax4.bar(
            df["Modelo"],
            df["Average_Performance"],
            color=[
                "red" if x < 0.7 else "yellow" if x < 0.8 else "green"
                for x in df["Average_Performance"]
            ],
        )
        ax4.set_title("Rendimiento Promedio", fontweight="bold")
        ax4.axhline(y=0.7, color="r", linestyle="--", alpha=0.7, label="Bajo (0.7)")
        ax4.axhline(
            y=0.8, color="orange", linestyle="--", alpha=0.7, label="Bueno (0.8)"
        )
        ax4.legend()
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        # 5. Gap de Generalización
        ax5 = axes[1, 1]
        ax5.bar(
            df["Modelo"],
            df["Generalization_Gap"],
            color=["red" if x > 0.05 else "green" for x in df["Generalization_Gap"]],
        )
        ax5.set_title("Gap de Generalización (|Val - Test|)", fontweight="bold")
        ax5.axhline(
            y=0.05, color="orange", linestyle="--", alpha=0.7, label="Umbral (0.05)"
        )
        ax5.legend()
        ax5.tick_params(axis="x", rotation=45)
        ax5.grid(True, alpha=0.3)

        # 6. Estabilidad
        ax6 = axes[1, 2]
        ax6.bar(
            df["Modelo"],
            df["Stability_Score"],
            color=[
                "green" if x > 0.95 else "yellow" if x > 0.9 else "orange"
                for x in df["Stability_Score"]
            ],
        )
        ax6.set_title("Puntuación de Estabilidad", fontweight="bold")
        ax6.axhline(y=0.95, color="g", linestyle="--", alpha=0.7, label="Excelente")
        ax6.axhline(y=0.9, color="orange", linestyle="--", alpha=0.7, label="Bueno")
        ax6.legend()
        ax6.tick_params(axis="x", rotation=45)
        ax6.grid(True, alpha=0.3)

        # 7. Coeficiente de Variación
        ax7 = axes[2, 0]
        ax7.bar(
            df["Modelo"],
            df["CV_Performance"],
            color=[
                "green" if x < 0.05 else "yellow" if x < 0.1 else "red"
                for x in df["CV_Performance"]
            ],
        )
        ax7.set_title("Coeficiente de Variación", fontweight="bold")
        ax7.set_ylabel("CV (menor es mejor)")
        ax7.tick_params(axis="x", rotation=45)
        ax7.grid(True, alpha=0.3)

        # 8. Eficiencia (Accuracy/Tiempo)
        ax8 = axes[2, 1]
        ax8.bar(df["Modelo"], df["Efficiency_Score"])
        ax8.set_title("Eficiencia (Accuracy/Tiempo)", fontweight="bold")
        ax8.set_ylabel("Score de Eficiencia")
        ax8.tick_params(axis="x", rotation=45)
        ax8.grid(True, alpha=0.3)

        # 9. Tiempo Total de Inferencia
        ax9 = axes[2, 2]
        ax9.bar(
            df["Modelo"],
            df["Total_Time"],
            color=[
                "green" if x < df["Total_Time"].median() else "orange"
                for x in df["Total_Time"]
            ],
        )
        ax9.set_title("Tiempo Total de Inferencia", fontweight="bold")
        ax9.set_ylabel("Tiempo (segundos)")
        ax9.tick_params(axis="x", rotation=45)
        ax9.grid(True, alpha=0.3)

        plt.suptitle(
            "Análisis Avanzado de Overfitting y Robustez",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join(
                    "figures", "advanced_overfitting_analysis_transformers.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_problem_classification_enhanced(self, save_fig=False):
        """Clasificación mejorada de problemas de fitting"""
        df = pd.DataFrame(self.results)

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Distribución de problemas
        ax1 = fig.add_subplot(gs[0, 0])
        problem_counts = df["Problem_Type"].value_counts()
        colors_map = {
            "Good_Fit": "#2ecc71",
            "Slight_Overfitting": "#f39c12",
            "Overfitting": "#e74c3c",
            "Severe_Overfitting": "#c0392b",
            "Underfitting": "#9b59b6",
            "Severe_Underfitting": "#8e44ad",
            "Poor_Generalization": "#e67e22",
        }
        colors = [colors_map.get(label, "gray") for label in problem_counts.index]
        ax1.pie(
            problem_counts.values,
            labels=problem_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax1.set_title("Distribución de Problemas de Fitting", fontweight="bold")

        # 2. Mapa Overfitting vs Performance
        ax2 = fig.add_subplot(gs[0, 1])
        scatter_colors = [
            colors_map.get(problem, "gray") for problem in df["Problem_Type"]
        ]
        ax2.scatter(
            df["Overfitting_Accuracy"],
            df["Average_Performance"],
            c=scatter_colors,
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
        )

        for i, model in enumerate(df["Modelo"]):
            ax2.annotate(
                model,
                (df.iloc[i]["Overfitting_Accuracy"], df.iloc[i]["Average_Performance"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax2.set_xlabel("Gap de Overfitting (Train - Val)", fontweight="bold")
        ax2.set_ylabel("Rendimiento Promedio", fontweight="bold")
        ax2.set_title("Mapa de Problemas de Fitting", fontweight="bold")
        ax2.axhline(
            y=0.7, color="r", linestyle="--", alpha=0.5, label="Umbral Underfitting"
        )
        ax2.axvline(
            x=0.05,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Overfitting Ligero",
        )
        ax2.axvline(
            x=0.1, color="r", linestyle="--", alpha=0.5, label="Overfitting Severo"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Nivel de Confianza
        ax3 = fig.add_subplot(gs[0, 2])
        confidence_counts = df["Confidence_Level"].value_counts()
        conf_colors = {
            "Muy_Alto": "#2ecc71",
            "Alto": "#3498db",
            "Medio": "#f39c12",
            "Bajo": "#e74c3c",
        }
        conf_plot_colors = [
            conf_colors.get(label, "gray") for label in confidence_counts.index
        ]
        ax3.bar(
            range(len(confidence_counts)),
            confidence_counts.values,
            color=conf_plot_colors,
            alpha=0.8,
        )
        ax3.set_xticks(range(len(confidence_counts)))
        ax3.set_xticklabels(confidence_counts.index, rotation=45)
        ax3.set_title("Distribución por Nivel de Confianza", fontweight="bold")
        ax3.set_ylabel("Cantidad de Modelos")
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Heatmap de Correlaciones
        ax4 = fig.add_subplot(gs[1, :])
        correlation_cols = [
            "Overfitting_Accuracy",
            "Average_Performance",
            "Stability_Score",
            "Generalization_Gap",
            "Efficiency_Score",
            "CV_Performance",
        ]
        corr_matrix = df[correlation_cols].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax4,
            cbar_kws={"shrink": 0.8},
        )
        ax4.set_title(
            "Matriz de Correlación entre Indicadores", fontweight="bold", pad=20
        )

        plt.suptitle(
            "Análisis Exhaustivo de Clasificación de Modelos",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        if save_fig:
            plt.savefig(
                os.path.join(
                    "figures", "problem_classification_enhanced_transformers.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_performance_radar(self, save_fig=False):
        """Gráfico radar comparativo de rendimiento"""
        df = pd.DataFrame(self.results)

        # Seleccionar métricas para el radar
        metrics = [
            "Test_Accuracy",
            "Test_F1",
            "Test_Precision",
            "Test_Recall",
            "Test_ROC_AUC",
            "Test_MCC",
        ]

        n_models = len(df)
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

        for idx, row in df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=row["Modelo"],
                color=colors[idx],
                alpha=0.7,
            )
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("Test_", "") for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title(
            "Comparación Multidimensional de Rendimiento (Test Set)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        if save_fig:
            plt.savefig(
                os.path.join("figures", "performance_radar_tranformers.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_error_analysis(self, save_fig=False):
        """Análisis detallado de errores"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Tasa de Falsos Positivos
        ax1 = axes[0, 0]
        ax1.bar(df["Modelo"], df["Test_FPR"], color="#e74c3c", alpha=0.7)
        ax1.set_title("Tasa de Falsos Positivos (FPR)", fontweight="bold")
        ax1.set_ylabel("FPR")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Tasa de Falsos Negativos
        ax2 = axes[0, 1]
        ax2.bar(df["Modelo"], df["Test_FNR"], color="#e67e22", alpha=0.7)
        ax2.set_title("Tasa de Falsos Negativos (FNR)", fontweight="bold")
        ax2.set_ylabel("FNR")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Especificidad
        ax3 = axes[0, 2]
        ax3.bar(df["Modelo"], df["Test_Specificity"], color="#3498db", alpha=0.7)
        ax3.set_title("Especificidad (True Negative Rate)", fontweight="bold")
        ax3.set_ylabel("Especificidad")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Valor Predictivo Negativo
        ax4 = axes[1, 0]
        ax4.bar(df["Modelo"], df["Test_NPV"], color="#9b59b6", alpha=0.7)
        ax4.set_title("Valor Predictivo Negativo (NPV)", fontweight="bold")
        ax4.set_ylabel("NPV")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3, axis="y")

        # 5. Log Loss
        ax5 = axes[1, 1]
        ax5.bar(df["Modelo"], df["Test_Log_Loss"], color="#e74c3c", alpha=0.7)
        ax5.set_title("Log Loss (menor es mejor)", fontweight="bold")
        ax5.set_ylabel("Log Loss")
        ax5.tick_params(axis="x", rotation=45)
        ax5.grid(True, alpha=0.3, axis="y")

        # 6. Balance Precision-Recall
        ax6 = axes[1, 2]
        x = np.arange(len(df))
        width = 0.35
        ax6.bar(
            x - width / 2, df["Test_Precision"], width, label="Precision", alpha=0.8
        )
        ax6.bar(x + width / 2, df["Test_Recall"], width, label="Recall", alpha=0.8)
        ax6.set_title("Balance Precision-Recall", fontweight="bold")
        ax6.set_ylabel("Score")
        ax6.set_xticks(x)
        ax6.set_xticklabels(df["Modelo"], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Análisis Detallado de Errores y Métricas Complementarias",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join("figures", "error_analysis_transformers.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_ranking_comparison(self, save_fig=False):
        """Ranking de modelos por diferentes criterios"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        criteria = [
            ("Test_Accuracy", "Accuracy en Test"),
            ("Test_F1", "F1-Score en Test"),
            ("Stability_Score", "Estabilidad"),
            ("Efficiency_Score", "Eficiencia"),
        ]

        for idx, (criterion, title) in enumerate(criteria):
            ax = axes[idx // 2, idx % 2]

            df_sorted = df.sort_values(by=criterion, ascending=False)
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))

            bars = ax.barh(df_sorted["Modelo"], df_sorted[criterion], color=colors)

            # Añadir valores en las barras
            for i, (bar, value) in enumerate(zip(bars, df_sorted[criterion])):
                ax.text(
                    value + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_xlabel(criterion.replace("_", " "), fontweight="bold")
            ax.set_title(f"Ranking por {title}", fontweight="bold", fontsize=12)
            ax.grid(True, alpha=0.3, axis="x")
            ax.invert_yaxis()

        plt.suptitle(
            "Rankings de Modelos por Diferentes Criterios",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join("figures", "ranking_comparison_transformers.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_train_val_test_trends(self, save_fig=False):
        """Tendencias de rendimiento entre conjuntos"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        metrics = ["Accuracy", "F1", "Precision", "Recall"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            for i, row in df.iterrows():
                train_val_test = [
                    row[f"Train_{metric}"],
                    row[f"Validation_{metric}"],
                    row[f"Test_{metric}"],
                ]

                ax.plot(
                    ["Train", "Validation", "Test"],
                    train_val_test,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=row["Modelo"],
                    alpha=0.7,
                )

            ax.set_title(f"Tendencia de {metric}", fontweight="bold", fontsize=12)
            ax.set_ylabel(metric)
            ax.set_xlabel("Conjunto de Datos")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

        plt.suptitle(
            "Tendencias de Rendimiento: Train → Validation → Test",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join("figures", "train_val_test_trends_transformers.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_statistical_significance(self, save_fig=False):
        """Análisis de significancia estadística entre modelos"""
        if len(self.predictions) < 2:
            print("⚠️  Se necesitan al menos 2 modelos para análisis estadístico")
            return

        df = pd.DataFrame(self.results)
        models = df["Modelo"].tolist()
        n_models = len(models)

        # Crear matriz de p-values usando McNemar's test
        p_value_matrix = np.ones((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_i = models[i]
                model_j = models[j]

                if model_i in self.predictions and model_j in self.predictions:
                    pred_i = self.predictions[model_i]["y_pred"]
                    pred_j = self.predictions[model_j]["y_pred"]

                    # Tabla de contingencia para McNemar
                    n01 = np.sum((pred_i == 0) & (pred_j == 1))
                    n10 = np.sum((pred_i == 1) & (pred_j == 0))

                    if (n01 + n10) > 0:
                        chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
                        p_value = 1 - stats.chi2.cdf(chi2, 1)
                        p_value_matrix[i, j] = p_value
                        p_value_matrix[j, i] = p_value

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            p_value_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=models,
            yticklabels=models,
            ax=ax,
            vmin=0,
            vmax=0.1,
            center=0.05,
            square=True,
            cbar_kws={"label": "p-value"},
        )

        ax.set_title(
            "Matriz de Significancia Estadística (McNemar's Test)\n"
            + "p < 0.05 = diferencia significativa",
            fontweight="bold",
            fontsize=13,
            pad=15,
        )

        plt.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join("figures", "statistical_significance_transformers.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def _save_report_to_file(self, df):
        """Guarda el reporte en archivo CSV y texto"""
        # Guardar DataFrame completo
        df.to_csv("reports/model_evaluation_results_transformers.csv", index=False)
        print("💾 Resultados guardados en: reports/model_evaluation_results.csv")

        # Crear resumen en texto
        with open("reports/evaluation_summary.txt", "w", encoding="utf-8") as f:
            f.write("RESUMEN DE EVALUACIÓN DE MODELOS\n")
            f.write("=" * 80 + "\n\n")

            for _, row in df.iterrows():
                f.write(f"Modelo: {row['Modelo']}\n")
                f.write(f"  Test Accuracy: {row['Test_Accuracy']:.4f}\n")
                f.write(f"  Test F1: {row['Test_F1']:.4f}\n")
                f.write(f"  Problema: {row['Problem_Type']}\n")
                f.write(f"  Confianza: {row['Confidence_Level']}\n")
                f.write("-" * 80 + "\n")

        print("📄 Resumen guardado en: reports/evaluation_summary_transformers.txt")


def main():
    """Función principal optimizada con evaluación paralela"""
    print("=" * 100)
    print("🚀 SISTEMA AVANZADO DE EVALUACIÓN DE MODELOS")
    print("=" * 100)

    # Crear directorios si no existen
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\n🔄 Cargando datasets...")
    train_df, valid_df, test_df = load_datasets()

    x_train = train_df["texto"].astype(str).values
    y_train = train_df["clase"].astype(int).values
    x_val = valid_df["texto"].astype(str).values
    y_val = valid_df["clase"].astype(int).values
    x_test = test_df["texto"].astype(str).values
    y_test = test_df["clase"].astype(int).values

    print(f"   ✅ Train: {len(x_train)} muestras")
    print(f"   ✅ Validation: {len(x_val)} muestras")
    print(f"   ✅ Test: {len(x_test)} muestras")

    evaluator = AdvancedModelEvaluator(x_train, y_train, x_val, y_val, x_test, y_test)

    models_dir = "models"

    # Definir modelos
    torch_models = {
        "MBERT": {
            "path": os.path.join(models_dir, "mbert_pytorch_best_model.pt"),
            "tokenizer": "bert-base-multilingual-cased",
        },
        "BETO": {
            "path": os.path.join(models_dir, "beto_pytorch_best_model.pt"),
            "tokenizer": "dccuchile/bert-base-spanish-wwm-cased",
        },
    }

    # Evaluar modelos
    print("\n" + "=" * 100)
    print("🔍 EVALUANDO MODELOS...")
    print("=" * 100 + "\n")

    # Pausa ligera para liberar CPU
    time.sleep(2)
    clear_memory()

    # Evaluar pytorch models
    for name, info in torch_models.items():
        path = info["path"]
        tokenizer = info["tokenizer"]
        if os.path.exists(path):
            evaluator.evaluate_torch_model_complete(
                name,
                path,
                tokenizer,
            )
        else:
            print(f"⚠️ Modelo no encontrado: {path}")

    # Generar reporte completo
    print("\n" + "=" * 100)
    print("📊 GENERANDO REPORTE EXHAUSTIVO...")
    print("=" * 100)

    df_results = evaluator.generate_comprehensive_report()
    df_results.to_csv("reports/transformer_results_summary.csv", index=False)

    # Interpretación avanzada de modelos
    print("\n" + "=" * 120)
    print("🧠 INTERPRETANDO RESULTADOS Y PROPÓSITOS DE LOS MODELOS")
    print("=" * 120)
    evaluator.interpret_model_purposes()

    # Generar todas las visualizaciones
    print("\n" + "=" * 100)
    print("📈 GENERANDO VISUALIZACIONES AVANZADAS...")
    print("=" * 100 + "\n")

    visualizations = [
        ("Comparación Exhaustiva", evaluator.plot_comprehensive_comparison),
        ("Matrices de Confusión", evaluator.plot_confusion_matrices),
        (
            "Análisis Avanzado de Overfitting",
            evaluator.plot_advanced_overfitting_analysis,
        ),
        ("Clasificación de Problemas", evaluator.plot_problem_classification_enhanced),
        ("Radar de Rendimiento", evaluator.plot_performance_radar),
        ("Análisis de Errores", evaluator.plot_error_analysis),
        ("Rankings Comparativos", evaluator.plot_ranking_comparison),
        ("Tendencias Train/Val/Test", evaluator.plot_train_val_test_trends),
        ("Significancia Estadística", evaluator.plot_statistical_significance),
    ]

    for viz_name, viz_function in visualizations:
        try:
            print(f"   📊 Generando: {viz_name}...")
            viz_function(save_fig=True)
            print(f"   ✅ {viz_name} completado")
        except Exception as e:
            print(f"   ⚠️  Error en {viz_name}: {str(e)}")

    print("\n" + "=" * 100)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("=" * 100)
    print("\n📁 Archivos generados:")
    print("   • Figuras guardadas en: ./figures/")
    print("   • Reportes guardados en: ./reports/")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
