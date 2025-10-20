import os
import gc
import time
import torch
import psutil
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch_directml_native
import matplotlib.pyplot as plt


from scipy import stats
from keras.models import load_model
from transformers import AutoTokenizer
from keras.layers import TextVectorization
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

# ===============================
# CONFIGURACIONES DE RENDIMIENTO
# ===============================
num_threads = max(4, psutil.cpu_count(logical=True))
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

print(f"🧩 CPU optimizado: usando {num_threads} threads paralelos")


# --- Función auxiliar para estimar RAM disponible ---
def available_memory_gb():
    return psutil.virtual_memory().available / (1024**3)


# --- Liberar todo lo posible entre frameworks ---
def clear_memory(full=True):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if full:
        tf.keras.backend.clear_session()
    gc.collect()


def to_text_list(batch):
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

    def evaluate_sklearn_model_complete(self, name, model_path):
        """Evalúa modelos de scikit-learn con métricas extendidas"""
        print(f"⏳ Evaluando {name}...")
        model = joblib.load(model_path)

        datasets = {
            "Train": (self.x_train, self.y_train),
            "Validation": (self.x_val, self.y_val),
            "Test": (self.x_test, self.y_test),
        }

        model_results = {"Modelo": name, "Tipo": "sklearn"}
        cm_dict = {}

        for dataset_name, (X, y) in datasets.items():
            start_time = time.time()
            y_pred = model.predict(X)
            prediction_time = time.time() - start_time

            try:
                y_proba = model.predict_proba(X)[:, 1]
            except Exception:
                y_proba = y_pred.astype(float)

            metrics = self._calculate_extended_metrics(
                y, y_pred, y_proba, prediction_time
            )

            for metric, value in metrics.items():
                model_results[f"{dataset_name}_{metric}"] = value

            cm_dict[dataset_name] = confusion_matrix(y, y_pred)

            # Guardar predicciones para análisis posterior
            if dataset_name == "Test":
                self.predictions[name] = {
                    "y_true": y,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                }

        del X, y, y_pred, y_proba
        gc.collect()

        self.confusion_matrices[name] = cm_dict

        # Indicadores avanzados
        advanced_indicators = self._calculate_advanced_indicators(model_results)
        model_results.update(advanced_indicators)

        self.results.append(model_results)
        print(f"✅ {name} evaluado")
        clear_memory()
        return model

    @staticmethod
    def prepare_vectorizer(texts, max_tokens=30000, output_seq_len=200):
        """Prepara vectorizador de texto optimizado"""
        vectorizer = TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=output_seq_len,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
        )
        vectorizer.adapt(texts)
        return vectorizer

    def evaluate_keras_model_complete(
        self, name, model_path, max_tokens=30000, seq_len=200
    ):
        """Evalúa modelos de Keras con métricas extendidas"""
        print(f"⏳ Evaluando {name}...")
        model = load_model(model_path)

        vectorizer = self.prepare_vectorizer(
            self.x_train, max_tokens=max_tokens, output_seq_len=seq_len
        )

        datasets = {
            "Train": (self.x_train, self.y_train),
            "Validation": (self.x_val, self.y_val),
            "Test": (self.x_test, self.y_test),
        }

        model_results = {"Modelo": name, "Tipo": "keras"}
        cm_dict = {}

        for dataset_name, (X, y) in datasets.items():
            X_vec = vectorizer(np.array(X)).numpy()

            start_time = time.time()
            # Lote dinámico para no saturar RAM/GPU
            adaptive_batch = min(
                256, max(32, int(len(X_vec) / (available_memory_gb() * 50 + 1)))
            )
            y_proba = model.predict(
                X_vec, verbose=0, batch_size=adaptive_batch
            ).flatten()

            y_pred = (y_proba > 0.5).astype(int)
            prediction_time = time.time() - start_time

            metrics = self._calculate_extended_metrics(
                y, y_pred, y_proba, prediction_time
            )

            for metric, value in metrics.items():
                model_results[f"{dataset_name}_{metric}"] = value

            cm_dict[dataset_name] = confusion_matrix(y, y_pred)

            if dataset_name == "Test":
                self.predictions[name] = {
                    "y_true": y,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.confusion_matrices[name] = cm_dict

        advanced_indicators = self._calculate_advanced_indicators(model_results)
        model_results.update(advanced_indicators)

        self.results.append(model_results)
        print(f"✅ {name} evaluado")
        clear_memory()
        return model

    @staticmethod
    def get_best_device(run_benchmark=True):
        """
        Compara todos los dispositivos disponibles y selecciona el mejor.
        Si run_benchmark=True, hace pruebas de velocidad reales.
        Retorna: (device_string, device_object, memoria_disponible_gb)
        """
        devices_info = []

        # ====== CONFIGURACIÓN CRÍTICA PARA CPU ======
        cpu_threads = psutil.cpu_count(logical=True)  # Usa hilos lógicos (4 en tu caso)
        cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_threads

        print(f"💻 CPU detectado: {cpu_cores_physical} núcleos físicos, {cpu_threads} hilos lógicos")

        # Configurar threads de PyTorch para máximo rendimiento
        # Configurar threads de PyTorch para máximo rendimiento
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)

        # Configurar OpenMP/MKL si está disponible
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_threads)

        print(f"🔧 CPU configurado para usar {cpu_threads} hilos (threads)")
        print(f"   - torch.get_num_threads(): {torch.get_num_threads()}")
        print(f"   - torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

        # --- 1. Revisar CUDA (NVIDIA GPUs) ---
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)

                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_mem = (props.total_memory - torch.cuda.memory_allocated(i)) / (
                    1024**3
                )

                tflops_estimate = (
                    props.multi_processor_count * props.major * 100
                ) / 1000

                devices_info.append(
                    {
                        "type": "cuda",
                        "id": i,
                        "name": props.name,
                        "device_str": f"cuda:{i}",
                        "device_obj": torch.device(f"cuda:{i}"),
                        "total_memory_gb": total_mem,
                        "free_memory_gb": free_mem,
                        "compute_capability": props.major + props.minor * 0.1,
                        "tflops_estimate": tflops_estimate,
                        "cores": props.multi_processor_count * 128,
                        "score": 0,
                    }
                )
                print(f"🎮 GPU {i}: {props.name}")
                print(f"   Memoria: {free_mem:.2f}GB libre / {total_mem:.2f}GB total")
                print(
                    f"   Compute Capability: {props.major}.{props.minor} | ~{tflops_estimate:.1f} TFLOPS"
                )

        # --- 2. Revisar DirectML (AMD/Intel GPUs en Windows) ---
        try:
            dml_device = torch_directml_native.device()

            devices_info.append(
                {
                    "type": "dml",
                    "id": 0,
                    "name": "DirectML Device",
                    "device_str": "dml",
                    "device_obj": dml_device,
                    "total_memory_gb": 4.0,
                    "free_memory_gb": 3.0,
                    "compute_capability": 5.0,
                    "tflops_estimate": 2.0,
                    "cores": 1024,
                    "score": 0,
                }
            )
            print("🎮 DirectML Device detectado")
            print("   Memoria estimada: 3GB | ~2 TFLOPS (estimado)")
        except (ImportError, Exception):
            pass

        # --- 3. CPU siempre disponible ---
        cpu_ram = psutil.virtual_memory().available / (1024**3)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_ghz = cpu_freq.max / 1000 if cpu_freq else 3.0

        cpu_gflops = cpu_threads * cpu_freq_ghz * 32

        devices_info.append(
            {
                "type": "cpu",
                "id": 0,
                "name": f"CPU ({cpu_cores_physical} cores, {cpu_threads} threads @ {cpu_freq_ghz:.1f}GHz)",
                "device_str": "cpu",
                "device_obj": torch.device("cpu"),
                "total_memory_gb": cpu_ram,
                "free_memory_gb": cpu_ram,
                "compute_capability": 1.0,
                "tflops_estimate": cpu_gflops / 1000,
                "cores": cpu_threads,
                "num_workers": max(2, cpu_threads // 2),
                "score": 0,
            }
        )
        print(f"💻 CPU: {cpu_threads} threads @ {cpu_freq_ghz:.1f}GHz")
        print(f"   RAM disponible: {cpu_ram:.2f}GB | ~{cpu_gflops:.0f} GFLOPS")

        # --- 4. Benchmark real si se solicita ---
        if run_benchmark and len(devices_info) > 1:
            print("\n🏃 Ejecutando benchmark rápido en cada dispositivo...")

            for dev_info in devices_info:
                try:
                    device = dev_info["device_obj"]

                    size = 1024
                    torch.manual_seed(42)
                    a = torch.randn(size, size)
                    b = torch.randn(size, size)

                    a_dev = a.to(device)
                    b_dev = b.to(device)

                    _ = torch.matmul(a_dev, b_dev)

                    if dev_info["type"] == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    for _ in range(10):
                        _ = torch.matmul(a_dev, b_dev)

                    if dev_info["type"] == "cuda":
                        torch.cuda.synchronize()

                    elapsed = time.perf_counter() - start
                    ops_per_sec = (10 * 2 * size**3) / elapsed / 1e9

                    dev_info["benchmark_gflops"] = ops_per_sec
                    print(
                        f"   {dev_info['device_str']}: {ops_per_sec:.1f} GFLOPS (real)"
                    )

                    del a_dev, b_dev
                    if dev_info["type"] == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"   {dev_info['device_str']}: Benchmark falló ({e})")
                    dev_info["benchmark_gflops"] = 0

        # --- 5. Calcular scores finales ---
        print("\n📊 Calculando scores finales...")

        for dev_info in devices_info:
            memory_score = dev_info["free_memory_gb"] * 100

            if run_benchmark and "benchmark_gflops" in dev_info:
                compute_score = dev_info["benchmark_gflops"] * 5
            else:
                compute_score = dev_info["tflops_estimate"] * 1000 * 3

            type_bonus = {
                "cuda": 500,
                "dml": 200,
                "cpu": 0,
            }

            dev_info["score"] = (
                memory_score + compute_score + type_bonus[dev_info["type"]]
            )

            print(
                f"   {dev_info['device_str']}: Score = {dev_info['score']:.0f} "
                + f"(mem:{memory_score:.0f} + compute:{compute_score:.0f} + bonus:{type_bonus[dev_info['type']]})"
            )

        # --- 6. Seleccionar el mejor ---
        if not devices_info:
            cpu_threads = psutil.cpu_count(logical=True)
            return "cpu", torch.device("cpu"), cpu_ram, max(2, cpu_threads // 2)

        best = max(devices_info, key=lambda x: x["score"])

        print(f"\n✅ MEJOR DISPOSITIVO: {best['name']} ({best['device_str']})")
        print(f"   Memoria disponible: {best['free_memory_gb']:.2f}GB")
        if "benchmark_gflops" in best:
            print(f"   Rendimiento medido: {best['benchmark_gflops']:.1f} GFLOPS")

        num_workers = best.get("num_workers", 0)
        return best["device_str"], best["device_obj"], best["free_memory_gb"], num_workers

    @staticmethod
    def calculate_optimal_batch_size(
        device_str, available_memory_gb, model_type="bert"
    ):
        """
        Calcula el batch size óptimo según el dispositivo y memoria disponible.
        """
        if model_type.lower() in ["mbert", "bert", "beto"]:
            memory_per_sample_gb = 0.05
        else:
            memory_per_sample_gb = 0.03

        safety_factors = {
            "cuda": 0.7,
            "dml": 0.4,
            "cpu": 0.8,
        }

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
        """
        Evalúa modelos de Transformers con selección automática del mejor dispositivo.
        """
        print(f"\n{'=' * 60}")
        print(f"⏳ Evaluando {name}...")
        print(f"{'=' * 60}")

        # --- Selección inteligente de dispositivo ---
        if device:
            device_str = device
            device_obj = torch.device(device)
            available_mem = 4.0
            num_workers = 4 
            print(f"🖥️ Usando dispositivo especificado: {device_str}")
        else:
            device_str, device_obj, available_mem, num_workers = self.get_best_device(
                run_benchmark=True
            )

        # --- Calcular batch size óptimo ---
        if batch_size is None:
            batch_size = self.calculate_optimal_batch_size(
                device_str,
                available_mem,
                model_type="mbert" if "mbert" in name.lower() else "bert",
            )
        else:
            print(f"📊 Usando batch size especificado: {batch_size}")

        # --- Cargar tokenizer ---
        print(f"📚 Cargando tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # --- Cargar modelo ---
        print(f"🤖 Cargando modelo desde: {model_path}")
        if "beto" in name.lower():
            model = BETOClassifier(tokenizer_name)
        else:
            model = mBERTClassifier(tokenizer_name)

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # --- Mover modelo al dispositivo seleccionado ---
        print(f"🚀 Moviendo modelo a {device_str}...")
        model.to(device_obj)
        model.eval()

        # --- Configuración de optimización ---
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False

        if device_str == "dml" or available_mem < 4:
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

        # --- Evaluación por conjunto ---
        for dataset_name, (X, y_true) in datasets.items():
            print(f"\n📂 Procesando conjunto: {dataset_name} ({len(X)} muestras)")
            y_pred_list, y_proba_list = [], []
            start_time = time.time()

            total_batches = (len(X) + batch_size - 1) // batch_size
            current_batch_size = batch_size

            batch_idx = 0
            while batch_idx < len(X):
                try:
                    current_batch = (batch_idx // current_batch_size) + 1
                    if current_batch % 10 == 0:
                        print(f"   Batch {current_batch}/{total_batches}...", end="\r")

                    texts_batch = to_text_list(
                        X[batch_idx : batch_idx + current_batch_size]
                    )

                    inputs = tokenizer(
                        texts_batch,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    )

                    inputs = {k: v.to(device_obj) for k, v in inputs.items()}

                    # Verificar si el modelo acepta token_type_ids
                    if "token_type_ids" in inputs:
                        try:
                            with torch.no_grad():
                                _ = model(
                                    input_ids=inputs["input_ids"][:1],
                                    attention_mask=inputs["attention_mask"][:1],
                                )
                            del inputs["token_type_ids"]
                        except TypeError:
                            pass

                    # --- Inferencia ---
                    with torch.no_grad():
                        outputs = model(**inputs)

                        logits = (
                            outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        )

                        logits = logits.view(logits.size(0), -1)

                        if logits.shape[1] == 1:
                            probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        else:
                            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                        preds = (probs > 0.5).astype(int)

                    y_pred_list.extend(preds)
                    y_proba_list.extend(probs)

                    del inputs, outputs, logits
                    batch_idx += current_batch_size

                except RuntimeError as e:
                    if (
                        "out of memory" in str(e).lower()
                        or "could not allocate" in str(e).lower()
                    ):
                        print(
                            f"\n⚠️ Error de memoria en batch {current_batch}, reduciendo batch size"
                        )

                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"   Nuevo batch_size: {current_batch_size}")

                        gc.collect()
                        if device_str.startswith("cuda"):
                            torch.cuda.empty_cache()

                        # Continuar con el mismo batch_idx (reintentar)
                        continue
                    else:
                        raise e

            print(
                f"\n   ✅ {dataset_name} completado en {time.time() - start_time:.2f}s"
            )

            prediction_time = time.time() - start_time

            # --- Calcular métricas ---
            y_pred = np.array(y_pred_list)
            y_proba = np.array(y_proba_list)
            y_true = np.array(y_true)

            metrics = self._calculate_extended_metrics(
                y_true, y_pred, y_proba, prediction_time
            )
            for metric, value in metrics.items():
                model_results[f"{dataset_name}_{metric}"] = value

            cm_dict[dataset_name] = confusion_matrix(y_true, y_pred)

            if dataset_name == "Test":
                self.predictions[name] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                }

        # --- Limpieza final ---
        del model
        gc.collect()
        if device_str.startswith("cuda"):
            torch.cuda.empty_cache()

        # --- Resultados finales ---
        self.confusion_matrices[name] = cm_dict
        advanced_indicators = self._calculate_advanced_indicators(model_results)
        model_results.update(advanced_indicators)

        self.results.append(model_results)
        print(f"\n✅ {name} evaluado correctamente en {device_str}")
        print(f"{'=' * 60}\n")

        return None

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
            "MCC": matthews_corrcoef(
                y_true, y_pred
            ),  # Matthews Correlation Coefficient
            "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
            "Time": prediction_time,
        }

        # Métricas de la matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["NPV"] = (
            tn / (tn + fn) if (tn + fn) > 0 else 0
        )  # Negative Predictive Value
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        return metrics

    def _calculate_advanced_indicators(self, model_results):
        """Calcula indicadores avanzados de overfitting y robustez"""
        train_acc = model_results["Train_Accuracy"]
        val_acc = model_results["Validation_Accuracy"]
        test_acc = model_results["Test_Accuracy"]

        train_f1 = model_results["Train_F1"]
        val_f1 = model_results["Validation_F1"]
        test_f1 = model_results["Test_F1"]

        # Gaps de overfitting
        overfitting_acc = train_acc - val_acc
        overfitting_f1 = train_f1 - val_f1

        # Consistencia entre conjuntos
        avg_performance = (train_acc + val_acc + test_acc) / 3
        generalization_gap = abs(val_acc - test_acc)

        # Variabilidad (robustez)
        stability_score = 1 - np.std([train_acc, val_acc, test_acc])
        cv_performance = np.std([train_acc, val_acc, test_acc]) / np.mean(
            [train_acc, val_acc, test_acc]
        )

        # Análisis de ROC AUC
        train_auc = model_results["Train_ROC_AUC"]
        val_auc = model_results["Validation_ROC_AUC"]
        test_auc = model_results["Test_ROC_AUC"]
        auc_gap = train_auc - val_auc

        # Eficiencia (tiempo)
        total_time = (
            model_results["Train_Time"]
            + model_results["Validation_Time"]
            + model_results["Test_Time"]
        )
        efficiency_score = test_acc / (total_time + 1e-6)  # Accuracy por segundo

        # Clasificación de problema
        problem_type = self._classify_fitting_problem(
            train_acc, val_acc, test_acc, overfitting_acc
        )

        # Nivel de confianza (basado en métricas)
        confidence_level = self._calculate_confidence_level(model_results)

        return {
            "Overfitting_Accuracy": overfitting_acc,
            "Overfitting_F1": overfitting_f1,
            "Overfitting_AUC": auc_gap,
            "Average_Performance": avg_performance,
            "Generalization_Gap": generalization_gap,
            "Stability_Score": stability_score,
            "CV_Performance": cv_performance,
            "Efficiency_Score": efficiency_score,
            "Total_Time": total_time,
            "Problem_Type": problem_type,
            "Confidence_Level": confidence_level,
            "Variance_Metric": np.var([train_acc, val_acc, test_acc]),
            "Test_F1": test_f1,
            "Test_ROC_AUC": test_auc,
        }

    def _classify_fitting_problem(self, train_acc, val_acc, test_acc, overfitting_gap):
        """Clasificación mejorada del tipo de fitting"""
        if overfitting_gap > 0.15:
            return "Severe_Overfitting"
        elif overfitting_gap > 0.1:
            return "Overfitting"
        elif train_acc < 0.65 and val_acc < 0.65:
            return "Severe_Underfitting"
        elif train_acc < 0.75 and val_acc < 0.75:
            return "Underfitting"
        elif overfitting_gap > 0.05:
            return "Slight_Overfitting"
        elif abs(val_acc - test_acc) > 0.05:
            return "Poor_Generalization"
        return "Good_Fit"

    def _calculate_confidence_level(self, model_results):
        """Calcula nivel de confianza del modelo"""
        test_acc = model_results["Test_Accuracy"]
        test_f1 = model_results["Test_F1"]
        test_mcc = model_results["Test_MCC"]
        test_auc = model_results["Test_ROC_AUC"]

        confidence = (test_acc + test_f1 + test_mcc + test_auc) / 4

        if confidence >= 0.85:
            return "Muy_Alto"
        elif confidence >= 0.75:
            return "Alto"
        elif confidence >= 0.65:
            return "Medio"
        else:
            return "Bajo"

    def interpret_model_purposes(self):
        """
        Analiza los resultados globales y sugiere para qué tipo de tarea
        destaca cada modelo, según métricas clave y comportamiento general.
        Muestra las 10 categorías aunque no haya un modelo ganador.
        """

        if not hasattr(self, "results") or not self.results:
            print("⚠️ No hay resultados para analizar.")
            return

        df = pd.DataFrame(self.results)
        df.columns = [c.replace(" ", "_") for c in df.columns]

        print("\n" + "=" * 120)
        print("🧩 INTERPRETACIÓN AVANZADA DE LOS MODELOS")
        print("=" * 120 + "\n")

        interpretaciones = {}

        def add_interpretacion(categoria, modelo=None, descripcion=None):
            """Agrega interpretación o marca como no detectada."""
            if modelo is not None:
                interpretaciones[categoria] = (modelo, descripcion)
            else:
                interpretaciones[categoria] = (
                    "⚠️ Sin modelo destacado",
                    "No se detectó un modelo sobresaliente en esta categoría.",
                )

        # 1️⃣ Mejor para detectar Fake News → Recall más alto
        if "Test_Recall" in df:
            mejor_fake = df.loc[df["Test_Recall"].idxmax(), "Modelo"]
            add_interpretacion(
                "Detección de Fake News",
                mejor_fake,
                "Posee el mejor Recall en test, ideal para detectar noticias falsas sin omitir casos importantes.",
            )
        else:
            add_interpretacion("Detección de Fake News")

        # 2️⃣ Mejor para detectar Real News → Precisión más alta
        if "Test_Precision" in df:
            mejor_real = df.loc[df["Test_Precision"].idxmax(), "Modelo"]
            add_interpretacion(
                "Detección de Real News",
                mejor_real,
                "Alcanza la mayor Precisión, reduciendo falsos positivos. Perfecto para validar contenido legítimo.",
            )
        else:
            add_interpretacion("Detección de Real News")

        # 3️⃣ Modelo más equilibrado entre validación y prueba
        if "Validation_F1" in df and "Test_F1" in df:
            idx = (df["Validation_F1"] - df["Test_F1"]).abs().idxmin()
            mejor_equilibrado = df.loc[idx, "Modelo"]
            add_interpretacion(
                "Modelo más equilibrado",
                mejor_equilibrado,
                "Presenta consistencia entre validación y prueba, indicando buena estabilidad y generalización.",
            )
        else:
            add_interpretacion("Modelo más equilibrado")

        # 4️⃣ Modelo más robusto (menos overfitting)
        if "Overfitting_F1" in df:
            mejor_general = df.loc[df["Overfitting_F1"].abs().idxmin(), "Modelo"]
            add_interpretacion(
                "Modelo más robusto",
                mejor_general,
                "Tiene el menor sobreajuste. Ideal para entornos con datos no vistos o cambiantes.",
            )
        else:
            add_interpretacion("Modelo más robusto")

        # 5️⃣ Modelo más rápido en predicción
        if "Test_Time" in df:
            mejor_rapido_pred = df.loc[df["Test_Time"].idxmin(), "Modelo"]
            add_interpretacion(
                "Más rápido en predicción",
                mejor_rapido_pred,
                "Tiempo de inferencia más bajo. Ideal para sistemas en tiempo real o móviles.",
            )
        else:
            add_interpretacion("Más rápido en predicción")

        # 6️⃣ Modelo más eficiente en entrenamiento
        if "Train_Time" in df:
            mejor_entrenamiento = df.loc[df["Train_Time"].idxmin(), "Modelo"]
            add_interpretacion(
                "Más eficiente en entrenamiento",
                mejor_entrenamiento,
                "Entrena en menos tiempo. Útil cuando se requieren actualizaciones frecuentes del modelo.",
            )
        else:
            add_interpretacion("Más eficiente en entrenamiento")

        # 7️⃣ Modelo con mejor AUC (discriminación)
        if "Test_ROC_AUC" in df:
            mejor_auc = df.loc[df["Test_ROC_AUC"].idxmax(), "Modelo"]
            add_interpretacion(
                "Mayor capacidad de discriminación (AUC)",
                mejor_auc,
                "Excelente balance entre sensibilidad y especificidad. Distingue claramente entre clases.",
            )
        else:
            add_interpretacion("Mayor capacidad de discriminación (AUC)")

        # 8️⃣ Modelo más consistente (menor desviación entre métricas)
        posibles = [c for c in ["Test_F1", "Test_Precision", "Test_Recall"] if c in df]
        if posibles:
            df["Consistency_Score"] = df[posibles].std(axis=1)
            idx = df["Consistency_Score"].idxmin()
            consistente = df.loc[idx, "Modelo"]
            add_interpretacion(
                "Modelo más consistente",
                consistente,
                "Mantiene equilibrio entre métricas clave (Precision, Recall, F1). Ideal para decisiones balanceadas.",
            )
        else:
            add_interpretacion("Modelo más consistente")

        # 9️⃣ Modelo con mejor rendimiento global (F1)
        if "Test_F1" in df:
            mejor_f1 = df.loc[df["Test_F1"].idxmax(), "Modelo"]
            add_interpretacion(
                "Rendimiento global (F1)",
                mejor_f1,
                "Obtiene el mayor F1, ofreciendo el mejor rendimiento global en clasificación.",
            )
        else:
            add_interpretacion("Rendimiento global (F1)")

        # 🔟 Modelo más interpretable
        if any(df["Modelo"].str.contains("Decision", case=False)):
            add_interpretacion(
                "Ideal para interpretación y explicación",
                "Decision Tree",
                "Permite entender las decisiones internas. Recomendado para auditorías o explicación de resultados.",
            )
        else:
            add_interpretacion("Ideal para interpretación y explicación")

        # ───────────────────────────────────────────────
        # 💬 Mostrar resultados
        print("🧠 Interpretaciones de modelos detectadas:\n")
        for categoria, (modelo, descripcion) in interpretaciones.items():
            print(f" • 🏷️ {categoria}:")
            print(f"    → **{modelo}** → {descripcion}\n")

        print("=" * 120)
        print("💡 Sugerencia general de uso:")
        print("   - Usa el modelo más equilibrado como predeterminado.")
        print("   - Ofrece al usuario seleccionar según su objetivo:")
        print("       • Precisión alta → Detección de Real News")
        print("       • Recall alto → Detección de Fake News")
        print("       • Estabilidad → Modelo más equilibrado")
        print("       • Velocidad → Predicción o entrenamiento rápido")
        print("       • Interpretabilidad → Árbol de decisión")
        print("=" * 120 + "\n")

        resumen = pd.DataFrame(
            [
                {"Categoría": k, "Modelo": v[0], "Descripción": v[1]}
                for k, v in interpretaciones.items()
            ]
        )
        resumen.to_csv("reports/model_purpose_summary.csv", index=False)

        return interpretaciones

    # ==================== VISUALIZACIONES ====================

    def plot_comprehensive_comparison(self, save_fig=False):
        """Comparación exhaustiva de todos los modelos"""
        df = pd.DataFrame(self.results)

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

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
            ax = fig.add_subplot(gs[idx // 3, idx % 3])

            train_data = df[f"Train_{metric}"].values
            val_data = df[f"Validation_{metric}"].values
            test_data = df[f"Test_{metric}"].values

            x = np.arange(len(df))
            width = 0.25

            ax.bar(
                x - width, train_data, width, label="Train", alpha=0.8, color="#3498db"
            )
            ax.bar(x, val_data, width, label="Validation", alpha=0.8, color="#e74c3c")
            ax.bar(
                x + width, test_data, width, label="Test", alpha=0.8, color="#2ecc71"
            )

            ax.set_title(f"{metric.replace('_', ' ')}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Modelos", fontsize=9)
            ax.set_ylabel(metric.replace("_", " "), fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(df["Modelo"], rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

        plt.suptitle(
            "Comparación Exhaustiva de Métricas - Train/Validation/Test",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        if save_fig:
            plt.savefig(
                os.path.join("figures", "comprehensive_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_confusion_matrices(self, save_fig=False):
        """Visualiza matrices de confusión de todos los modelos"""
        n_models = len(self.confusion_matrices)
        fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models))

        if n_models == 1:
            axes = axes.reshape(1, -1)

        for idx, (model_name, cm_dict) in enumerate(self.confusion_matrices.items()):
            for col_idx, (dataset, cm) in enumerate(cm_dict.items()):
                ax = axes[idx, col_idx]

                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax, square=True
                )
                ax.set_title(f"{model_name} - {dataset}", fontweight="bold")
                ax.set_xlabel("Predicción")
                ax.set_ylabel("Real")

        plt.suptitle(
            "Matrices de Confusión por Modelo y Conjunto",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join("figures", "confusion_matrices.png"),
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
                os.path.join("figures", "advanced_overfitting_analysis.png"),
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
                os.path.join("figures", "problem_classification_enhanced.png"),
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
                os.path.join("figures", "performance_radar.png"),
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
                os.path.join("figures", "error_analysis.png"),
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
                os.path.join("figures", "ranking_comparison.png"),
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
                os.path.join("figures", "train_val_test_trends.png"),
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
                os.path.join("figures", "statistical_significance.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def generate_comprehensive_report(self):
        """Genera reporte exhaustivo y profesional"""
        df = pd.DataFrame(self.results)

        print("\n" + "=" * 100)
        print("📊 REPORTE EXHAUSTIVO DE EVALUACIÓN DE MODELOS DE MACHINE LEARNING")
        print("=" * 100)

        # 1. RESUMEN EJECUTIVO
        print("\n" + "─" * 100)
        print("📋 RESUMEN EJECUTIVO")
        print("─" * 100)

        best_model = df.loc[df["Test_Accuracy"].idxmax()]
        most_stable = df.loc[df["Stability_Score"].idxmax()]
        most_efficient = df.loc[df["Efficiency_Score"].idxmax()]

        print(
            f"\n🏆 Mejor Modelo (Test Accuracy): {best_model['Modelo']} ({best_model['Test_Accuracy']:.4f})"
        )
        print(
            f"⚖️  Modelo Más Estable: {most_stable['Modelo']} (Score: {most_stable['Stability_Score']:.4f})"
        )
        print(
            f"⚡ Modelo Más Eficiente: {most_efficient['Modelo']} (Score: {most_efficient['Efficiency_Score']:.4f})"
        )

        print("\n📈 Estadísticas Generales:")
        print(
            f"   • Accuracy promedio (Test): {df['Test_Accuracy'].mean():.4f} ± {df['Test_Accuracy'].std():.4f}"
        )
        print(
            f"   • F1-Score promedio (Test): {df['Test_F1'].mean():.4f} ± {df['Test_F1'].std():.4f}"
        )
        print(
            f"   • ROC-AUC promedio (Test): {df['Test_ROC_AUC'].mean():.4f} ± {df['Test_ROC_AUC'].std():.4f}"
        )
        print(
            f"   • MCC promedio (Test): {df['Test_MCC'].mean():.4f} ± {df['Test_MCC'].std():.4f}"
        )

        # 2. ANÁLISIS DETALLADO POR MODELO
        print("\n" + "─" * 100)
        print("🔍 ANÁLISIS DETALLADO POR MODELO")
        print("─" * 100)

        for idx, row in df.iterrows():
            print(f"\n{'=' * 90}")
            print(f"🤖 MODELO: {row['Modelo']} ({row['Tipo'].upper()})")
            print(f"{'=' * 90}")

            # Métricas principales
            print("\n📊 Métricas de Rendimiento:")
            print(
                f"   {'Conjunto':<15} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}"
            )
            print(f"   {'-' * 75}")
            print(
                f"   {'Train':<15} {row['Train_Accuracy']:<10.4f} {row['Train_F1']:<10.4f} {row['Train_Precision']:<10.4f} {row['Train_Recall']:<10.4f} {row['Train_ROC_AUC']:<10.4f}"
            )
            print(
                f"   {'Validation':<15} {row['Validation_Accuracy']:<10.4f} {row['Validation_F1']:<10.4f} {row['Validation_Precision']:<10.4f} {row['Validation_Recall']:<10.4f} {row['Validation_ROC_AUC']:<10.4f}"
            )
            print(
                f"   {'Test':<15} {row['Test_Accuracy']:<10.4f} {row['Test_F1']:<10.4f} {row['Test_Precision']:<10.4f} {row['Test_Recall']:<10.4f} {row['Test_ROC_AUC']:<10.4f}"
            )

            # Métricas avanzadas
            print("\n📈 Métricas Avanzadas (Test):")
            print(f"   • Matthews Correlation Coefficient (MCC): {row['Test_MCC']:.4f}")
            print(f"   • Cohen's Kappa: {row['Test_Cohen_Kappa']:.4f}")
            print(f"   • Balanced Accuracy: {row['Test_Balanced_Accuracy']:.4f}")
            print(f"   • Especificidad: {row['Test_Specificity']:.4f}")
            print(f"   • Valor Predictivo Negativo (NPV): {row['Test_NPV']:.4f}")
            print(f"   • Log Loss: {row['Test_Log_Loss']:.4f}")

            # Análisis de Overfitting/Underfitting
            print("\n⚠️  Análisis de Overfitting/Underfitting:")
            print(f"   • Tipo de Problema: {row['Problem_Type'].replace('_', ' ')}")
            print(f"   • Gap Overfitting (Accuracy): {row['Overfitting_Accuracy']:.4f}")
            print(f"   • Gap Overfitting (F1): {row['Overfitting_F1']:.4f}")
            print(f"   • Gap Overfitting (AUC): {row['Overfitting_AUC']:.4f}")
            print(
                f"   • Gap de Generalización (|Val-Test|): {row['Generalization_Gap']:.4f}"
            )

            # Interpretación del problema
            problem = row["Problem_Type"]
            if problem == "Severe_Overfitting":
                print(
                    "   🔴 CRÍTICO: Overfitting severo detectado. Reducir complejidad urgentemente."
                )
            elif problem == "Overfitting":
                print(
                    "   🟠 ADVERTENCIA: Overfitting moderado. Aplicar regularización fuerte."
                )
            elif problem == "Slight_Overfitting":
                print(
                    "   🟡 ATENCIÓN: Overfitting ligero. Considerar regularización adicional."
                )
            elif problem == "Severe_Underfitting":
                print(
                    "   🔴 CRÍTICO: Underfitting severo. Aumentar complejidad del modelo."
                )
            elif problem == "Underfitting":
                print(
                    "   🟠 ADVERTENCIA: Underfitting detectado. Modelo demasiado simple."
                )
            elif problem == "Poor_Generalization":
                print(
                    "   🟡 ATENCIÓN: Pobre generalización. Revisar distribución de datos."
                )
            else:
                print(
                    "   🟢 EXCELENTE: Modelo bien calibrado y generaliza correctamente."
                )

            # Estabilidad y Robustez
            print("\n🎯 Estabilidad y Robustez:")
            print(f"   • Stability Score: {row['Stability_Score']:.4f}")
            print(f"   • Coeficiente de Variación: {row['CV_Performance']:.4f}")
            print(f"   • Varianza entre conjuntos: {row['Variance_Metric']:.6f}")
            print(f"   • Rendimiento Promedio: {row['Average_Performance']:.4f}")
            print(
                f"   • Nivel de Confianza: {row['Confidence_Level'].replace('_', ' ')}"
            )

            # Eficiencia
            print("\n⚡ Eficiencia y Tiempo:")
            print(f"   • Tiempo Total de Inferencia: {row['Total_Time']:.4f} segundos")
            print(f"   • Tiempo Train: {row['Train_Time']:.4f}s")
            print(f"   • Tiempo Validation: {row['Validation_Time']:.4f}s")
            print(f"   • Tiempo Test: {row['Test_Time']:.4f}s")
            print(f"   • Score de Eficiencia: {row['Efficiency_Score']:.4f}")

            # Análisis de Errores
            print("\n❌ Análisis de Errores (Test):")
            print(f"   • Tasa de Falsos Positivos (FPR): {row['Test_FPR']:.4f}")
            print(f"   • Tasa de Falsos Negativos (FNR): {row['Test_FNR']:.4f}")

            # Recomendaciones específicas
            print("\n💡 Recomendaciones:")
            recommendations = []

            if row["Overfitting_Accuracy"] > 0.1:
                recommendations.append("   • Aplicar regularización L1/L2 más fuerte")
                recommendations.append(
                    "   • Reducir complejidad del modelo (menos capas/neuronas)"
                )
                recommendations.append("   • Aumentar Dropout")
                recommendations.append("   • Aplicar Early Stopping más agresivo")
            elif row["Overfitting_Accuracy"] > 0.05:
                recommendations.append("   • Ajustar hiperparámetros de regularización")
                recommendations.append("   • Implementar Data Augmentation")

            if row["Average_Performance"] < 0.7:
                recommendations.append("   • Aumentar complejidad del modelo")
                recommendations.append("   • Agregar más features relevantes")
                recommendations.append("   • Revisar ingeniería de características")

            if row["Generalization_Gap"] > 0.05:
                recommendations.append(
                    "   • Verificar distribución de datos train/val/test"
                )
                recommendations.append("   • Considerar técnicas de ensemble")

            if row["Test_FPR"] > 0.2:
                recommendations.append(
                    "   • Ajustar threshold de clasificación para reducir FP"
                )
                recommendations.append("   • Revisar balance de clases")

            if row["Total_Time"] > df["Total_Time"].median() * 2:
                recommendations.append(
                    "   • Optimizar arquitectura para mejor eficiencia"
                )
                recommendations.append("   • Considerar cuantización o pruning")

            if not recommendations:
                recommendations.append(
                    "   ✅ Modelo bien optimizado, continuar monitoreando"
                )

            for rec in recommendations:
                print(rec)

        # 3. COMPARATIVA GLOBAL
        print("\n" + "─" * 100)
        print("📊 COMPARATIVA GLOBAL")
        print("─" * 100)

        print("\n🏅 Top 3 por Métrica:")
        metrics_rank = {
            "Accuracy": "Test_Accuracy",
            "F1-Score": "Test_F1",
            "ROC-AUC": "Test_ROC_AUC",
            "MCC": "Test_MCC",
            "Estabilidad": "Stability_Score",
            "Eficiencia": "Efficiency_Score",
        }

        for metric_name, metric_col in metrics_rank.items():
            top3 = df.nlargest(3, metric_col)[["Modelo", metric_col]]
            print(f"\n   {metric_name}:")
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                print(f"      {i}. {row['Modelo']:<25} {row[metric_col]:.4f}")

        # 4. DISTRIBUCIÓN DE PROBLEMAS
        print("\n" + "─" * 100)
        print("⚠️  DISTRIBUCIÓN DE PROBLEMAS DE FITTING")
        print("─" * 100)

        problem_counts = df["Problem_Type"].value_counts()
        print(f"\n   Total de modelos evaluados: {len(df)}")
        for problem, count in problem_counts.items():
            percentage = (count / len(df)) * 100
            print(
                f"   • {problem.replace('_', ' '):<25}: {count} modelos ({percentage:.1f}%)"
            )

        # 5. RECOMENDACIONES FINALES
        print("\n" + "─" * 100)
        print("💡 RECOMENDACIONES FINALES PARA EL PROYECTO")
        print("─" * 100)

        print("\n1️⃣  MODELO RECOMENDADO PARA PRODUCCIÓN:")
        # Calcular score ponderado
        df["Production_Score"] = (
            df["Test_Accuracy"] * 0.3
            + df["Test_F1"] * 0.25
            + df["Stability_Score"] * 0.25
            + df["Efficiency_Score"] * 0.1
            + (1 - df["Overfitting_Accuracy"].clip(0, 0.2) / 0.2) * 0.1
        )
        best_production = df.loc[df["Production_Score"].idxmax()]
        print(f"   → {best_production['Modelo']}")
        print("     Razones:")
        print(f"     • Accuracy en Test: {best_production['Test_Accuracy']:.4f}")
        print(f"     • F1-Score: {best_production['Test_F1']:.4f}")
        print(f"     • Estabilidad: {best_production['Stability_Score']:.4f}")
        print(
            f"     • Problema de fitting: {best_production['Problem_Type'].replace('_', ' ')}"
        )

        print("\n2️⃣  MODELOS QUE REQUIEREN ATENCIÓN:")
        problematic = df[
            df["Problem_Type"].isin(
                [
                    "Overfitting",
                    "Severe_Overfitting",
                    "Underfitting",
                    "Severe_Underfitting",
                ]
            )
        ]
        if len(problematic) > 0:
            for _, model in problematic.iterrows():
                print(
                    f"   • {model['Modelo']}: {model['Problem_Type'].replace('_', ' ')}"
                )
        else:
            print("   ✅ Todos los modelos tienen buen ajuste")

        print("\n3️⃣  PRÓXIMOS PASOS:")
        print("   • Realizar validación cruzada k-fold para mayor robustez")
        print("   • Implementar ensemble de los mejores modelos")
        print("   • Analizar casos de error para mejorar features")
        print("   • Monitorear rendimiento en datos nuevos (concept drift)")
        print("   • Documentar hiperparámetros finales y pipeline completo")

        print("\n" + "=" * 100)
        print("✅ REPORTE COMPLETO GENERADO")
        print("=" * 100 + "\n")

        # Guardar reporte en archivo
        self._save_report_to_file(df)

        return df

    def _save_report_to_file(self, df):
        """Guarda el reporte en archivo CSV y texto"""
        # Guardar DataFrame completo
        df.to_csv("reports/model_evaluation_results.csv", index=False)
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

        print("📄 Resumen guardado en: reports/evaluation_summary.txt")


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
    sklearn_models = {
        "Logistic Regression": os.path.join(
            models_dir, "logistic_regression_best_model.pkl"
        ),
        "SVC": os.path.join(models_dir, "svc_best_model.pkl"),
        "Naive Bayes": os.path.join(models_dir, "naive_bayes_best_model.pkl"),
        "Decision Tree": os.path.join(models_dir, "decision_tree_best_model.pkl"),
        "Random Forest": os.path.join(models_dir, "random_forest_best_model.pkl"),
    }

    keras_models = {
        "LSTM": os.path.join(models_dir, "lstm_best_model.keras"),
        "BiLSTM": os.path.join(models_dir, "bilstm_best_model.keras"),
        "GRU": os.path.join(models_dir, "gru_best_model.keras"),
        "CNN": os.path.join(models_dir, "cnn_best_model.keras"),
        "Hibrido": os.path.join(
            models_dir, "hybrid_cnn_bilstm_gru_attention_best_model.keras"
        ),
    }

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

    # Evaluar sklearn models
    for name, path in sklearn_models.items():
        if os.path.exists(path):
            evaluator.evaluate_sklearn_model_complete(name, path)
        else:
            print(f"⚠️  Modelo no encontrado: {path}")

    # Pausa ligera para liberar CPU
    time.sleep(2)
    clear_memory()

    # Evaluar keras models
    for name, path in keras_models.items():
        if os.path.exists(path):
            evaluator.evaluate_keras_model_complete(name, path)
        else:
            print(f"⚠️  Modelo no encontrado: {path}")

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
    df_results.to_csv("reports/model_results_summary.csv", index=False)

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
