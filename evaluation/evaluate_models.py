import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.layers import TextVectorization
from training.utils_common import load_datasets
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

warnings.filterwarnings("ignore")

# Configuración de visualización
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class OverfittingDetector:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.results = []

    def evaluate_sklearn_model_complete(self, name, model_path):
        """Evalúa modelos de scikit-learn en train, validation y test"""
        print(f"Evaluando {name}...")
        model = joblib.load(model_path)

        datasets = {
            "Train": (self.x_train, self.y_train),
            "Validation": (self.x_val, self.y_val),
            "Test": (self.x_test, self.y_test),
        }

        model_results = {"Modelo": name}

        for dataset_name, (X, y) in datasets.items():
            start_time = time.time()
            y_pred = model.predict(X)
            prediction_time = time.time() - start_time

            try:
                y_proba = model.predict_proba(X)[:, 1]
            except Exception:
                y_proba = y_pred.astype(float)

            metrics = self._calculate_metrics_for_dataset(
                y, y_pred, y_proba, prediction_time
            )

            for metric, value in metrics.items():
                model_results[f"{dataset_name}_{metric}"] = value

        overfitting_indicators = self._calculate_overfitting_indicators(model_results)
        model_results.update(overfitting_indicators)
        self.results.append(model_results)

        return model

    @staticmethod
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

    def evaluate_keras_model_complete(
        self, name, model_path, max_tokens=30000, seq_len=200
    ):
        """Evalúa modelos de Keras en train, validation y test"""
        print(f"Evaluando {name}...")
        model = load_model(model_path)

        # 🔑 Preparar vectorizador SOLO con los datos de entrenamiento
        vectorizer = self.prepare_vectorizer(
            self.x_train, max_tokens=max_tokens, output_seq_len=seq_len
        )

        datasets = {
            "Train": (self.x_train, self.y_train),
            "Validation": (self.x_val, self.y_val),
            "Test": (self.x_test, self.y_test),
        }

        model_results = {"Modelo": name}

        for dataset_name, (X, y) in datasets.items():
            # Transformar los textos en secuencias enteras padded
            X_vec = vectorizer(np.array(X)).numpy()

            start_time = time.time()
            y_proba = model.predict(X_vec, verbose=0).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            prediction_time = time.time() - start_time

            metrics = self._calculate_metrics_for_dataset(
                y, y_pred, y_proba, prediction_time
            )

            for metric, value in metrics.items():
                model_results[f"{dataset_name}_{metric}"] = value

        overfitting_indicators = self._calculate_overfitting_indicators(model_results)
        model_results.update(overfitting_indicators)
        self.results.append(model_results)

        return model

    def _calculate_metrics_for_dataset(self, y_true, y_pred, y_proba, prediction_time):
        """Calcula métricas para un dataset específico"""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_proba),
            "Log_Loss": log_loss(y_true, y_proba),
            "Time": prediction_time,
        }

    def _calculate_overfitting_indicators(self, model_results):
        """Calcula indicadores de overfitting y underfitting"""
        train_acc = model_results["Train_Accuracy"]
        val_acc = model_results["Validation_Accuracy"]
        test_acc = model_results["Test_Accuracy"]

        train_f1 = model_results["Train_F1"]
        val_f1 = model_results["Validation_F1"]

        overfitting_acc = train_acc - val_acc
        overfitting_f1 = train_f1 - val_f1
        avg_performance = (train_acc + val_acc + test_acc) / 3
        generalization_gap = abs(val_acc - test_acc)

        problem_type = self._classify_fitting_problem(
            train_acc, val_acc, test_acc, overfitting_acc
        )

        return {
            "Overfitting_Accuracy": overfitting_acc,
            "Overfitting_F1": overfitting_f1,
            "Average_Performance": avg_performance,
            "Generalization_Gap": generalization_gap,
            "Problem_Type": problem_type,
            "Stability_Score": 1 - np.std([train_acc, val_acc, test_acc]),
        }

    def _classify_fitting_problem(self, train_acc, val_acc, test_acc, overfitting_gap):
        """Clasifica el tipo de fitting"""
        if overfitting_gap > 0.1:
            return "Overfitting"
        elif train_acc < 0.7 and val_acc < 0.7:
            return "Underfitting"
        elif overfitting_gap > 0.05:
            return "Slight_Overfitting"
        return "Good_Fit"

    def plot_train_val_test_comparison(self, save_fig=False):
        """Gráfico comparativo de rendimiento en los 3 conjuntos"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics = ["Accuracy", "F1", "ROC_AUC", "Log_Loss"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Datos para cada conjunto
            train_data = df[f"Train_{metric}"].values
            val_data = df[f"Validation_{metric}"].values
            test_data = df[f"Test_{metric}"].values

            x = np.arange(len(df))
            width = 0.25

            # Barras para cada conjunto
            ax.bar(x - width, train_data, width, label="Train", alpha=0.8)
            ax.bar(x, val_data, width, label="Validation", alpha=0.8)
            ax.bar(x + width, test_data, width, label="Test", alpha=0.8)

            # Configuración del gráfico
            ax.set_title(f"{metric} - Comparación Train/Val/Test")
            ax.set_xlabel("Modelos")
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(df["Modelo"], rotation=45)
            ax.legend()

        plt.tight_layout()
        if save_fig:
            plt.savefig(
                os.path.join("figures", "train_val_test_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_overfitting_indicators(self, save_fig=False):
        """Gráfico de indicadores de overfitting"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Gap de Overfitting (Accuracy)
        ax1 = axes[0, 0]
        bars = ax1.bar(df["Modelo"], df["Overfitting_Accuracy"])
        ax1.set_title("Gap de Overfitting (Train - Validation Accuracy)")
        ax1.set_ylabel("Diferencia de Accuracy")
        ax1.axhline(
            y=0.1,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="Umbral Overfitting (0.1)",
        )
        ax1.axhline(
            y=0.05,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Umbral Ligero (0.05)",
        )
        ax1.legend()

        # Colorear barras según severidad
        for i, bar in enumerate(bars):
            if df.iloc[i]["Overfitting_Accuracy"] > 0.1:
                bar.set_color("red")
            elif df.iloc[i]["Overfitting_Accuracy"] > 0.05:
                bar.set_color("orange")
            else:
                bar.set_color("green")

        # 2. Rendimiento Promedio
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df["Modelo"], df["Average_Performance"])
        ax2.set_title("Rendimiento Promedio (Underfitting)")
        ax2.set_ylabel("Accuracy Promedio")
        ax2.axhline(
            y=0.7,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="Umbral Underfitting (0.7)",
        )
        ax2.legend()

        # Colorear barras según rendimiento
        for i, bar in enumerate(bars2):
            if df.iloc[i]["Average_Performance"] < 0.7:
                bar.set_color("red")
            else:
                bar.set_color("green")

        # 3. Gap de Generalización
        ax3 = axes[1, 0]
        ax3.bar(df["Modelo"], df["Generalization_Gap"])
        ax3.set_title("Gap de Generalización (|Val - Test|)")
        ax3.set_ylabel("Diferencia Absoluta")
        ax3.axhline(
            y=0.05,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Umbral Aceptable (0.05)",
        )
        ax3.legend()

        # 4. Puntuación de Estabilidad
        ax4 = axes[1, 1]
        ax4.bar(df["Modelo"], df["Stability_Score"])
        ax4.set_title("Puntuación de Estabilidad")
        ax4.set_ylabel("Estabilidad (1 = perfecto)")
        ax4.axhline(
            y=0.95, color="g", linestyle="--", alpha=0.7, label="Excelente (0.95)"
        )
        ax4.legend()

        # Rotar etiquetas
        for ax in axes.flat:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save_fig:
            plt.savefig(
                os.path.join("figures", "overfitting_indicators.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_problem_classification(self, save_fig=False):
        """Gráfico de clasificación de problemas de fitting"""
        df = pd.DataFrame(self.results)

        # Contar tipos de problemas
        problem_counts = df["Problem_Type"].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Gráfico de pastel
        colors = {
            "Good_Fit": "green",
            "Slight_Overfitting": "orange",
            "Overfitting": "red",
            "Underfitting": "purple",
        }

        ax1.pie(
            problem_counts.values,
            labels=problem_counts.index,
            autopct="%1.1f%%",
            colors=[colors.get(label, "gray") for label in problem_counts.index],
        )
        ax1.set_title("Distribución de Problemas de Fitting")

        # 2. Scatter plot: Overfitting vs Performance
        scatter_colors = [colors.get(problem, "gray") for problem in df["Problem_Type"]]

        ax2.scatter(
            df["Overfitting_Accuracy"],
            df["Average_Performance"],
            c=scatter_colors,
            s=100,
            alpha=0.7,
        )

        # Añadir etiquetas a los puntos
        for i, model in enumerate(df["Modelo"]):
            ax2.annotate(
                model,
                (df.iloc[i]["Overfitting_Accuracy"], df.iloc[i]["Average_Performance"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax2.set_xlabel("Gap de Overfitting (Train - Val)")
        ax2.set_ylabel("Rendimiento Promedio")
        ax2.set_title("Mapa de Problemas de Fitting")
        ax2.axhline(
            y=0.7, color="r", linestyle="--", alpha=0.5, label="Umbral Underfitting"
        )
        ax2.axvline(
            x=0.05,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Umbral Overfitting Ligero",
        )
        ax2.axvline(
            x=0.1, color="r", linestyle="--", alpha=0.5, label="Umbral Overfitting"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_fig:
            plt.savefig(
                os.path.join("figures", "problem_classification.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_learning_curves(self, model, model_name, save_fig=False):
        """Genera curvas de aprendizaje para detectar overfitting visualmente"""
        print(f"Generando curva de aprendizaje para {model_name}...")

        # Determinar tamaños de entrenamiento
        train_sizes = np.linspace(0.1, 1.0, 10)

        # Calcular curvas de aprendizaje
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            train_sizes=train_sizes,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )

        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Crear gráfico
        plt.figure(figsize=(10, 6))

        # Plotear curvas
        plt.plot(
            train_sizes_abs, train_mean, "o-", color="blue", label="Training Score"
        )
        plt.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )

        plt.plot(
            train_sizes_abs, val_mean, "o-", color="red", label="Cross-Validation Score"
        )
        plt.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color="red",
        )

        # Configuración
        plt.xlabel("Tamaño del Conjunto de Entrenamiento")
        plt.ylabel("Accuracy")
        plt.title(f"Curva de Aprendizaje - {model_name}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        # Añadir interpretación
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            plt.text(
                0.02,
                0.95,
                "Overfitting Detectado",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=10,
                color="white",
            )
        elif val_mean[-1] < 0.7:
            plt.text(
                0.02,
                0.95,
                "Underfitting Detectado",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="purple", alpha=0.7),
                fontsize=10,
                color="white",
            )
        else:
            plt.text(
                0.02,
                0.95,
                "Buen Ajuste",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                fontsize=10,
                color="white",
            )

        if save_fig:
            plt.savefig(
                os.path.join(
                    "figures", f"learning_curve_{model_name.replace(' ', '_')}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def generate_overfitting_report(self):
        """Genera reporte completo sobre overfitting/underfitting"""
        df = pd.DataFrame(self.results)

        print("=" * 90)
        print("🔍 REPORTE DE ANÁLISIS DE OVERFITTING/UNDERFITTING")
        print("=" * 90)

        print("\n📊 RESUMEN POR MODELO:")
        print("-" * 90)

        for _, row in df.iterrows():
            print(f"\n🤖 {row['Modelo']}:")
            print(f"  • Train Accuracy: {row['Train_Accuracy']:.4f}")
            print(f"  • Val Accuracy:   {row['Validation_Accuracy']:.4f}")
            print(f"  • Test Accuracy:  {row['Test_Accuracy']:.4f}")
            print(f"  • Gap Overfitting: {row['Overfitting_Accuracy']:.4f}")
            print(f"  • Problema: {row['Problem_Type']}")

            # Interpretación
            if row["Problem_Type"] == "Overfitting":
                print("  ⚠️  ALTO OVERFITTING - Reducir complejidad del modelo")
            elif row["Problem_Type"] == "Underfitting":
                print("  ⚠️  UNDERFITTING - Aumentar complejidad del modelo")
            elif row["Problem_Type"] == "Slight_Overfitting":
                print("  ⚡ OVERFITTING LIGERO - Considerar regularización")
            else:
                print("  ✅ BUEN AJUSTE - Modelo bien calibrado")

        print("\n📈 ESTADÍSTICAS GENERALES:")
        print("-" * 50)
        print(
            f"• Modelos con overfitting: {sum(df['Problem_Type'].isin(['Overfitting', 'Slight_Overfitting']))}"
        )
        print(
            f"• Modelos con underfitting: {sum(df['Problem_Type'] == 'Underfitting')}"
        )
        print(f"• Modelos con buen ajuste: {sum(df['Problem_Type'] == 'Good_Fit')}")
        print(f"• Gap promedio overfitting: {df['Overfitting_Accuracy'].mean():.4f}")
        print(f"• Rendimiento promedio general: {df['Average_Performance'].mean():.4f}")

        print("\n💡 RECOMENDACIONES:")
        print("-" * 30)

        # Modelo con mejor balance
        best_balance = df.loc[df["Stability_Score"].idxmax(), "Modelo"]
        print(f"• Modelo más estable: {best_balance}")

        # Modelo con overfitting
        overfitted = df[df["Problem_Type"].isin(["Overfitting", "Slight_Overfitting"])]
        if len(overfitted) > 0:
            print(
                f"• Modelos que necesitan regularización: {', '.join(overfitted['Modelo'])}"
            )

        # Modelo con underfitting
        underfitted = df[df["Problem_Type"] == "Underfitting"]
        if len(underfitted) > 0:
            print(
                f"• Modelos que necesitan más complejidad: {', '.join(underfitted['Modelo'])}"
            )

        print("=" * 90)


def main():
    print("🔄 Cargando datasets...")

    train_df, valid_df, test_df = load_datasets()

    x_train = train_df["texto"].astype(str).values
    y_train = train_df["clase"].astype(int).values
    x_val = valid_df["texto"].astype(str).values
    y_val = valid_df["clase"].astype(int).values
    x_test = test_df["texto"].astype(str).values
    y_test = test_df["clase"].astype(int).values

    detector = OverfittingDetector(x_train, y_train, x_val, y_val, x_test, y_test)

    models_dir = "models"

    # Modelos
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

    # Evaluar modelos
    print("\n🔍 Evaluando modelos para detectar overfitting...")

    models_for_curves = {}

    # Evaluar sklearn
    for name, path in sklearn_models.items():
        detector.evaluate_sklearn_model_complete(name, path)

    # Evaluar keras
    for name, path in keras_models.items():
        detector.evaluate_keras_model_complete(name, path)

    detector.generate_overfitting_report()

    # Crear visualizaciones específicas para overfitting
    print("\n📊 Generando visualizaciones de overfitting...")

    detector.plot_train_val_test_comparison(save_fig=True)
    detector.plot_overfitting_indicators(save_fig=True)
    detector.plot_problem_classification(save_fig=True)

    # Generar curvas de aprendizaje para modelos sklearn
    for name, model in models_for_curves.items():
        detector.plot_learning_curves(model, name, save_fig=True)

    print("✅ Análisis completo de overfitting/underfitting terminado!")


if __name__ == "__main__":
    main()
