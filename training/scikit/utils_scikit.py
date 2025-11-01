import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from training.utils_common import (
    save_figure,
    load_previous_results,
    _params_to_frozenset,
)
from sklearn.model_selection import ParameterSampler, ParameterGrid


def save_best_model_sklearn(model, score, BEST_SCORE_PATH, BEST_MODEL_PATH):
    """Guarda el mejor modelo si mejora el anterior."""
    if os.path.exists(BEST_SCORE_PATH):
        with open(BEST_SCORE_PATH, "r") as f:
            best_score = float(f.read().strip())
    else:
        best_score = -1

    if score > best_score:
        joblib.dump(model, BEST_MODEL_PATH)
        with open(BEST_SCORE_PATH, "w") as f:
            f.write(str(score))
        print(f"[INFO] Nuevo mejor modelo guardado con score {score:.4f}")
    else:
        print(
            f"[INFO] El modelo actual ({score:.4f}) no supera al mejor ({best_score:.4f})."
        )


def plot_grid_search_results(grid, filename="grid_results.png", param_x="param_clf__C"):
    """
    Grafica resultados de RandomizedSearch para un parámetro dado.

    Args:
        grid: objeto RandomizedSearchCV ya entrenado
        filename: nombre del archivo donde guardar la figura
        param_x: nombre del hiperparámetro en grid.cv_results_ para el eje X
    """
    results = pd.DataFrame(grid.cv_results_)

    if param_x not in results.columns:
        raise ValueError(f"El parámetro {param_x} no está en los resultados.")

    fig, ax = plt.subplots(figsize=(10, 6))

    if "param_tfidf__ngram_range" in results.columns:
        for ngram in results["param_tfidf__ngram_range"].unique():
            subset = results[results["param_tfidf__ngram_range"] == ngram]
            subset = subset.sort_values(param_x)
            ax.plot(
                subset[param_x],
                subset["mean_test_score"],
                marker="o",
                label=f"N-gram {ngram}",
            )
    else:
        # Si no se probó con n-gramas, grafica solo contra param_x
        results = results.sort_values(param_x)
        ax.plot(results[param_x], results["mean_test_score"], marker="o")

    # Si param_x es 'C' o 'gamma', conviene logscale
    if any(k in param_x.lower() for k in ["c", "gamma"]):
        ax.set_xscale("log")

    ax.set_title(f"Resultados de RandomizedSearch ({param_x})")
    ax.set_xlabel(param_x)
    ax.set_ylabel("F1 ponderado (media CV)")
    ax.legend()
    save_figure(fig, filename)
    plt.close(fig)


# =============================
# FUNCIONES DE REGISTRO
# =============================
def save_results(cv_results, history_file, scoring_name="f1_macro"):
    """Guarda combinaciones probadas y sus resultados."""
    history = load_previous_results(history_file)

    # Crear set de combinaciones existentes para búsqueda rápida
    existing_combos = {_params_to_frozenset(h["params"]) for h in history}

    new_entries = []
    for params, score in zip(cv_results["params"], cv_results["mean_test_score"]):
        combo = _params_to_frozenset(params)

        if combo not in existing_combos:
            record = {
                "timestamp": datetime.now().isoformat(),
                "scoring": scoring_name,
                "params": params,
                "mean_test_score": float(score),
            }
            new_entries.append(record)
            existing_combos.add(combo)  # Evitar duplicados en el mismo batch

    if new_entries:
        history.extend(new_entries)
        with open(history_file, "w", encoding="utf8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(
            f"[INFO] {len(new_entries)} nuevas combinaciones registradas en {history_file}"
        )
    else:
        print("[INFO] No se registraron nuevas combinaciones (todas ya probadas).")


def _is_continuous_distribution(value):
    """Verifica si un valor es una distribución continua (loguniform, uniform, etc.)."""
    # Verificar si es una distribución de scipy.stats
    if hasattr(value, "rvs"):
        # loguniform, uniform, y otras distribuciones continuas
        continuous_types = ("loguniform", "uniform", "expon", "norm")
        dist_name = getattr(value, "dist", type(value).__name__.lower())
        if hasattr(dist_name, "name"):
            dist_name = dist_name.name
        return any(ct in str(dist_name).lower() for ct in continuous_types)
    return False


def get_unique_param_samples(param_distributions, history_file, n_iter, random_state):
    """
    Genera combinaciones únicas que no hayan sido probadas antes.

    Esta función primero intenta generar TODAS las combinaciones posibles del espacio
    de búsqueda. Si hay demasiadas (>10000) o hay distribuciones continuas, usa muestreo aleatorio.

    Args:
        param_distributions: dict con distribuciones de parámetros
        history_file: archivo JSON con historial de combinaciones probadas
        n_iter: número máximo de combinaciones a devolver
        random_state: semilla para reproducibilidad

    Returns:
        list: lista de diccionarios con combinaciones únicas
    """
    history = load_previous_results(history_file)

    # Convertir historial a set de combinaciones
    tried_combinations = {_params_to_frozenset(h["params"]) for h in history}

    print(f"[INFO] {len(tried_combinations)} combinaciones únicas ya probadas.")

    # Verificar si hay distribuciones continuas
    has_continuous = False
    discrete_params = {}

    for key, value in param_distributions.items():
        if _is_continuous_distribution(value):
            has_continuous = True
            print(f"[INFO] Parámetro '{key}' es continuo, se usará muestreo aleatorio")
            # No intentamos discretizar distribuciones continuas
            discrete_params[key] = value
        elif hasattr(value, "rvs"):  # Es una distribución discreta (randint, etc.)
            if hasattr(value, "a") and hasattr(value, "b"):
                # Para randint: genera lista de valores posibles
                discrete_params[key] = list(range(value.a, value.b))
            else:
                # Si no podemos extraer el rango, usar muestreo
                discrete_params[key] = [
                    value.rvs(random_state=random_state) for _ in range(20)
                ]
        else:
            # Ya es una lista
            discrete_params[key] = value if isinstance(value, list) else [value]

    # Si hay distribuciones continuas, solo podemos usar muestreo aleatorio
    if has_continuous:
        print("[INFO] Espacio con parámetros CONTINUOS, usando MUESTREO ALEATORIO")

        # Estimar combinaciones para parámetros discretos solamente
        discrete_only = {
            k: v for k, v in discrete_params.items() if isinstance(v, list)
        }

        if discrete_only:
            total_discrete = 1
            for values in discrete_only.values():
                total_discrete *= len(values)
            print(f"[INFO] Combinaciones discretas posibles: {total_discrete:,}")

        saturation_ratio = len(tried_combinations) / max(
            total_discrete if discrete_only else 1000, 1
        )

        if saturation_ratio > 0.5:
            max_attempts = n_iter * 100
        elif saturation_ratio > 0.1:
            max_attempts = n_iter * 50
        else:
            max_attempts = n_iter * 10

        print(f"[INFO] Saturación estimada: {saturation_ratio * 100:.2f}%")
        print(
            f"[INFO] Intentando generar {n_iter} combinaciones únicas (máx {max_attempts} intentos)"
        )

        sampler = ParameterSampler(
            param_distributions,
            n_iter=max_attempts,
            random_state=random_state,
        )

        unique_samples = []
        attempts = 0

        for params in sampler:
            attempts += 1
            combo = _params_to_frozenset(params)

            if combo not in tried_combinations:
                grid_params = {k: [v] for k, v in params.items()}
                unique_samples.append(grid_params)
                tried_combinations.add(combo)

            if len(unique_samples) >= n_iter:
                break

        if len(unique_samples) < n_iter:
            print(
                f"[WARNING] Solo se pudieron generar {len(unique_samples)} combinaciones únicas de {n_iter} solicitadas"
            )
            print(f"[INFO] Intentos realizados: {attempts}")
            print("[INFO] Espacio de búsqueda posiblemente saturado")
        else:
            print(
                f"[INFO] {len(unique_samples)} combinaciones únicas generadas en {attempts} intentos"
            )

        return unique_samples

    # Calcular número total de combinaciones posibles (solo para parámetros discretos)
    total_combinations = 1
    for values in discrete_params.values():
        if isinstance(values, list):
            total_combinations *= len(values)

    print(f"[INFO] Espacio de búsqueda: {total_combinations:,} combinaciones posibles")

    # Estrategia 1: Si hay pocas combinaciones (<= 10000), usar ParameterGrid exhaustivo
    if total_combinations <= 10000:
        print("[INFO] Usando búsqueda EXHAUSTIVA (todas las combinaciones)")
        all_combinations = list(ParameterGrid(discrete_params))
        print(f"[INFO] {len(all_combinations)} combinaciones generadas")

        # Filtrar las ya probadas
        unique_samples = []
        for params in all_combinations:
            combo = _params_to_frozenset(params)
            if combo not in tried_combinations:
                grid_params = {k: [v] for k, v in params.items()}
                unique_samples.append(grid_params)
                tried_combinations.add(combo)

        print(f"[INFO] {len(unique_samples)} combinaciones no probadas encontradas")

        # Limitar a n_iter si hay más
        if len(unique_samples) > n_iter:
            import random

            random.seed(random_state)
            unique_samples = random.sample(unique_samples, n_iter)
            print(f"[INFO] Seleccionadas {n_iter} combinaciones aleatoriamente")

        return unique_samples

    # Estrategia 2: Si hay muchas combinaciones, usar muestreo aleatorio inteligente
    else:
        print("[INFO] Espacio muy grande, usando MUESTREO ALEATORIO")

        # Calcular cuántos intentos necesitamos
        saturation_ratio = len(tried_combinations) / total_combinations

        if saturation_ratio > 0.5:
            max_attempts = n_iter * 100  # Espacio muy saturado
        elif saturation_ratio > 0.1:
            max_attempts = n_iter * 50  # Moderadamente saturado
        else:
            max_attempts = n_iter * 10  # Poco explorado

        print(f"[INFO] Saturación del espacio: {saturation_ratio * 100:.2f}%")
        print(
            f"[INFO] Intentando generar {n_iter} combinaciones únicas (máx {max_attempts} intentos)"
        )

        sampler = ParameterSampler(
            param_distributions,
            n_iter=max_attempts,
            random_state=random_state,
        )

        unique_samples = []
        attempts = 0

        for params in sampler:
            attempts += 1
            combo = _params_to_frozenset(params)

            if combo not in tried_combinations:
                grid_params = {k: [v] for k, v in params.items()}
                unique_samples.append(grid_params)
                tried_combinations.add(combo)

            if len(unique_samples) >= n_iter:
                break

        if len(unique_samples) < n_iter:
            print(
                f"[WARNING] Solo se pudieron generar {len(unique_samples)} combinaciones únicas de {n_iter} solicitadas"
            )
            print(f"[INFO] Intentos realizados: {attempts}")
            print("[INFO] Espacio de búsqueda posiblemente saturado")

            # Calcular cuántas combinaciones quedan sin probar
            remaining = total_combinations - len(tried_combinations)
            print(f"[INFO] Combinaciones restantes estimadas: {remaining:,}")
        else:
            print(
                f"[INFO] {len(unique_samples)} combinaciones únicas generadas en {attempts} intentos"
            )

        return unique_samples
