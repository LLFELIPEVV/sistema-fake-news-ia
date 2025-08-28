import numpy as np
import pandas as pd

datasets = [
    r"datasets\fake_news_corpus_spanish\test (1).csv",
    r"datasets\FakeNewsSpanish_Kaggle2\spanishFakeNews.csv",
    r"datasets\FakeNewsSpanish_Kaggle2\testSpanishFakeNews.csv",
    r"datasets\Spanish Fake and Real News\spanishFakeNews.csv",
    r"datasets\Spanish Fake and Real News\testSpanishFakeNews.csv",
    r"datasets\Spanish Political Fake News\D57000_complete.csv",
]

nombres_columnas_texto = [c.lower() for c in ["TEXT", "texto", "Descripcion", "Text"]]
nombres_columnas_clase = [c.lower() for c in ["CATEGORY", "clase", "Label", "class"]]

dataframes_procesados = []


def union_datasets(rutas):
    for ruta_dataset in rutas:
        try:
            df = pd.read_csv(f"{ruta_dataset}")
        except Exception:
            df = pd.read_csv(f"{ruta_dataset}", sep=";")

        # Normalizar columnas a minúsculas para mayor flexibilidad
        print(f"Titulo: {ruta_dataset}, Columnas: {df.columns}")
        df.columns = [col.lower().strip() for col in df.columns]

        # Detectar columna de texto
        col_texto = next((c for c in df.columns if c in nombres_columnas_texto), None)
        # Detectar columna de clase
        col_clase = next((c for c in df.columns if c in nombres_columnas_clase), None)

        if col_texto is None or col_clase is None:
            print(f"⚠️ No se encontraron columnas adecuadas en {ruta_dataset}")
            continue

        # Renombrar a nombres estándar
        df = df.rename(columns={col_texto: "texto", col_clase: "clase"})

        # Solo conservar columnas necesarias
        df = df[["texto", "clase"]]

        # Normalizar valores de clase (ejemplo: "FAKE", "REAL", 0, 1, etc.)
        df["clase"] = (
            df["clase"]
            .astype(str)
            .str.lower()
            .map({"fake": 0, "falso": 0, "0": 0, "real": 1, "verdadero": 1, "1": 1})
        )

        # Eliminar filas con valores de clase no reconocidos
        df = df.dropna(subset=["clase"])

        df["clase"] = df["clase"].astype(np.int8)
        df["texto"] = df["texto"].astype(str)

        dataframes_procesados.append(df)

    df_final = pd.concat(dataframes_procesados, ignore_index=True)
    print(df_final.head())
    print(f"Total de registros unificados: {len(df_final)}")

    return df_final
