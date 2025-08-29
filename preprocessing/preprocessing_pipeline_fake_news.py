import os
import numpy as np
import pandas as pd

DATASETS = [
    r"datasets\fake_news_corpus_spanish\test (1).csv",
    r"datasets\FakeNewsSpanish_Kaggle2\spanishFakeNews.csv",
    r"datasets\FakeNewsSpanish_Kaggle2\testSpanishFakeNews.csv",
    r"datasets\Spanish Fake and Real News\spanishFakeNews.csv",
    r"datasets\Spanish Fake and Real News\testSpanishFakeNews.csv",
    r"datasets\Spanish Political Fake News\D57000_complete.csv",
]

COLUMNAS_TEXTO = [c.lower() for c in ["TEXT", "texto", "Descripcion", "Text"]]
COLUMNAS_CLASE = [c.lower() for c in ["CATEGORY", "clase", "Label", "class"]]


def unir_datasets(rutas):
    """
    Lee múltiples datasets, unifica columnas de texto y clase,
    y devuelve un DataFrame consolidado.
    """
    dataframes = []

    for ruta in rutas:
        try:
            df = pd.read_csv(ruta)
        except Exception:
            df = pd.read_csv(ruta, sep=";")

        # Normalizar nombres de columnas
        print(f"Título: {ruta}, Columnas: {df.columns.tolist()}")
        df.columns = [col.lower().strip() for col in df.columns]

        # Detectar columnas relevantes
        col_texto = next((c for c in df.columns if c in COLUMNAS_TEXTO), None)
        col_clase = next((c for c in df.columns if c in COLUMNAS_CLASE), None)

        if col_texto is None or col_clase is None:
            print(f"⚠️ Columnas no encontradas en {ruta}")
            continue

        # Renombrar columnas estándar
        df = df.rename(columns={col_texto: "texto", col_clase: "clase"})
        df = df[["texto", "clase"]]

        # Normalizar valores de clase
        df["clase"] = (
            df["clase"]
            .astype(str)
            .str.lower()
            .map({"fake": 0, "falso": 0, "0": 0, "real": 1, "verdadero": 1, "1": 1})
        )

        # Eliminar filas inválidas
        df = df.dropna(subset=["clase"])

        # Ajustar tipos
        df["clase"] = df["clase"].astype(np.int8)
        df["texto"] = df["texto"].astype(str)

        dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)
    print(df_final.head())
    print(f"✅ Total de registros unificados: {len(df_final)}")

    return df_final


def guardar_parquet(dataframe, ruta):
    """
    Guarda un DataFrame en formato Parquet.
    Si el directorio no existe, lo crea automáticamente.
    """
    directorio = os.path.dirname(ruta)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio, exist_ok=True)

    dataframe.to_parquet(ruta, engine="pyarrow", index=False)
    print(f"✅ Datos guardados en {ruta}")


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza y normalización inicial a un DataFrame.

    - Elimina valores nulos y duplicados
    - Elimina textos vacíos
    - Estandariza codificación a UTF-8
    """
    df = df.dropna(subset=["texto", "clase"])
    df = df.drop_duplicates(subset=["texto"])
    df = df[df["texto"].str.strip() != ""]
    df["texto"] = df["texto"].apply(
        lambda x: x.encode("utf-8", "ignore").decode("utf-8", "ignore")
    )
    print(f"✅ Registros después de la normalización: {len(df)}")
    return df


def estandarizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica estandarización del texto:

    - Conversión a minúsculas
    - Eliminación de caracteres especiales
    - Reducción de espacios múltiples a uno solo
    - Normalización de tildes (sin acentos)
    - Eliminación de etiquetas HTML
    - Sustitución de URLs, menciones, hashtags, emojis y números por tokens especiales
    - Filtrado de textos con menos de 3 palabras
    """
    df = df.copy()
    df["texto"] = df["texto"].str.lower()
    df["texto"] = df["texto"].str.replace(r"[^\w\s]", "", regex=True)
    df["texto"] = df["texto"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["texto"] = (
        df["texto"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    df["texto"] = df["texto"].str.replace(r"<.*?>", "", regex=True)
    df["texto"] = df["texto"].str.replace(r"http\S+|www\S+", " <URL> ", regex=True)
    df["texto"] = df["texto"].str.replace(r"@\w+", " <USER> ", regex=True)
    df["texto"] = df["texto"].str.replace(r"#\w+", " <HASHTAG> ", regex=True)
    df["texto"] = df["texto"].str.replace(r"[^\x00-\x7F]+", " <EMOJI> ", regex=True)
    df["texto"] = df["texto"].str.replace(r"\d+", " <NUM> ", regex=True)
    # Filtrar textos demasiado cortos
    df = df[df["texto"].str.split().str.len() > 2]

    print(f"✅ Registros después de la estandarización: {len(df)}")
    print("✅ Registros por clase después de la estandarización:")
    print(df["clase"].value_counts(normalize=True))
    return df


if __name__ == "__main__":
    df_unificado = unir_datasets(DATASETS)
    guardar_parquet(df_unificado, "data/raw/fake_news_unificado.parquet")

    df_normalizado = normalizar_dataframe(df_unificado)
    df_estandarizado = estandarizar_texto(df_normalizado)
    guardar_parquet(df_estandarizado, "data/processed/fake_news_estandarizado.parquet")
