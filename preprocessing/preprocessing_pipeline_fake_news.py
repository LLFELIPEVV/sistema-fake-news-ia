import numpy as np
import pandas as pd

from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASETS = [
    Path("datasets/fake_news_corpus_spanish/test (1).csv"),
    Path("datasets/FakeNewsSpanish_Kaggle2/spanishFakeNews.csv"),
    Path("datasets/FakeNewsSpanish_Kaggle2/testSpanishFakeNews.csv"),
    Path("datasets/Spanish Fake and Real News/spanishFakeNews.csv"),
    Path("datasets/Spanish Fake and Real News/testSpanishFakeNews.csv"),
    Path("datasets/Spanish Political Fake News/D57000_complete.csv"),
]

COLUMNAS_TEXTO = [c.lower() for c in ["TEXT", "texto", "Descripcion", "Text"]]
COLUMNAS_CLASE = [c.lower() for c in ["CATEGORY", "clase", "Label", "class"]]

CLASES_MAP = {
    "fake": 0,
    "falso": 0,
    "0": 0,
    "false": 0,
    "negativo": 0,
    "real": 1,
    "verdadero": 1,
    "1": 1,
    "true": 1,
    "positivo": 1,
}


def unir_datasets(rutas: list[Path]) -> pd.DataFrame:
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

        print(f"Título: {ruta}, Columnas: {df.columns.tolist()}")
        df.columns = [col.lower().strip() for col in df.columns]

        col_texto = next((c for c in df.columns if c in COLUMNAS_TEXTO), None)
        col_clase = next((c for c in df.columns if c in COLUMNAS_CLASE), None)

        if col_texto is None or col_clase is None:
            print(f"⚠️ Columnas no encontradas en {ruta}")
            continue

        df = df.rename(columns={col_texto: "texto", col_clase: "clase"})
        df = df[["texto", "clase"]]

        df["clase"] = (
            df["clase"].astype(str).str.lower().map(lambda x: CLASES_MAP.get(x, np.nan))
        )
        df = df.dropna(subset=["clase"])
        df["clase"] = df["clase"].astype(np.int8)
        df["texto"] = df["texto"].astype(str)

        dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)
    print(df_final.head())
    print(f"✅ Total de registros unificados: {len(df_final)}")

    return df_final


def guardar_parquet(dataframe: pd.DataFrame, ruta: Path):
    """
    Guarda un DataFrame en formato Parquet.
    Si el directorio no existe, lo crea automáticamente.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(ruta, engine="pyarrow", index=False)
    print(f"✅ Datos guardados en {ruta}")


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Detectar columnas
    columnas = df.columns.tolist()
    tiene_texto = "texto" in columnas
    tiene_clase = "clase" in columnas

    # Validar columnas esenciales
    if not tiene_texto:
        raise ValueError("El DataFrame debe contener la columna 'texto'.")

    # Dropna selectivo
    if tiene_clase:
        df = df.dropna(subset=["texto", "clase"])
    else:
        df = df.dropna(subset=["texto"])

    # Eliminar duplicados por texto
    df = df.drop_duplicates(subset=["texto"])

    # Eliminar textos vacíos o en blanco
    df = df[df["texto"].astype(str).str.strip() != ""]

    # Normalizar codificación
    df["texto"] = df["texto"].apply(
        lambda x: str(x).encode("utf-8", "ignore").decode("utf-8", "ignore")
    )

    print(f"✅ Registros después de la normalización: {len(df)}")

    # Si tiene clases, mostrar resumen
    if tiene_clase:
        print("✅ Registros por clase después de la normalización:")
        print(df["clase"].value_counts(dropna=False))

    return df


def estandarizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Verificar existencia de columna texto
    if "texto" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'texto'.")

    patrones = {
        r"[^\w\s]": "",
        r"\s+": " ",
        r"<.*?>": "",
        r"http\S+|www\S+": " <URL> ",
        r"@\w+": " <USER> ",
        r"#\w+": " <HASHTAG> ",
        r"[^\x00-\x7F]+": " <EMOJI> ",
        r"\d+": " <NUM> ",
    }

    # Limpieza general
    df["texto"] = df["texto"].astype(str).str.lower()
    df["texto"] = (
        df["texto"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )

    for patron, reemplazo in patrones.items():
        df["texto"] = df["texto"].str.replace(patron, reemplazo, regex=True)

    # Filtrar textos muy cortos
    df["texto"] = df["texto"].str.strip()
    df = df[df["texto"].str.split().str.len() > 2]

    print(f"✅ Registros después de la estandarización: {len(df)}")

    # Si hay columna clase, mostrar resumen
    if "clase" in df.columns:
        print("✅ Registros por clase después de la estandarización:")
        print(df["clase"].value_counts(normalize=True))

    return df


def dividir_datos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["texto"], df["clase"], test_size=0.3, random_state=42, stratify=df["clase"]
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    _reportar_division(y_train, y_valid, y_test)

    return (
        pd.DataFrame({"texto": X_train, "clase": y_train}),
        pd.DataFrame({"texto": X_valid, "clase": y_valid}),
        pd.DataFrame({"texto": X_test, "clase": y_test}),
    )


def _reportar_division(y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series):
    print(f"✅ Registros en train: {len(y_train)}")
    print(f"✅ Registros en valid: {len(y_valid)}")
    print(f"✅ Registros en test: {len(y_test)}")

    print("✅ Registros por clase en train:")
    print(y_train.value_counts(normalize=True))
    print("✅ Registros por clase en valid:")
    print(y_valid.value_counts(normalize=True))
    print("✅ Registros por clase en test:")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    df_unificado = unir_datasets(DATASETS)
    guardar_parquet(df_unificado, Path("data/raw/fake_news_unificado.parquet"))

    df_normalizado = normalizar_dataframe(df_unificado)
    df_estandarizado = estandarizar_texto(df_normalizado)
    guardar_parquet(
        df_estandarizado, Path("data/processed/fake_news_estandarizado.parquet")
    )

    train_df, valid_df, test_df = dividir_datos(df_estandarizado)
    guardar_parquet(train_df, Path("data/splits/fake_news_train.parquet"))
    guardar_parquet(valid_df, Path("data/splits/fake_news_valid.parquet"))
    guardar_parquet(test_df, Path("data/splits/fake_news_test.parquet"))
