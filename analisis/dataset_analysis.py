import io
import pandas as pd


def analizar_dataset(
    primer_dataset, segundo_dataset=None, col_texto="text", col_clase="clase"
):
    """
    Analiza uno o dos datasets y muestra información relevante para proyectos de NLP.

    Parámetros:
    -----------
    primer_dataset : str
        Ruta al primer dataset (CSV).
    segundo_dataset : str, opcional
        Ruta al segundo dataset (CSV). Se concatena con el primero si se proporciona.
    col_texto : str, opcional
        Nombre de la columna que contiene el texto de las noticias.
    col_clase : str, opcional
        Nombre de la columna que contiene la etiqueta (fake/real).
    """

    # Cargar datasets
    df1 = pd.read_csv(primer_dataset)
    df2 = pd.read_csv(segundo_dataset) if segundo_dataset else None
    df = pd.concat([df1, df2], ignore_index=True) if df2 is not None else df1

    print("=" * 60)
    print("📊 ANÁLISIS INICIAL DEL DATASET")
    print("=" * 60)

    # Tamaño
    print("\n➡️ Tamaño del dataset:", df.shape)

    # Columnas
    print("\n➡️ Columnas del dataset:")
    print(df.columns.tolist())

    # Primeras filas
    print("\n➡️ Primeras 5 filas:")
    print(df.head())

    # Info general
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print("\n➡️ Información del dataset:")
    print(info_str)

    # Descripción numérica
    print("\n➡️ Descripción estadística de variables numéricas:")
    print(df.describe(include="all").transpose())

    # Nulos
    print("\n➡️ Valores nulos por columna:")
    print(df.isnull().sum())

    # Duplicados
    print("\n➡️ Filas duplicadas totales:", df.duplicated().sum())

    # Balance de clases
    if col_clase in df.columns:
        print("\n➡️ Balance de clases:")
        print(df[col_clase].value_counts(normalize=True))
    else:
        print(f"\n⚠️ No se encontró la columna '{col_clase}' en el dataset.")

    # Longitud del texto
    if col_texto in df.columns:
        df["longitud_texto"] = df[col_texto].astype(str).str.len()
        print("\n➡️ Estadísticas de la longitud de los textos:")
        print(df["longitud_texto"].describe())
    else:
        print(f"\n⚠️ No se encontró la columna '{col_texto}' en el dataset.")

    print("\n✅ Análisis completado.")


# Ejemplo de uso
analizar_dataset(
    r"datasets\noticias falsas en español\fakes1000.csv",
    col_texto="Text",
    col_clase="class",
)
