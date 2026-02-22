import pandas as pd
from pathlib import Path

# Rutas a los archivos parquet
PARQUETS = [
    Path(r"data\splits\fake_news_test.parquet"),
    Path(r"data\splits\fake_news_train.parquet"),
    Path(r"data\splits\fake_news_valid.parquet"),
]

# Columnas esperadas
TEXT_COL = "texto"
LABEL_COL = "clase"
FAKE_LABEL = 0

def load_and_sample_fake_news(parquet_paths, n_samples=10, random_state=None):
    dataframes = []
    for path in parquet_paths:
        df = pd.read_parquet(path, columns=[TEXT_COL, LABEL_COL])
        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)
    fake_df = full_df[full_df[LABEL_COL] == FAKE_LABEL]

    if len(fake_df) < n_samples:
        raise ValueError(f"No hay suficientes noticias falsas: {len(fake_df)} disponibles.")

    # Muestreo y eliminamos la columna de clase ya que no la usaremos para imprimir
    sample_df = fake_df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    return sample_df[TEXT_COL] # Retornamos solo la serie de texto

if __name__ == "__main__":
    texts = load_and_sample_fake_news(
        parquet_paths=PARQUETS, n_samples=10, random_state=42
    )

    # --- FORMATO DE IMPRESIÓN MEJORADO ---
    print("\n" + "="*60)
    print(" REPORTE DE NOTICIAS FALSAS (MUESTRA) ")
    print("="*60 + "\n")

    for i, noticia in enumerate(texts, 1):
        print(f"📰 NOTICIA #{i}")
        print("-" * 30)
        # Imprime el texto completo. 
        print(noticia) 
        print("\n" + "."*60 + "\n")