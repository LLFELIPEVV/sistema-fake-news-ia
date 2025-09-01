import pandas as pd
import stopwordsiso as stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

spanish_stopwords = list(stopwords.stopwords("es"))


def load_data():
    """Carga los datasets preprocesados en formato parquet."""
    train_df = pd.read_parquet("data/splits/fake_news_train.parquet")
    valid_df = pd.read_parquet("data/splits/fake_news_valid.parquet")
    test_df = pd.read_parquet("data/splits/fake_news_test.parquet")
    return train_df, valid_df, test_df


def vectorize_text(train_df, valid_df, test_df):
    """Vectoriza los textos usando TF-IDF."""
    vectorizer = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), stop_words=spanish_stopwords
    )

    X_train = vectorizer.fit_transform(train_df["texto"])
    X_valid = vectorizer.transform(valid_df["texto"])
    X_test = vectorizer.transform(test_df["texto"])

    return X_train, X_valid, X_test, vectorizer


if __name__ == "__main__":
    # Cargar datos
    train_df, valid_df, test_df = load_data()

    # Datos crudos
    print("=== Datos crudos ===")
    print(train_df.head())

    # Vectorizar textos
    X_train, X_valid, X_test, vectorizer = vectorize_text(train_df, valid_df, test_df)

    print("\n=== Datos vectorizados (matriz dispersa → DataFrame) ===")
    feature_names = vectorizer.get_feature_names_out()
    vectorized_df = pd.DataFrame(X_train[:5].toarray(), columns=feature_names)
    print(
        vectorized_df.iloc[:, :10]
    )  # solo las 10 primeras columnas para ver algo legible

    # Variables objetivo
    y_train = train_df["clase"]
    y_valid = valid_df["clase"]
    y_test = test_df["clase"]

    # Modelo de ejemplo
    log_reg_model = LogisticRegression(random_state=42)
