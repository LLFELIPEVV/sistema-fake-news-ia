import os

APP_NAME = "Fake News Detection API"
VERSION = "1.0.0"
DEBUG = True

DEFAULT_MODEL = "CNN"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

MODELS_ENABLED = {
    "Naive Bayes": True,
    "Random Forest": True,
    "CNN": True,
    "Hibrido": True,
}

if __name__ == "__main__":
    print(f"Ruta: {BASE_DIR}")
    print(f"Ruta: {PROJECT_DIR}")
    print(f"Ruta: {MODELS_DIR}")
    print(f"Archivos: {os.listdir(MODELS_DIR)}")
