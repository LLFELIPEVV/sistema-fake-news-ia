#!/usr/bin/env bash
set -e

echo "📦 Descargando modelos desde Hugging Face..."

mkdir -p models

python - << 'EOF'
from huggingface_hub import hf_hub_download

REPO_ID = "LFELIPEV/sistema-fake-news-ia"

files = [
    "cnn_best_model.keras",
    "cnn_best_score.txt",
    "hybrid_cnn_bilstm_gru_attention_best_model.keras",
    "hybrid_cnn_bilstm_gru_attention_best_score.txt",
    "naive_bayes_best_model.pkl",
    "naive_bayes_best_score.txt",
    "random_forest_best_model.pkl",
    "random_forest_best_score.txt"
]

for f in files:
    hf_hub_download(
        repo_id=REPO_ID,
        filename=f,
        local_dir="models",
        local_dir_use_symlinks=False
    )

print("✅ Modelos descargados correctamente")
EOF
