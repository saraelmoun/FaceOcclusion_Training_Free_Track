#!/bin/bash
# Télécharge les features figées (4 modèles, 8 fichiers .npy) + le checkpoint TabICL
# depuis Hugging Face (dataset public).
# Place tout dans features/ (là où p6_tabicl_icl.py les lit).
HF_REPO="${1:-saraelmoun/facepredict-icl-features}"
pip install -q huggingface_hub
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$HF_REPO", repo_type="dataset", local_dir="features",
                  allow_patterns=["*.npy", "*.ckpt"])
print("Features figées (8 fichiers .npy) + checkpoint TabICL téléchargés dans features/")
PY
