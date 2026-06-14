#!/bin/bash
# Récupère TOUTES les ressources nécessaires à la régénération des features (voie
# from-scratch), SAUF les crops (= dataset du challenge, non redistribuable).
set -e
HF_REPO="${1:-saraelmoun/facepredict-icl-extractor-weights}"
pip install -q huggingface_hub

# 1) repo CLIB-FIQA (code clip + models), épinglé au commit utilisé
if [ ! -d external/CLIB-FIQA ]; then
  git clone https://github.com/oufuzhao/CLIB-FIQA.git external/CLIB-FIQA
  git -C external/CLIB-FIQA checkout aa02294
fi

# 2) poids des extracteurs depuis HF
python - <<PY
from huggingface_hub import hf_hub_download
import shutil
repo="$HF_REPO"
for f,dst in [("RN50.pt","external/CLIB-FIQA/weights/RN50.pt"),
              ("CLIB-FIQA_R50.pth","external/CLIB-FIQA/weights/CLIB-FIQA_R50.pth"),
              ("faceptor_checkpoint_rank0_iter_50000.pth.tar","external/faceptor_checkpoint_rank0_iter_50000.pth.tar")]:
    import os; os.makedirs(os.path.dirname(dst),exist_ok=True)
    p=hf_hub_download(repo_id=repo,repo_type="model",filename=f)
    shutil.copy(p,dst); print("ok",dst)
PY
echo "Ressources extracteurs prêtes. Reste à fournir les crops (CROPS_DIR=...) puis: python src/extract.py"
