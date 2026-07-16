#!/bin/bash
# Run this on a node WITH internet access to set up the conda env + cache models.
# The cluster compute nodes have NO internet.
#
# Usage (from login node or local machine):
#   bash scripts/cluster/setup_env.sh

set -euo pipefail

echo "=== Setting up locked diffusiontuner environment ==="

module load Python/Anaconda_v03.2023

python -m pip install --user "uv==0.11.28"
python -m uv python install 3.11
python -m uv sync --frozen --group dev --extra gpu --extra reward
# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "=== Pre-caching models (REQUIRED: no internet on compute nodes) ==="
FLUX_MODEL_ID="black-forest-labs/FLUX.2-klein-base-4B"
FLUX_MODEL_REVISION="a3b4f4849157f664bdbc776fd7453c2783562f4d"
VLM_MODEL_ID="Qwen/Qwen3.5-9B"
VLM_MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"

python - "$FLUX_MODEL_ID" "$FLUX_MODEL_REVISION" "$VLM_MODEL_ID" "$VLM_MODEL_REVISION" <<'PY'
from huggingface_hub import snapshot_download
import sys

for repo_id, revision in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    print(f"Caching {repo_id}@{revision}")
    snapshot_download(repo_id=repo_id, revision=revision)
PY

echo "Locked environment ready. Submit jobs with: sbatch scripts/cluster/sft.sbatch"
