#!/bin/bash
#SBATCH --job-name=dpo
#SBATCH --error=slurm-dpo-%j.err.log
#SBATCH --output=slurm-dpo-%j.out.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
# BF16 configs require A100/H100-class hardware. Do not submit this profile to V100 nodes.
#SBATCH --constraint="[type_e|type_h]"

set -euo pipefail

# ── Environment ──
cd "$SLURM_SUBMIT_DIR" || exit
mkdir -p runs
test -f .venv/bin/activate || { echo "Run scripts/cluster/setup_env.sh first" >&2; exit 2; }
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Launch ──
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_PROCESSES="${SLURM_GPUS_ON_NODE:-4}"
CONFIG_PATH="${CONFIG_PATH:-configs/dpo.json}"
python -c 'import json,re,sys; p=json.load(open(sys.argv[1])); r=p.get("model_revision"); raise SystemExit(0 if isinstance(r,str) and re.fullmatch(r"[0-9a-f]{40}",r) else "an immutable 40-character model_revision is required for cluster runs")' "$CONFIG_PATH"
RUN_DIR="$(python -m scripts.run_manifest init --stage dpo --config "$CONFIG_PATH" --command "accelerate launch -m src.training.dpo_trainer --config $CONFIG_PATH")"
python -m scripts.preflight_runtime \
    --stage dpo \
    --config "$CONFIG_PATH" \
    --manifest "$RUN_DIR/manifest.json" \
    --json

accelerate launch \
    --config_file configs/accelerate/multi_gpu.yaml \
    --num_processes "$NUM_PROCESSES" \
    -m src.training.dpo_trainer \
    --config "$CONFIG_PATH"

python -m scripts.run_manifest note "$RUN_DIR/manifest.json" "SLURM training command completed"
