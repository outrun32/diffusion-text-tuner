#!/bin/bash
# SLURM array job: shard VLM scoring across GPUs.
#
# Each task scores a disjoint slice of images, writing to a per-shard CSV.
# After all tasks finish, run scripts/cluster/merge_scores.sh to combine.
#
# Usage:
#   sbatch --array=0-3 scripts/cluster/score_images.sbatch   # 4 GPUs
#   sbatch --array=0-7 scripts/cluster/score_images.sbatch   # 8 GPUs
#
# After completion:
#   bash scripts/cluster/merge_scores.sh outputs/generated 8

#SBATCH --job-name=score_imgs
#SBATCH --error=slurm-score_imgs-%A_%a.err.log
#SBATCH --output=slurm-score_imgs-%A_%a.out.log
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# BF16 VLM scoring requires A100/H100-class hardware.
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

# ── Config ──
IMAGES_DIR="outputs/generated/images"
TEXT_EMBEDS_DIR="outputs/generated/text_embeds"
OUTPUT_CSV="outputs/generated/scores.csv"
VLM_MODEL="Qwen/Qwen3.5-9B"
VLM_REVISION="${VLM_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
[[ "$VLM_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "VLM_REVISION must be an immutable 40-character commit SHA" >&2; exit 2; }

N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}

echo "Scoring shard $SHARD_IDX / $N_SHARDS"

shopt -s nullglob
GENERATION_MANIFESTS=(runs/generation/shard-*.json)
if (( ${#GENERATION_MANIFESTS[@]} == 0 )); then
    echo "No generation shard manifests found under runs/generation" >&2
    exit 2
fi
SOURCE_ARGS=()
for manifest in "${GENERATION_MANIFESTS[@]}"; do
    SOURCE_ARGS+=(--source_manifest "$manifest")
done
RUN_DIR="$(python -m scripts.run_manifest init --stage score --command "python -m scripts.score_images --shard_idx $SHARD_IDX --num_shards $N_SHARDS")"

# ── Run ──
python -m scripts.score_images \
    --images_dir "$IMAGES_DIR" \
    --text_embeds_dir "$TEXT_EMBEDS_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --vlm_model_id "$VLM_MODEL" \
    --vlm_model_revision "$VLM_REVISION" \
    --scorer vlm \
    --shard_idx "$SHARD_IDX" \
    --num_shards "$N_SHARDS" \
    --manifest_path "$RUN_DIR/manifest.json" \
    "${SOURCE_ARGS[@]}" \
    --resume

python -m scripts.run_manifest note "$RUN_DIR/manifest.json" "SLURM scoring shard completed"
