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
#   bash scripts/cluster/merge_scores.sh

#SBATCH --job-name=score_imgs
#SBATCH --error=runs/score_imgs-%A_%a.err
#SBATCH --output=runs/score_imgs-%A_%a.log
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# V100 32GB (type_a/b/c), A100 80GB (type_e), H100 (type_h — update when confirmed)
#SBATCH --constraint="[type_a|type_b|type_c]"

# ── Environment ──
module load Python/Anaconda_v03.2023
source activate diffusiontuner

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

# ── Config ──
IMAGES_DIR="outputs/generated/images"
TEXT_EMBEDS_DIR="outputs/generated/text_embeds"
OUTPUT_CSV="outputs/generated/scores.csv"
VLM_MODEL="Qwen/Qwen3.5-9B"

N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}

echo "Scoring shard $SHARD_IDX / $N_SHARDS"

# ── Run ──
python -m scripts.score_images \
    --images_dir "$IMAGES_DIR" \
    --text_embeds_dir "$TEXT_EMBEDS_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --vlm_model_id "$VLM_MODEL" \
    --shard_idx "$SHARD_IDX" \
    --num_shards "$N_SHARDS" \
    --resume
