#!/bin/bash
# SLURM array job: shard image generation across GPUs.
#
# Splits N_PROMPTS evenly across SLURM_ARRAY_TASK_COUNT jobs,
# each job uses 1 GPU to generate its slice.
#
# Usage:
#   sbatch --array=0-3 scripts/cluster/generate_images.sbatch   # 4 GPUs
#   sbatch --array=0-15 scripts/cluster/generate_images.sbatch  # 16 GPUs
#   sbatch --array=0-31 scripts/cluster/generate_images.sbatch  # 32 GPUs (8×H100)
#
# Adjust --constraint for your card type:
#   type_a|type_b|type_c  = V100 32GB (29 nodes × 4)
#   type_e                = A100 80GB (6 nodes × 8)
#   type_h                = H100 (placeholder — update when confirmed)

#SBATCH --job-name=gen_imgs
#SBATCH --error=slurm-gen_imgs-%A_%a.err.log
#SBATCH --output=slurm-gen_imgs-%A_%a.out.log
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# BF16 generation requires A100/H100-class hardware.
#SBATCH --constraint="[type_e|type_h]"

set -euo pipefail

# ── Environment ──
cd "$SLURM_SUBMIT_DIR" || exit
mkdir -p runs
mkdir -p runs/generation
test -f .venv/bin/activate || { echo "Run scripts/cluster/setup_env.sh first" >&2; exit 2; }
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Config ──
N_PROMPTS=15000
PROMPTS_FILE="data/prompts_simple.jsonl"
OUTPUT_DIR="outputs/generated"
MODEL_ID="black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION="${MODEL_REVISION:-a3b4f4849157f664bdbc776fd7453c2783562f4d}"
[[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "MODEL_REVISION must be an immutable 40-character commit SHA" >&2; exit 2; }
VERSIONS=5
STEPS=50
GUIDANCE=4.0
RESOLUTION=512
SEED=42

# ── Compute shard boundaries ──
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

CHUNK_SIZE=$(( (N_PROMPTS + N_TASKS - 1) / N_TASKS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))

# Clamp end
if [ "$END_IDX" -gt "$N_PROMPTS" ]; then
    END_IDX=$N_PROMPTS
fi

echo "Task $TASK_ID / $N_TASKS: prompts [$START_IDX, $END_IDX)"

python -m scripts.preflight_runtime \
    --stage generate \
    --prompts "$PROMPTS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --json
RUN_DIR="$(python -m scripts.run_manifest init --stage generate --command "python -m scripts.generate_images --start_idx $START_IDX --end_idx $END_IDX")"

# ── Run ──
python -m scripts.generate_images \
    --prompts "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --model_revision "$MODEL_REVISION" \
    --versions_per_prompt "$VERSIONS" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$GUIDANCE" \
    --resolution "$RESOLUTION" \
    --seed "$SEED" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --shard_index "$TASK_ID" \
    --num_shards "$N_TASKS" \
    --manifest_path "runs/generation/shard-${TASK_ID}-of-${N_TASKS}.json" \
    --run_manifest_path "$RUN_DIR/manifest.json" \
    --device cuda

python -m scripts.run_manifest note "$RUN_DIR/manifest.json" "SLURM generation shard completed"
