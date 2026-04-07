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
#SBATCH --error=runs/gen_imgs-%A_%a.err
#SBATCH --output=runs/gen_imgs-%A_%a.log
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
N_PROMPTS=15000
PROMPTS_FILE="data/prompts_simple.jsonl"
OUTPUT_DIR="outputs/generated"
MODEL_ID="black-forest-labs/FLUX.2-klein-base-4B"
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

# ── Run ──
python -m scripts.generate_images \
    --prompts "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --versions_per_prompt "$VERSIONS" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$GUIDANCE" \
    --resolution "$RESOLUTION" \
    --seed "$SEED" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX"
