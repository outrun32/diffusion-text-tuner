#!/usr/bin/env bash
# Precompute text embeddings, then run ReFL training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS="${PROMPTS:-data/prompts_llm.jsonl}"
TEXT_EMBEDS_DIR="${TEXT_EMBEDS_DIR:-outputs/text_embeds}"
MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-4B-Base}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/refl}"
EXPERIMENT="${EXPERIMENT:-refl_v1}"
NUM_STEPS="${NUM_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-5}"
RESOLUTION="${RESOLUTION:-512}"
LORA_RANK="${LORA_RANK:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"  # Small run for testing

# ── Step 1: Precompute text embeddings (if needed) ─────────────────────────
if [ ! -d "$TEXT_EMBEDS_DIR" ] || [ -z "$(ls -A $TEXT_EMBEDS_DIR 2>/dev/null)" ]; then
    echo "=== Step 1: Precomputing text embeddings ==="
    python -c "
from src.training.flux2_utils import precompute_text_embeddings
precompute_text_embeddings(
    prompts_path='${PROMPTS}',
    output_dir='${TEXT_EMBEDS_DIR}',
    model_id='${MODEL}',
    batch_size=4,
)
"
else
    echo "=== Step 1: Text embeddings already exist at $TEXT_EMBEDS_DIR, skipping ==="
fi

# ── Step 2: Run ReFL training ──────────────────────────────────────────────
echo ""
echo "=== Step 2: Starting ReFL training ==="
python -m src.training.refl_trainer \
    --model-id "$MODEL" \
    --vlm-model-id "$VLM_MODEL" \
    --text-embeds-dir "$TEXT_EMBEDS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT" \
    --num-training-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --resolution "$RESOLUTION" \
    --lora-rank "$LORA_RANK" \
    --num-samples "$NUM_SAMPLES"
