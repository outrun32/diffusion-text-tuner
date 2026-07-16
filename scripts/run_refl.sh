#!/usr/bin/env bash
# Precompute text embeddings, then run ReFL training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS="${PROMPTS:-data/prompts_simple.jsonl}"
TEXT_EMBEDS_DIR="${TEXT_EMBEDS_DIR:-outputs/text_embeds}"
MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-base-4B}"
MODEL_REVISION="${MODEL_REVISION:-a3b4f4849157f664bdbc776fd7453c2783562f4d}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.5-9B}"
VLM_MODEL_REVISION="${VLM_MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/refl}"
EXPERIMENT="${EXPERIMENT:-refl_v1}"
NUM_STEPS="${NUM_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-5}"
RESOLUTION="${RESOLUTION:-512}"
LORA_RANK="${LORA_RANK:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"  # Small run for testing

RUN_DIR="$(python -m scripts.run_manifest init --stage refl --command "python -m src.training.refl_trainer --model-id $MODEL --output-dir $OUTPUT_DIR")"

python -m scripts.preflight_runtime \
    --stage refl \
    --prompts "$PROMPTS" \
    --output-dir "$OUTPUT_DIR" \
    --manifest "$RUN_DIR/manifest.json" \
    --json

# ── Step 1: Precompute text embeddings (if needed) ─────────────────────────
if [ ! -d "$TEXT_EMBEDS_DIR" ] || [ -z "$(ls -A "$TEXT_EMBEDS_DIR" 2>/dev/null)" ]; then
    echo "=== Step 1: Precomputing text embeddings ==="
    python -m scripts.precompute_text_embeddings \
        --prompts "$PROMPTS" \
        --output-dir "$TEXT_EMBEDS_DIR" \
        --model-id "$MODEL" \
        --model-revision "$MODEL_REVISION" \
        --batch-size 4 \
        --device cuda
else
    echo "=== Step 1: Text embeddings already exist at $TEXT_EMBEDS_DIR, skipping ==="
fi

# ── Step 2: Run ReFL training ──────────────────────────────────────────────
echo ""
echo "=== Step 2: Starting ReFL training ==="
python -m src.training.refl_trainer \
    --model-id "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --vlm-model-id "$VLM_MODEL" \
    --vlm-model-revision "$VLM_MODEL_REVISION" \
    --text-embeds-dir "$TEXT_EMBEDS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT" \
    --num-training-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --resolution "$RESOLUTION" \
    --lora-rank "$LORA_RANK" \
    --num-samples "$NUM_SAMPLES"

python -m scripts.run_manifest note "$RUN_DIR/manifest.json" "ReFL training command completed"
