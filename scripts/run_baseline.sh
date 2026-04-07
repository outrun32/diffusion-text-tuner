#!/usr/bin/env bash
# Generate baseline images and evaluate rewards.
# Run on VPS / GPU machine with CUDA.
set -eu pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS="${PROMPTS:-data/prompts_llm.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline}"
MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-4B}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-42}"

# ── Step 1: Generate baseline images ───────────────────────────────────────
echo "=== Step 1: Generating baseline images ==="
python -m src.evaluation.generate_baseline \
    --prompts "$PROMPTS" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"

# ── Step 2: Evaluate with PaddleOCR ───────────────────────────────────────
echo ""
echo "=== Step 2: Evaluating with PaddleOCR ==="
python -m src.evaluation.evaluate_rewards \
    --metadata "$OUTPUT_DIR/metadata.jsonl" \
    --output "$OUTPUT_DIR/scores.jsonl" \
    --reward paddleocr

# ── Step 3: Evaluate with Qwen yes-prob ───────────────────────────────────
echo ""
echo "=== Step 3: Evaluating with Qwen yes-prob ==="
python -m src.evaluation.evaluate_rewards \
    --metadata "$OUTPUT_DIR/metadata.jsonl" \
    --output "$OUTPUT_DIR/scores_qwen.jsonl" \
    --reward qwen_yes_prob

echo ""
echo "=== Done ==="
