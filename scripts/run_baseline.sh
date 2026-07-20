#!/usr/bin/env bash
# Generate one base-model candidate per prompt and score it with the thesis reward.
# Linux/CUDA only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PROMPTS="${PROMPTS:-data/prompts_simple.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline}"
MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-base-4B}"
MODEL_REVISION="${MODEL_REVISION:-a3b4f4849157f664bdbc776fd7453c2783562f4d}"
VLM_REVISION="${VLM_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"
OCR_DEVICE="${OCR_DEVICE:-cpu}"

python -m scripts.preflight_runtime \
    --stage generate \
    --prompts "$PROMPTS" \
    --output-dir "$OUTPUT_DIR" \
    --json
RUN_DIR="$(python -m scripts.run_manifest init --stage generate --command "python -m scripts.generate_images --prompts $PROMPTS --output_dir $OUTPUT_DIR")"

python -m scripts.generate_images \
    --prompts "$PROMPTS" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL" \
    --model_revision "$MODEL_REVISION" \
    --versions_per_prompt 1 \
    --batch_size 1 \
    --num_inference_steps "$STEPS" \
    --guidance_scale 4.0 \
    --resolution 512 \
    --seed "$SEED" \
    --manifest_path "$OUTPUT_DIR/generation.manifest.json" \
    --run_manifest_path "$RUN_DIR/manifest.json" \
    --device cuda

python -m scripts.score_images \
    --images_dir "$OUTPUT_DIR/images" \
    --text_embeds_dir "$OUTPUT_DIR/text_embeds" \
    --output_csv "$OUTPUT_DIR/scores.csv" \
    --scorer both \
    --vlm_model_revision "$VLM_REVISION" \
    --ocr_device "$OCR_DEVICE" \
    --product_formula product \
    --source_manifest "$OUTPUT_DIR/generation.manifest.json"

echo "Baseline scores: $OUTPUT_DIR/scores.csv"
python -m scripts.run_manifest note "$RUN_DIR/manifest.json" "Baseline generation and scoring completed"
