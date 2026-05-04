# Command Catalog

This catalog is the standard execution surface for the Diffusion Text Tuner thesis toolkit. Default automation is CPU-safe; GPU, model-access, OCR, SLURM, and manual diagnostics are explicit opt-in commands.

## Setup

Use Python 3.11 and the committed uv project files for reproducible local tooling:

```bash
uv sync --group dev
```

Install optional dependency groups only for the workflows you intend to run:

```bash
uv sync --group dev --extra gpu
uv sync --group dev --extra ocr --extra reward
uv sync --group dev --extra synthesis
uv sync --group dev --extra plotting --extra analysis
```

For cluster environments that still use the historical conda bootstrap, run:

```bash
bash scripts/cluster/setup_env.sh
```

## CPU-safe default commands

These commands are safe defaults for local automation and broad verification. They do not intentionally launch CUDA, FLUX, Qwen, PaddleOCR, OCR, or SLURM work.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Smoke checks

Smoke checks are explicit preflight commands. The import check should remain lightweight; CUDA, model access, OCR, and cache checks are opt-in diagnostics before long jobs.

```bash
python -m scripts.smoke_environment --check imports
python -m scripts.smoke_environment --check cuda --allow-missing
python -m scripts.smoke_environment --check model-access --allow-missing
python -m scripts.smoke_environment --check ocr --allow-missing
python -m scripts.smoke_environment --check cache --allow-missing
```

## Local pipeline commands

Run these from the repository root after installing the appropriate optional dependencies and preparing required model/cache access. Commands write generated artifacts under ignored runtime roots such as `data/`, `outputs/`, and `runs/` unless you choose different paths.

### Prompt generation

```bash
python -m src.prompt_pipeline.generate \
    --n 1000 \
    --output data/prompts_simple.jsonl \
    --seed 42 \
    --no-llm
```

### Image generation

```bash
python -m scripts.generate_images \
    --prompts data/prompts_simple.jsonl \
    --output_dir outputs/generated \
    --versions_per_prompt 3 \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --resolution 512 \
    --seed 42
```

### Reward scoring

```bash
python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv \
    --scorer vlm \
    --resume
```

### SFT training

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.sft_trainer \
    --config configs/sft.json
```

### DPO training

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.dpo_trainer \
    --config configs/dpo.json
```

### Masked-SFT training

```bash
python -m src.training.masked_sft_trainer \
    --config configs/masked_sft.json
```

### Synthetic data

```bash
python -m scripts.synth.build_dataset \
    --num 1000 \
    --config configs/synth/cyrillic.yaml \
    --raw-dir data/synth_cyrillic/raw \
    --out-masked data/synth_cyrillic/masked_sft \
    --out-anyword data/synth_cyrillic/anyword \
    --seed 42
```

Bake latents and text embeddings only when GPU/model dependencies are available:

```bash
python -m scripts.synth.build_dataset \
    --num 1000 \
    --skip-render \
    --bake-latents \
    --encode-text \
    --device cuda
```

### Evaluation

```bash
python -m src.evaluation.generate_baseline \
    --prompts data/prompts_simple.jsonl \
    --output-dir outputs/baseline \
    --num-samples 500 \
    --seed 42

python -m src.evaluation.evaluate_rewards \
    --metadata outputs/baseline/metadata.jsonl \
    --output outputs/eval/rewards.jsonl \
    --reward vlm ocr
```

### Thesis plotting

```bash
python scripts/thesis_plots/plot_sft_losses.py
python scripts/thesis_plots/plot_dpo_metrics.py
```

## SLURM command variants

The cluster launchers mirror the local generation, scoring, SFT, and DPO stages while writing logs under ignored `runs/` paths. Review each `.sbatch` file for node constraints, prompt counts, config paths, and cache assumptions before submission.

```bash
bash scripts/cluster/setup_env.sh

sbatch --array=0-15 scripts/cluster/generate_images.sbatch

sbatch --array=0-7 scripts/cluster/score_images.sbatch
bash scripts/cluster/merge_scores.sh

sbatch scripts/cluster/sft.sbatch
sbatch scripts/cluster/dpo.sbatch
```

## Manual diagnostics

Manual diagnostics are not part of default pytest discovery. Run them only in an environment with the required CUDA, model, cache, and generated-artifact prerequisites.

```bash
python scripts/diagnose_gradient_flow.py
python scripts/diagnose_grad_magnitude.py
```

OCR/VLM experiments under `experiments/ocr_reward_tests/test_*.py` remain research scripts rather than CPU-safe tests. Do not add them to default test discovery.

## Generated artifacts and git safety

Keep generated outputs, runs, logs, tensors, checkpoints, and generated images out of git. In particular, do not commit artifacts from `outputs/`, `runs/`, generated `data/synth_cyrillic/` payloads, model caches, `.pt` tensors, checkpoint weights, or private prompt/output logs.

Before committing source or documentation changes, inspect the working tree and stage only intentional files:

```bash
git status --short
```

Generated artifacts may contain prompt text, target text, model outputs, local paths, and reward scores. Treat them as environment-specific research outputs unless they are intentionally tiny fixtures or documentation assets.
