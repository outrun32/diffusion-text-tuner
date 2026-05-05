# diffusion-text-tuner

Improving multilingual text rendering in diffusion models via SFT + DPO fine-tuning.

Targets **FLUX.2-klein-base-4B** with a focus on **Cyrillic/Russian** text accuracy, evaluated by a Qwen3.5-9B VLM reward model.

## Execution Surface

Start with the Phase 1 execution docs before launching experiments:

- [`docs/pipeline_inventory.md`](docs/pipeline_inventory.md) separates supported toolkit entry points from historical experiments and manual diagnostics.
- [`docs/commands.md`](docs/commands.md) lists setup, CPU-safe tests, lint/format commands, smoke checks, local commands, SLURM variants, manual diagnostics, and generated-artifact safety notes.
- [`docs/runtime_contracts.md`](docs/runtime_contracts.md) defines canonical runtime paths, artifact schemas, manifest expectations, and generated-artifact git safety.
- [`docs/training_comparability.md`](docs/training_comparability.md) explains CPU-safe Phase 5 comparability checks for baseline, SFT, DPO, masked-SFT, combined, and curriculum training approaches.
- [`configs/experiments/README.md`](configs/experiments/README.md) documents the config-family and naming contract for new SFT, DPO, masked-SFT, reward, synthesis, evaluation, and ablation variants.
- Phase 3 data curriculum and quality docs cover prompt curriculum configs, prompt dataset validation, synthetic quality inspection, materialized SFT/DPO selections, and generated-vs-synthetic source comparison:
  [`docs/data_curriculum.md`](docs/data_curriculum.md),
  [`docs/dataset_quality.md`](docs/dataset_quality.md),
  [`docs/synthetic_quality.md`](docs/synthetic_quality.md),
  [`docs/data_selection.md`](docs/data_selection.md), and
  [`docs/data_source_comparison.md`](docs/data_source_comparison.md).

Default automated tests are **CPU-safe**. GPU, model-access, OCR, SLURM, and gradient diagnostics are explicit **opt-in** commands so broad test discovery does not accidentally launch expensive work.

Phase 4 CPU-safe characterization tests can be discovered from [`docs/commands.md`](docs/commands.md), [`docs/runtime_contracts.md`](docs/runtime_contracts.md), and Makefile aliases. Run the complete characterization suite with:

```bash
make characterization-test
```

Focused aliases are available for `characterization-runtime`, `characterization-datasets`, `characterization-objectives`, `characterization-prompts`, and `characterization-rewards`. These commands cover config/artifact, dataset/collator, objective math/DPO, prompt determinism, and reward wrapper fake tests without loading CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, SynthTIGER, or external model weights.

Phase 5 training comparability commands are CPU-safe and compare recorded manifests
without launching training, CUDA, model, OCR, image, tensor, or checkpoint work.
Use `python -m scripts.compare_run_manifests` for raw manifest diffs,
`python -m scripts.check_training_comparability` for controlled-field mismatch
checks, and `python -m scripts.compare_training_runs` or `make compare-training-runs`
for the integrated baseline/SFT/DPO/masked-SFT/combined/curriculum comparison report.

Run the manifest and preflight steps before long-running GPU/model work: create or inspect a local run manifest, run the appropriate preflight check, and confirm generated artifacts will land under ignored runtime roots such as `outputs/` and `runs/`:

```bash
python -m scripts.run_manifest init --stage sft --config configs/sft.json --command "accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/sft.json"
python -m scripts.run_manifest inspect runs/<run_id>/manifest.json
python -m scripts.preflight_runtime --stage sft --config configs/sft.json --manifest runs/<run_id>/manifest.json --json
```

Preflight validates readiness only; it does not launch generation, scoring, training, synthesis, evaluation, CUDA, model downloads, or OCR work.

Phase 3 report and comparison workflows are CPU-safe by default. generated reports, images, tensors, contact sheets, selections, and comparisons should remain out of git under ignored runtime roots such as `runs/`, `outputs/`, and generated `data/` subtrees unless they are intentionally tiny reviewed fixtures or documentation assets.

## Pipeline

```
prompts (HF dataset)
    │
    ▼
generate_images.py  ──→  latents/ + images/ + text_embeds/
    │
    ▼
score_images.py     ──→  scores.csv  (VLM P(yes) per image)
    │
    ▼
sft_trainer.py      ──→  SFT LoRA   (flow-matching MSE on high-reward samples)
    │
    ▼
dpo_trainer.py      ──→  DPO LoRA   (preference optimization on winner/loser pairs)
```

## Setup

```bash
conda create -n diffusiontuner python=3.11 -y
conda activate diffusiontuner
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers diffusers peft bitsandbytes tqdm pillow
```

## Quick Start

```bash
# 1. Generate images (3 versions per prompt, single GPU)
python -m scripts.generate_images \
    --prompts data/prompts_simple.jsonl \
    --output_dir outputs/generated \
    --versions_per_prompt 3

# 2. Score with VLM
python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv

# 3. SFT
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.sft_trainer --config configs/sft.json

# 4. DPO
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.dpo_trainer --config configs/dpo.json
```

## Cluster (SLURM)

```bash
# Image generation — sharded across GPUs
sbatch --array=0-15 scripts/cluster/generate_images.sbatch

# Scoring — sharded, then merge
sbatch --array=0-7 scripts/cluster/score_images.sbatch
bash scripts/cluster/merge_scores.sh

# Training
sbatch scripts/cluster/sft.sbatch
sbatch scripts/cluster/dpo.sbatch
```

## Project Structure

```
configs/
  sft.json                    # SFT hyperparameters
  dpo.json                    # DPO hyperparameters
  accelerate/                 # HF Accelerate configs (1/4/8 GPU)
data/                         # Prompt datasets (small test files in git)
scripts/
  generate_images.py          # Image generation (single/multi-GPU)
  generate_simple_dataset.py  # Prompt dataset generation via LLM
  score_images.py             # VLM scoring
  cluster/                    # SLURM sbatch scripts
src/
  prompt_pipeline/            # LLM-based prompt generation
  training/
    config.py                 # SFT / DPO / ReFL config dataclasses
    dataset.py                # SFT / DPO PyTorch datasets
    sft_trainer.py            # SFT trainer (accelerate)
    dpo_trainer.py            # DPO trainer (accelerate)
    refl_trainer.py           # ReFL trainer (legacy)
    rewards.py                # Qwen VLM reward model
    flux2_utils.py            # FLUX-specific latent/text utilities
  evaluation/                 # Baseline generation & reward evaluation
experiments/                  # One-off experiments & research scripts
```

## Dataset

Prompt datasets are hosted on HuggingFace: [`Outrun32/cyrillic-prompts-15k`](https://huggingface.co/datasets/Outrun32/cyrillic-prompts-15k)

```python
from datasets import load_dataset
ds = load_dataset("Outrun32/cyrillic-prompts-15k")
```
