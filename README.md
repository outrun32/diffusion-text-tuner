# diffusion-text-tuner

Improving multilingual text rendering in diffusion models via SFT + DPO fine-tuning.

Targets **FLUX.2-klein-base-4B** with a focus on **Cyrillic/Russian** text accuracy, evaluated by a Qwen3.5-9B VLM reward model.

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
