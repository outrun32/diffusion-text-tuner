# Reward-filtered self-training for Cyrillic text rendering

This repository contains the code for a bachelor-thesis study on adapting an open diffusion transformer to **Cyrillic/Russian text rendering** without full retraining.

**Main result:** in this setup, **reward-filtered generated-output self-training** is the most reliable adaptation route. DPO-style preference refinement trains stably and can improve some automatic scores, but it does not cleanly supersede product-score self-training.

The project uses **FLUX.2-klein-base-4B** with LoRA adapters. Generated image candidates are scored with a Qwen VLM reward, a PaddleOCR/CER reward, and their product. High-scoring candidates become self-training data; scored winner/loser variants become DPO-style preference pairs.

## Thesis claim in one sentence

> Reward-filtered generated-output self-training can substantially improve Cyrillic text rendering in an open diffusion transformer, while DPO-style refinement is more fragile under noisy and off-policy preference data.

## What is in this repo

The repository is a research codebase, not a packaged application. It includes:

- prompt generation for Cyrillic/multilingual text-rendering prompts;
- FLUX candidate generation with saved images, latents, and text embeddings;
- VLM/OCR/product reward scoring;
- LoRA SFT/self-training and DPO-style training;
- synthetic masked-SFT data tooling;
- held-out benchmark and diagnostic report utilities;
- final experiment configs used for the thesis runs.

Large generated artifacts are intentionally **not** committed. Local training outputs, checkpoints, generated images, tensors, and benchmark runs live under ignored paths such as `outputs/`, `runs/`, and generated `data/` subtrees.

## Method overview

```text
Prompt set
   │
   ▼
Base FLUX candidate generation
   │  3 image variants per prompt in the final training pool
   ▼
Automatic scoring
   │  VLM reward, OCR reward, product reward
   ├───────────────► reward-filtered SFT / self-training
   │                    imitate high-scoring generated samples
   ▼
Preference pairs
   │  best-vs-worst scored variants per prompt
   ▼
DPO-style refinement
      pairwise objective using flow-matching MSE as a surrogate
```

The self-training objective is standard flow-matching MSE on selected generated samples:

```text
L_SFT = || v_theta(x_sigma, t, c) - (epsilon - x_0) ||^2
```

The DPO-style objective uses a pairwise surrogate margin from winner/loser flow-matching MSE:

```text
L_DPO = - E log sigmoid(beta(t) * (Delta_w - Delta_l))
Delta_i = - (L_theta(x_i, t) - L_ref(x_i, t))
```

This is **not canonical language-model DPO**. The preference margin is a diffusion flow-matching surrogate for relative fit.

## Reward signals

| Reward | Role | Notes |
|---|---|---|
| VLM | Qwen yes/no judge for exact target text presence | Can preserve scene-level quality, but may miss subtle character errors. |
| OCR | PaddleOCR CER and CTC entropy | Makes character errors visible, but can fail on stylized text. |
| Product | `VLM * OCR` in the final runs | High only when both evaluators agree; useful but biased toward easier samples. |

For final analysis, strict CER, homoglyph-normalized CER, exact match, and script-mixing diagnostics are reported separately. This matters because Latin/Cyrillic homoglyph normalization can hide script-mixing if used as the only metric.

## Final experiments

Final SFT/self-training configs:

```text
configs/experiments/sft/sft_vlm_final.json
configs/experiments/sft/sft_ocr_final.json
configs/experiments/sft/sft_product_final.json
```

Final DPO configs:

```text
configs/experiments/dpo/dpo_vlm_final.json
configs/experiments/dpo/dpo_ocr_final.json
configs/experiments/dpo/dpo_product_final.json
```

Fixed qualitative sample suite:

```text
configs/experiments/evaluation/cyrillic_final_sample_suite.json
```

Prompts in that suite:

```text
МИР
ЮЛЯ
ПОДЪЕЗД
ЩУКА
СЪЕМКА
ДОБРО ПОЖАЛОВАТЬ
```

## Final held-out benchmark summary

A 120-prompt held-out diagnostic benchmark was used for the final defense analysis. Each model generated one image per prompt with matched settings.

| Run | Strict CER ↓ | Homoglyph-normalized CER ↓ | Strict exact ↑ | Normalized exact ↑ | OCR score ↑ | VLM score ↑ | Product ↑ | Script-mix ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.984 | 0.859 | 33.3% | 41.7% | 0.560 | 0.658 | 0.392 | 23.3% |
| Product SFT | 0.238 | 0.126 | 40.8% | 50.0% | 0.635 | 0.681 | 0.444 | 16.7% |
| Product DPO | 0.287 | 0.168 | 44.2% | 52.5% | 0.639 | 0.704 | 0.470 | 17.5% |

Interpretation:

- Product SFT gives the strongest CER improvement and is the safest demo checkpoint.
- Product DPO improves some automatic exact/VLM/product scores, but its CER is worse than Product SFT.
- DPO pairs are generated from base-model candidates while the DPO policy starts from an SFT checkpoint, so the preference data is off-policy relative to initialization.

## Product selection bias

The product reward is useful as a selector, but it changes the training-data distribution.

| Quantity | All candidates | Product-selected | Delta |
|---|---:|---:|---:|
| Mean target length | 14.551 | 10.536 | -4.015 |
| Median target length | 15.000 | 8.000 | -7.000 |
| Hard-glyph share | 93.9% | 88.2% | -5.7% |
| Tier 1 share | 35.2% | 62.2% | +27.0% |
| Tier 2 share | 45.1% | 37.6% | -7.6% |
| Tier 3 share | 19.7% | 0.2% | -19.5% |

So Product SFT is the best demonstration checkpoint here, but not proof that the product reward is universally best.

## Setup

Python 3.11 is expected.

```bash
# using uv
uv sync --extra gpu --extra reward --extra plotting --extra test

# or with conda/pip
conda create -n diffusiontuner python=3.11 -y
conda activate diffusiontuner
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers diffusers peft bitsandbytes tqdm pillow paddleocr matplotlib pandas
```

Some workflows need separate environment handling for PaddleOCR depending on CUDA/Paddle wheel availability.

## Core commands

Generate prompt candidates:

```bash
python -m scripts.generate_images \
  --prompts data/prompts_simple.jsonl \
  --output_dir outputs/generated \
  --model_id black-forest-labs/FLUX.2-klein-base-4B \
  --versions_per_prompt 3 \
  --num_inference_steps 50 \
  --guidance_scale 4.0 \
  --resolution 512
```

Score generated candidates:

```bash
python -m scripts.score_images \
  --images_dir outputs/generated/images \
  --text_embeds_dir outputs/generated/text_embeds \
  --output_csv outputs/generated/scores.csv \
  --scorer both
```

Run reward-filtered self-training:

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  -m src.training.sft_trainer \
  --config configs/experiments/sft/sft_product_final.json
```

Run DPO-style refinement:

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  -m src.training.dpo_trainer \
  --config configs/experiments/dpo/dpo_product_final.json
```

Build final diagnostic reports from recorded score files:

```bash
python -m scripts.final_benchmark make-prompts \
  --output runs/final_benchmark/prompts.jsonl \
  --count-per-slice 20

python -m scripts.final_benchmark product-bias \
  --prompts data/prompts_simple.jsonl \
  --scores outputs/generated/scores_product.csv \
  --output-json runs/final_reports/product_bias.json

python -m scripts.final_benchmark dpo-provenance \
  --prompts data/prompts_simple.jsonl \
  --scores outputs/generated/scores_product.csv \
  --reward-name product \
  --winner-threshold 0.3 \
  --output-json runs/final_reports/dpo_product_provenance.json
```

## Repository structure

```text
configs/                 Experiment, accelerate, prompt, and final-run configs
scripts/                 Thin CLI entry points and cluster helpers
src/generation/          FLUX candidate generation pipeline
src/scoring/             Batch reward-scoring pipeline
src/training/            Dataset loaders, losses, SFT, DPO, masked-SFT trainers
src/evaluation/          Reward contracts, diagnostics, benchmark reports
src/prompt_pipeline/     Prompt/data generation components
src/data_quality/        Prompt/synthetic quality and source-comparison helpers
docs/                    Runtime, reward, evaluation, and command notes
tests/                   Lightweight characterization tests
experiments/             Small historical OCR/VLM experiments and fixtures
```

Ignored local artifact roots:

```text
outputs/   generated images, latents, text embeddings, scores, checkpoints
runs/      local benchmark/report outputs
data/      large generated datasets and local backgrounds, except small source resources
```

## Dataset

Prompt datasets used in development were also published on Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("Outrun32/cyrillic-prompts-15k")
```

## Limitations

- Final evidence is Cyrillic/Russian only, even though the toolkit is multilingual-oriented.
- Reward-filtered self-training can only amplify patterns that the base model can generate at least occasionally.
- OCR and VLM rewards are useful but imperfect; neither is treated as ground truth.
- DPO-style refinement is sensitive to noisy, weakly separated, and off-policy preference pairs.
- Synthetic masked-SFT tooling exists, but the final reported result emphasizes generated-output self-training and DPO-style refinement.

## Bottom line

The reliable route in this study is not full retraining and not DPO alone. It is:

```text
base-generated candidates → automatic reward filtering → LoRA self-training
```

For Cyrillic text rendering with FLUX.2 Klein, this route produced the clearest and most defensible improvement.
