<!-- refreshed: 2026-05-04 -->
# Architecture

**Analysis Date:** 2026-05-04

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                  CLI / Batch Pipeline Layer                  │
├──────────────────┬──────────────────┬───────────────────────┤
│ Prompt creation  │ Image generation │ Reward scoring        │
│ `src/prompt_...` │ `scripts/gene...`│ `scripts/score...`    │
├──────────────────┴──────────────────┴───────────────────────┤
│ Synthetic data builder                                       │
│ `scripts/synth/build_dataset.py`                             │
└──────────────┬──────────────────────────┬────────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Training / Evaluation Layer                  │
│ `src/training/sft_trainer.py`                                │
│ `src/training/dpo_trainer.py`                                │
│ `src/training/masked_sft_trainer.py`                         │
│ `src/evaluation/`                                            │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Local Filesystem Artifacts / Models             │
│ `data/` + `outputs/` + `configs/` + Hugging Face models      │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Prompt pipeline | Builds multilingual/Cyrillic prompt JSONL records from sampled text, style, and scene metadata. | `src/prompt_pipeline/generate.py` |
| LLM client | Wraps local Transformers, vLLM, or MLX backends for phrase and scene generation. | `src/prompt_pipeline/llm_client.py` |
| Image generator | Runs FLUX.2 Klein to create prompt variants, save PNGs, and save pre-encoded latents/text embeddings. | `scripts/generate_images.py` |
| Reward scorer | Scores generated images with Qwen VLM and/or PaddleOCR and writes `scores.csv`. | `scripts/score_images.py` |
| Training config | Defines dataclass schemas/defaults for SFT, masked SFT, DPO, ReFL, and LoRA config. | `src/training/config.py` |
| Dataset loaders | Loads precomputed latents, text embeddings, CSV scores, preference pairs, masked-SFT tensors, and resolution buckets. | `src/training/dataset.py` |
| FLUX utilities | Encodes/decodes images, packs/unpacks latents, prepares position IDs, and precomputes Qwen text embeddings. | `src/training/flux2_utils.py` |
| SFT trainer | Trains LoRA adapters with standard flow-matching MSE on high-reward samples. | `src/training/sft_trainer.py` |
| DPO trainer | Trains policy LoRA against frozen reference LoRA using diffusion DPO preference pairs. | `src/training/dpo_trainer.py` |
| Masked SFT trainer | Trains region-weighted LoRA using latent-space text masks and multi-rank LoRA groups. | `src/training/masked_sft_trainer.py` |
| Reward models | Implements Qwen yes-probability and OCR/CER reward implementations. | `src/training/rewards.py` |
| Synthetic data builder | Renders SynthTIGER samples, fans out layouts, bakes latents, and encodes text. | `scripts/synth/build_dataset.py` |
| Evaluation tools | Generates baselines and evaluates reward models on metadata/image outputs. | `src/evaluation/generate_baseline.py`, `src/evaluation/evaluate_rewards.py` |
| Cluster launchers | Run generation, scoring, and training on SLURM. | `scripts/cluster/` |

## Pattern Overview

**Overall:** File-backed machine-learning research pipeline with CLI entry points, dataclass/JSON configuration, PyTorch datasets, and Accelerate-based trainer modules.

**Key Characteristics:**
- Artifacts flow through local directories (`data/` → `outputs/generated/` → `outputs/*/checkpoints`) rather than a database or service API.
- Expensive model components are loaded only inside command/training functions to reduce import-time memory cost; examples are `from diffusers import Flux2KleinPipeline` inside `scripts/generate_images.py` and `src/training/*_trainer.py`.
- Training consumes precomputed latents and prompt embeddings from disk, so trainers in `src/training/` do not keep the FLUX text encoder loaded after setup.
- Config is split between dataclass defaults in `src/training/config.py` and JSON overrides in `configs/*.json`.
- CLI modules use `python -m ...` execution and parse arguments at the module boundary.

## Layers

**Prompt/Data Generation:**
- Purpose: Create text-rendering prompts and optional synthetic masked-SFT datasets.
- Location: `src/prompt_pipeline/`, `scripts/generate_simple_dataset.py`, `scripts/synth/`, and `data/`.
- Contains: Random samplers, scene/style/text assemblers, LLM wrapper, SynthTIGER rendering, data conversion, and dataset download scripts.
- Depends on: `transformers`, optional `vllm`, optional `mlx_lm`, `synthtiger`, `Pillow`, and local data files under `data/`.
- Used by: `scripts/generate_images.py`, `scripts/synth/build_dataset.py`, and training configs pointing at generated files.

**Image Generation and Scoring:**
- Purpose: Generate FLUX image variants and compute reward scores for supervised/preference training.
- Location: `scripts/generate_images.py`, `scripts/score_images.py`, and `src/training/rewards.py`.
- Contains: FLUX pipeline execution, latent/text embedding persistence, Qwen VLM scoring, PaddleOCR scoring, resume/sharding controls.
- Depends on: `diffusers`, `torch`, `transformers`, `bitsandbytes`, `paddleocr`, `Pillow`, `torchvision`.
- Used by: SFT/DPO datasets in `src/training/dataset.py` and configs in `configs/sft.json` and `configs/dpo.json`.

**Training:**
- Purpose: Fine-tune FLUX.2 Klein LoRA adapters with SFT, masked SFT, DPO, or legacy ReFL.
- Location: `src/training/`.
- Contains: Config dataclasses, datasets, losses, latent utilities, reward models, trainer loops, sampling, checkpointing.
- Depends on: `torch`, `accelerate`, `peft`, `diffusers`, `transformers`, local JSON configs in `configs/`, and file artifacts under `outputs/` or `data/synth_cyrillic/`.
- Used by: Local/SLURM launch commands in `README.md` and `scripts/cluster/`.

**Evaluation and Experiments:**
- Purpose: Generate baseline images, measure reward functions, and test OCR/VLM reward approaches.
- Location: `src/evaluation/` and `experiments/ocr_reward_tests/`.
- Contains: Baseline generation, Qwen/PaddleOCR scoring scripts, one-off experiment scripts, and committed sample assets in `experiments/assets/`.
- Depends on: `diffusers`, `transformers`, `paddleocr`, `torch`, `Pillow`.
- Used by: Research comparison and reward-model validation workflows.

**Configuration and Orchestration:**
- Purpose: Store hyperparameters, Accelerate process topology, synthetic data config, and SLURM batch commands.
- Location: `configs/`, `configs/accelerate/`, `configs/synth/`, and `scripts/cluster/`.
- Contains: JSON training configs, YAML Accelerate configs, SynthTIGER configs, `.sbatch` jobs, and shell helpers.
- Depends on: Runtime commands consuming the files.
- Used by: `accelerate launch`, `python -m scripts.synth.build_dataset`, and cluster submission commands.

## Data Flow

### Primary SFT/DPO Request Path

1. Generate or download prompt JSONL records under `data/` with `src/prompt_pipeline/generate.py` or `scripts/download_dataset.py`.
2. Generate multiple FLUX image variants per prompt and save `{latents,text_embeds,images}` under `outputs/generated/` via `scripts/generate_images.py`.
3. Score generated PNGs and write `outputs/generated/scores.csv` with `scripts/score_images.py`.
4. Load high-reward samples with `SFTDataset` from `src/training/dataset.py` and train LoRA through `src/training/sft_trainer.py` using `configs/sft.json`.
5. Load winner/loser pairs with `DPODataset` from `src/training/dataset.py` and train policy/reference LoRA through `src/training/dpo_trainer.py` using `configs/dpo.json`.
6. Save checkpoints and sample images under trainer-specific `output_dir` values such as `outputs/sft` and `outputs/dpo`.

### Masked Synthetic SFT Flow

1. Render raw synthetic text images, masks, metadata, and index files with `scripts/synth/build_dataset.py` and the template `scripts/synth/synthtiger_template.py`.
2. Fan out raw samples into masked-SFT and AnyWord-compatible layouts under `data/synth_cyrillic/` with `scripts/synth/build_dataset.py`.
3. Bake image/mask pairs into `latents/{sample_id}.pt`, `text_embeds/{sample_id}.pt`, and `shapes.csv` through `scripts/synth/build_dataset.py` and `src/training/flux2_utils.py`.
4. Load `MaskedSFTDataset` and optional `ResolutionBucketSampler` from `src/training/dataset.py`.
5. Train region-weighted flow matching with `masked_flow_matching_loss` from `src/training/losses.py` inside `src/training/masked_sft_trainer.py`.

**State Management:**
- Application state is primarily filesystem state: JSONL/CSV files, `.pt` tensors, generated images, checkpoints, logs, and manifests.
- Training state is held in process by PyTorch/Accelerate and persisted through PEFT/optimizer checkpoints under `outputs/*/checkpoints`.
- No database, central state server, or web session state exists.

## Key Abstractions

**Config Dataclasses:**
- Purpose: Provide typed defaults for model IDs, paths, optimizer settings, LoRA targets, sampling, logging, and hardware options.
- Examples: `SFTConfig`, `DPOConfig`, `MaskedSFTConfig`, `ReflConfig`, `LoraConfig`, and `MultiRankLoraConfig` in `src/training/config.py`.
- Pattern: Dataclasses with JSON override loading in trainer `main()` functions.

**PyTorch Datasets and Collators:**
- Purpose: Convert file-backed `.pt` and CSV data into tensors for trainer loops.
- Examples: `SFTDataset`, `DPODataset`, `MaskedSFTDataset`, `sft_collate_fn`, `dpo_collate_fn`, and `masked_sft_collate_fn` in `src/training/dataset.py`.
- Pattern: Dataset classes own file discovery/loading; collators own padding and batch assembly.

**FLUX Latent Utilities:**
- Purpose: Keep latent shape, patchification, BN normalization, position ID, and text-embedding code shared across generation, synthetic data, sampling, and training.
- Examples: `src/training/flux2_utils.py`.
- Pattern: Stateless tensor utility functions.

**Reward Model Objects:**
- Purpose: Encapsulate VLM/OCR model loading and scoring APIs.
- Examples: `QwenYesProbReward` and `OcrCerEntropyReward` in `src/training/rewards.py`; evaluation duplicates live in `src/evaluation/evaluate_rewards.py`.
- Pattern: Class initializes model client once, exposes per-image/batch scoring methods.

**Trainer Modules:**
- Purpose: Combine config, model loading, dataset creation, optimizer/scheduler, train loop, sampling, logging, and checkpointing.
- Examples: `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, and `src/training/refl_trainer.py`.
- Pattern: `train(cfg)` function plus `main()` CLI wrapper.

## Entry Points

**Prompt Generation:**
- Location: `src/prompt_pipeline/generate.py`
- Triggers: `python -m src.prompt_pipeline.generate ...`
- Responsibilities: Generate prompt JSONL, optionally use LLM backends, expand scene pools, and write data files under `data/`.

**Dataset Download:**
- Location: `scripts/download_dataset.py`
- Triggers: `python scripts/download_dataset.py --output data/prompts_simple.jsonl`
- Responsibilities: Load `Outrun32/cyrillic-prompts-15k` from Hugging Face and write local JSONL prompt records.

**Image Generation:**
- Location: `scripts/generate_images.py`
- Triggers: `python -m scripts.generate_images ...` or `scripts/cluster/generate_images.sbatch`
- Responsibilities: Load FLUX pipeline, generate image variants, save PNGs, encode latents, and save prompt embeddings.

**Scoring:**
- Location: `scripts/score_images.py`
- Triggers: `python -m scripts.score_images ...` or `scripts/cluster/score_images.sbatch`
- Responsibilities: Score PNGs with VLM/OCR rewards, support sharding/resume, and write CSV scores.

**SFT Training:**
- Location: `src/training/sft_trainer.py`
- Triggers: `accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/sft.json`
- Responsibilities: Train LoRA on high-reward samples, log metrics, sample images, and save checkpoints.

**DPO Training:**
- Location: `src/training/dpo_trainer.py`
- Triggers: `accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.dpo_trainer --config configs/dpo.json`
- Responsibilities: Train policy LoRA against frozen reference on preference pairs.

**Masked SFT Training:**
- Location: `src/training/masked_sft_trainer.py`
- Triggers: `python -m src.training.masked_sft_trainer --config configs/masked_sft.json` or an Accelerate launch command.
- Responsibilities: Train multi-rank LoRA with region-weighted latent mask loss and validation/eval suite support.

**Synthetic Dataset Build:**
- Location: `scripts/synth/build_dataset.py`
- Triggers: `python -m scripts.synth.build_dataset ...`
- Responsibilities: Render SynthTIGER samples, prepare masked and AnyWord layouts, bake latents, and encode text embeddings.

## Architectural Constraints

- **Threading:** Python CLI processes with PyTorch CUDA execution. DataLoader worker processes are used in trainer modules; SynthTIGER rendering uses worker counts supplied to `scripts/synth/build_dataset.py`.
- **Global state:** Module-level loggers are used across `src/training/*.py`, `scripts/*.py`, and `src/prompt_pipeline/*.py`; `src/training/rewards.py` sets `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` at import time.
- **Circular imports:** No confirmed circular dependency chain was detected in fast mapping. Notable trainer reuse exists: `src/training/dpo_trainer.py` imports sampling helpers from `src/training/sft_trainer.py`; `src/training/masked_sft_trainer.py` imports `compute_sigma`/`decode_latents` from `src/training/sft_trainer.py` and `FlowMatchScheduler` from `src/training/refl_trainer.py`.
- **GPU memory:** Main code paths assume large CUDA models; trainers explicitly delete text encoders/tokenizers after embeddings are no longer needed in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py`.
- **Filesystem artifacts:** `.gitignore` excludes `outputs/`, `runs/`, tensors, weights, generated images, and logs; new production data should not be committed unless intentionally small fixtures.

## Anti-Patterns

### Adding New Pipeline Logic Directly to Large Trainer Modules

**What happens:** Trainer modules such as `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py` already combine CLI parsing, model loading, optimizer setup, sampling, logging, and checkpointing.
**Why it's wrong:** Additional unrelated logic increases GPU-heavy module complexity and makes reuse/testing harder.
**Do this instead:** Put shared tensor/data logic into focused helpers like `src/training/flux2_utils.py`, `src/training/losses.py`, or `src/training/dataset.py`, then call those helpers from trainer `train(cfg)` functions.

### Recomputing Text Embeddings Inside Training Loops

**What happens:** FLUX text encoders are very large, and this codebase intentionally precomputes text embeddings in `scripts/generate_images.py` and `src/training/flux2_utils.py`.
**Why it's wrong:** Loading text encoders during every training step increases VRAM use and slows training.
**Do this instead:** Save embeddings under `outputs/generated/text_embeds/` or `data/synth_cyrillic/masked_sft/text_embeds/` and load them through dataset classes in `src/training/dataset.py`.

### Committing Generated Artifacts

**What happens:** Generated tensors, checkpoints, logs, images, and large prompt files are ignored in `.gitignore`.
**Why it's wrong:** These files are large, environment-specific, and can include experimental output that should not be source-controlled.
**Do this instead:** Keep generated artifacts under ignored paths such as `outputs/`, `runs/`, and `data/synth_cyrillic/`; commit only source code, small configs, docs, tests, and intentional fixtures.

## Error Handling

**Strategy:** Fail fast for missing required files/configurations, skip recoverable missing samples, and log progress/warnings for long-running batch jobs.

**Patterns:**
- Raise `FileNotFoundError`, `RuntimeError`, or `ValueError` for missing datasets, empty sample sets, bad LoRA module matches, and invalid config options in `src/training/dataset.py`, `src/training/masked_sft_trainer.py`, and `src/training/losses.py`.
- Skip incomplete data records with warnings in `src/training/dataset.py`, `scripts/score_images.py`, and `scripts/synth/build_dataset.py`.
- Use `subprocess.run(..., check=True)` in `scripts/synth/build_dataset.py` so render failures stop the build.
- Use resume/skip behavior for idempotent long jobs in `scripts/generate_images.py` and `scripts/score_images.py`.

## Cross-Cutting Concerns

**Logging:** Use module-level `logging.getLogger(__name__)` with `logging.basicConfig(...)` in CLI `main()` functions such as `scripts/generate_images.py`, `scripts/score_images.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/prompt_pipeline/generate.py`.
**Validation:** Use dataclass defaults plus JSON config loading in `src/training/config.py` and trainer `main()` functions; validate file presence in dataset constructors and synthetic builders.
**Authentication:** No app auth exists; external Hugging Face access is left to the runtime environment.
**Reproducibility:** Seeds are threaded through prompt generation, image generation, and trainer configs in `src/prompt_pipeline/generate.py`, `scripts/generate_images.py`, and `src/training/config.py`.
**Project Skills:** No `.claude/skills/` or `.agents/skills/` directory was detected in this repository.

---

*Architecture analysis: 2026-05-04*
