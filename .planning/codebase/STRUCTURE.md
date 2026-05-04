# Codebase Structure

**Analysis Date:** 2026-05-04

## Directory Layout

```text
diffusion-text-tuner/
├── configs/                 # Training, evaluation, Accelerate, and synthetic-data configs
│   ├── accelerate/          # Hugging Face Accelerate YAML topology configs
│   └── synth/               # SynthTIGER rendering configuration
├── data/                    # Prompt data, scene/style resources, backgrounds, fonts, synthetic data roots
├── docs/                    # Research/reference notes
├── experiments/             # One-off reward/OCR experiments and small committed assets
├── outputs/                 # Ignored generated images, latents, scores, samples, checkpoints
├── runs/                    # Ignored training/evaluation run logs
├── scripts/                 # CLI scripts for pipeline stages, cluster jobs, plots, profiling, synthesis
│   ├── cluster/             # SLURM submission and merge helpers
│   ├── synth/               # Synthetic masked-SFT dataset rendering/build tools
│   └── thesis_plots/        # Plotting scripts for thesis/research outputs
├── src/                     # Importable project code
│   ├── evaluation/          # Baseline generation and reward evaluation
│   ├── prompt_pipeline/     # Prompt generation components and local LLM client
│   └── training/            # Configs, datasets, losses, FLUX utilities, rewards, trainers
├── tests/                   # Lightweight tests
├── README.md                # Pipeline, setup, and quick-start documentation
└── .gitignore               # Ignores generated data, weights, logs, environments, and secrets
```

## Directory Purposes

**`configs/`:**
- Purpose: Store runtime hyperparameters and launcher configuration.
- Contains: SFT/DPO/masked-SFT JSON configs, eval suite JSON, Accelerate YAML, SynthTIGER YAML.
- Key files: `configs/sft.json`, `configs/dpo.json`, `configs/masked_sft.json`, `configs/eval_suite.json`, `configs/accelerate/single_gpu.yaml`, `configs/accelerate/multi_gpu.yaml`, `configs/synth/cyrillic.yaml`.

**`data/`:**
- Purpose: Store source prompt datasets, prompt-generation resources, fonts/backgrounds, and synthetic data roots.
- Contains: JSONL prompts, Russian frequency/thematic files, scene seed JSON, fonts/backgrounds, `data/synth_cyrillic/`.
- Key files: `data/prompts_simple_test.jsonl`, `data/ru_freq_50k.txt`, `data/thematic.json`, `data/scenes_seed.json`, `data/backgrounds/unsplash_meta/README.md`.

**`scripts/`:**
- Purpose: Provide executable pipeline steps, utilities, experiments, plotting, profiling, and cluster launchers.
- Contains: Image generation, scoring, dataset download/generation, baseline scripts, shell helpers, synthetic builders, and profiling scripts.
- Key files: `scripts/generate_images.py`, `scripts/score_images.py`, `scripts/download_dataset.py`, `scripts/generate_simple_dataset.py`, `scripts/profile_step.py`, `scripts/cluster/`, `scripts/synth/`.

**`scripts/cluster/`:**
- Purpose: Run pipeline stages on SLURM.
- Contains: `.sbatch` jobs for image generation, scoring, SFT, DPO, merge, and environment setup.
- Key files: `scripts/cluster/generate_images.sbatch`, `scripts/cluster/score_images.sbatch`, `scripts/cluster/sft.sbatch`, `scripts/cluster/dpo.sbatch`, `scripts/cluster/merge_scores.sh`, `scripts/cluster/setup_env.sh`.

**`scripts/synth/`:**
- Purpose: Build synthetic Cyrillic masked-SFT datasets.
- Contains: SynthTIGER template/runner, dataset builder, caption builder, masked dataset filter, fixture generation, background fetch, and word sampling.
- Key files: `scripts/synth/build_dataset.py`, `scripts/synth/synthtiger_template.py`, `scripts/synth/run_synthtiger.py`, `scripts/synth/filter_masked_dataset.py`, `scripts/synth/make_fixture.py`, `scripts/synth/fetch_unsplash.py`, `scripts/synth/word_sampler.py`.

**`src/training/`:**
- Purpose: Own training-time abstractions and trainer entry points.
- Contains: Dataclass configs, file-backed datasets/collators, latent utilities, rewards, loss functions, and SFT/DPO/masked-SFT/ReFL trainers.
- Key files: `src/training/config.py`, `src/training/dataset.py`, `src/training/flux2_utils.py`, `src/training/losses.py`, `src/training/rewards.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, `src/training/refl_trainer.py`.

**`src/prompt_pipeline/`:**
- Purpose: Generate prompt JSONL datasets for text-rendering tasks.
- Contains: LLM wrapper, scene/style/text generators, assembler, config constants, and main CLI.
- Key files: `src/prompt_pipeline/generate.py`, `src/prompt_pipeline/llm_client.py`, `src/prompt_pipeline/config.py`, `src/prompt_pipeline/assembler.py`, `src/prompt_pipeline/scene_pool.py`, `src/prompt_pipeline/style_generator.py`, `src/prompt_pipeline/text_generator.py`.

**`src/evaluation/`:**
- Purpose: Generate baselines and compute reward metrics outside training.
- Contains: Baseline image generation and reward evaluation scripts.
- Key files: `src/evaluation/generate_baseline.py`, `src/evaluation/evaluate_rewards.py`.

**`experiments/`:**
- Purpose: Hold one-off OCR/VLM reward experiments and small assets.
- Contains: OCR reward tests and `experiments/assets/` sample images allowed by `.gitignore`.
- Key files: `experiments/ocr_reward_tests/test_qwen_yes_prob.py`, `experiments/ocr_reward_tests/test_paddleocr.py`, `experiments/ocr_reward_tests/test_paddleocr_v5.py`, `experiments/ocr_reward_tests/test_trocr.py`.

**`tests/`:**
- Purpose: Store lightweight test coverage for core utilities.
- Contains: One script-style test module.
- Key files: `tests/test_losses.py`.

## Key File Locations

**Entry Points:**
- `src/prompt_pipeline/generate.py`: Prompt dataset generation CLI.
- `scripts/download_dataset.py`: Hugging Face prompt dataset download CLI.
- `scripts/generate_images.py`: FLUX image, latent, and text-embedding generation CLI.
- `scripts/score_images.py`: Reward scoring CLI for generated images.
- `src/training/sft_trainer.py`: SFT LoRA trainer CLI.
- `src/training/dpo_trainer.py`: DPO LoRA trainer CLI.
- `src/training/masked_sft_trainer.py`: Masked-SFT LoRA trainer CLI.
- `src/evaluation/generate_baseline.py`: Baseline generation CLI.
- `src/evaluation/evaluate_rewards.py`: Reward evaluation CLI.
- `scripts/synth/build_dataset.py`: Synthetic dataset build CLI.

**Configuration:**
- `src/training/config.py`: Dataclass defaults and schema for all training modes.
- `configs/sft.json`: SFT runtime config.
- `configs/dpo.json`: DPO runtime config.
- `configs/masked_sft.json`: Masked-SFT runtime config.
- `configs/eval_suite.json`: Masked-SFT validation/eval suite items.
- `configs/accelerate/`: Accelerate YAML configs for local and multi-GPU launches.
- `configs/synth/cyrillic.yaml`: SynthTIGER Cyrillic render config.

**Core Logic:**
- `src/training/dataset.py`: File-backed PyTorch datasets and collators.
- `src/training/flux2_utils.py`: FLUX latent/text utility functions.
- `src/training/losses.py`: Masked flow-matching loss and mask downsampling.
- `src/training/rewards.py`: Qwen/PaddleOCR reward implementations.
- `src/prompt_pipeline/llm_client.py`: Local LLM backend abstraction.
- `src/prompt_pipeline/assembler.py`: Prompt assembly from text/scene/style components.

**Testing:**
- `tests/test_losses.py`: Tests for masked flow-matching losses.
- `experiments/ocr_reward_tests/`: Experimental scripts, not formal test-suite coverage.

## Naming Conventions

**Files:**
- `snake_case.py`: Python modules use snake_case, e.g. `src/training/masked_sft_trainer.py`, `src/training/flux2_utils.py`, and `scripts/generate_images.py`.
- `<mode>.json`: Training configs use short mode names and variants, e.g. `configs/sft.json`, `configs/dpo_ocr.json`, `configs/masked_sft_strict.json`.
- `.sbatch`: Cluster jobs use stage names, e.g. `scripts/cluster/score_images.sbatch` and `scripts/cluster/dpo.sbatch`.

**Directories:**
- Domain-oriented package directories under `src/`: `src/training/`, `src/prompt_pipeline/`, and `src/evaluation/`.
- Pipeline/tooling directories under `scripts/`: `scripts/cluster/`, `scripts/synth/`, and `scripts/thesis_plots/`.
- Artifact directories use stage names under ignored roots, e.g. `outputs/generated/`, `outputs/sft/`, `outputs/dpo/`, and `data/synth_cyrillic/`.

## Where to Add New Code

**New Training Feature:**
- Primary code: Add shared data/loss/utility logic to `src/training/dataset.py`, `src/training/losses.py`, or `src/training/flux2_utils.py`.
- Trainer integration: Add mode-specific orchestration to `src/training/<mode>_trainer.py`; keep `train(cfg)` as the main orchestration function and `main()` as CLI-only parsing.
- Config: Add dataclass fields to `src/training/config.py` and JSON runtime configs under `configs/`.
- Tests: Add utility/loss tests under `tests/`, following `tests/test_losses.py`.

**New Prompt Generation Feature:**
- Primary code: Add reusable sampling/assembly logic under `src/prompt_pipeline/`.
- CLI behavior: Extend `src/prompt_pipeline/generate.py` only for command-line orchestration and file output.
- Data resources: Place small committed resources under `data/`; keep generated large JSONL files ignored unless intentionally published.

**New Scoring or Reward Feature:**
- Primary code: Add reward model class or helpers to `src/training/rewards.py` if needed by training/scoring.
- Batch CLI integration: Extend `scripts/score_images.py` to select and write the reward output.
- Evaluation-only logic: Put research comparison code in `src/evaluation/` or `experiments/ocr_reward_tests/`.

**New Synthetic Dataset Feature:**
- Rendering/template changes: Add to `scripts/synth/synthtiger_template.py` or related helpers in `scripts/synth/`.
- Build pipeline changes: Add orchestration to `scripts/synth/build_dataset.py`.
- Config changes: Add render config under `configs/synth/`.

**New Cluster Job:**
- Implementation: Add `.sbatch` files under `scripts/cluster/`.
- Shared setup: Reuse or update `scripts/cluster/setup_env.sh`.
- Local CLI parity: Ensure the same stage can run with `python -m ...` or `accelerate launch ...` from `README.md` patterns.

**Utilities:**
- Shared ML/tensor helpers: `src/training/flux2_utils.py` or a new focused module under `src/training/`.
- Research plotting: `scripts/thesis_plots/` or `scripts/plot_metrics.py`.
- One-off diagnostics/profiling: `scripts/` with clear CLI args and no import-time model loading.

## Special Directories

**`outputs/`:**
- Purpose: Generated images, latents, embeddings, scores, checkpoints, samples, and logs.
- Generated: Yes.
- Committed: No; ignored by `.gitignore`.

**`runs/`:**
- Purpose: Training/evaluation run logs or TensorBoard-like outputs.
- Generated: Yes.
- Committed: No; ignored by `.gitignore`.

**`data/synth_cyrillic/`:**
- Purpose: Synthetic masked-SFT and AnyWord-format datasets generated by `scripts/synth/build_dataset.py`.
- Generated: Yes.
- Committed: Partially possible for metadata/config-like files, but generated tensors/images should remain ignored by `.gitignore` patterns.

**`data/backgrounds/unsplash_meta/`:**
- Purpose: Documentation/metadata for Unsplash backgrounds.
- Generated: No for docs; background assets may be external/generated.
- Committed: Metadata docs are committed; generated/downloaded images follow `.gitignore` image rules unless explicitly unignored.

**`experiments/assets/`:**
- Purpose: Small image assets for OCR/VLM experiments.
- Generated: No for committed fixtures.
- Committed: Yes for `.png` and `.jpg` due `.gitignore` exceptions.

**`.planning/codebase/`:**
- Purpose: GSD codebase maps consumed by planning/execution workflows.
- Generated: Yes.
- Committed: Project workflow dependent; current mapping writes `STACK.md`, `INTEGRATIONS.md`, `ARCHITECTURE.md`, and `STRUCTURE.md`.

---

*Structure analysis: 2026-05-04*
