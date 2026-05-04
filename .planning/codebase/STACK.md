# Technology Stack

**Analysis Date:** 2026-05-04

## Languages

**Primary:**
- Python 3.11 - Primary runtime for prompt generation, image generation, scoring, training, evaluation, and synthetic dataset tooling in `src/`, `scripts/`, `tests/`, and `experiments/ocr_reward_tests/`. Python 3.11 is specified in `README.md:28`.

**Secondary:**
- Bash / SLURM shell scripts - Cluster orchestration lives in `scripts/cluster/*.sbatch`, `scripts/cluster/merge_scores.sh`, and top-level helpers such as `scripts/run_all_experiments.sh`.
- JSON / YAML - Experiment and runtime configuration lives in `configs/*.json`, `configs/accelerate/*.yaml`, and `configs/synth/cyrillic.yaml`.

## Runtime

**Environment:**
- Conda environment named `diffusiontuner` with Python 3.11 is the documented local setup in `README.md:28-31`.
- CUDA GPU execution is assumed by the main pipeline: `scripts/generate_images.py` moves the FLUX pipeline to `cuda`, `scripts/score_images.py` scores tensors on `cuda`, and `src/training/*.py` use Hugging Face Accelerate devices.
- No committed `requirements.txt`, `pyproject.toml`, `setup.py`, or lockfile is present at the repository root.

**Package Manager:**
- pip - Installation is documented through direct `pip install` commands in `README.md:30-31`.
- Lockfile: missing. No `requirements.txt`, `uv.lock`, `poetry.lock`, `Pipfile.lock`, or conda environment file is committed.

## Frameworks

**Core:**
- PyTorch - Tensor, dataset, dataloader, loss, and gradient execution throughout `src/training/dataset.py`, `src/training/losses.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py`.
- Hugging Face Diffusers - FLUX.2 Klein model loading and pipeline execution via `Flux2KleinPipeline` in `scripts/generate_images.py`, `src/training/flux2_utils.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, and `src/evaluation/generate_baseline.py`.
- Hugging Face Transformers - Qwen text generation and VLM reward models in `src/prompt_pipeline/llm_client.py`, `src/training/rewards.py`, and `src/evaluation/evaluate_rewards.py`.
- PEFT - LoRA adapters for FLUX transformer tuning in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, and `src/training/refl_trainer.py`.
- Hugging Face Accelerate - Single/multi-GPU training launcher integration in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py` with configs in `configs/accelerate/`.

**Testing:**
- Python script-style tests - `tests/test_losses.py` is the only committed test file; no pytest configuration file is present.
- PyTorch assertions - `tests/test_losses.py` validates tensor loss behavior for `src/training/losses.py`.

**Build/Dev:**
- tqdm - Progress bars in CLI pipelines such as `scripts/generate_images.py`, `scripts/score_images.py`, `src/prompt_pipeline/generate.py`, and `scripts/synth/build_dataset.py`.
- Pillow - Image IO and conversion in `scripts/generate_images.py`, `scripts/score_images.py`, `src/training/rewards.py`, and synthetic-data scripts under `scripts/synth/`.
- torchvision - Image-to-tensor conversion in `scripts/generate_images.py` and `scripts/score_images.py`.
- tensorboard via Accelerate - Training logs are configured with `log_with="tensorboard"` and `project_dir=cfg.output_dir` in `src/training/sft_trainer.py` and `src/training/dpo_trainer.py`.

## Key Dependencies

**Critical:**
- `torch` - Required for all model execution, dataloading, losses, latent utilities, and tests in `src/training/` and `tests/test_losses.py`.
- `diffusers` - Provides `Flux2KleinPipeline` and FLUX model classes used by generation, text embedding, baseline, and trainer modules in `scripts/` and `src/`.
- `transformers` - Provides Qwen VLM, text-generation, tokenization, and schedule helpers in `src/training/rewards.py`, `src/prompt_pipeline/llm_client.py`, and trainer schedulers.
- `accelerate` - Required for distributed and mixed-precision training in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py`.
- `peft` - Required for LoRA injection, adapter loading, and checkpointing in `src/training/*.py`.
- `bitsandbytes` - Required by the 4-bit Qwen VLM reward loader through `BitsAndBytesConfig` in `src/training/rewards.py`.

**Infrastructure:**
- `datasets` - Downloads hosted Hugging Face datasets in `scripts/download_dataset.py`.
- `paddleocr` - OCR reward path for scoring and evaluation in `src/training/rewards.py`, `src/evaluation/evaluate_rewards.py`, and `experiments/ocr_reward_tests/`.
- `vllm` - Optional high-throughput local LLM backend in `src/prompt_pipeline/llm_client.py`.
- `mlx_lm` - Optional Apple Silicon local LLM backend in `src/prompt_pipeline/llm_client.py`.
- `synthtiger` CLI/runtime - Synthetic text rendering is invoked by `scripts/synth/build_dataset.py` through `scripts/synth/run_synthtiger.py` and template/config files in `scripts/synth/` and `configs/synth/`.

## Configuration

**Environment:**
- Configuration is primarily file-based JSON under `configs/`, with dataclass defaults in `src/training/config.py`.
- Accelerate runtime topology is configured under `configs/accelerate/` and referenced by launch commands in `README.md:50-55`.
- `.env` files are ignored by `.gitignore:137-145`; no `.env` file was detected during mapping and no secret file contents were read.
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` is set programmatically in `src/training/rewards.py` for PaddleOCR model loading.

**Build:**
- No package build configuration is committed. Treat the repository as a runnable source tree using `python -m ...` module invocations.
- Key experiment configs: `configs/sft.json`, `configs/dpo.json`, `configs/masked_sft.json`, `configs/eval_suite.json`, and product/OCR/VLM variants under `configs/`.
- Synthetic dataset config: `configs/synth/cyrillic.yaml`.
- Cluster configs: `scripts/cluster/*.sbatch` and `scripts/cluster/setup_env.sh`.

## Platform Requirements

**Development:**
- Python 3.11 conda environment from `README.md`.
- CUDA-capable GPU with PyTorch CUDA wheels; README uses CUDA 12.1 install URL.
- Hugging Face model access for `black-forest-labs/FLUX.2-klein-base-4B`, `Qwen/Qwen3.5-9B`, and prompt datasets referenced in `README.md`, `src/training/config.py`, and `scripts/download_dataset.py`.
- Optional PaddleOCR, vLLM, MLX, and synthtiger dependencies are required only for their specific code paths in `src/training/rewards.py`, `src/prompt_pipeline/llm_client.py`, and `scripts/synth/`.

**Production:**
- No web service or production server deployment target is present.
- Operational target is local GPU or SLURM cluster batch execution through `scripts/cluster/`.
- Generated model outputs, tensors, images, logs, and runs are intentionally ignored by `.gitignore:173-199`.

---

*Stack analysis: 2026-05-04*
