# Codebase Concerns

**Analysis Date:** 2026-05-04

## Tech Debt

**Missing dependency manifest and tool configuration:**
- Issue: The repository documents `pip install` commands in `README.md`, but no committed `requirements.txt`, `pyproject.toml`, environment file, lockfile, lint config, formatter config, or pytest config exists at the repository root.
- Files: `README.md`, `.gitignore`, repository root.
- Impact: Reproducing the exact training/evaluation environment is fragile, contributors cannot run a standard lint/test command, and optional dependencies such as `paddleocr`, `vllm`, `mlx_lm`, `synthtiger`, `requests`, and `pytest` are not discoverable from a single source of truth.
- Fix approach: Add a dependency manifest such as `pyproject.toml` or `requirements*.txt` with grouped optional extras for training, OCR, synthesis, MLX, vLLM, and tests; add formatter/linter/test config with documented commands.

**Large trainer modules combine many responsibilities:**
- Issue: Trainer modules own model loading, config loading, dataset setup, optimizer/scheduler setup, sampling, logging, checkpointing, and train loops in one file.
- Files: `src/training/masked_sft_trainer.py`, `src/training/refl_trainer.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`.
- Impact: Adding features increases already-large files, makes focused unit testing difficult, and encourages copy/paste across trainer modes.
- Fix approach: Extract shared trainer utilities into focused modules such as `src/training/checkpointing.py`, `src/training/sampling.py`, `src/training/schedulers.py`, and `src/training/config_io.py`; keep trainer files as orchestration layers.

**Duplicated reward implementations:**
- Issue: Qwen/PaddleOCR reward logic exists in both training and evaluation paths.
- Files: `src/training/rewards.py`, `src/evaluation/evaluate_rewards.py`, `scripts/score_images.py`, `experiments/ocr_reward_tests/`.
- Impact: Reward prompts, token handling, OCR settings, and scoring semantics can drift between training, scoring, and evaluation.
- Fix approach: Move shared reward interfaces and model implementations to `src/training/rewards.py` or a new `src/rewards/` package; make evaluation/scoring import the shared classes instead of duplicating logic.

**Manual diagnostic scripts are named like formal tests:**
- Issue: Expensive CUDA/model-loading diagnostics use `test_*.py` names outside `tests/` and execute work at import time.
- Files: `scripts/test_gradient_flow.py`, `scripts/test_grad_magnitude.py`, `experiments/ocr_reward_tests/test_qwen_yes_prob.py`, `experiments/ocr_reward_tests/test_paddleocr.py`.
- Impact: Broad test discovery can accidentally launch multi-GB model downloads/loads, require local paths, or fail on CPU-only machines.
- Fix approach: Rename manual diagnostics to `check_*.py` or `diagnose_*.py`, guard execution behind `main()`, and reserve `tests/test_*.py` for lightweight automated tests.

## Known Bugs

**Hardcoded local absolute path in Qwen reward experiment:**
- Symptoms: The experiment uses a developer-specific path and fails outside that workstation unless edited.
- Files: `experiments/ocr_reward_tests/test_qwen_yes_prob.py`.
- Trigger: Run `python experiments/ocr_reward_tests/test_qwen_yes_prob.py` on any environment without `/Users/udmurtpsycho/Dev/diffusion-text-tuner`.
- Workaround: Edit `BASE_DIR` manually or run from the original local path.

**Gradient diagnostic assumes generated embeddings exist in an older output layout:**
- Symptoms: The script loads `outputs/text_embeds/000000.pt`, while the current README/config pipeline commonly uses `outputs/generated/text_embeds`.
- Files: `scripts/test_gradient_flow.py`, `scripts/test_grad_magnitude.py`, `README.md`, `configs/sft.json`.
- Trigger: Run the diagnostic scripts before creating `outputs/text_embeds/000000.pt` exactly where the script expects it.
- Workaround: Create/copy the expected embedding file or edit the diagnostic path before running.

## Security Considerations

**External image download trusts manifest URLs:**
- Risk: The Unsplash fetcher downloads arbitrary URLs from a TSV manifest and writes response bodies as images without domain allowlisting or content-length limits.
- Files: `scripts/synth/fetch_unsplash.py`.
- Current mitigation: Uses a timeout, verifies the resulting file can be opened by Pillow, and deletes invalid files on failure.
- Recommendations: Validate URL scheme/host against expected Unsplash domains, stream to disk with a maximum size, and reject non-image content types before writing final files.

**Local file deserialization relies on generated `.pt` artifacts:**
- Risk: `torch.load` can be unsafe when reading untrusted pickle-backed files; this project loads many generated `.pt` artifacts from local data/output directories.
- Files: `src/training/dataset.py`, `src/training/flux2_utils.py`, `src/training/refl_trainer.py`, `src/training/sft_trainer.py`, `src/training/masked_sft_trainer.py`, `scripts/test_gradient_flow.py`, `scripts/test_grad_magnitude.py`.
- Current mitigation: Core dataset/trainer loads mostly use `weights_only=True`, which reduces pickle exposure for tensor-only artifacts.
- Recommendations: Keep `weights_only=True` on all `torch.load` calls, do not load `.pt` files from untrusted sources, and document artifact trust boundaries in README or setup docs.

**Generated logs and artifacts may contain prompt text:**
- Risk: Prompt text, target text, image metadata, and scores can be written into CSV/JSONL/log files under ignored directories; these may become sensitive if private datasets are used.
- Files: `scripts/score_images.py`, `scripts/generate_images.py`, `src/evaluation/evaluate_rewards.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `.gitignore`.
- Current mitigation: `outputs/`, `runs/`, `*.log`, tensors, checkpoints, and generated images are ignored by `.gitignore`.
- Recommendations: Keep generated artifacts out of commits, avoid placing private prompts under committed `data/` paths, and sanitize logs before sharing.

## Performance Bottlenecks

**Repeated text embedding/sample setup can load large FLUX components:**
- Problem: Sample-state setup and text embedding precomputation load FLUX pipeline/text encoder components and can be slow or memory-heavy.
- Files: `src/training/sft_trainer.py`, `src/training/flux2_utils.py`, `scripts/generate_images.py`, `scripts/synth/build_dataset.py`.
- Cause: Text encoders and full pipelines are large; some helper paths create temporary prompt files and run pipeline-level embedding methods.
- Improvement path: Cache sample prompt embeddings per config, standardize embedding artifact paths, and isolate text-encoder-only loading behind reusable utilities.

**Resolution bucket fallback scans latent files:**
- Problem: `ResolutionBucketSampler` loads every sample's latent file if `shapes.csv` lacks entries.
- Files: `src/training/dataset.py`, `scripts/synth/build_dataset.py`.
- Cause: Shape metadata is optional and fallback reads `.pt` files one by one at sampler initialization.
- Improvement path: Ensure `scripts/synth/build_dataset.py` always writes complete `shapes.csv`; validate it in `MaskedSFTDataset` and fail fast or provide a separate repair command for missing shapes.

**Qwen VLM scoring is inherently slow and memory-heavy:**
- Problem: VLM scoring loads Qwen image-text models and scores images one at a time by default.
- Files: `scripts/score_images.py`, `src/training/rewards.py`, `src/evaluation/evaluate_rewards.py`.
- Cause: Large VLM inference, image preprocessing, and default `batch_size=1` for memory safety.
- Improvement path: Use sharding options already present in `scripts/score_images.py`, add batched VLM scoring where safe, and prefer OCR scorer for fast screening when quality requirements allow it.

## Fragile Areas

**FLUX latent geometry and tensor shape contracts:**
- Files: `src/training/flux2_utils.py`, `src/training/losses.py`, `src/training/dataset.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`.
- Why fragile: Packing, patchifying, BN normalization, position IDs, and mask-grid downsampling must stay aligned with FLUX.2 Klein latent conventions.
- Safe modification: Add small tensor-shape unit tests under `tests/` before changing utility functions; keep shape comments up to date in `src/training/flux2_utils.py` and `src/training/losses.py`.
- Test coverage: `tests/test_losses.py` covers masked loss and mask downsampling only; no tests cover latent pack/unpack round-trips, text IDs, or decode/encode shape contracts.

**DPO objective sign/scaling and reference model behavior:**
- Files: `src/training/dpo_trainer.py`, `configs/dpo.json`.
- Why fragile: `time_dependent_beta` returns a negative beta, policy/reference losses are transformed into log-ratios, and reference model is a deep copy moved separately to the device.
- Safe modification: Add deterministic unit tests for `compute_sigma`, `time_dependent_beta`, and `compute_dpo_loss` with a small fake model before changing DPO math.
- Test coverage: No formal tests cover `src/training/dpo_trainer.py`.

**Prompt generation depends on mutable random/coverage state:**
- Files: `src/prompt_pipeline/text_generator.py`, `src/prompt_pipeline/generate.py`, `src/prompt_pipeline/scene_pool.py`, `src/prompt_pipeline/style_generator.py`.
- Why fragile: Character coverage, cached weights, random seeds, deduplication, and optional LLM fallback interact to determine dataset distribution.
- Safe modification: Add tests for deterministic output under fixed seeds and for coverage-cache refresh behavior before changing sampling logic.
- Test coverage: No formal tests cover `src/prompt_pipeline/`.

## Scaling Limits

**Single-node filesystem artifact pipeline:**
- Current capacity: Designed around local/SLURM filesystem directories such as `data/`, `outputs/generated/`, `outputs/sft/`, and `data/synth_cyrillic/`.
- Limit: Very large datasets/checkpoints stress local disk, directory listing, CSV manifests, and manual merge workflows.
- Scaling path: Add sharded manifests, explicit dataset indexes, resumable validation checks, and object-store/HF dataset publishing steps for large artifacts.

**Trainer memory footprint:**
- Current capacity: Uses bf16, LoRA, gradient checkpointing, and Accelerate configs under `configs/accelerate/` to fit large FLUX/Qwen workloads.
- Limit: Full FLUX pipeline, Qwen reward models, reference DPO model copies, and VAE/text encoder components can exceed GPU memory depending on mode.
- Scaling path: Keep text embeddings precomputed, unload unused pipeline components promptly, validate quantization settings, and maintain separate configs for single-GPU/multi-GPU/low-memory modes.

## Dependencies at Risk

**Unpinned ML dependency stack:**
- Risk: `diffusers`, `transformers`, `accelerate`, `peft`, `torch`, `bitsandbytes`, `paddleocr`, `vllm`, and `mlx_lm` APIs change frequently.
- Impact: Model loading, Qwen chat templates, FLUX pipeline methods, PEFT LoRA target matching, and PaddleOCR constructor arguments can break without code changes.
- Migration plan: Pin known-good versions in a manifest, add a smoke test for imports/config loading, and document CUDA/PyTorch compatibility.

**Gated/external model availability:**
- Risk: Code depends on Hugging Face model IDs such as `black-forest-labs/FLUX.2-klein-base-4B`, `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.5-4B`.
- Impact: Fresh environments fail if model names change, access is gated, or credentials are absent.
- Migration plan: Document required model access, support local model paths in configs, and add clearer preflight checks before long jobs.

## Missing Critical Features

**Automated CI/test gate:**
- Problem: No CI config or standard test command is committed.
- Blocks: Safe refactoring of training utilities, prompt generation, dataset loaders, and reward logic.

**Formal integration fixtures:**
- Problem: Only `tests/test_losses.py` is automated; dataset loader, collator, config parsing, prompt generation, and reward wrappers lack small fixtures.
- Blocks: Confident changes to `src/training/dataset.py`, `src/prompt_pipeline/`, and trainer config parsing without running expensive GPU jobs.

**Centralized config/dependency documentation:**
- Problem: Config defaults live in `src/training/config.py`, runtime JSONs live in `configs/`, and setup dependencies live only in `README.md` prose.
- Blocks: Reliable onboarding and reproducible experiment reruns.

## Test Coverage Gaps

**Dataset loading and collators:**
- What's not tested: CSV parsing, `.pt` artifact loading, preference pair construction, prompt embedding padding, masked-SFT sample loading, and resolution buckets.
- Files: `src/training/dataset.py`.
- Risk: Shape/path regressions can break training only after expensive data preparation starts.
- Priority: High.

**Trainer math and config parsing:**
- What's not tested: `compute_sigma`, DPO beta/log-ratio math, config JSON override loading, scheduler choices, checkpoint/sample behavior, and Accelerate setup assumptions.
- Files: `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, `src/training/config.py`.
- Risk: Objective or config regressions can silently degrade model training.
- Priority: High.

**Prompt pipeline determinism and data quality:**
- What's not tested: scene/style/text sampling, fallback generation, LLM output cleanup, deduplication, and character coverage balancing.
- Files: `src/prompt_pipeline/generate.py`, `src/prompt_pipeline/text_generator.py`, `src/prompt_pipeline/llm_client.py`, `src/prompt_pipeline/assembler.py`, `src/prompt_pipeline/scene_pool.py`.
- Risk: Dataset distribution changes can reduce text-rendering quality or reproducibility without obvious failures.
- Priority: Medium.

**External reward/scoring wrappers:**
- What's not tested: VLM yes/no token extraction, OCR reward calculation, scoring CSV resume/sharding, and duplicated evaluation reward behavior.
- Files: `src/training/rewards.py`, `scripts/score_images.py`, `src/evaluation/evaluate_rewards.py`.
- Risk: Reward score changes can corrupt SFT filtering and DPO pair selection.
- Priority: High.

---

*Concerns audit: 2026-05-04*
