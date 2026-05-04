# Coding Conventions

**Analysis Date:** 2026-05-04

## Naming Patterns

**Files:**
- Use `snake_case.py` for Python modules: `src/training/masked_sft_trainer.py`, `src/training/flux2_utils.py`, `scripts/generate_images.py`, and `scripts/synth/build_dataset.py`.
- Use stage/mode names for training configs: `configs/sft.json`, `configs/dpo.json`, `configs/masked_sft.json`, `configs/sft_ocr.json`, and `configs/dpo_vlm.json`.
- Use `test_*.py` for formal and experimental tests: `tests/test_losses.py`, `scripts/test_gradient_flow.py`, and `experiments/ocr_reward_tests/test_qwen_yes_prob.py`.
- Use package `__init__.py` files to make source directories importable: `src/__init__.py`, `src/training/__init__.py`, `src/prompt_pipeline/__init__.py`, and `scripts/__init__.py`.

**Functions:**
- Use `snake_case` verbs for functions: `compute_sigma` in `src/training/sft_trainer.py`, `time_dependent_beta` in `src/training/dpo_trainer.py`, `masked_flow_matching_loss` in `src/training/losses.py`, and `precompute_text_embeddings` in `src/training/flux2_utils.py`.
- Use `load_*`, `setup_*`, `generate_*`, and `compute_*` prefixes consistently for workflow steps in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/prompt_pipeline/generate.py`.
- Use leading underscore for private helpers: `_assemble_ru` in `src/prompt_pipeline/assembler.py`, `_compute_weights` in `src/prompt_pipeline/text_generator.py`, and `_load_transformers` in `src/prompt_pipeline/llm_client.py`.

**Variables:**
- Use lowercase `snake_case` for regular variables and config fields: `latents_dir`, `text_embeds_dir`, `scores_csv`, `score_threshold`, and `gradient_accumulation_steps` in `src/training/config.py`.
- Use short mathematical/tensor names only inside dense model code: `B`, `S`, `C`, `x0`, `x_t`, `sigma`, and `noise` in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/losses.py`.
- Use explicit path suffixes for filesystem variables: `csv_path`, `embed_path`, `latent_path`, `output_dir`, and `samples_dir` in `scripts/generate_images.py`, `scripts/score_images.py`, and `src/training/sft_trainer.py`.

**Types:**
- Use `PascalCase` for dataclasses and classes: `SFTConfig`, `DPOConfig`, `MaskedSFTConfig`, `LoraConfig`, `MaskedSFTDataset`, `ResolutionBucketSampler`, and `QwenYesProbReward` in `src/training/`.
- Use built-in generic typing (`list[str]`, `dict[str, ...]`, `tuple[...]`) rather than `typing.List`/`typing.Dict`, as seen in `src/training/config.py`, `src/training/losses.py`, and `src/prompt_pipeline/llm_client.py`.
- Include `from __future__ import annotations` in new modules that use modern annotations and forward references; this pattern is used in `src/training/config.py`, `src/training/dataset.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, and `src/prompt_pipeline/generate.py`.

## Code Style

**Formatting:**
- Tool used: Not detected. There is no committed formatter configuration such as `pyproject.toml`, `setup.cfg`, `.ruff.toml`, `black.toml`, `.prettierrc`, or `biome.json` at the repository root.
- Key settings: Follow the current PEP 8-like style visible in `src/training/losses.py` and `src/training/dataset.py`: 4-space indentation, blank lines between top-level functions/classes, and readable line wrapping around tensors/config blocks.
- Use long section comments sparingly to separate major trainer phases, following `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py`.

**Linting:**
- Tool used: Not detected. No lint config or lint command is committed at the repository root.
- Key rules: Treat runtime imports and unused imports manually; several modules import optional heavy dependencies inside functions/classes to avoid import-time failures, e.g. `src/training/rewards.py`, `src/training/flux2_utils.py`, and `src/prompt_pipeline/llm_client.py`.
- New code should keep optional dependency imports local to the function/class that needs them, matching `src/training/rewards.py` and `src/prompt_pipeline/llm_client.py`.

## Import Organization

**Order:**
1. Module docstring and `from __future__ import annotations` when used, as in `src/training/dataset.py` and `src/training/sft_trainer.py`.
2. Standard library imports (`argparse`, `csv`, `json`, `logging`, `os`, `random`, `time`, `pathlib`) grouped together, as in `scripts/score_images.py` and `src/training/masked_sft_trainer.py`.
3. Third-party imports (`torch`, `accelerate`, `peft`, `PIL`, `tqdm`) grouped together, as in `src/training/sft_trainer.py` and `src/training/dpo_trainer.py`.
4. Local package imports using relative imports inside `src/` packages, as in `src/training/sft_trainer.py` and `src/prompt_pipeline/generate.py`.
5. Heavy optional imports inside functions where possible: `diffusers.Flux2KleinPipeline` in trainer loaders, `transformers` classes in `src/training/rewards.py`, and `vllm`/`mlx_lm` in `src/prompt_pipeline/llm_client.py`.

**Path Aliases:**
- No configured path alias system was detected.
- Use relative imports inside packages: `from .config import SFTConfig` in `src/training/sft_trainer.py` and `from .assembler import Assembler` in `src/prompt_pipeline/generate.py`.
- Use absolute imports from executable scripts: `from src.training.flux2_utils import encode_image` in `scripts/generate_images.py` and `scripts/synth/build_dataset.py`.

## Error Handling

**Patterns:**
- Fail fast on invalid configuration or missing required dataset files with explicit exceptions: `FileNotFoundError` and `RuntimeError` in `src/training/dataset.py`, `ValueError` in `src/training/losses.py`, and `ValueError` in `src/training/masked_sft_trainer.py`.
- Log and skip recoverable missing or incomplete data records: missing embeddings in `scripts/score_images.py`, missing metadata in `scripts/synth/build_dataset.py`, and missing text embeddings in `src/training/dataset.py`.
- Use resumable/idempotent behavior for long jobs: skip existing generated image/latent pairs in `scripts/generate_images.py`, skip already-scored CSV rows in `scripts/score_images.py`, and skip existing Unsplash downloads in `scripts/synth/fetch_unsplash.py`.
- Avoid broad `except Exception` in core training logic; broad catches currently appear in boundary/cleanup code such as image validation/download cleanup in `scripts/synth/fetch_unsplash.py`.

## Logging

**Framework:** Python `logging` plus occasional `print` in experimental/model-load utilities.

**Patterns:**
- Use `logger = logging.getLogger(__name__)` at module scope in new source modules, matching `src/training/dataset.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/prompt_pipeline/generate.py`, and `scripts/synth/build_dataset.py`.
- Configure logging only in CLI `main()` functions, not importable helpers, as in `scripts/generate_images.py`, `scripts/score_images.py`, and `src/training/sft_trainer.py`.
- Log progress, configuration-derived counts, checkpoint paths, and skip/resume events; examples include dataset counts in `src/training/dataset.py`, shard stats in `scripts/score_images.py`, and checkpoint saves in `src/training/sft_trainer.py`.
- Use `print` only for one-off experiments or low-level interactive diagnostics; current print-heavy files include `src/training/rewards.py`, `src/training/flux2_utils.py`, `src/evaluation/evaluate_rewards.py`, and scripts under `experiments/ocr_reward_tests/`.

## Comments

**When to Comment:**
- Comment tensor shapes and model math where it helps preserve correctness: `src/training/losses.py`, `src/training/flux2_utils.py`, `src/training/sft_trainer.py`, and `src/training/dpo_trainer.py`.
- Use module docstrings to describe pipeline role and CLI usage: `scripts/generate_images.py`, `scripts/score_images.py`, `scripts/synth/build_dataset.py`, and `src/training/sft_trainer.py`.
- Use section dividers for long training or pipeline modules: `src/training/masked_sft_trainer.py`, `src/training/sft_trainer.py`, and `src/prompt_pipeline/generate.py`.

**JSDoc/TSDoc:**
- Not applicable. This is a Python repository.
- Use Python docstrings for public functions/classes, following `src/training/losses.py`, `src/training/dataset.py`, `src/training/flux2_utils.py`, and `src/prompt_pipeline/llm_client.py`.

## Function Design

**Size:**
- Keep pure utility functions small and testable, as in `src/training/losses.py` and `src/training/flux2_utils.py`.
- Trainer orchestration functions are long by current convention; isolate new reusable logic into helpers instead of further expanding `train()` in `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, or `src/training/masked_sft_trainer.py`.

**Parameters:**
- Use dataclass config objects for training orchestration: `train(cfg: SFTConfig)` in `src/training/sft_trainer.py`, `train(cfg: DPOConfig)` in `src/training/dpo_trainer.py`, and `train(cfg: MaskedSFTConfig)` in `src/training/masked_sft_trainer.py`.
- Use explicit keyword-only parameters for multi-argument pipeline helpers where clarity matters, as in `render_phase(...)` in `scripts/synth/build_dataset.py`.
- Pass file paths as strings or `Path` objects consistently within each module; `scripts/synth/build_dataset.py` favors `Path`, while older scripts such as `scripts/generate_images.py` favor `str`/`os.path`.

**Return Values:**
- Return dictionaries for batch/sample payloads in datasets and collators, as in `src/training/dataset.py`.
- Return `(loss, metrics)` or `(loss, parts)` tuples for train-step computations, as in `src/training/losses.py` and `src/training/dpo_trainer.py`.
- Return `None` for disabled optional behavior, such as sample state setup in `src/training/sft_trainer.py`; document this in callers and guard with `is not None`.

## Module Design

**Exports:**
- There is no explicit `__all__` export convention.
- Modules expose importable classes/functions directly; e.g. `MaskedSFTDataset` from `src/training/dataset.py`, `masked_flow_matching_loss` from `src/training/losses.py`, and `LLMClient` from `src/prompt_pipeline/llm_client.py`.

**Barrel Files:**
- Barrel exports are not used. `__init__.py` files are empty or minimal in `src/`, `src/training/`, `src/prompt_pipeline/`, `src/evaluation/`, and `scripts/`.
- Import from concrete modules rather than package roots, matching `from src.training.flux2_utils import ...` in `scripts/generate_images.py` and `from .losses import masked_flow_matching_loss` in `src/training/masked_sft_trainer.py`.

---

*Convention analysis: 2026-05-04*
