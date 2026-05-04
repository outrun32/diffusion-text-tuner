# Phase 1 — Execution Surface and Pipeline Inventory Research

**Phase:** 1 — Execution Surface and Pipeline Inventory  
**Researched:** 2026-05-04  
**Status:** Ready for planning  
**Confidence:** HIGH for inventory/tooling/test-surface patterns; MEDIUM for exact CUDA/ML pins until target machines run smoke checks.

## Research Question

What must be known to plan Phase 1 so the repository gains a reproducible execution surface without destabilizing existing thesis research flows?

## Source Inputs

- `.planning/PROJECT.md` — brownfield thesis toolkit scope, constraints, and active requirements.
- `.planning/ROADMAP.md` — Phase 1 goal, success criteria, and requirement IDs.
- `.planning/REQUIREMENTS.md` — INV, ENV, and TEST requirements assigned to Phase 1.
- `.planning/codebase/*.md` — architecture, stack, structure, testing, integrations, conventions, and concerns.
- `.planning/research/SUMMARY.md` — project-level research recommending uv, pyproject, Ruff, pytest, command catalog, and diagnostic separation.
- `README.md`, `.gitignore`, `tests/test_losses.py`, `scripts/cluster/*.sbatch`, `scripts/cluster/setup_env.sh` — current setup, command, test, and SLURM surfaces.

## Key Findings

### Current Execution Surface

The repository already has runnable surfaces for prompt generation, image generation, reward scoring, SFT, DPO, masked-SFT, synthetic data, evaluation, plotting, and SLURM jobs, but they are spread across `README.md`, `scripts/`, `scripts/cluster/`, `src/`, `configs/`, `experiments/`, and thesis plotting scripts. The codebase map identifies supported entry points but no single user-facing inventory currently separates supported toolkit paths from experiments, legacy configs, diagnostics, and manual one-off scripts.

### Environment and Tooling Gap

`README.md` documents direct `pip install` commands, but the root has no committed `pyproject.toml`, `uv.lock`, `.python-version`, Ruff config, pytest config, lint command, or formatter command. `.gitignore` explicitly notes that `uv.lock` is generally recommended to commit, and it is not ignored. Phase 1 should add a committed Python 3.11 manifest and lock with optional dependency groups for GPU, OCR/reward, synthesis, vLLM, MLX, tests, linting, plotting, and analysis.

### Default Test Safety

The only formal test file is `tests/test_losses.py`, which is CPU-safe and includes a script fallback. However, expensive diagnostics are named like tests outside `tests/`: `scripts/test_gradient_flow.py`, `scripts/test_grad_magnitude.py`, and `experiments/ocr_reward_tests/test_*.py`. Pytest should be configured to discover only `tests/`, strict markers should distinguish `slow`, `gpu`, `model`, `ocr`, `integration`, and `manual`, and manual diagnostics should be documented as opt-in commands rather than default test-discovery targets.

### Smoke Check Surface

Phase 1 needs explicit smoke commands for:

- Python/runtime import checks that do not load FLUX, Qwen, PaddleOCR, or CUDA models by default.
- CUDA availability checks for target machines.
- Hugging Face model access/cache checks for FLUX and Qwen before long jobs.
- PaddleOCR availability checks for OCR/reward workflows.
- Cache/runtime path checks for `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, generated artifact roots, and SLURM `runs/` logs.

These checks should be explicit commands separate from default tests because Phase 1 must keep CPU-safe automation distinct from GPU/model/OCR diagnostics.

## Recommended Implementation Pattern

1. **Inventory first:** Create a durable documentation artifact that lists every pipeline family, supported entry point, current command, consumed inputs, produced artifacts, optimized objective, thesis role, and status classification.
2. **Manifest second:** Add `pyproject.toml`, `.python-version`, and `uv.lock` as the reproducible install/tooling contract. Use optional extras/groups so users can choose minimal, GPU, OCR/reward, synthesis, vLLM, MLX, test, lint, plotting, and analysis installs.
3. **Smoke checks third:** Add a small import-safe Python smoke CLI with tests for argument parsing and non-model-loading behavior. The script should use `importlib.util.find_spec` for dependency presence checks wherever possible and only import heavyweight modules in explicit smoke modes.
4. **Command catalog last:** Update README and add a command catalog/Makefile after the inventory, manifest, and smoke script exist. The command catalog should give comparable local and SLURM variants without changing existing supported entry points.

## Validation Architecture

Phase 1 validation should sample the new execution surface without running expensive ML work:

- **Quick command:** `python tests/test_losses.py` until pytest is installed, then `uv run pytest` after `pyproject.toml` and `uv.lock` exist.
- **Tooling checks:** `uv lock --check`, `uv run ruff check .`, `uv run ruff format --check .` after tooling is added.
- **Inventory checks:** Use Python string checks to verify docs contain all Phase 1 pipeline families and status categories.
- **Smoke checks:** Unit tests for `scripts/smoke_environment.py`; import/cuda/model/OCR smoke commands remain explicit and not part of default pytest.
- **Diagnostic discovery check:** `uv run pytest --collect-only` must collect only files under `tests/`; it must not collect `scripts/` or `experiments/ocr_reward_tests/` diagnostics.

## Security and Trust Notes

- No application authentication or API surface is added.
- External trust boundaries are local CLI input paths, Hugging Face model/dataset downloads, PaddleOCR model loading, and local `.pt` artifact deserialization.
- Phase 1 should document that generated artifacts may contain prompt text and must remain out of git.
- Smoke commands should not print secrets or tokens. If Hugging Face credentials are detected, commands should report presence/absence only, not values.
- `.pt` artifacts remain trusted local outputs; do not add commands that load untrusted `.pt` files by default.

## Out of Scope for This Phase

- Refactoring trainer, reward, generation, or dataset internals.
- Adding run manifests or strict artifact validators (Phase 2).
- Adding DPO/reward/dataset characterization tests beyond default test-surface configuration (Phase 4).
- Running real GPU model generation/training as a default verification step.

## Planning Implications

- Plan 1 should cover pipeline inventory and classification (INV-01 through INV-04).
- Plan 2 should cover reproducible dependency/tooling manifests (ENV-01, ENV-02, ENV-03, ENV-05, TEST-06).
- Plan 3 should cover explicit smoke checks and testable command behavior (ENV-04, TEST-06, TEST-07).
- Plan 4 should cover README/Makefile/command catalog, local-vs-SLURM parity, and diagnostic separation (ENV-03 through ENV-06, TEST-06, TEST-07, plus INV references).
