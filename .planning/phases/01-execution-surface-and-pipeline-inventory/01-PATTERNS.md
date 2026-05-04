# Phase 1 Pattern Map

**Phase:** 1 — Execution Surface and Pipeline Inventory  
**Generated:** 2026-05-04

## Files to Create or Modify

| Planned file | Role | Existing analogs / source of truth |
|--------------|------|------------------------------------|
| `docs/pipeline_inventory.md` | User-facing inventory of execution surfaces | `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`, `README.md` |
| `pyproject.toml` | Python package/tool/test/lint configuration | `.planning/research/SUMMARY.md`, `.planning/codebase/TESTING.md`, `.planning/codebase/CONCERNS.md` |
| `.python-version` | Python runtime pin | `README.md` setup uses Python 3.11 |
| `uv.lock` | Committed dependency lock | `.planning/research/SUMMARY.md`, `.gitignore` comments explicitly allow committing `uv.lock` |
| `scripts/smoke_environment.py` | Import-safe smoke-check CLI | CLI style in `scripts/generate_images.py`, `scripts/score_images.py`; optional import guidance in `.planning/codebase/CONVENTIONS.md` |
| `tests/test_smoke_environment.py` | CPU-safe tests for smoke CLI | `tests/test_losses.py` style: direct asserts, no heavy model loads |
| `docs/commands.md` | Standard command catalog | `README.md`, `scripts/cluster/*.sbatch`, `configs/accelerate/*.yaml` |
| `Makefile` | Short command aliases | New file; commands wrap uv/python/pytest/Ruff and existing CLI entry points |
| `scripts/diagnose_gradient_flow.py`, `scripts/diagnose_grad_magnitude.py` | Renamed manual CUDA/model diagnostics | `scripts/test_gradient_flow.py`, `scripts/test_grad_magnitude.py` |
| `README.md` | Front-door summary linking new execution docs | Existing README structure; keep concise and avoid replacing detailed docs |

## Concrete Patterns to Preserve

### Python CLI Modules

- Use `python -m scripts.<module>` style for scripts that are intended to run from repository root.
- Parse CLI arguments at module boundary with `argparse`.
- Configure logging in `main()`, not at import time.
- Keep heavy optional imports inside explicit functions/subcommands.

### Tests

- Keep formal automated tests under `tests/`.
- Follow `tests/test_losses.py`: small deterministic in-memory checks, direct assertions, no CUDA/model/OCR loading.
- Add pytest configuration so broad discovery does not collect `scripts/` or `experiments/ocr_reward_tests/`.

### Documentation

- Keep generated artifacts out of git and name ignored roots explicitly: `outputs/`, `runs/`, large `data/` outputs, tensors, checkpoints, logs, and generated images.
- Preserve current command entry points while documenting local and SLURM variants.
- Distinguish supported, experimental, historical, and manual diagnostic paths in tables.

## Landmines

- Do not run FLUX, Qwen, PaddleOCR, or CUDA diagnostics in default pytest.
- Do not commit generated images, tensors, checkpoints, logs, or large datasets.
- Do not replace current CLI entry points during Phase 1.
- Do not perform trainer/reward/dataset refactors before characterization phases.
