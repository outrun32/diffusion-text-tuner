---
phase: 01-execution-surface-and-pipeline-inventory
verified: 2026-05-04T13:48:11Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 9/12
  gaps_closed:
    - "Documented and Makefile Ruff commands now use `uv run --extra lint ruff ...` and pass on the scoped Phase 1 CPU-safe surface."
    - "Documented and Makefile smoke commands now use `uv run python -m scripts.smoke_environment ...`; import smoke passes through both docs command and Makefile target."
    - "Command catalog now explicitly states that masked-SFT, synthetic data, evaluation, and thesis plotting have local commands but no committed SLURM wrappers yet."
  gaps_remaining: []
  regressions: []
---

# Phase 1: Execution Surface and Pipeline Inventory Verification Report

**Phase Goal:** Users can understand the current toolkit, install it reproducibly, and run safe baseline commands without triggering expensive GPU/model work by accident.  
**Verified:** 2026-05-04T13:48:11Z  
**Status:** passed  
**Re-verification:** Yes — after gap closure

## Goal Achievement

Re-verification started from the three previously blocked truths and then quick-checked the previously passing Phase 1 truths for regression. The command-surface wiring gaps are closed: Ruff commands are reproducible through the lint extra, smoke checks are run through the uv Python 3.11 environment, and SLURM coverage for local-only flows is explicit rather than implied.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can open a current pipeline inventory that separates supported entry points from outdated, duplicate, experimental, and manual diagnostic scripts. | ✓ VERIFIED | `docs/pipeline_inventory.md:5-19` lists supported entry points; `docs/pipeline_inventory.md:21-56` separates manual diagnostics, experimental scripts, legacy/superseded configs, and supported commands. |
| 2 | User can see what prompt, generation, scoring, SFT, DPO, masked-SFT, synthetic, evaluation, plotting, and SLURM flows consume, produce, optimize/measure, and support in the thesis. | ✓ VERIFIED | The supported-entry-points table includes `Consumes`, `Produces`, `Optimizes / measures`, and `Thesis support` columns for the required flow families at `docs/pipeline_inventory.md:7-19`. |
| 3 | User can trace historical reward-filtered SFT/DPO, synthetic masked-MSE, OCR/VLM/product reward, and thesis plotting/report tracks. | ✓ VERIFIED | `docs/pipeline_inventory.md:58-74` contains the four historical tracks with concrete command/config references. |
| 4 | User can install Python 3.11 dependencies from committed manifests. | ✓ VERIFIED | `.python-version` contains `3.11`; `pyproject.toml:1-14` pins `requires-python = ">=3.11,<3.12"`; `uv.lock` is present; `PATH="/root/.local/bin:$PATH" uv lock --check` passed. |
| 5 | User can choose optional groups for GPU, OCR/reward, synthesis, vLLM/MLX, tests, linting, plotting, and analysis. | ✓ VERIFIED | `pyproject.toml:16-56` defines `gpu`, `ocr`, `reward`, `synthesis`, `vllm`, `mlx`, `test`, `lint`, `plotting`, and `analysis`. |
| 6 | User can run the standard CPU-safe pytest command without collecting expensive diagnostics. | ✓ VERIFIED | `PATH="/root/.local/bin:$PATH" uv run pytest` passed 11 tests; `uv run pytest --collect-only` collected only tests under `tests/` from `pyproject.toml` `testpaths`. |
| 7 | User can distinguish default automated tests from slow, GPU, model, OCR, integration, and manual diagnostics. | ✓ VERIFIED | `pyproject.toml:68-80` restricts discovery to `tests` and declares strict markers; `docs/commands.md:28-49` separates CPU-safe defaults from opt-in smoke checks; `docs/commands.md:187-196` separates manual diagnostics. |
| 8 | User can run documented Ruff lint and format commands from the committed command surface. | ✓ VERIFIED | `docs/commands.md:32-38` and `Makefile:9-13` use `uv run --extra lint ruff ...`; both `uv run --extra lint ruff check scripts/smoke_environment.py tests` and `uv run --extra lint ruff format --check scripts/smoke_environment.py tests` passed. `make lint` and `make format` also passed. |
| 9 | User can list explicit smoke checks and run an import-safe smoke check without loading CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, or SynthTIGER. | ✓ VERIFIED | `uv run python -m scripts.smoke_environment --list` printed `imports`, `cuda`, `cache`, `model-access`, and `ocr`; `uv run python -m scripts.smoke_environment --check imports` passed; `tests/test_smoke_environment.py:38-45` asserts heavy modules are not imported by smoke module import. |
| 10 | User can run documented smoke checks through docs/Makefile from the reproducible command surface. | ✓ VERIFIED | `docs/commands.md:44-50` and `Makefile:15-28` use `uv run python -m scripts.smoke_environment ...`; `make smoke-imports` passed under Python 3.11; direct CUDA/model/OCR/cache smoke commands with `--allow-missing` completed without requiring broad test discovery. |
| 11 | User can compare local and SLURM variants of supported pipeline flows. | ✓ VERIFIED | `docs/commands.md:52-156` documents local commands; `docs/commands.md:158-185` documents SLURM launchers for generation/scoring/SFT/DPO and explicitly states no committed SLURM wrapper yet for masked-SFT, synthetic data, evaluation, and thesis plotting. |
| 12 | User can run optional diagnostic commands without confusing them with CI-safe tests. | ✓ VERIFIED | `scripts/diagnose_gradient_flow.py` and `scripts/diagnose_grad_magnitude.py` exist with guarded `main()` functions; `scripts/test_grad*.py` is absent; diagnostics are listed under `docs/commands.md:187-196`. |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `docs/pipeline_inventory.md` | Inventory, classifications, flow inputs/outputs/optimization/thesis support, historical tracks, artifact safety | ✓ VERIFIED | Substantive inventory with required supported entry points, non-default diagnostics, historical tracks, and artifact safety notes. |
| `.python-version` | Python runtime pin | ✓ VERIFIED | Contains `3.11`. |
| `pyproject.toml` | Project metadata, optional dependencies, pytest config, Ruff config | ✓ VERIFIED | Defines Python 3.11, optional dependency groups, pytest `testpaths`, markers, and Ruff config. |
| `uv.lock` | Resolved dependency lock | ✓ VERIFIED | Exists and `uv lock --check` passed. |
| `scripts/smoke_environment.py` | Import-safe smoke CLI | ✓ VERIFIED | Defines checks, parser, runner, and guarded explicit CUDA/model/OCR/cache checks; import check enforces Python 3.11. |
| `tests/test_smoke_environment.py` | CPU-safe tests for smoke CLI | ✓ VERIFIED | Tests list output, `--list`, unknown check handling, and no heavy import side effects. |
| `docs/commands.md` | Command catalog | ✓ VERIFIED | Documents setup, CPU-safe pytest/Ruff commands, uv-wired smoke commands, local commands, explicit SLURM coverage, manual diagnostics, and artifact safety. |
| `Makefile` | Short aliases for setup/test/lint/format/smokes | ✓ VERIFIED | Targets are present and wired to `uv run`; `make lint`, `make format`, `make smoke-imports`, and dry-run of all smoke targets passed. |
| `README.md` | Front-door execution surface links | ✓ VERIFIED | Links `docs/pipeline_inventory.md` and `docs/commands.md` and states default tests are CPU-safe with opt-in diagnostics. |
| `scripts/diagnose_gradient_flow.py` | Guarded manual diagnostic | ✓ VERIFIED | Contains `def main()`; heavy imports are inside `main()`. |
| `scripts/diagnose_grad_magnitude.py` | Guarded manual diagnostic | ✓ VERIFIED | Contains `def main()`; heavy imports are inside `main()`. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `README.md` | `docs/pipeline_inventory.md`, `docs/commands.md` | Execution Surface links | ✓ WIRED | Links present at `README.md:11-12`. |
| `pyproject.toml` | `tests/` | pytest `testpaths` | ✓ WIRED | `pyproject.toml:68-72` restricts collection to `tests`; collect-only confirmed 11 tests under `tests/`. |
| `tests/test_smoke_environment.py` | `scripts/smoke_environment.py` | imports smoke helper functions | ✓ WIRED | Tests import/use `list_checks`, `main`, and `run_check`. |
| `Makefile` | `scripts/smoke_environment.py` | `smoke-*` targets | ✓ WIRED | Targets call `uv run python -m scripts.smoke_environment ...`; `make smoke-imports` passed and dry-run prints all smoke variants. |
| `docs/commands.md` | `scripts/cluster/*.sbatch` | SLURM command catalog | ✓ WIRED | Documents existing `generate_images.sbatch`, `score_images.sbatch`, `merge_scores.sh`, `sft.sbatch`, and `dpo.sbatch`; local-only/no-wrapper status is explicit for remaining flows. |

### Data-Flow Trace (Level 4)

No dynamic UI/data-rendering artifacts were introduced in this phase. Documentation artifacts are static; smoke CLI behavior was verified through command execution and tests.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Lockfile is current | `PATH="/root/.local/bin:$PATH" uv lock --check` | Resolved successfully | ✓ PASS |
| Default pytest discovery is CPU-safe | `PATH="/root/.local/bin:$PATH" uv run pytest --collect-only` | Collected 11 tests under `tests/` only | ✓ PASS |
| CPU-safe tests pass | `PATH="/root/.local/bin:$PATH" uv run pytest` | 11 passed | ✓ PASS |
| Scoped Ruff lint command runs | `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check scripts/smoke_environment.py tests` | All checks passed | ✓ PASS |
| Scoped Ruff format command runs | `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff format --check scripts/smoke_environment.py tests` | 3 files already formatted | ✓ PASS |
| Makefile lint target runs | `PATH="/root/.local/bin:$PATH" make lint` | All checks passed | ✓ PASS |
| Makefile format target runs | `PATH="/root/.local/bin:$PATH" make format` | 3 files already formatted | ✓ PASS |
| Smoke CLI lists checks | `PATH="/root/.local/bin:$PATH" uv run python -m scripts.smoke_environment --list` | Listed imports/cuda/cache/model-access/ocr | ✓ PASS |
| Import-safe smoke check runs | `PATH="/root/.local/bin:$PATH" uv run python -m scripts.smoke_environment --check imports` | Python 3.11 and local modules ok | ✓ PASS |
| Makefile smoke imports target runs | `PATH="/root/.local/bin:$PATH" make smoke-imports` | Python 3.11 and local modules ok | ✓ PASS |
| Makefile command aliases are reproducible | `PATH="/root/.local/bin:$PATH" make -n test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr smoke-cache` | Printed `uv run` / `uv run --extra lint` commands | ✓ PASS |
| Optional CUDA smoke is explicit | `PATH="/root/.local/bin:$PATH" uv run python -m scripts.smoke_environment --check cuda --allow-missing` | Completed explicitly; torch/CUDA reported ok in this environment | ✓ PASS |
| Optional model-access smoke is explicit | `PATH="/root/.local/bin:$PATH" uv run python -m scripts.smoke_environment --check model-access --allow-missing` | Completed; packages ok, auth/cache reported missing informationally | ✓ PASS |
| Optional OCR smoke is explicit | `PATH="/root/.local/bin:$PATH" uv run python -m scripts.smoke_environment --check ocr --allow-missing` | Completed with missing PaddleOCR warning but zero status due `--allow-missing` | ✓ PASS |
| Optional cache smoke is explicit | `PATH="/root/.local/bin:$PATH" uv run python -m scripts.smoke_environment --check cache --allow-missing` | Completed and reported env/path status | ✓ PASS |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|---|---|---|---|
| INV-01 | Inventory of runnable pipelines | ✓ SATISFIED | `docs/pipeline_inventory.md:5-19` lists prompt generation, image generation, reward scoring, SFT, DPO, masked-SFT, synthetic data, evaluation, plotting, and SLURM launchers. |
| INV-02 | Each pipeline shows optimization, inputs, outputs, thesis support | ✓ SATISFIED | `docs/pipeline_inventory.md:7-19` includes consumes/produces/optimizes-or-measures/thesis-support columns. |
| INV-03 | Outdated/duplicate/diagnostic/experimental scripts separate from supported entry points | ✓ SATISFIED | `docs/pipeline_inventory.md:21-56` separates diagnostics, experiments, legacy configs, and supported entry points. |
| INV-04 | Historical tracks traceable | ✓ SATISFIED | `docs/pipeline_inventory.md:58-74` traces reward-filtered SFT/DPO, synthetic masked-MSE, OCR/VLM/product rewards, and plotting/report flows. |
| ENV-01 | Install from committed Python 3.11 manifest | ✓ SATISFIED | `.python-version`, `pyproject.toml`, and `uv.lock` are present; lock check passed. |
| ENV-02 | Optional dependency groups | ✓ SATISFIED | Required optional groups exist in `pyproject.toml:16-56`. |
| ENV-03 | CPU-safe test command | ✓ SATISFIED | `uv run pytest` passed; collect-only stayed under `tests/`. |
| ENV-04 | Smoke commands for imports/CUDA/model/OCR/cache | ✓ SATISFIED | `docs/commands.md:40-50` and `Makefile:15-28` document/run uv-wired smoke commands; direct smoke checks completed with explicit opt-in. |
| ENV-05 | Format and lint with standard documented commands | ✓ SATISFIED | `docs/commands.md:32-38` and `Makefile:9-13` use `uv run --extra lint`; direct and Makefile Ruff checks passed. |
| ENV-06 | Local and SLURM variants of supported commands | ✓ SATISFIED | `docs/commands.md:52-185` provides local commands, existing SLURM launchers, and explicit no-wrapper status for flows without committed SLURM wrappers. |
| TEST-06 | Distinguish default CPU tests from slow/GPU/model/OCR/integration/manual diagnostics | ✓ SATISFIED | `pyproject.toml` testpaths/markers plus `docs/commands.md` separation and guarded diagnostic names. |
| TEST-07 | Optional diagnostic commands not confused with CI-safe tests | ✓ SATISFIED | `docs/commands.md:187-196`, guarded `diagnose_*.py` scripts, and no `scripts/test_grad*.py` diagnostics. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `scripts/smoke_environment.py` | 103 | Local `import torch` | ℹ Info | Intentional: only runs behind explicit `cuda` smoke check. |
| `scripts/smoke_environment.py` | 113 | Text `CUDA is not available through torch` | ℹ Info | Diagnostic warning text, not a placeholder or stub. |

### Human Verification Required

None for Phase 1 pass. Real target-machine CUDA/OCR/model credentials and SLURM scheduler behavior remain environment-specific caveats noted by the roadmap, but Phase 1 only requires a safe, explicit, documented command surface and import-safe smoke checks; those are verified.

### Gaps Summary

No blocking gaps remain. The previously failed Ruff, smoke-command, and SLURM-coverage truths are verified against the committed code and documentation.

---

_Verified: 2026-05-04T13:48:11Z_  
_Verifier: the agent (gsd-verifier)_
