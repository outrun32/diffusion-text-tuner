---
phase: 01-execution-surface-and-pipeline-inventory
plan: 02
subsystem: tooling
tags: [python, uv, pytest, ruff, dependencies]

# Dependency graph
requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: Pipeline inventory and diagnostic separation context from plan 01-01
provides:
  - Python 3.11 runtime pin in `.python-version`
  - uv-resolved dependency lock in `uv.lock`
  - `pyproject.toml` dependency groups, pytest discovery contract, and Ruff lint/format settings
affects: [phase-1-smoke-checks, phase-1-command-catalog, cpu-safe-tests, environment-setup]

# Tech tracking
tech-stack:
  added: [uv, pytest, pytest-cov, ruff]
  patterns: [optional-dependency-extras, uv-lockfile, pytest-testpaths, strict-pytest-markers]

key-files:
  created: [.python-version, pyproject.toml, uv.lock]
  modified: []

key-decisions:
  - "Use `pyproject.toml` plus committed `uv.lock` as the environment source of truth for Python 3.11."
  - "Keep expensive GPU, OCR, reward, synthesis, vLLM, and MLX stacks in optional extras so default test discovery stays CPU-safe."
  - "Use a uv dev dependency group for pytest and torch so the required `uv run pytest` command works without an extra CLI flag."

patterns-established:
  - "Default pytest discovery is restricted to `tests/` via `testpaths` and strict markers."
  - "Ruff configuration lives in `pyproject.toml` with Python 3.11 target and 100-character line length."

requirements-completed: [ENV-01, ENV-02, ENV-03, ENV-05, TEST-06]

# Metrics
duration: 4min
completed: 2026-05-04
---

# Phase 1 Plan 02: Python uv Tooling Summary

**Python 3.11 uv dependency contract with optional ML workflow extras, CPU-safe pytest discovery, and Ruff configuration**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-04T13:24:21Z
- **Completed:** 2026-05-04T13:28:04Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `.python-version` with the exact Python 3.11 runtime pin required by the plan.
- Added `pyproject.toml` with project metadata, baseline runtime dependencies, optional groups for GPU/OCR/reward/synthesis/vLLM/MLX/test/lint/plotting/analysis workflows, pytest config, and Ruff config.
- Generated and committed `uv.lock`, then verified lock freshness, CPU-safe pytest collection under `tests/`, and the existing tensor-loss test suite.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Python 3.11 uv project manifest with optional groups** - `b4a3383` (chore)
2. **Task 2: Lock dependencies and verify CPU-safe tooling configuration** - `2f4770e` (chore)

**Plan metadata:** pending final docs commit

## Files Created/Modified

- `.python-version` - Pins the repository runtime to Python 3.11.
- `pyproject.toml` - Defines project metadata, dependencies/extras, uv behavior, pytest discovery/markers, and Ruff lint/format settings.
- `uv.lock` - Records the resolved Python 3.11 dependency graph for reproducible uv installs.

## Decisions Made

- Used `pyproject.toml` plus `uv.lock` rather than a prose-only setup path so later phases can rely on committed manifests.
- Kept heavyweight or environment-specific stacks in optional extras, including `gpu`, `ocr`, `reward`, `synthesis`, `vllm`, and `mlx`, to avoid making default CPU tests depend on model/OCR backends.
- Added a uv dev dependency group containing pytest, pytest-cov, and torch so the plan's required `uv run pytest` command can execute the existing tensor tests directly.
- Set `tool.uv.package = false` because this brownfield repository is currently a runnable source tree rather than an installable Python package.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added uv dev dependencies for the exact pytest command**
- **Found during:** Task 1 (Create Python 3.11 uv project manifest with optional groups)
- **Issue:** The plan required `uv run pytest`, but placing pytest only in the optional `test` extra would require an extra CLI flag and the existing tests also need torch.
- **Fix:** Added a `dev` dependency group with `pytest`, `pytest-cov`, and `torch` while preserving the required optional `test` group.
- **Files modified:** `pyproject.toml`
- **Verification:** `uv run pytest --collect-only` collected 7 tests from `tests/test_losses.py`, and `uv run pytest` passed.
- **Committed in:** `b4a3383` and preserved through `2f4770e`

**2. [Rule 3 - Blocking] Added pytest import path for uv execution**
- **Found during:** Task 2 (Lock dependencies and verify CPU-safe tooling configuration)
- **Issue:** The first `uv run pytest --collect-only` failed with `ModuleNotFoundError: No module named 'src'` while importing `tests/test_losses.py`.
- **Fix:** Added `pythonpath = ["."]` to pytest configuration so source-tree imports work under uv without packaging the project.
- **Files modified:** `pyproject.toml`
- **Verification:** Re-ran `uv lock --check`, `uv run pytest --collect-only`, and `uv run pytest`; all passed.
- **Committed in:** `2f4770e`

---

**Total deviations:** 2 auto-fixed (1 missing critical, 1 blocking)
**Impact on plan:** Both fixes were required to make the plan's exact uv/pytest verification commands work while keeping default discovery restricted to `tests/`.

## Issues Encountered

- `uv` was not preinstalled in the environment. It was installed for execution with `python -m pip install --user uv`; no generated package artifacts were committed.
- The initial pytest collection failed before `pythonpath = ["."]` was added; the final verification passed.

## Verification Results

- `python -c "from pathlib import Path; import tomllib; ..."` - passed manifest acceptance checks.
- `uv lock --check` - passed.
- `uv run pytest --collect-only` - passed; collected 7 tests from `tests/test_losses.py` only.
- `uv run pytest` - passed; 7 tests passed.
- Collection did not include `scripts/test_gradient_flow.py`, `scripts/test_grad_magnitude.py`, or `experiments/ocr_reward_tests/`.

## Known Stubs

None found in `.python-version`, `pyproject.toml`, or `uv.lock`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-03 can build explicit smoke checks on top of the committed Python 3.11 uv environment.
- Plan 01-04 can document setup, test, lint, and format commands using the new pyproject and lockfile.
- Hardware-specific CUDA/PyTorch/PaddleOCR/vLLM/SynthTIGER validation remains intentionally deferred to smoke and local/SLURM checks.

## Self-Check: PASSED

- Found `.python-version`, `pyproject.toml`, `uv.lock`, and this summary file.
- Found task commits `b4a3383` and `2f4770e` in git history.

---
*Phase: 01-execution-surface-and-pipeline-inventory*
*Completed: 2026-05-04*
