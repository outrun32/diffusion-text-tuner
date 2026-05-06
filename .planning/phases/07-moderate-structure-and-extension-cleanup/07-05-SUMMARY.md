---
phase: 07-moderate-structure-and-extension-cleanup
plan: 05
subsystem: training-metric-plotting
tags: [plotting, training-metrics, cli-wrapper, import-safe, cpu-safe-tests]

requires:
  - phase: 07-moderate-structure-and-extension-cleanup
    provides: Phase 7 structure rules and prior thin CLI seam patterns
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Training metrics and run-comparison context
provides:
  - Import-safe `src.plotting.training_metrics` module for metric loading, smoothing, summaries, and rendering
  - Thin `scripts/plot_metrics.py` CLI wrapper preserving direct script and module invocation
  - CPU-safe plotting seam tests for helper behavior, lazy matplotlib imports, and CLI delegation

tech-stack:
  added: []
  patterns: [thin-cli-wrapper, import-safe-plotting-seam, lazy-plotting-backend, dataclass-metrics]

key-files:
  created:
    - src/plotting/__init__.py
    - src/plotting/training_metrics.py
    - tests/test_plotting_pipeline_contracts.py
  modified:
    - scripts/plot_metrics.py

key-decisions:
  - "Keep `scripts/plot_metrics.py` as a compatibility wrapper that parses existing arguments, re-exports `load_metrics` and `smooth`, and delegates rendering to `src.plotting.training_metrics.plot_training_metrics`."
  - "Import NumPy for pure helper math but keep Matplotlib backend selection and `pyplot` imports inside `plot_training_metrics` so tests and future helpers stay display-free at import time."
  - "Preserve the existing `training_curves.png` output name, chart layout, and printed summary fields while returning the output path for importable callers."

patterns-established:
  - "Future training-metric plotting variants should import `load_metrics`, `smooth`, `summarize_metrics`, and `plot_training_metrics` instead of adding plotting logic to CLI scripts."
  - "Generated plot images remain caller-selected runtime artifacts and are not committed."

requirements-completed: [STR-05, STR-06]

duration: 5min 25s
completed: 2026-05-06T16:09:02Z
---

# Phase 7 Plan 05: Training Metric Plotting Seam Summary

**Import-safe training metric plotting helpers with preserved `python scripts/plot_metrics.py` and `python -m scripts.plot_metrics` commands.**

## Performance

- **Duration:** 5 min 25 sec
- **Started:** 2026-05-06T16:03:37Z
- **Completed:** 2026-05-06T16:09:02Z
- **Tasks:** 3
- **Files modified:** 4 task files plus this summary and planning metadata

## Accomplishments

- Added `src.plotting.training_metrics` with a frozen `TrainingMetrics` dataclass, explicit CSV column loading, deterministic smoothing, summary calculation, and `plot_training_metrics(...) -> Path` rendering.
- Moved Matplotlib import/backend setup into `plot_training_metrics`, keeping helper imports CPU-safe and display-free.
- Replaced `scripts/plot_metrics.py` with a thin argparse wrapper that preserves `metrics_csv`, optional `--output-dir`, `load_metrics`, `smooth`, and `plot` compatibility surfaces.
- Added CPU-safe plotting contract tests using tiny temporary CSV fixtures and fake Matplotlib modules; tests avoid training, GPU, model, OCR, and generated run artifacts.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_plotting_pipeline_contracts.py -x` failed with `ModuleNotFoundError: No module named 'src.plotting'` after adding initial plotting contracts.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_plotting_pipeline_contracts.py -q` passed after adding `src.plotting.training_metrics` and public exports.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_plotting_pipeline_contracts.py -x` failed because the historical script imported Matplotlib at module import time and had no `main(argv)` delegation seam.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_plotting_pipeline_contracts.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/plotting/training_metrics.py scripts/plot_metrics.py tests/test_plotting_pipeline_contracts.py` passed after thinning the wrapper and fixing plan-owned lint issues.

## Task Commits

1. **Task 1 RED: Plotting module contracts** - `586e880` (`test`)
2. **Task 2 GREEN: Importable plotting helpers** - `79d77f7` (`feat`)
3. **Task 3 RED: Thin CLI delegation contract** - `25ba224` (`test`)
4. **Task 3 GREEN: Thin plotting CLI wrapper** - `bce01c1` (`feat`)
5. **Task 3 fix: Direct script execution** - `000aeae` (`fix`)

## Files Created/Modified

- `src/plotting/__init__.py` - Public import-safe exports for the plotting seam.
- `src/plotting/training_metrics.py` - Reusable metric loading, smoothing, summary, and plotting implementation.
- `scripts/plot_metrics.py` - Thin compatibility CLI wrapper for direct and module execution.
- `tests/test_plotting_pipeline_contracts.py` - CPU-safe tests for metric helpers, lazy Matplotlib imports, plotting outputs, and CLI delegation.

## Decisions Made

- Used a frozen dataclass for loaded metric columns so downstream plotting/report variants receive a stable typed shape.
- Preserved explicit CSV column parsing and numeric conversion failures rather than adding permissive defaults for malformed metrics.
- Kept output directory semantics caller-selected: when `--output-dir` is omitted, plots are written next to the metrics CSV as before.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed plan-owned test and Ruff issues during Task 2/3 verification**
- **Found during:** Task 2 and Task 3 verification
- **Issue:** NumPy convolution returned `3.9999999999999996` for the planned `[2, 3, 4]` smoothing contract, the fake Matplotlib package needed package-like attributes for lazy import testing, and Ruff required `collections.abc.Sequence`, used re-export declarations, and `zip(..., strict=True)`.
- **Fix:** Rounded smoothing outputs to 12 decimals, made the fake Matplotlib module package-like in tests, added `__all__` for compatibility re-exports, modernized imports, and made the speed zip strict.
- **Files modified:** `src/plotting/training_metrics.py`, `scripts/plot_metrics.py`, `tests/test_plotting_pipeline_contracts.py`
- **Verification:** Plotting contract tests and Ruff passed.
- **Committed in:** `79d77f7`, `bce01c1`

**2. [Rule 1 - Bug] Preserved direct script execution after moving implementation into `src`**
- **Found during:** Overall verification
- **Issue:** `python scripts/plot_metrics.py --help` failed with `ModuleNotFoundError: No module named 'src'` because direct script execution put `scripts/` rather than the repository root on `sys.path`.
- **Fix:** Added the repository root to `sys.path` only when the wrapper is executed directly, leaving `python -m scripts.plot_metrics` behavior unchanged.
- **Files modified:** `scripts/plot_metrics.py`
- **Verification:** Both direct script and module help commands passed.
- **Committed in:** `000aeae`

---

**Total deviations:** 2 auto-fixed (1 blocking/lint/test issue, 1 direct-execution bug)
**Impact on plan:** Fixes were limited to plan-owned files and preserved the required plotting command behavior.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder/coming-soon/not-available markers or UI-facing empty/mock data paths in plan-owned files.

## Threat Flags

None. The plan threat model covered the new surfaces: metric CSV parsing keeps explicit numeric failures, plots write only to caller-selected output locations, and Matplotlib/backend work remains inside `plot_training_metrics`.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_plotting_pipeline_contracts.py -q` — passed, 6 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/plotting/training_metrics.py scripts/plot_metrics.py tests/test_plotting_pipeline_contracts.py` — passed.
- `PATH="/root/.local/bin:$PATH" uv run python scripts/plot_metrics.py --help` — passed.
- `PATH="/root/.local/bin:$PATH" uv run python -m scripts.plot_metrics --help` — passed.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`, so STATE/ROADMAP/REQUIREMENTS updates were prepared manually.
- The worktree contained unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 7 Plan 06 can include `src.plotting.training_metrics` in the final extension-point registry as the training-metric plotting seam.
- Future plotting variants can import `TrainingMetrics`, `load_metrics`, `smooth`, `summarize_metrics`, and `plot_training_metrics` without editing `scripts/plot_metrics.py`.

## TDD Gate Compliance

- RED and GREEN commits are present for both the module seam and CLI wrapper work.
- No refactor-only commit was required beyond the lint/direct-execution fix commit.

## Self-Check: PASSED

- Found all created/modified task files: `src/plotting/__init__.py`, `src/plotting/training_metrics.py`, `scripts/plot_metrics.py`, `tests/test_plotting_pipeline_contracts.py`, and this summary.
- Found task commits `586e880`, `79d77f7`, `25ba224`, `bce01c1`, and `000aeae` in git history.
- Required verification commands passed, plotting contract tests remain CPU-safe, and no generated images/tensors/checkpoints/logs were committed.

---
*Phase: 07-moderate-structure-and-extension-cleanup*
*Completed: 2026-05-06T16:09:02Z*
