---
phase: 04-cpu-safe-characterization-tests
plan: 03
subsystem: training-objective-characterization
tags: [cpu-safe-tests, dpo-objective, tensor-math, scheduler, latent-geometry]

requires:
  - phase: 04-cpu-safe-characterization-tests
    provides: Plan 04-01 committed-config/tiny-artifact tests and Plan 04-02 dataset/collator contracts
  - phase: 03-data-curriculum-and-dataset-quality
    provides: DPO winner/loser selection semantics that objective tests must preserve
provides:
  - CPU-safe characterization tests for masked losses, latent packing, scheduler updates, sigma/beta schedules, and DPO objective sign behavior
  - Import-safe pure DPO objective helpers delegated by `src.training.dpo_trainer`
  - Full-suite import isolation fixes required for reliable CPU-safe pytest execution
affects: [phase-4-characterization-tests, phase-5-trainer-comparability, dpo-training-objective]

tech-stack:
  added: []
  patterns: [pure-objective-helper-extraction, cpu-safe-tensor-fixtures, import-side-effect-isolation]

key-files:
  created:
    - tests/test_training_objective_math.py
    - src/training/dpo_objective.py
  modified:
    - src/training/dpo_trainer.py
    - tests/test_smoke_environment.py
    - tests/test_synthetic_quality.py

key-decisions:
  - "Extract DPO sigma, beta, and objective math into an import-safe helper that only depends on PyTorch."
  - "Preserve the existing negative beta convention and document its winner/loser consequences with explicit numeric tests."
  - "Keep trainer-facing `compute_sigma`, `time_dependent_beta`, and `compute_dpo_loss` import-compatible through delegation."

patterns-established:
  - "DPO objective helpers accept precomputed per-sample losses so sign/beta behavior can be tested without models, CUDA, or trainer imports."
  - "Import-side-effect tests compare newly imported modules or restore preloaded modules instead of corrupting global `sys.modules` state."

requirements-completed: [TEST-03, TRN-01]

metrics:
  duration: 5min
  completed: 2026-05-05T18:20:33Z
  tasks: 3
  files: 6
---

# Phase 4 Plan 03: Objective Math, Scheduler, Latent Geometry, and DPO Sign/Beta Summary

**Import-safe DPO objective helpers and deterministic CPU tensor tests now lock sigma/beta schedules, latent geometry, scheduler updates, masked losses, and current winner/loser sign behavior before Phase 5 trainer comparability work.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-05T18:15:38Z
- **Completed:** 2026-05-05T18:20:33Z
- **Tasks:** 3
- **Files modified:** 6 including this summary

## Accomplishments

- Added `tests/test_training_objective_math.py` with tiny CPU-safe tensor coverage for latent patchify/unpatchify/pack ordering, `FlowMatchScheduler.step`/`step_to_zero`, masked flow loss, mask-grid downsampling, sigma monotonicity, beta scaling, and DPO objective cases.
- Added `src/training/dpo_objective.py`, a pure PyTorch helper module exporting `compute_sigma`, `time_dependent_beta`, and `compute_dpo_objective` without trainer, CUDA, model, OCR, or filesystem imports.
- Updated `src/training/dpo_trainer.py` to delegate public DPO schedule/objective math to the pure helper while preserving the trainer-facing function names and `compute_dpo_loss` signature.
- Made full default pytest robust by isolating import-side-effect assertions that previously removed or depended on globally preloaded heavy modules.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_objective_math.py -x` initially failed during collection with `ModuleNotFoundError: No module named 'src.training.dpo_objective'` after a test-import setup issue was corrected. This confirmed the intended missing-helper RED before trainer code was edited.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_objective_math.py tests/test_losses.py -q` passed with 17 tests after adding the pure helper and trainer delegation.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_objective_math.py tests/test_losses.py && PATH="/root/.local/bin:$PATH" uv run pytest -q` passed with 17 targeted tests and 148 full-suite tests after import isolation fixes.

## Task Commits

1. **Task 1: Specify objective math and latent geometry before implementation** - `488303d` (`test`)
2. **Task 2: Implement pure DPO objective helpers and trainer delegation** - `c48cf9c` (`feat`)
3. **Task 3: Guard masked loss and DPO regression in the full CPU-safe suite** - `d155200` (`fix`)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_training_objective_math.py` - CPU-safe characterization tests for latent geometry, scheduler updates, masked losses, sigma/beta schedules, DPO numeric cases, and trainer delegation.
- `src/training/dpo_objective.py` - Import-safe pure DPO sigma, beta, log-ratio, loss, reward-margin, and accuracy helpers.
- `src/training/dpo_trainer.py` - Delegates DPO objective math to `src.training.dpo_objective` while preserving public function names and training-loop API.
- `tests/test_smoke_environment.py` - Restores preloaded modules after import-side-effect checks so later torch tests are not corrupted.
- `tests/test_synthetic_quality.py` - Checks synthetic-quality CPU-safe imports against newly imported modules instead of global prior state.
- `.planning/phases/04-cpu-safe-characterization-tests/04-03-SUMMARY.md` - This execution summary.

## Decisions Made

- Preserved the existing negative DPO beta schedule exactly rather than changing winner/loser semantics during characterization.
- Returned detailed tensor metrics from the pure objective helper for tests, while keeping trainer metrics as scalar floats for existing logging and accelerator calls.
- Treated full-suite import-state fixes as blocking correctness fixes because the required default pytest command otherwise failed after torch was removed from `sys.modules` mid-session.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Avoided optional OCR import while testing `FlowMatchScheduler`**
- **Found during:** Task 1 RED verification
- **Issue:** Importing `src.training.refl_trainer` pulled in `src.training.rewards`, which currently imports optional Paddle/PaddleX internals at module import time and failed with `ModuleNotFoundError: No module named 'paddlex'` before reaching the intended DPO-helper RED.
- **Fix:** Kept the scheduler test CPU-safe by stubbing `src.training.rewards` only for the scheduler import path; the broader rewards import-safety refactor remains assigned to Plan 04-05.
- **Files modified:** `tests/test_training_objective_math.py`
- **Verification:** RED rerun then failed for the intended missing `src.training.dpo_objective` module.
- **Committed in:** `488303d`

**2. [Rule 3 - Blocking] Isolated smoke import-side-effect test from global torch state**
- **Found during:** Task 3 full-suite verification
- **Issue:** `tests/test_smoke_environment.py` removed `torch` from `sys.modules` and did not restore it, causing later torch tensor tests in the same pytest process to fail with duplicate Triton library registration errors.
- **Fix:** Restored preloaded modules after the import-side-effect assertion.
- **Files modified:** `tests/test_smoke_environment.py`
- **Verification:** Full `PATH="/root/.local/bin:$PATH" uv run pytest -q` progressed past torch/Triton failures.
- **Committed in:** `d155200`

**3. [Rule 3 - Blocking] Made synthetic-quality import assertions order-independent**
- **Found during:** Task 3 full-suite verification
- **Issue:** `tests/test_synthetic_quality.py` asserted `transformers` was absent from global `sys.modules`, which was order-dependent once the smoke test restored preloaded modules.
- **Fix:** Asserted that synthetic-quality inspection did not newly import `paddleocr`, `diffusers`, or `transformers` during the function call.
- **Files modified:** `tests/test_synthetic_quality.py`
- **Verification:** Full `PATH="/root/.local/bin:$PATH" uv run pytest -q` passed with 148 tests.
- **Committed in:** `d155200`

---

**Total deviations:** 3 auto-fixed (3 Rule 3 blocking fixes)
**Impact on plan:** All fixes were required to achieve the plan's CPU-safe test posture and required full-suite verification; no GPU/model/OCR behavior was changed.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found only local test accumulator dictionaries and expected empty fixture assertions, not placeholder data flows or unimplemented behavior.

## Threat Flags

None. The plan threat model covered numeric fixtures crossing into pure objective helpers and trainer delegation; the implementation introduced no new network endpoints, auth paths, persistent artifact schemas, or external file trust boundaries.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_objective_math.py -x` — RED failed as intended on missing `src.training.dpo_objective` after the scheduler import setup was fixed.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_objective_math.py tests/test_losses.py -q` — passed, 17 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_objective_math.py tests/test_losses.py && PATH="/root/.local/bin:$PATH" uv run pytest -q` — passed, 17 targeted tests and 148 full-suite tests.

## Deferred Issues

- The worktree still contains unrelated pre-existing dirty and untracked files in training, scripts, configs, docs, data, and thesis directories. They were left untouched and excluded from all plan commits except for the plan-owned DPO objective/test files and blocking test-isolation fixes.
- `src/training/rewards.py` still imports Paddle/PaddleX internals at module import time; this was not refactored because Phase 4 Plan 04-05 owns reward wrapper import-safety work. The objective scheduler test uses a local stub to stay CPU-safe until that plan executes.
- No CUDA/model/OCR diagnostics were run, per plan constraints.

## Next Phase Readiness

- Plan 04-04 can proceed with fixed-seed prompt-generation determinism tests on top of a default suite that now passes after torch-heavy imports have occurred.
- Phase 5 trainer comparability work can depend on pure DPO objective helpers and numeric tests before changing DPO modes, pair construction, or trainer structure.

## Self-Check: PASSED

- Found `tests/test_training_objective_math.py`, `src/training/dpo_objective.py`, `src/training/dpo_trainer.py`, `tests/test_smoke_environment.py`, `tests/test_synthetic_quality.py`, and this summary file.
- Found task commits `488303d`, `c48cf9c`, and `d155200` in git history.
- Required verification commands passed.

---
*Phase: 04-cpu-safe-characterization-tests*  
*Completed: 2026-05-05T18:20:33Z*
