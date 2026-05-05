---
phase: 04-cpu-safe-characterization-tests
plan: 01
subsystem: runtime-characterization
tags: [cpu-safe-tests, config-validation, artifact-contracts, pytorch-fixtures]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Strict runtime config loading and CPU-safe artifact validators
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Phase 3 artifact-schema context for later characterization plans
provides:
  - CPU-safe characterization tests for committed SFT, DPO, and masked-SFT config loading
  - Tiny tmp_path artifact fixtures covering prompt, scores, generated, and masked-SFT layouts
  - Keyword-compatible `validate_artifacts(..., require_ready=True)` readiness gate
affects: [phase-4-characterization-tests, phase-5-trainer-refactors, runtime-preflight]

tech-stack:
  added: []
  patterns: [pytest-tmp-path-fixtures, weights-only-cpu-tensor-inspection, aggregate-readiness-errors]

key-files:
  created:
    - tests/test_characterization_config_artifacts.py
    - configs/masked_sft.json
  modified:
    - src/runtime/artifacts.py

key-decisions:
  - "Use real root training configs as characterization inputs so later trainer/runtime refactors preserve committed config behavior."
  - "Keep all artifact characterization fixtures under pytest `tmp_path` and inspect only tiny trusted local tensor dictionaries on CPU."
  - "Support both legacy mapping-based and keyword-based `require_ready` calls for artifact validators to match the documented public interface."

metrics:
  duration: 4min
  completed: 2026-05-05T18:07:16Z
  tasks: 2
  files: 4
---

# Phase 4 Plan 01: Committed Config and Tiny Artifact Characterization Summary

**CPU-safe characterization now locks committed training config validation and tiny artifact path/shape contracts before Phase 5 trainer and evaluation refactors.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T18:04:14Z
- **Completed:** 2026-05-05T18:07:16Z
- **Tasks:** 2
- **Files modified:** 4 including this summary

## Accomplishments

- Added `tests/test_characterization_config_artifacts.py` with CPU-safe coverage for SFT, DPO, and masked-SFT root config loading through `load_stage_config`.
- Asserted malformed/unsafe config errors include the config path and failing field while not echoing secret-like raw values.
- Added strict unknown-field characterization so future config changes cannot silently accept unexpected toggles.
- Added tmp_path artifact fixtures for prompt JSONL, scores CSV, generated image/latent/text-embedding layouts, and masked-SFT latent/mask/embed/shapes contracts.
- Verified tensor inspection uses tiny trusted local dictionaries through `torch.load(..., map_location="cpu", weights_only=True)`.
- Updated `validate_artifacts` to accept the documented keyword form `require_ready=True` while preserving the existing mapping-based flag.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py -x` passed immediately after config tests were added. This was an expected characterization outcome because the core config contract already existed; no production config-loader change was made.
- **Task 1 GREEN/verification:** The same command passed with 5 tests covering real SFT, DPO, and masked-SFT configs plus strict, secret-safe invalid-config behavior.
- **Task 2 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py -x` failed as intended with `TypeError: validate_artifacts() got an unexpected keyword argument 'require_ready'` after adding artifact tests.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py tests/test_runtime_artifacts.py -q` passed with 17 tests after the narrow runtime artifact signature fix.

## Task Commits

1. **Task 1: Characterize committed training config validation** - `dbbfc2e` (`test`)
2. **Task 2 RED: Characterize tiny artifact path and shape contracts** - `de13737` (`test`)
3. **Task 2 GREEN: Support keyword artifact readiness checks** - `3b798b1` (`feat`)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_characterization_config_artifacts.py` - CPU-safe characterization tests for committed configs and tiny artifact contracts.
- `configs/masked_sft.json` - Committed masked-SFT root config required by the characterization tests and Plan 04-01 context.
- `src/runtime/artifacts.py` - Public `validate_artifacts` signature now accepts `paths=None` and keyword-only `require_ready` while preserving existing behavior.
- `.planning/phases/04-cpu-safe-characterization-tests/04-01-SUMMARY.md` - This execution summary.

## Decisions Made

- Treated the previously untracked `configs/masked_sft.json` as plan-critical because Plan 04-01 explicitly requires all three root training configs to load as committed fixtures.
- Kept generated images, tensors, scores, and JSONL data out of git by creating all artifact fixtures under `tmp_path` during tests.
- Limited production code changes to a backward-compatible public function signature correction; no artifact validation semantics were loosened.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Committed missing masked-SFT root config required by characterization tests**
- **Found during:** Task 1
- **Issue:** `configs/masked_sft.json` existed in the worktree but was untracked, while the plan required committed SFT, DPO, and masked-SFT config characterization.
- **Fix:** Staged and committed only the plan-referenced `configs/masked_sft.json` alongside the config characterization tests.
- **Files modified:** `configs/masked_sft.json`, `tests/test_characterization_config_artifacts.py`
- **Commit:** `dbbfc2e`

**2. [Rule 1 - Bug] Aligned artifact validator signature with documented interface**
- **Found during:** Task 2 RED verification
- **Issue:** `validate_artifacts(..., require_ready=True)` raised `TypeError` even though the plan interface documented a keyword-only readiness flag.
- **Fix:** Added `paths=None` and keyword-only `require_ready=False`, preserving mapping-based `{"require_ready": True}` compatibility.
- **Files modified:** `src/runtime/artifacts.py`
- **Commit:** `3b798b1`

**3. [Rule 3 - Blocking] Fixed Ruff import-order issue in the new test file**
- **Found during:** Task 2 verification
- **Issue:** Ruff reported an unsorted import block in `tests/test_characterization_config_artifacts.py`.
- **Fix:** Ran Ruff auto-fix on the targeted test file and re-ran lint/tests.
- **Files modified:** `tests/test_characterization_config_artifacts.py`
- **Commit:** `3b798b1`

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found only implementation accumulators/defaults (`[]`, `{}`, `None`, and empty newline strings) in test/runtime code, not placeholder UI/data flows.

## Threat Flags

None. The plan threat model covered local config loading and local artifact inspection; the implementation preserves secret-safe config errors and CPU-only `weights_only=True` tensor loading.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py tests/test_runtime_artifacts.py -q` — passed before changes, 32 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py -x` — Task 1 characterization passed immediately, 5 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py -x` — Task 2 RED failed as expected on missing `require_ready` keyword support.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py tests/test_runtime_artifacts.py -q` — passed, 17 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_characterization_config_artifacts.py tests/test_runtime_config_io.py tests/test_runtime_artifacts.py -q` — passed, 40 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/artifacts.py tests/test_characterization_config_artifacts.py` — passed.

## Deferred Issues

- The worktree contains unrelated pre-existing dirty and untracked files in training, scripts, configs, docs, and data directories. They were left untouched and excluded from commits except for plan-required `configs/masked_sft.json`.
- No GPU/model/OCR diagnostics were run, per plan constraints.

## Next Phase Readiness

- Plan 04-02 can build dataset, collator, selection, and resolution-bucket characterization on top of the committed config/artifact baseline.
- Phase 5 trainer/evaluation refactors can use this test file to detect runtime config and artifact contract drift without CUDA/model/OCR dependencies.

## Self-Check: PASSED

- Found `tests/test_characterization_config_artifacts.py`, `configs/masked_sft.json`, `src/runtime/artifacts.py`, and this summary file.
- Found task commits `dbbfc2e`, `de13737`, and `3b798b1` in git history.
- Required verification commands passed.

---
*Phase: 04-cpu-safe-characterization-tests*  
*Completed: 2026-05-05T18:07:16Z*
