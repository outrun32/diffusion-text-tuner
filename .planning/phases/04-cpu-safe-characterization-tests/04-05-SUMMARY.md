---
phase: 04-cpu-safe-characterization-tests
plan: 05
subsystem: reward-wrapper-characterization
tags: [cpu-safe-tests, rewards, qwen, ocr, paddleocr, import-safety]

requires:
  - phase: 04-cpu-safe-characterization-tests
    provides: Prior CPU-safe characterization posture, objective tests, and prompt-generation import-safety patterns
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Reward-filtered selection context and generated score-file semantics
provides:
  - Fake-based CPU-safe reward wrapper characterization tests for Qwen and OCR rewards
  - Import-safe `src.training.rewards` module behavior without PaddleOCR/PaddleX/Qwen/Transformers at collection time
  - Lazy OCR CTC capture patch setup behind `OcrCerEntropyReward` initialization
  - Scoring script import-safety checks that keep reward classes inside scorer selection paths
affects: [phase-4-characterization-tests, phase-6-reward-evaluation-validity, scoring-boundaries]

tech-stack:
  added: []
  patterns: [lazy-optional-dependency-imports, fake-reward-wrapper-tests, tmp-png-lifecycle-tests]

key-files:
  created:
    - tests/test_reward_wrapper_contracts.py
  modified:
    - src/training/rewards.py
    - scripts/score_images.py

key-decisions:
  - "Keep OCR/Paddle dependency setup lazy so default pytest can import reward helpers without optional OCR packages."
  - "Use object.__new__ and local fakes for Qwen/OCR wrapper tests instead of initializing models, CUDA, PaddleOCR, or Transformers."
  - "Preserve scoring CLI reward imports inside scorer branches to keep import-only collection CPU-safe."

patterns-established:
  - "Reward wrapper tests characterize public methods and formulas with deterministic tiny tensors, arrays, and PIL images."
  - "Optional OCR monkey-patching is idempotent and installed only when real OCR reward initialization is requested."

requirements-completed: [TEST-05]

metrics:
  duration: 4min
  completed: 2026-05-05T18:33:20Z
  tasks: 3
  files: 4
---

# Phase 4 Plan 05: Import-Safe Fake Reward Wrapper Tests Summary

**Fake Qwen/OCR reward tests and lazy PaddleOCR setup now lock reward wrapper behavior without model, OCR, CUDA, or optional dependency loading during default pytest.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T18:29:30Z
- **Completed:** 2026-05-05T18:33:20Z
- **Tasks:** 3
- **Files modified:** 4 including this summary

## Accomplishments

- Added `tests/test_reward_wrapper_contracts.py` with 160 lines of CPU-safe reward characterization coverage.
- Proved `src.training.rewards` imports without `paddleocr`, `paddlex`, `transformers`, Qwen, CUDA, or model weights.
- Characterized `_normalize_homoglyphs`, `_char_error_rate`, and `_ctc_entropy_stats` with deterministic tiny inputs.
- Characterized `QwenYesProbReward.score_batch` ordering and tensor output using `object.__new__` plus a fake `score_single`, with no model initialization.
- Characterized `OcrCerEntropyReward.score` with fake OCR output and seeded raw CTC predictions, including CER, entropy, `min_p`, `frac_unc`, and `reward_ocr` formula behavior.
- Verified `score_pil` writes a temporary PNG for scoring and deletes it afterward.
- Verified `scripts.score_images` remains import-safe and keeps `QwenYesProbReward`/`OcrCerEntropyReward` imports inside scorer selection branches.
- Moved Paddle/PaddleX CTC monkey-patching behind lazy OCR setup while preserving existing OCR reward formula and public wrapper methods.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py -x` failed as intended with `ModuleNotFoundError: No module named 'paddlex.inference'; 'paddlex' is not a package` after import-safety and helper tests were written.
- **Task 1 GREEN:** The same targeted command passed with 3 tests after moving Paddle/PaddleX setup behind lazy OCR initialization and fixing a test expectation for homoglyph-normalized CER.
- **Task 2 characterization:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py -q` passed with 6 tests after adding fake Qwen and fake OCR wrapper coverage. Existing wrapper behavior satisfied the characterized contracts once Task 1 made imports safe.
- **Task 3 characterization:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py && PATH="/root/.local/bin:$PATH" uv run pytest -q` passed with 8 targeted tests and 163 full-suite tests after adding scoring import-safety tests.

## Task Commits

1. **Task 1 RED: Prove reward module import safety and pure helper behavior** - `ba49452` (`test`)
2. **Task 1 GREEN: Make reward helpers import safe** - `588f615` (`feat`)
3. **Task 2: Characterize Qwen and OCR wrappers with fakes** - `ba6df6f` (`test`)
4. **Task 3: Guard scoring script compatibility after reward import changes** - `a5c8923` (`test`)
5. **Refactor/lint cleanup: Tidy reward scoring wrappers** - `af7aecb` (`style`)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_reward_wrapper_contracts.py` - CPU-safe fake/mock tests for reward import safety, pure helpers, Qwen batch ordering, OCR formula behavior, temporary PNG cleanup, and scoring import boundaries.
- `src/training/rewards.py` - Lazy optional OCR/Paddle dependency setup, import-safe helper/module behavior, OCR CER/entropy reward wrapper, and Qwen batch scoring support without constructor-side globals.
- `scripts/score_images.py` - Scoring boundary with lazy VLM/OCR reward imports inside scorer selection paths and OCR-compatible CSV fields.
- `.planning/phases/04-cpu-safe-characterization-tests/04-05-SUMMARY.md` - This execution summary.

## Decisions Made

- Kept `numpy` and PIL importable at module scope because they are lightweight project dependencies used by pure helper tests; only optional OCR/model stacks are lazy-loaded.
- Installed the PaddleX CTC decoder patch through an idempotent `_ensure_ctc_capture_patch()` called by `OcrCerEntropyReward.__init__`, preserving the existing raw-prediction capture semantics while avoiding import-time optional dependencies.
- Used source inspection for `scripts.score_images` import-boundary tests because the plan explicitly required reward class imports to remain inside scorer selection paths without instantiating models.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Made optional OCR dependency setup lazy**
- **Found during:** Task 1 RED verification
- **Issue:** `src.training.rewards` imported PaddleX internals at module import time, breaking default pytest collection when OCR extras are unavailable.
- **Fix:** Added lazy, idempotent `_ensure_ctc_capture_patch()` and invoked it from `OcrCerEntropyReward.__init__` before importing PaddleOCR.
- **Files modified:** `src/training/rewards.py`
- **Verification:** Targeted reward tests passed and full default pytest passed without OCR/model packages.
- **Committed in:** `588f615`

**2. [Rule 3 - Blocking] Included plan-relevant scoring boundary changes already present in the worktree**
- **Found during:** Task 3
- **Issue:** `scripts/score_images.py` had uncommitted scorer-selection changes at executor start, and the plan explicitly required the scoring boundary to remain compatible with lazy reward class imports.
- **Fix:** Added import-safety tests for that boundary and committed only the plan-owned scoring file alongside those tests, leaving unrelated dirty files untouched.
- **Files modified:** `scripts/score_images.py`, `tests/test_reward_wrapper_contracts.py`
- **Verification:** Targeted reward tests and full pytest passed.
- **Committed in:** `a5c8923`

**3. [Rule 3 - Blocking] Fixed task-owned Ruff issues after verification**
- **Found during:** Post-task lint check
- **Issue:** Ruff flagged an unused `json` import, unnecessary read modes, local import ordering, and a long OCR reward expression in plan-owned files.
- **Fix:** Removed the unused import/read modes, sorted the local imports, and wrapped the reward expression without behavior changes.
- **Files modified:** `scripts/score_images.py`, `src/training/rewards.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check tests/test_reward_wrapper_contracts.py src/training/rewards.py scripts/score_images.py` passed; targeted tests passed afterward.
- **Committed in:** `af7aecb`

---

**Total deviations:** 3 auto-fixed (1 Rule 2 missing critical, 2 Rule 3 blocking fixes)
**Impact on plan:** All fixes were required for import-safe default tests and scoring compatibility; no Qwen, PaddleOCR, CUDA, VLM, OCR engine, vLLM, or MLX initialization was introduced.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found one Qwen chat-template image marker string in `src/training/rewards.py`; it is a processor placeholder required by the existing Qwen message-building contract, not an unwired data stub.

## Threat Flags

None. The plan threat model covered fake model/OCR output into reward wrappers and optional dependency imports into default pytest. The implementation introduces no new network endpoints, auth paths, persistent artifact schemas, or unplanned trust boundaries.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py -x` — RED failed as intended on import-time PaddleX dependency before production code was changed.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py -x` — passed after lazy OCR setup, 3 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py -q` — passed after fake wrapper tests, 6 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py && PATH="/root/.local/bin:$PATH" uv run pytest -q` — passed after scoring import-safety tests, 8 targeted tests and 163 full-suite tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check tests/test_reward_wrapper_contracts.py src/training/rewards.py scripts/score_images.py` — passed.
- Final `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py -q && PATH="/root/.local/bin:$PATH" uv run pytest -q` — passed, 8 targeted tests and 163 full-suite tests.

## Deferred Issues

- The worktree still contains unrelated pre-existing dirty and untracked files in training, configs, docs, data, and scripts. They were left untouched and excluded from all commits except for plan-owned `src/training/rewards.py` and `scripts/score_images.py` changes.
- No GPU/model/OCR diagnostics were run, per plan constraints.

## Next Phase Readiness

- Phase 4 Plan 06 can publish characterization commands and docs with reward wrapper coverage now included in the default CPU-safe suite.
- Phase 6 reward/evaluation centralization can use these tests as behavior locks for Qwen batch delegation, OCR CER/entropy formulas, and scoring import boundaries.

## Self-Check: PASSED

- Found `tests/test_reward_wrapper_contracts.py`, `src/training/rewards.py`, `scripts/score_images.py`, and this summary file.
- Found task commits `ba49452`, `588f615`, `ba6df6f`, `a5c8923`, and `af7aecb` in git history.
- Required verification commands passed.

---
*Phase: 04-cpu-safe-characterization-tests*  
*Completed: 2026-05-05T18:33:20Z*
