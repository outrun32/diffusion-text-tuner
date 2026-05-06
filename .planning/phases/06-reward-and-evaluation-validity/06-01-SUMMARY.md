---
phase: 06-reward-and-evaluation-validity
plan: 01
subsystem: reward-evaluation-contracts
tags: [cpu-safe-tests, rewards, evaluation, product-score, metadata, import-safety]

requires:
  - phase: 04-cpu-safe-characterization-tests
    provides: Import-safe fake reward wrapper characterization for Qwen/VLM and OCR reward boundaries
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Missing-evidence and comparability conventions for training/run reports
provides:
  - Canonical RewardResult row contract for shared score/evaluation/report records
  - Missing-aware ProductScoreFormula and compute_product_score formula implementation
  - Score metadata helper carrying formula details, thresholds, scorer versions, and manifest links
  - CPU-safe tests and docs drift assertions for reward evaluation contracts
affects: [phase-6-reward-evaluation-validity, scoring-outputs, evaluation-diagnostics, thesis-reports]

tech-stack:
  added: []
  patterns: [standard-library-only-contract-module, frozen-dataclass-records, weighted-geometric-product-score, docs-drift-tests]

key-files:
  created:
    - src/evaluation/reward_interface.py
    - tests/test_evaluation_reward_interface.py
    - docs/reward_evaluation.md
  modified: []

key-decisions:
  - "Use a weighted geometric product over normalized VLM, OCR, CER-quality, entropy-quality, and exact-text terms."
  - "Compute a numeric score from available evidence while marking incomplete rows with missing_components and formula_complete=false."
  - "Keep reward metadata creation pure and secret-safe by storing manifest links and scorer versions without opening local files or cache paths."

patterns-established:
  - "Canonical reward rows serialize nested metrics and scorer metadata as deterministic JSON strings for CSV/JSON sidecars."
  - "Product score formulas carry weights, thresholds, scorer_versions, entropy_scale, and formula name as explicit metadata."
  - "Docs drift tests require reward_evaluation.md to mention exact exported names and canonical field names."

requirements-completed: [EVAL-01, EVAL-02, STR-03]

metrics:
  duration: 5min
  completed: 2026-05-06T14:26:22Z
  tasks: 3
  files: 4
---

# Phase 6 Plan 01: Canonical Reward Interface and Product Formula Summary

**CPU-safe canonical reward rows, weighted product-score metadata, and missing-evidence accounting for VLM/OCR/CER/entropy/exact-text evaluation.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-06T14:21:18Z
- **Completed:** 2026-05-06T14:26:22Z
- **Tasks:** 3
- **Files modified:** 4 including this summary

## Accomplishments

- Added `src/evaluation/reward_interface.py` as a standard-library-only module exporting `RewardResult`, `ProductScoreFormula`, `compute_product_score`, and `build_score_metadata` without loading Qwen, PaddleOCR, OCR engines, CUDA, vLLM, MLX, Diffusers, Transformers, torch, PIL, or model weights.
- Implemented deterministic reward row serialization covering `sample_id`, `version`, `target_text`, component scores, exact/text metrics, scorer metadata, thresholds, `missing_components`, and `manifest_path`.
- Implemented a documented weighted geometric product formula using `score_vlm`, `score_ocr`, `cer_quality`, `entropy_quality`, and `exact_text_match`, with threshold flags and explicit missing-evidence reporting.
- Added CPU-safe tests for canonical schema, product formula math, missing evidence, metadata sidecars, import safety, and documentation drift.
- Published `docs/reward_evaluation.md` describing field names, formula terms, scorer versions, thresholds, manifest links, missing evidence semantics, and generated-artifact safety.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -x` failed as intended with `ModuleNotFoundError: No module named 'src.evaluation.reward_interface'` after contract tests were written.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -q` passed with 5 tests after implementing the import-safe reward interface.
- **Task 3 RED:** The targeted pytest command failed as intended with `FileNotFoundError: docs/reward_evaluation.md` after adding docs drift assertions.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/reward_interface.py tests/test_evaluation_reward_interface.py` passed with 6 tests and Ruff clean.

## Task Commits

1. **Task 1 RED: Specify canonical reward records and product formula** - `168ebfe` (`test`)
2. **Task 2 GREEN: Implement import-safe reward interface module** - `202e7de` (`feat`)
3. **Task 3 RED: Add reward docs drift assertion** - `bd8fd31` (`test`)
4. **Task 3 GREEN: Document reward schema and product score formula** - `a7d68c3` (`docs`)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `src/evaluation/reward_interface.py` - Pure reward contract module with frozen dataclasses, product formula computation, threshold flags, metadata helper, and missing-evidence accounting.
- `tests/test_evaluation_reward_interface.py` - CPU-safe TDD tests for row serialization, formula math, metadata, missing evidence, import safety, and docs drift.
- `docs/reward_evaluation.md` - Contract documentation for canonical fields, formula terms, scorer versions, thresholds, manifest links, and generated-artifact safety.
- `.planning/phases/06-reward-and-evaluation-validity/06-01-SUMMARY.md` - This execution summary.

## Decisions Made

- Used a weighted geometric product so product scores reward agreement across normalized VLM, OCR, CER-quality, entropy-quality, and exact-text evidence rather than adding unrelated scales linearly.
- Normalized `cer` into `cer_quality = 1.0 - cer` and `entropy` into `entropy_quality = exp(-entropy_scale * entropy)` so all product terms are higher-is-better.
- Kept numeric scores available for incomplete rows, but made comparability explicit through `missing_components` and `formula_complete`; downstream reports should reject or flag incomplete rows.
- Stored `source_manifest_paths` and `scorer_versions` as metadata strings without inspecting local files, cache paths, environment variables, or secrets.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed plan-owned Ruff issues**
- **Found during:** Task 3 verification
- **Issue:** Ruff flagged import ordering, `typing` collection imports, and long lines in plan-owned Python files after the docs drift test and module implementation were added.
- **Fix:** Applied Ruff import ordering and wrapped long expressions without changing behavior.
- **Files modified:** `src/evaluation/reward_interface.py`, `tests/test_evaluation_reward_interface.py`
- **Verification:** Targeted pytest and Ruff command passed afterward.
- **Committed in:** `a7d68c3`

---

**Total deviations:** 1 auto-fixed (1 Rule 3 blocking fix)
**Impact on plan:** Lint cleanup was required for the plan's Task 3 verification. No scope expansion, generated artifacts, model/OCR/CUDA imports, or heavy runtime work were introduced.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no `TODO`, `FIXME`, placeholder, coming-soon text, hardcoded empty UI data, or unwired mock data in plan-created/modified files.

## Threat Flags

None. The plan threat model covered untrusted score records entering product formula computation and manifest/scorer metadata entering reports. The implementation adds no network endpoints, auth paths, file-reading paths, schema changes at external trust boundaries, or unplanned security surface.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -x` — RED failed as intended on missing `src.evaluation.reward_interface` before implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -q` — passed with 5 tests after implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -x` — RED failed as intended on missing `docs/reward_evaluation.md` before documentation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_reward_interface.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/reward_interface.py tests/test_evaluation_reward_interface.py` — passed with 6 tests and Ruff clean.
- `python - <<'PY' ... import src.evaluation.reward_interface ... PY` — printed `False` for newly loaded `transformers`, `paddleocr`, or `diffusers`.

## TDD Gate Compliance

- RED gate commits exist: `168ebfe` and `bd8fd31`.
- GREEN gate commits exist after their RED commits: `202e7de` and `a7d68c3`.
- No refactor-only commit was needed beyond lint cleanup included in Task 3's GREEN commit.

## Deferred Issues

- The worktree still contains unrelated pre-existing dirty and untracked files in training, configs, docs, data, scripts, and tests. They were left untouched and excluded from all plan commits.
- The GSD SDK CLI was unavailable in this environment (`gsd-sdk: command not found` and no local `node_modules/@gsd-build/sdk`), so state/roadmap/requirements updates were applied manually instead of through SDK query handlers.
- No GPU/model/OCR diagnostics were run, per plan constraints.

## Next Phase Readiness

- Phase 6 Plan 04 can wire scoring/evaluation outputs to `RewardResult`, `ProductScoreFormula`, and `build_score_metadata` without duplicating product formula logic.
- Phase 6 diagnostic and thesis-output plans can use `missing_components`, `threshold_flags`, `source_manifest_paths`, and `scorer_versions` to keep missing evidence explicit.

## Self-Check: PASSED

- Found `src/evaluation/reward_interface.py`, `tests/test_evaluation_reward_interface.py`, `docs/reward_evaluation.md`, and this summary file.
- Found task commits `168ebfe`, `202e7de`, `bd8fd31`, and `a7d68c3` in git history.
- Required targeted pytest, Ruff, and import-safety verification commands passed.

---
*Phase: 06-reward-and-evaluation-validity*  
*Completed: 2026-05-06T14:26:22Z*
