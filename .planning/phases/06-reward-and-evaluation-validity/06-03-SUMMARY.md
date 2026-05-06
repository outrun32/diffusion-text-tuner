---
phase: 06-reward-and-evaluation-validity
plan: 03
subsystem: evaluation-diagnostics
tags: [evaluation, russian-slices, gold-benchmark, reward-validity, cpu-safe-tests]

requires:
  - phase: 06-reward-and-evaluation-validity
    provides: Canonical reward result and held-out evaluation contracts from Plans 06-01 and 06-02
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Russian prompt/data quality context and generated-artifact safety policy
provides:
  - CPU-safe Russian text difficulty slice classification in `src.evaluation.slices`
  - Metadata-only gold diagnostic benchmark loading, schema validation, and prediction agreement reports
  - Tiny committed JSONL gold diagnostic fixture with no generated binary assets
  - Evaluation diagnostics documentation with docs drift tests
affects: [phase-6-scoring-outputs, phase-6-reward-diagnostics, phase-6-thesis-bundles]

tech-stack:
  added: []
  patterns: [pure-python-evaluation-contracts, aggregate-schema-validation, explicit-missing-evidence, docs-drift-tests]

key-files:
  created:
    - src/evaluation/slices.py
    - src/evaluation/gold_benchmark.py
    - tests/test_evaluation_slices_gold.py
    - tests/fixtures/evaluation/gold_diagnostic.jsonl
    - docs/evaluation_diagnostics.md
  modified: []

key-decisions:
  - "Keep slice and gold benchmark diagnostics CPU-safe and metadata-only so default tests never load FLUX, Qwen, PaddleOCR, CUDA, model weights, generated images, tensors, checkpoints, or logs."
  - "Treat missing predictions and reward disagreements as explicit diagnostic evidence rather than hidden pass/fail conditions."
  - "Use aggregate JSONL schema validation for gold benchmarks so users can fix all visible malformed or missing fields before relying on diagnostic evidence."

patterns-established:
  - "Slice labels are deterministic pure-Python tags derived from `target_text` and optional prompt metadata fields."
  - "Gold diagnostic reports join predictions by `sample_id` and preserve source benchmark paths, missing predictions, agreement counters, and per-slice summaries."

requirements-completed: [EVAL-05, EVAL-07]

duration: 5min 10s
completed: 2026-05-06T14:45:14Z
---

# Phase 6 Plan 03: Russian Slice and Gold Diagnostic Benchmark Summary

**CPU-safe Russian difficulty slicing and metadata-only gold diagnostic validation for reward/evaluation evidence checks.**

## Performance

- **Duration:** 5min 10s
- **Started:** 2026-05-06T14:40:04Z
- **Completed:** 2026-05-06T14:45:14Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added `classify_text_slices` and `summarize_slices` for deterministic Russian text difficulty labels covering rare Cyrillic letters, word/phrase length, digits, punctuation, mixed case, multiline text, and optional font/style/scene/background metadata.
- Added `load_gold_benchmark`, `evaluate_gold_predictions`, and `format_gold_report_markdown` for small JSONL gold diagnostic benchmarks with aggregate schema validation and explicit missing/disagreement evidence.
- Committed a tiny metadata-only gold fixture at `tests/fixtures/evaluation/gold_diagnostic.jsonl`; the directory contains only JSONL text metadata and no generated image/tensor/checkpoint/log assets.
- Documented all supported slice labels, gold schema fields, report semantics, and generated-artifact safety guidance in `docs/evaluation_diagnostics.md`, guarded by docs drift tests.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_slices_gold.py -x` failed with `ModuleNotFoundError: No module named 'src.evaluation.slices'` after adding slice tests.
- **Task 2 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_slices_gold.py -x` failed with `ModuleNotFoundError: No module named 'src.evaluation.gold_benchmark'` after adding gold benchmark tests and fixture.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_slices_gold.py -q` passed with 9 tests after implementing slices and gold benchmark contracts.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_slices_gold.py -q` failed with `FileNotFoundError: docs/evaluation_diagnostics.md` after adding docs drift assertions.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_slices_gold.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/slices.py src/evaluation/gold_benchmark.py tests/test_evaluation_slices_gold.py` passed after adding docs and lint cleanup.

## Task Commits

1. **Task 1 RED: Slice classification tests** - `6742ed7` (`test`)
2. **Task 2 RED: Gold benchmark tests and fixture** - `44ce1fd` (`test`)
3. **Task 2 GREEN: Slice and gold benchmark contracts** - `41728ec` (`feat`)
4. **Task 3 RED: Evaluation diagnostics docs drift test** - `7a45c26` (`test`)
5. **Task 3 GREEN: Evaluation diagnostics docs** - `5548dd9` (`docs`)

## Files Created/Modified

- `src/evaluation/slices.py` - CPU-safe Russian text difficulty slice classifier and aggregate slice summary helper.
- `src/evaluation/gold_benchmark.py` - Gold diagnostic JSONL loader, schema validator, prediction evaluator, per-slice report builder, and Markdown renderer.
- `tests/test_evaluation_slices_gold.py` - TDD and docs drift coverage for import safety, slices, gold schema validation, agreement reports, missing predictions, and docs contracts.
- `tests/fixtures/evaluation/gold_diagnostic.jsonl` - Tiny metadata-only gold diagnostic fixture covering rare Cyrillic, digits/punctuation, mixed case, and multiline examples.
- `docs/evaluation_diagnostics.md` - User-facing slice and gold diagnostic contract guide with artifact safety notes.

## Decisions Made

- Kept diagnostics import-safe and pure Python, matching Phase 6 constraints and preserving CPU-safe default testing.
- Used `sample_id` as the only join key for gold predictions so missing predictions are deterministic and explicit.
- Normalized OCR text agreement by stripping punctuation/spacing differences for diagnostic comparison while still reporting exact-match and OCR-detection expectation disagreements separately.
- Limited the committed gold fixture to JSONL metadata paths; no image binaries were added.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed Ruff import ordering and line length before final verification**
- **Found during:** Task 3 (Document slice and gold diagnostic contracts)
- **Issue:** Ruff flagged import ordering in `tests/test_evaluation_slices_gold.py` and a long Markdown table string in `src/evaluation/gold_benchmark.py`.
- **Fix:** Applied Ruff's safe import-order fix for the plan-owned test and split the long Markdown table header string without changing behavior.
- **Files modified:** `tests/test_evaluation_slices_gold.py`, `src/evaluation/gold_benchmark.py`
- **Verification:** Task 3 verification command passed.
- **Commit:** `5548dd9`

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking)
**Impact on plan:** Verification-only cleanup on plan-owned files; no scope expansion.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan. Empty list/dict initializers in implementation are runtime accumulators, not UI or contract stubs.

## Threat Flags

None. The plan threat model covered the new JSONL benchmark trust boundary, prediction-record diagnostic summaries, and metadata-only fixture safety. Mitigations were implemented through aggregate schema validation, source benchmark path reporting, missing prediction/per-slice counts, and fixture text-only verification.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_slices_gold.py -q` — passed, 10 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/slices.py src/evaluation/gold_benchmark.py tests/test_evaluation_slices_gold.py` — passed.
- Fixture safety check — `tests/fixtures/evaluation/` contains only `gold_diagnostic.jsonl`; no generated images/tensors/checkpoints/logs were committed.

## Issues Encountered

- `gsd-sdk` initialization/state queries produced no output in this environment; planning state and roadmap updates were prepared directly in tracked planning files.
- The worktree had unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## User Setup Required

None - no external service configuration required.

## Next Plan Readiness

- Plan 06-04 can use the slice labels and gold report fields when wiring canonical scoring/evaluation outputs.
- Plan 06-05 can build disagreement diagnostics from `evaluate_gold_predictions` outputs and per-slice summaries.
- Later thesis bundle work can cite the gold diagnostic source path and missing/disagreement counters as explicit evidence.

## Self-Check: PASSED

- Found all created/modified task files: `src/evaluation/slices.py`, `src/evaluation/gold_benchmark.py`, `tests/test_evaluation_slices_gold.py`, `tests/fixtures/evaluation/gold_diagnostic.jsonl`, `docs/evaluation_diagnostics.md`, and this summary.
- Found task commits `6742ed7`, `44ce1fd`, `41728ec`, `7a45c26`, and `5548dd9` in git history.
- Required verification commands passed, and the fixture directory contains only JSONL metadata.

---
*Phase: 06-reward-and-evaluation-validity*
*Completed: 2026-05-06T14:45:14Z*
