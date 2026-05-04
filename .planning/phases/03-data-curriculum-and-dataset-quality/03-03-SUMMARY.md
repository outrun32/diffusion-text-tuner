---
phase: 03-data-curriculum-and-dataset-quality
plan: 03
subsystem: synthetic-dataset-quality
tags: [synthetic-data, masked-sft, quality-reports, manifests, contact-sheets, cpu-safe-tests]

requires:
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Prompt-side dataset manifest helpers and source hashing from Plan 03-02
  - phase: 02-runtime-contracts-and-run-provenance
    provides: CPU-safe runtime provenance and generated-artifact safety patterns
provides:
  - CPU-safe synthetic masked-SFT quality inspection in `src.data_quality.synthetic_quality`
  - Thin `scripts/inspect_synthetic_dataset.py` CLI for reports, manifests, optional OCR summaries, and contact sheets
  - User documentation for synthetic layouts, filters, OCR handoff, manifests, and artifact safety
affects: [phase-3-selection-artifacts, phase-3-source-comparison, phase-3-runtime-docs]

tech-stack:
  added: []
  patterns: [pil-only-image-inspection, deterministic-json-reports, optional-ocr-handoff, explicit-output-paths, runtime-manifest-provenance]

key-files:
  created:
    - src/data_quality/synthetic_quality.py
    - scripts/inspect_synthetic_dataset.py
    - tests/test_synthetic_quality.py
    - docs/synthetic_quality.md
  modified:
    - src/data_quality/__init__.py

key-decisions:
  - "Keep default synthetic quality inspection CPU-safe by using PIL/CSV/JSON only and accepting OCR results only as optional precomputed files."
  - "Treat reports, manifests, and contact sheets as explicit runtime outputs under user-specified paths rather than committed generated artifacts."
  - "Reuse `dataset-manifest/v1` for synthetic provenance while hashing safe text/CSV/JSONL sources and referencing generated images/tensors by path."

requirements-completed: [DATA-04, DATA-05]

duration: 7min
completed: 2026-05-04T15:38:25Z
---

# Phase 3 Plan 03: Synthetic Masked-SFT Quality Inspection Summary

**PIL-only synthetic masked-SFT quality reports with optional OCR result summaries, dataset manifests, threshold filters, and contact sheets.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-04T15:31:10Z
- **Completed:** 2026-05-04T15:38:25Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added `SyntheticQualityReport` and `inspect_synthetic_dataset(...)` for CPU-safe inspection of masked-SFT `index.csv`, `prompts.jsonl`, `shapes.csv`, raw images, raw masks, and raw metadata.
- Computed sample/missing-file counts, mask area fraction, bbox height/area fraction, foreground/background contrast, character coverage, font coverage, resolution distribution, accepted/rejected counts, and threshold rejection reasons.
- Added optional OCR CSV/JSONL ingestion for exact-match rate and CER-style summaries without importing PaddleOCR or any model stack.
- Added `scripts/inspect_synthetic_dataset.py` to write JSON reports, optional `dataset-manifest/v1` manifests, and PIL-only contact sheets to explicit user-provided output paths.
- Documented input layouts, report schema, threshold meanings, optional OCR handoff, contact sheets, manifest provenance, and generated-artifact safety in `docs/synthetic_quality.md`.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` failed with `ModuleNotFoundError: No module named 'src.data_quality.synthetic_quality'` after adding metric/report tests.
- **Task 1 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` passed after implementing synthetic quality metrics and public exports.
- **Task 2 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` failed with `ModuleNotFoundError: No module named 'scripts.inspect_synthetic_dataset'` after adding CLI/contact-sheet tests.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` passed after adding contact-sheet generation and the CLI.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` failed with `FileNotFoundError: docs/synthetic_quality.md` after adding docs coverage.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` passed after adding synthetic quality documentation.

## Task Commits

1. **Task 1 RED: Synthetic quality metrics tests** - `ac3bfa6` (`test`)
2. **Task 1 GREEN: Synthetic quality metrics implementation** - `7062bff` (`feat`)
3. **Task 2 RED: Synthetic inspection CLI tests** - `ba5298c` (`test`)
4. **Task 2 GREEN: Inspection CLI and contact sheets** - `bb98b3c` (`feat`)
5. **Task 3 RED: Synthetic quality docs test** - `6af5d4e` (`test`)
6. **Task 3 GREEN: Synthetic quality documentation** - `2d02032` (`docs`)

## Files Created/Modified

- `src/data_quality/synthetic_quality.py` - CPU-safe synthetic quality report dataclass, dataset inspector, optional OCR summary parsing, threshold filtering, and contact-sheet helper.
- `scripts/inspect_synthetic_dataset.py` - Thin CLI for JSON reports, manifests, optional OCR inputs, contact sheets, thresholds, and bounded sample inspection.
- `tests/test_synthetic_quality.py` - TDD coverage for metrics, missing/heavy import safety, threshold rejections, optional OCR ingestion, CLI outputs, contact sheets, exit codes, and docs drift.
- `docs/synthetic_quality.md` - User-facing synthetic quality/report/manifest/OCR/contact-sheet guide and artifact safety rules.
- `src/data_quality/__init__.py` - Public exports for `SyntheticQualityReport` and `inspect_synthetic_dataset`.

## Decisions Made

- Used PIL image/mask inspection only, avoiding NumPy, Torch tensor loading, PaddleOCR, diffusers, transformers, SynthTIGER, and CUDA in the default inspection path.
- Reused Plan 03-02 dataset manifest helpers so synthetic reports share the same `dataset-manifest/v1` provenance contract as prompt datasets.
- Made contact sheets opt-in and tied to explicit user-provided paths because they expose generated images and prompt/target text.
- Returned nonzero from the CLI when missing files or threshold rejections remain, while keeping OCR summaries optional evidence rather than a default test dependency.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed Ruff line-length issues in new synthetic quality files**
- **Found during:** Task 1 through Task 3 verification
- **Issue:** Ruff reported line-length violations in new implementation, CLI, and test additions.
- **Fix:** Applied Ruff formatting and wrapped long strings/calls while preserving the tested behavior.
- **Files modified:** `src/data_quality/synthetic_quality.py`, `scripts/inspect_synthetic_dataset.py`, `tests/test_synthetic_quality.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/data_quality/synthetic_quality.py scripts/inspect_synthetic_dataset.py tests/test_synthetic_quality.py` passed.
- **Committed in:** `7062bff`, `bb98b3c`, `2d02032`

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan.

## Threat Flags

None. The plan threat model covered local generated dataset inputs, explicit report/contact-sheet paths, and denial-of-service limits; mitigations were implemented through missing-file accounting, explicit output paths, optional `--max-samples`, opt-in contact sheets, and no OCR/model loading in default inspection.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` — passed, 5 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/data_quality/synthetic_quality.py scripts/inspect_synthetic_dataset.py tests/test_synthetic_quality.py` — passed.
- Default module import safety is asserted by tests: `paddleocr`, `diffusers`, and `transformers` are not imported by synthetic quality inspection.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirements updates were prepared manually.
- The worktree had unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## Next Plan Readiness

- Plan 03-04 can consume synthetic quality manifests and filtering stats when materializing selected SFT samples or DPO preference pairs.
- Plan 03-05 can compare generated reward-filtered data against synthetic masked-SFT data using the synthetic report/manifest fields added here.
- Plan 03-06 can link `docs/synthetic_quality.md` and add command aliases after the remaining Phase 3 implementation contracts exist.

## Self-Check: PASSED

- Found all created/modified task files: `src/data_quality/synthetic_quality.py`, `scripts/inspect_synthetic_dataset.py`, `tests/test_synthetic_quality.py`, `docs/synthetic_quality.md`, `src/data_quality/__init__.py`, and this summary.
- Found task commits `ac3bfa6`, `7062bff`, `ba5298c`, `bb98b3c`, `6af5d4e`, and `2d02032` in git history.
- Required verification commands passed, default inspection remains CPU-safe, and no generated reports/contact sheets/images/tensors were committed.

---
*Phase: 03-data-curriculum-and-dataset-quality*
*Completed: 2026-05-04T15:38:25Z*
