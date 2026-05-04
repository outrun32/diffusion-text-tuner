---
phase: 03-data-curriculum-and-dataset-quality
plan: 05
subsystem: data-source-comparison
tags: [source-comparison, generated-data, synthetic-data, data-quality, cli, docs, cpu-safe-tests]

requires:
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Prompt quality reports, dataset manifests, synthetic quality reports, selected samples, and preference pairs from Plans 03-02 through 03-04
  - phase: 02-runtime-contracts-and-run-provenance
    provides: CPU-safe artifact/provenance conventions and generated-artifact safety boundaries
provides:
  - Generated-vs-synthetic comparison report contracts in `src.data_quality.source_comparison`
  - `scripts/compare_data_sources.py` CLI for JSON and Markdown comparison summaries
  - User guidance for interpreting reward-filtered generated-image data against synthetic masked-SFT data
affects: [phase-3-runtime-docs, phase-5-training-comparability, phase-6-evaluation-validity]

tech-stack:
  added: []
  patterns: [pure-json-jsonl-comparison, explicit-evidence-gaps, source-hash-provenance, thin-cli, tdd]

key-files:
  created:
    - src/data_quality/source_comparison.py
    - scripts/compare_data_sources.py
    - tests/test_data_source_comparison.py
    - docs/data_source_comparison.md
  modified:
    - src/data_quality/__init__.py

key-decisions:
  - "Keep source comparison metadata-only and CPU-safe by reading JSON/JSONL reports and manifests without inspecting generated images or tensors."
  - "Treat every comparison input as optional evidence and expose missing evidence explicitly instead of fabricating unavailable metrics."
  - "Encode interpretation rules as report fields so thesis notes can separate expected strengths/failures from final Phase 6 evaluation evidence."

requirements-completed: [DATA-07]

duration: 6min
completed: 2026-05-04T15:56:26Z
---

# Phase 3 Plan 05: Data Source Comparison Summary

**Metadata-only generated reward-filtered versus synthetic masked-SFT comparison reports with explicit evidence gaps, provenance, and thesis interpretation guidance.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-04T15:50:21Z
- **Completed:** 2026-05-04T15:56:26Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added `DataSourceComparison` and `compare_data_sources(...)` in `src.data_quality.source_comparison`.
- Aggregated generated prompt counts, selected-sample counts, preference-pair counts, synthetic counts, accepted/rejected synthetic samples, rare-character overlap/gaps, distribution differences, generated score summaries, DPO margin summaries, synthetic mask/contrast/OCR health, and input provenance.
- Added explicit `evidence_available` and `evidence_missing` fields for optional prompt quality reports, selected samples, preference pairs, generated manifests, synthetic quality reports, and synthetic manifests.
- Added `scripts/compare_data_sources.py` for JSON comparison reports, Markdown summaries, and concise stdout summaries.
- Documented command examples, input/output schemas, artifact safety, expected help/failure interpretations, and thesis caveats in `docs/data_source_comparison.md`.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_data_source_comparison.py -q` failed with `ModuleNotFoundError: No module named 'src.data_quality.source_comparison'` after adding metrics/provenance tests only.
- **Task 1 GREEN:** The same test command passed with 2 tests after implementing `src/data_quality/source_comparison.py` and exporting `DataSourceComparison` / `compare_data_sources`.
- **Task 2 RED:** The same test command failed with `ModuleNotFoundError: No module named 'scripts.compare_data_sources'` after adding CLI JSON/Markdown/missing-evidence tests.
- **Task 2 GREEN:** The same test command passed with 4 tests after adding `scripts/compare_data_sources.py`.
- **Task 3 RED:** The same test command failed with `FileNotFoundError: docs/data_source_comparison.md` after adding docs coverage.
- **Task 3 GREEN:** The same test command passed with 5 tests after adding source comparison documentation.

## Task Commits

1. **Task 1 RED: Source comparison metric tests** - `992b1d7` (`test`)
2. **Task 1 GREEN: Source comparison metrics implementation** - `869721a` (`feat`)
3. **Task 2 RED: Comparison CLI tests** - `083c950` (`test`)
4. **Task 2 GREEN: Comparison CLI implementation** - `5e5a9ff` (`feat`)
5. **Task 3 RED: Documentation coverage test** - `741f6ae` (`test`)
6. **Task 3 GREEN: Source comparison documentation** - `dcbca10` (`docs`)

## Files Created/Modified

- `src/data_quality/source_comparison.py` - CPU-safe comparison report dataclass, JSON/JSONL input parsing, provenance hashing, counts, coverage gaps, score summaries, synthetic health, and expected help/failure sections.
- `src/data_quality/__init__.py` - Public exports for `DataSourceComparison` and `compare_data_sources`.
- `scripts/compare_data_sources.py` - Thin CLI for optional evidence paths, JSON report writes, Markdown summaries, and stdout summaries.
- `tests/test_data_source_comparison.py` - TDD coverage for comparison metrics, missing optional evidence, CLI outputs, Markdown summaries, and docs drift.
- `docs/data_source_comparison.md` - User-facing DATA-07 workflow, schema, interpretation, generated-artifact safety, and thesis caveats.

## Decisions Made

- Kept comparison tooling CPU-safe and metadata-only by default; it never opens generated images/tensors and relies on prior Phase 3 reports/manifests for evidence.
- Returned unavailable metrics as `None` or empty maps with explicit `evidence_missing` entries, preserving scientific honesty when a run lacks one artifact family.
- Recorded SHA-256 provenance for every parsed evidence file so comparison conclusions can trace back to exact JSON/JSONL artifacts.
- Included interpretation sections in machine-readable reports to make expected help/failure assumptions inspectable before training comparisons.

## Deviations from Plan

None - plan executed as written. Lint line-length cleanup was applied before committing the relevant task implementations and did not change scope.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan. Empty dictionaries/lists in reports represent unavailable optional evidence, not placeholder UI data.

## Threat Flags

None. The plan threat model covered report parsing and conclusions. Mitigations were implemented through minimal schema/path parsing, explicit missing evidence, per-input provenance, SHA-256 hashes, and aggregate interpretation sections.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_data_source_comparison.py -q` — passed, 5 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/data_quality/source_comparison.py src/data_quality/__init__.py scripts/compare_data_sources.py tests/test_data_source_comparison.py` — passed.
- Temporary CLI smoke under `/tmp/opencode/source-comparison-smoke` — passed; generated JSON and Markdown reports from tiny `selected_samples.jsonl` and `synthetic-quality.json` fixtures.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirements updates were prepared manually.
- `uv` is available only with `/root/.local/bin` added to `PATH` in this shell, so verification commands used `PATH="/root/.local/bin:$PATH"`.
- The worktree had unrelated pre-existing dirty and untracked files before execution, including generated data/thesis artifacts and trainer changes; they were left untouched and excluded from all commits.

## User Setup Required

None - no external service configuration required.

## Next Plan Readiness

- Plan 03-06 can link `docs/data_source_comparison.md`, add command aliases, and update runtime contract docs now that the comparison module and CLI exist.
- Phase 5 training comparability can consume comparison reports to document which data source was expected to help/fail before running SFT, DPO, masked-SFT, or combined variants.
- Phase 6 evaluation should validate whether the expected strengths/failures actually correspond to rendered-text quality on held-out prompts.

## Self-Check: PASSED

- Found all created/modified task files: `src/data_quality/source_comparison.py`, `src/data_quality/__init__.py`, `scripts/compare_data_sources.py`, `tests/test_data_source_comparison.py`, `docs/data_source_comparison.md`, and this summary.
- Found task commits `992b1d7`, `869721a`, `083c950`, `5e5a9ff`, `741f6ae`, and `dcbca10` in git history.
- Required verification commands passed, CLI smoke outputs stayed under `/tmp/opencode`, and no generated images/tensors were inspected or committed.

---
*Phase: 03-data-curriculum-and-dataset-quality*
*Completed: 2026-05-04T15:56:26Z*
