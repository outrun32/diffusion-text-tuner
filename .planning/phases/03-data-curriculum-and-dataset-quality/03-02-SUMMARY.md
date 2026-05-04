---
phase: 03-data-curriculum-and-dataset-quality
plan: 02
subsystem: prompt-dataset-quality
tags: [prompt-validation, dataset-manifests, provenance, cli, cpu-safe-tests]

requires:
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Prompt curriculum configs and record provenance from Plan 03-01
  - phase: 02-runtime-contracts-and-run-provenance
    provides: CPU-safe reproducibility helpers and generated-artifact safety patterns
provides:
  - CPU-safe prompt JSONL quality validation reports in `src.data_quality.prompt_validation`
  - Dataset manifest creation/loading and safe source hashing in `src.data_quality.manifests`
  - `scripts/validate_prompt_dataset.py` report/manifest CLI
  - User-facing prompt quality and dataset manifest documentation
affects: [phase-3-synthetic-quality, phase-3-selection-artifacts, phase-3-runtime-docs]

tech-stack:
  added: []
  patterns: [frozen-dataclasses, deterministic-json-reports, line-numbered-validation, safe-source-hashing, thin-cli]

key-files:
  created:
    - src/data_quality/prompt_validation.py
    - src/data_quality/manifests.py
    - scripts/validate_prompt_dataset.py
    - tests/test_prompt_dataset_quality.py
    - docs/dataset_quality.md
  modified:
    - src/data_quality/__init__.py

key-decisions:
  - "Keep prompt dataset validators CPU-safe and deterministic by using pure-Python JSONL parsing and aggregate heuristics only."
  - "Hash small safe text/CSV/JSON/JSONL source inputs by default while referencing generated binary tensors/images by path unless explicitly marked safe."
  - "Treat prompt validation warnings as non-blocking by default, with `--strict-warnings` for stricter local gates."

requirements-completed: [DATA-02, DATA-04]

duration: 8min
completed: 2026-05-04T15:29:27Z
---

# Phase 3 Plan 02: Prompt Dataset Quality and Manifest Summary

**CPU-safe prompt JSONL validation with deterministic quality reports, dataset provenance manifests, and a thin CLI for local prompt dataset gates.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-04T15:21:52Z
- **Completed:** 2026-05-04T15:29:27Z
- **Tasks:** 3
- **Files modified:** 6 task files plus this summary and planning metadata

## Accomplishments

- Added `PromptQualityReport` and `validate_prompt_dataset(...)` for line-numbered malformed JSONL/missing-field errors, length buckets, script coverage, rare-character coverage, duplicate rate, content/style distributions, and deterministic naturalness/malformed heuristics.
- Added threshold support for min/max target length, required rare-character coverage, duplicate-rate ceilings, allowed scripts, expected content/style distributions, and warning/error aggregation.
- Added `DatasetManifest`, `hash_source_file`, `create_dataset_manifest`, `load_dataset_manifest`, and `write_dataset_manifest` for prompt-side DATA-04 provenance.
- Reused Phase 2 `collect_git_state` and `collect_model_revisions` while keeping generated binary tensors/images referenced rather than hashed by default.
- Added `scripts/validate_prompt_dataset.py` to write JSON reports/manifests, print stdout JSON when no report path is supplied, and return deterministic exit codes for errors or strict warnings.
- Documented prompt validation commands for simple/full/curriculum datasets, report fields, manifest fields, and generated-artifact safety in `docs/dataset_quality.md`.

## RED/GREEN Evidence

- **Task 1 RED:** `uv run pytest tests/test_prompt_dataset_quality.py -q` failed with `ModuleNotFoundError: No module named 'src.data_quality.prompt_validation'` after adding prompt quality report tests.
- **Task 1 GREEN:** `uv run pytest tests/test_prompt_dataset_quality.py -q` passed after implementing prompt validation contracts and exports.
- **Task 2 RED:** `uv run pytest tests/test_prompt_dataset_quality.py -q` failed with `ModuleNotFoundError: No module named 'src.data_quality.manifests'` after adding dataset manifest provenance tests.
- **Task 2 GREEN:** `uv run pytest tests/test_prompt_dataset_quality.py -q` passed after implementing dataset manifest helpers and safe source hashing.
- **Task 3 RED:** `uv run pytest tests/test_prompt_dataset_quality.py -q` failed because `scripts.validate_prompt_dataset` and `docs/dataset_quality.md` did not exist after adding CLI/docs tests.
- **Task 3 GREEN:** `uv run pytest tests/test_prompt_dataset_quality.py -q` passed after adding the validation CLI and documentation.

## Task Commits

1. **Task 1 RED: Prompt quality validation tests** - `8b9f73f` (`test`)
2. **Task 1 GREEN: Prompt quality validation contracts** - `63345f8` (`feat`)
3. **Task 2 RED: Dataset manifest provenance tests** - `978ce44` (`test`)
4. **Task 2 GREEN: Dataset manifest helpers** - `8e40897` (`feat`)
5. **Task 3 RED: Prompt validation CLI/docs tests** - `f15292d` (`test`)
6. **Task 3 GREEN: Prompt validation CLI and docs** - `bb98dd6` (`feat`)
7. **Verification fix: Script-safe imports and lint cleanup** - `3fea0be` (`fix`)

## Files Created/Modified

- `src/data_quality/prompt_validation.py` - CPU-safe prompt JSONL validation report contract and threshold checks.
- `src/data_quality/manifests.py` - Dataset manifest dataclass, safe source hashing, manifest creation/loading/writing helpers.
- `scripts/validate_prompt_dataset.py` - Thin CLI for prompt report and manifest generation.
- `tests/test_prompt_dataset_quality.py` - TDD coverage for prompt reports, manifest provenance, source hashing, CLI behavior, and docs.
- `docs/dataset_quality.md` - User-facing validation/manifest command guide and generated-artifact safety notes.
- `src/data_quality/__init__.py` - Public exports for prompt quality and dataset manifest helpers.

## Decisions Made

- Used aggregate warnings/errors rather than fail-fast parsing so users can fix all visible dataset quality issues before expensive generation/training.
- Kept full prompt text out of reports; only small duplicate `target_text` examples are included for duplicate diagnosis.
- Made warnings non-blocking unless `--strict-warnings` is set, matching research workflow needs where distribution warnings may be inspected without blocking all runs.
- Added script-path import bootstrapping to the CLI so both `uv run python scripts/validate_prompt_dataset.py ...` and module imports used by tests work from the repo root.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed script-path imports and Ruff formatting before final verification**
- **Found during:** Task 3 verification
- **Issue:** Running the CLI by file path could not import `src`, and Ruff reported import ordering and line-length issues in new files.
- **Fix:** Added repo-root insertion for script execution, annotated the intentional post-bootstrap imports, organized exports, and wrapped long lines in helpers/tests.
- **Files modified:** `scripts/validate_prompt_dataset.py`, `src/data_quality/__init__.py`, `src/data_quality/manifests.py`, `src/data_quality/prompt_validation.py`, `tests/test_prompt_dataset_quality.py`
- **Commit:** `3fea0be`

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan.

## Threat Flags

None. The plan threat model covered prompt JSONL validation, report/manifest filesystem writes, and source-hashing denial-of-service boundaries; mitigations were implemented through line-by-line parsing, aggregate reports without full prompt text, and safe hashing defaults for small text inputs only.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_prompt_dataset_quality.py -q` — passed, 11 tests.
- `PATH="/root/.local/bin:$PATH" uv run python scripts/validate_prompt_dataset.py --input <tmp>/prompts.jsonl --report <tmp>/report.json --manifest <tmp>/manifest.json --required-rare-characters ё --min-rare-character-coverage 1.0` — passed; generated report/manifest under `/tmp/opencode` only.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/data_quality scripts/validate_prompt_dataset.py tests/test_prompt_dataset_quality.py` — passed.
- `PATH="/root/.local/bin:$PATH" uv run pytest -q` — passed, 92 CPU-safe tests.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirements updates were therefore prepared manually.
- The worktree had unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## Next Plan Readiness

- Plan 03-03 can reuse `DatasetManifest` for synthetic quality manifests and the source-hashing policy for generated synthetic artifacts.
- Plan 03-04 can reuse the manifest helper for selected sample and preference pair artifacts.
- Plan 03-06 can link `docs/dataset_quality.md` and add command aliases once the remaining Phase 3 implementation contracts exist.

## Self-Check: PASSED

- Found all created/modified task files: `src/data_quality/prompt_validation.py`, `src/data_quality/manifests.py`, `scripts/validate_prompt_dataset.py`, `tests/test_prompt_dataset_quality.py`, `docs/dataset_quality.md`, `src/data_quality/__init__.py`, and this summary.
- Found task commits `8b9f73f`, `63345f8`, `978ce44`, `8e40897`, `f15292d`, `bb98dd6`, and `3fea0be` in git history.
- Required verification commands passed, and generated report/manifest smoke outputs stayed under `/tmp/opencode` rather than git.

---
*Phase: 03-data-curriculum-and-dataset-quality*
*Completed: 2026-05-04T15:29:27Z*
