---
phase: 05-training-objective-and-pipeline-comparability
plan: 02
subsystem: run-manifest-comparability
tags: [run-manifests, diff, comparability, cli, cpu-safe-tests]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Local run manifests, config snapshots, secret-safe environment metadata, and manifest loader APIs.
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Explicit Phase 5 selection modes from Plan 05-01 for comparison-grade config snapshots.
provides:
  - CPU-safe `src.runtime.manifest_diff.compare_run_manifests` for categorized local run comparisons.
  - Deterministic Markdown rendering for manifest diffs.
  - `python -m scripts.compare_run_manifests` CLI with JSON default, `--markdown`, and `--output` support.
  - CPU-safe pytest coverage for config, data source, reward, seed, inference, metric, artifact, secret/cache-presence, CLI, and malformed-input behavior.
affects: [phase-5-comparability, run-provenance, thesis-evidence-review, runtime-contract-docs]

tech-stack:
  added: []
  patterns: [pure-json-diffing, categorized-manifest-changes, presence-only-sensitive-metadata, direct-main-cli-tests]

key-files:
  created:
    - src/runtime/manifest_diff.py
    - scripts/compare_run_manifests.py
    - tests/test_runtime_manifest_diff.py
  modified:
    - docs/runtime_contracts.md

key-decisions:
  - "Keep run-manifest comparison CPU-safe by reading only local manifest/config JSON dictionaries, metrics, and output metadata without importing torch, diffusers, transformers, OCR, or model stacks."
  - "Classify config changes into data source, reward, seed, inference, and general config sections while treating manifest metrics and outputs as their own comparison sections."
  - "Preserve secret/cache privacy in diff output by comparing only env/cache presence booleans and omitting raw cache path metadata."

requirements-completed: [RUN-02]

duration: 4min
completed: 2026-05-05T19:08:43Z
---

# Phase 5 Plan 02: Run Manifest Diff Surface Summary

**CPU-safe local run-manifest diffs with categorized JSON/Markdown output for comparison-grade training evidence.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T19:05:06Z
- **Completed:** 2026-05-05T19:08:43Z
- **Tasks:** 3
- **Files modified:** 4 task files plus this summary and planning metadata

## Accomplishments

- Added RED-first CPU-safe tests using pytest `tmp_path` manifests for `config_changes`, `data_source_changes`, `reward_changes`, `seed_changes`, `inference_changes`, `metric_changes`, and `artifact_changes`.
- Implemented `src.runtime.manifest_diff` with `DIFF_SCHEMA_VERSION = "run-manifest-diff/v1"`, `compare_run_manifests(...)`, and `format_manifest_diff_markdown(...)`.
- Added `scripts/compare_run_manifests.py` with `--left`, `--right`, `--markdown`, and `--output`; tests call `main(argv)` directly for JSON stdout, Markdown file output, and malformed input errors.
- Documented exact JSON and Markdown comparison commands in `docs/runtime_contracts.md` under a new `Run manifest diff` section.
- Ensured the new code path uses only JSON/path/runtime-manifest helpers and does not import heavy ML/OCR libraries.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifest_diff.py -x` failed during collection with `ModuleNotFoundError: No module named 'src.runtime.manifest_diff'` after adding tests only.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifest_diff.py -q` passed with 4 tests after adding the pure diff module and early CLI backing needed by the committed test file.
- **Task 3 verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifest_diff.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/manifest_diff.py scripts/compare_run_manifests.py tests/test_runtime_manifest_diff.py` passed.
- **Final verification:** The same targeted pytest and Ruff command passed after all task commits.

## Task Commits

Each planned task was committed atomically where possible:

1. **Task 1: Specify manifest diff categories** - `fadf80a` (test)
2. **Task 2: Implement pure manifest diff module** - `5a47a0e` (feat)
3. **Task 3: Add manifest diff CLI and docs** - `1b89c16` (docs)

## Files Created/Modified

- `tests/test_runtime_manifest_diff.py` - CPU-safe fixtures and assertions for categorized diff output, Markdown headings, CLI JSON/Markdown behavior, and malformed-input handling.
- `src/runtime/manifest_diff.py` - Pure manifest diff model, categorization rules, presence-only environment/cache comparison, stable JSON-serializable payloads, and deterministic Markdown formatting.
- `scripts/compare_run_manifests.py` - Argparse CLI for comparing local manifests and writing JSON or Markdown reports.
- `docs/runtime_contracts.md` - Runtime documentation with exact `python -m scripts.compare_run_manifests` examples.

## Decisions Made

- Kept manifest diffing local and CPU-safe; the implementation delegates manifest parsing to `load_run_manifest` and performs only dictionary traversal.
- Added `left_manifest_path` and `right_manifest_path` to diff payloads and Markdown for provenance, alongside `left_run_id` and `right_run_id`.
- Included an `environment_changes` JSON section for presence-only env/cache metadata while keeping the required seven comparison sections as the Markdown evidence surface.
- Treated uncategorized config keys such as `stage`, `score_threshold`, `pair_construction_mode`, and `output_dir` as `config_changes`; manifest `outputs` remain the source for `artifact_changes`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Implemented CLI backing during Task 2 GREEN**
- **Found during:** Task 2
- **Issue:** The Task 1 test file included CLI imports and direct `main(argv)` tests, so module-only implementation would leave the Task 2 verification target red.
- **Fix:** Added the import-safe `scripts/compare_run_manifests.py` backing together with the diff module, then used Task 3 for documentation and final lint cleanup.
- **Files modified:** `scripts/compare_run_manifests.py`, `tests/test_runtime_manifest_diff.py`, `src/runtime/manifest_diff.py`
- **Commit:** `5a47a0e`

**2. [Rule 1 - Bug] Corrected expected general config coverage**
- **Found during:** Task 2 GREEN verification
- **Issue:** The initial RED expectation omitted changed uncategorized config keys (`pair_construction_mode`, `output_dir`) even though the plan requires all other config changes to be reported in `config_changes`.
- **Fix:** Updated the test expectation so uncategorized config differences are explicitly asserted rather than silently ignored.
- **Files modified:** `tests/test_runtime_manifest_diff.py`
- **Commit:** `5a47a0e`

**3. [Rule 3 - Blocking] Fixed Ruff violations in new diff files**
- **Found during:** Task 3 verification
- **Issue:** Ruff flagged long lines, import ordering, and `typing.Mapping` usage in the new module/CLI/test files.
- **Fix:** Wrapped long argparse/Markdown lines, imported `Mapping` from `collections.abc`, and sorted imports.
- **Files modified:** `src/runtime/manifest_diff.py`, `scripts/compare_run_manifests.py`, `tests/test_runtime_manifest_diff.py`
- **Commit:** `1b89c16`

---

**Total deviations:** 3 auto-fixed.
**Impact on plan:** All deviations supported the planned CPU-safe manifest diff behavior and did not change existing manifest creation or trainer behavior.

## Issues Encountered

- `gsd-sdk` was not available under local `node_modules` or `PATH`, so state, roadmap, and requirement updates were applied manually instead of through SDK query handlers.
- The worktree had substantial unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from task and metadata commits.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder/coming-soon/not-available text in the files created or modified by this plan. Empty dictionary defaults are normal diff payload structures, not UI/data-source stubs.

## Threat Flags

None. The local manifest JSON to diff report and CLI args to filesystem read/write trust boundaries were covered by the plan threat model. Mitigations include reusing `load_run_manifest`, returning code 2 for malformed input, adding manifest paths/run IDs for provenance, and rendering secret/cache metadata as presence booleans only.

## User Setup Required

None - no external service credentials, GPU, OCR, model cache, or network access are required.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifest_diff.py -x` — RED failed as expected before implementation with missing `src.runtime.manifest_diff`.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifest_diff.py -q` — passed with 4 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/manifest_diff.py scripts/compare_run_manifests.py tests/test_runtime_manifest_diff.py` — passed.
- `git status --short` — no generated `runs/`, output images/tensors/checkpoints/logs, or runtime comparison reports from this plan were left untracked; unrelated pre-existing dirty/untracked files remain untouched.

## Next Phase Readiness

- Plan 05-03 can consume `compare_run_manifests(...)` output to build controlled comparability/mismatch checks over seeds, inference settings, data sources, rewards, metrics, and artifact paths.
- Plan 05-06 can publish integrated command aliases and docs drift tests around `python -m scripts.compare_run_manifests`.

## Self-Check: PASSED

- Found created/modified files: `src/runtime/manifest_diff.py`, `scripts/compare_run_manifests.py`, `tests/test_runtime_manifest_diff.py`, `docs/runtime_contracts.md`, and this summary.
- Found task commits: `fadf80a`, `5a47a0e`, and `1b89c16`.
- Required verification commands passed, and no generated runtime artifacts were committed.

---
*Phase: 05-training-objective-and-pipeline-comparability*
*Completed: 2026-05-05T19:08:43Z*
