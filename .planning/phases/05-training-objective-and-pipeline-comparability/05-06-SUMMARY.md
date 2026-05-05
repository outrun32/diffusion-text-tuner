---
phase: 05-training-objective-and-pipeline-comparability
plan: 06
subsystem: integrated-training-comparison-command-surface
tags: [training-comparability, run-manifests, cli, docs, makefile, cpu-safe, tdd]

requires:
  - phase: 05-training-objective-and-pipeline-comparability
    provides: CPU-safe run-manifest diff module and CLI from Plan 05-02.
  - phase: 05-training-objective-and-pipeline-comparability
    provides: CPU-safe training comparability mismatch detector and CLI from Plan 05-03.
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Explicit config choice snapshots and shared trainer seams from Plans 05-04 and 05-05.
provides:
  - Integrated `python -m scripts.compare_training_runs` CLI with JSON and Markdown output.
  - `compare-training-runs` Makefile alias using `LEFT_MANIFEST` and `RIGHT_MANIFEST`.
  - Docs drift tests covering CLI behavior, exact command strings, README links, and Makefile dry-run output.
  - Phase 5 command documentation for baseline, SFT, DPO, masked-SFT, combined, and curriculum comparison posture.
affects: [phase-5-comparability, phase-6-evaluation-validity, thesis-run-review]

tech-stack:
  added: []
  patterns: [cli-composition, deterministic-json-markdown, docs-drift-tests, cpu-safe-run-review]

key-files:
  created:
    - scripts/compare_training_runs.py
    - tests/test_training_comparison_docs.py
  modified:
    - docs/training_comparability.md
    - docs/commands.md
    - README.md
    - Makefile

key-decisions:
  - "Keep integrated training-run comparison CPU-safe by composing existing manifest diff and comparability helpers rather than duplicating comparison logic."
  - "Return exit code 1 for blocking comparability mismatches unless `--allow-blocking` is explicitly provided, after still writing the requested report."
  - "Publish Phase 5 comparison commands through docs, README, and Makefile aliases with docs drift tests guarding exact strings."

requirements-completed: [TRN-05, RUN-02, TRN-07, STR-04]

metrics:
  duration: 3min 2s
  completed: 2026-05-05T19:33:07Z
  tasks: 3
  files: 6
---

# Phase 5 Plan 06: Integrated Training Comparison Command Surface Summary

**Integrated CPU-safe training-run comparison reports now combine manifest diffs and controlled comparability mismatches behind one documented CLI and Makefile alias.**

## Performance

- **Duration:** 3 min 2 sec
- **Started:** 2026-05-05T19:30:05Z
- **Completed:** 2026-05-05T19:33:07Z
- **Tasks:** 3
- **Files modified:** 6 task files plus this summary and planning metadata

## Accomplishments

- Added `tests/test_training_comparison_docs.py` with TDD coverage for integrated JSON output, Markdown output, blocking mismatch exit semantics, exact docs/README strings, and `make -n compare-training-runs` dry-run output.
- Added `scripts/compare_training_runs.py`, which emits `schema_version: training-run-comparison/v1` and combines `manifest_diff` from `compare_run_manifests` with `comparability` from `compare_training_manifests`.
- Published the Phase 5 training comparability command surface in `docs/commands.md`, including selection materialization, raw manifest diff, controlled comparability checks, and integrated comparison report commands.
- Added the `compare-training-runs` Makefile alias with `LEFT_MANIFEST`, `RIGHT_MANIFEST`, and `TRAINING_RUN_COMPARISON` variables.
- Updated `README.md` and `docs/training_comparability.md` so baseline, SFT, DPO, masked-SFT, combined, and curriculum comparison workflows point to the integrated CPU-safe command.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparison_docs.py -x` failed during collection with `ImportError: cannot import name 'compare_training_runs' from 'scripts'`, confirming the integrated CLI/docs tests were written before implementation.
- **Task 2 GREEN:** After adding `scripts/compare_training_runs.py`, the integrated CLI behavior tests passed once the same test file's planned docs drift assertions were unblocked by the Task 3 command surface work.
- **Task 3 verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparison_docs.py tests/test_training_comparability.py tests/test_runtime_manifest_diff.py -q && make -n compare-training-runs LEFT_MANIFEST=runs/a/manifest.json RIGHT_MANIFEST=runs/b/manifest.json && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check scripts/compare_training_runs.py tests/test_training_comparison_docs.py` passed.

## Task Commits

1. **Task 1: Specify integrated comparison CLI and docs drift** - `dc2c202` (`test`)
2. **Task 2: Implement integrated training comparison CLI** - `a338a0d` (`feat`)
3. **Task 3: Publish Phase 5 command surface** - `e702b0d` (`docs`)

## Files Created/Modified

- `tests/test_training_comparison_docs.py` - CPU-safe pytest fixtures and docs drift tests for integrated training-run comparison behavior.
- `scripts/compare_training_runs.py` - Argparse CLI combining manifest diff and comparability reports with deterministic JSON/Markdown rendering and blocking-aware exit codes.
- `docs/commands.md` - Adds `Phase 5 training comparability` with exact command strings and CPU-safe comparison posture.
- `docs/training_comparability.md` - Adds the integrated run comparison command, schema shape, Markdown headings, and exit-code behavior.
- `README.md` - Links to `docs/training_comparability.md` and names the Phase 5 comparison commands.
- `Makefile` - Adds `compare-training-runs` alias and manifest/report variables.

## Decisions Made

- Composed the existing Phase 5 helper APIs instead of duplicating field comparison logic, keeping `src.runtime.manifest_diff` and `src.training.comparability` as the sources of truth.
- Kept generated reports under caller-provided paths and documented `runs/comparisons/` as the ignored runtime output target.
- Used direct `main(argv)` pytest coverage for CLI behavior and `make -n` for alias verification so tests remain CPU-safe and do not create real `runs/` artifacts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Completed docs surface before Task 2 verification could be fully green**
- **Found during:** Task 2
- **Issue:** The plan's Task 1 test file intentionally included docs/README/Makefile drift assertions, while Task 2's verification command ran the entire file. The integrated CLI tests passed after implementation, but the same target still failed on the planned Task 3 docs surface.
- **Fix:** Implemented the docs/README/Makefile command surface needed by Task 3 before recording the final green verification, then committed the CLI and docs in separate task commits.
- **Files modified:** `docs/commands.md`, `docs/training_comparability.md`, `README.md`, `Makefile`, `scripts/compare_training_runs.py`, `tests/test_training_comparison_docs.py`
- **Commit:** `a338a0d`, `e702b0d`

---

**Total deviations:** 1 auto-fixed blocking sequencing issue.
**Impact on plan:** The deviation did not change scope; it only handled the plan's shared test target ordering while preserving separate CLI and docs commits.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found only optional `argv: list[str] | None = None`, CLI `print(..., end="")`, and stdout assertions in tests; these are normal CLI/test constructs, not placeholder or unwired data behavior.

## Threat Flags

None. The new local manifest-to-report file access boundary, blocking mismatch exit behavior, and docs/Makefile command surface were covered by the plan threat model. The implementation only reads local manifests through existing loaders and writes optional user-specified reports; it adds no network endpoints, auth paths, or generated artifact readers.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparison_docs.py -x` — **failed as expected for RED** before implementation with missing `scripts.compare_training_runs`.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparison_docs.py -q` — **passed** (`4 passed`).
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparison_docs.py tests/test_training_comparability.py tests/test_runtime_manifest_diff.py -q` — **passed** (`15 passed`).
- `make -n compare-training-runs LEFT_MANIFEST=runs/a/manifest.json RIGHT_MANIFEST=runs/b/manifest.json` — **passed**, printing the integrated CLI command with both manifest variables.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check scripts/compare_training_runs.py tests/test_training_comparison_docs.py` — **passed**.

## Deferred Issues

- Unrelated pre-existing dirty and untracked files remain in `src/training/config.py`, trainer modules, configs, generated data roots, thesis docs, scripts, loss helpers, and loss tests. They were present before execution and were left untouched/excluded from all Plan 05-06 commits.
- `gsd-sdk` did not return usable state-handler output in this checkout, so planning state files were updated directly instead of via SDK query handlers.

## TDD Gate Compliance

- RED gate: `dc2c202` added failing tests for the missing integrated CLI/docs surface.
- GREEN gate: `a338a0d` added the integrated CLI implementation, and `e702b0d` completed the docs command surface required for the shared test target to pass.
- No separate refactor commit was needed.

## Next Phase Readiness

- Phase 5 Plan 05-06 is complete; Phase 5 has 6/6 plans implemented and is ready for phase-level verification/transition before Phase 6 reward and evaluation validity work.

## Self-Check: PASSED

- Found created/modified files: `scripts/compare_training_runs.py`, `tests/test_training_comparison_docs.py`, `docs/training_comparability.md`, `docs/commands.md`, `README.md`, `Makefile`, and this summary.
- Found task commits in git history: `dc2c202`, `a338a0d`, and `e702b0d`.
- Required CPU-safe pytest, Makefile dry-run, and Ruff verification commands passed.

---
*Phase: 05-training-objective-and-pipeline-comparability*  
*Completed: 2026-05-05T19:33:07Z*
