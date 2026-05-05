---
phase: 05-training-objective-and-pipeline-comparability
plan: 03
subsystem: training-comparability
tags: [training, comparability, manifests, configs, cli, cpu-safe, tdd]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Runtime config loaders and run manifest loading contracts
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Phase 5 run-manifest diff tooling from Plan 05-02
provides:
  - CPU-safe controlled-field comparison for training configs and manifests
  - Blocking-aware CLI for comparability reports
  - User guide for training/inference mismatch checks before run comparison
affects: [phase-05, phase-06-evaluation-validity, thesis-comparison-workflows]

tech-stack:
  added: []
  patterns: [pure-dict-comparison, import-safe-cli, deterministic-markdown-reporting]

key-files:
  created:
    - src/training/comparability.py
    - scripts/check_training_comparability.py
    - tests/test_training_comparability.py
    - docs/training_comparability.md
  modified: []

key-decisions:
  - "Treat model, prompt, seed, inference, data-source, and reward differences as blocking comparability mismatches."
  - "Treat training step count, metric availability, and sample artifact path differences as warnings so reports surface them without blocking every exploratory comparison."
  - "Keep comparability checks CPU-safe by comparing dictionaries, dataclasses, config snapshots, and manifest metadata only."

patterns-established:
  - "Controlled-field reports use schema_version training-comparability/v1 and deterministic blocking/warning lists."
  - "CLI exits 1 only for blocking mismatches unless --allow-blocking is passed."

requirements-completed: [TRN-05, TRN-06]

duration: 4min
completed: 2026-05-05
---

# Phase 5 Plan 03: Training Comparability Mismatch Detector Summary

**CPU-safe training comparability reports for configs and manifests with blocking inference/prompt/data/reward mismatch detection and documented CLI workflows.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T19:10:56Z
- **Completed:** 2026-05-05T19:14:43Z
- **Tasks:** 3
- **Files modified:** 4 plan files plus this summary/state metadata

## Accomplishments

- Added `src.training.comparability` with pure comparison functions for dictionaries, dataclasses, config snapshots, and loaded manifests.
- Added deterministic Markdown/JSON reports with exact labels, controlled field groups, blocking mismatches, warnings, and explicit `missing_left` / `missing_right` reasons.
- Added `scripts/check_training_comparability.py` for CPU-safe config or manifest comparison with `--allow-blocking`, `--markdown`, and `--output` support.
- Published `docs/training_comparability.md` explaining controlled fields, training/inference mismatches, config/manifest commands, and generated artifact safety.

## Task Commits

Each task was committed atomically:

1. **Task 1: Specify mismatch and controlled-field checks** - `1c94b90` (test)
2. **Task 2: Implement pure comparability report logic** - `eda05f0` (feat)
3. **Task 3: Add comparability CLI and guide** - `b4c9a41` (feat)

**Plan metadata:** pending final documentation commit

## Files Created/Modified

- `tests/test_training_comparability.py` - TDD tests for report shape, missing fields, Markdown rendering, and CLI exit behavior.
- `src/training/comparability.py` - Import-safe controlled-field comparison and Markdown formatting logic.
- `scripts/check_training_comparability.py` - Argparse CLI that delegates config loading/manifest comparison to pure comparability logic.
- `docs/training_comparability.md` - User guide for controlled comparison fields and CPU-safe commands.

## Decisions Made

- Model ID, prompt/seed, inference settings, data-source paths, and reward/scorer differences are blocking because they undermine controlled comparison claims.
- Training step count, metric columns, and sample artifact path differences are warnings so the report identifies interpretation risks without making every exploratory run non-comparable.
- Manifest comparison reuses `load_run_manifest` and compares metadata/config snapshots only, preserving the CPU-safe boundary and avoiding generated image/tensor/checkpoint reads.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected the initial RED test fixture for model mismatch**
- **Found during:** Task 2 (Implement pure comparability report logic)
- **Issue:** The test asserted a `model_id` mismatch but initially did not override the right-side `model_id`, so the assertion did not match the intended behavior.
- **Fix:** Added the right-side model ID override in `tests/test_training_comparability.py`.
- **Files modified:** `tests/test_training_comparability.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparability.py -q`
- **Committed in:** `eda05f0`

**2. [Rule 3 - Blocking] Fixed Ruff violations before Task 3 acceptance**
- **Found during:** Task 3 (Add comparability CLI and guide)
- **Issue:** Ruff reported line-length violations and a Python 3.11 typing import modernization warning.
- **Fix:** Wrapped long lines and imported `Mapping` from `collections.abc`.
- **Files modified:** `src/training/comparability.py`, `scripts/check_training_comparability.py`, `tests/test_training_comparability.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/comparability.py scripts/check_training_comparability.py tests/test_training_comparability.py`
- **Committed in:** `b4c9a41`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking verification issue)
**Impact on plan:** Both fixes were directly required for intended tests and acceptance criteria; no scope creep.

## Issues Encountered

- The GSD SDK CLI was not installed in `node_modules` or on `PATH`, so state/roadmap/requirements updates were applied manually while preserving the required metadata changes.
- Pre-existing dirty and untracked files were present before execution; they were left untouched and excluded from all commits.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None found in plan-created or plan-modified files. The grep match for `print(output, end="")` is a normal CLI output control argument, not a data/rendering stub.

## Threat Flags

None - the new CLI reads local config/manifest JSON and writes optional reports as described in the plan threat model; no network endpoints, auth paths, or generated artifact readers were introduced.

## Verification

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparability.py -q` — **passed** (`7 passed`)
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/comparability.py scripts/check_training_comparability.py tests/test_training_comparability.py` — **passed**

## TDD Gate Compliance

- RED gate: `1c94b90` added failing comparability tests; targeted pytest failed with `ModuleNotFoundError: No module named 'src.training.comparability'` before implementation.
- GREEN gate: `eda05f0` added the pure implementation and targeted tests passed.
- Task 3 extended tests/docs/CLI and passed targeted pytest plus Ruff in `b4c9a41`.

## Self-Check: PASSED

- Found created files: `src/training/comparability.py`, `scripts/check_training_comparability.py`, `tests/test_training_comparability.py`, `docs/training_comparability.md`, and this summary.
- Found task commits in git history: `1c94b90`, `eda05f0`, and `b4c9a41`.

## Next Phase Readiness

- Plan 05-04 can use the comparability schema and CLI when wiring explicit SFT, DPO, and masked-SFT config choices into snapshots.
- Plan 05-06 can publish integrated command/docs surfaces that compose run-manifest diff and comparability reports.

---
*Phase: 05-training-objective-and-pipeline-comparability*
*Completed: 2026-05-05*
