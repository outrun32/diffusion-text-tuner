---
phase: 01-execution-surface-and-pipeline-inventory
plan: 03
subsystem: testing
tags: [pytest, smoke-checks, cli, environment, reproducibility]

requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: Python 3.11 uv/pytest tooling and CPU-safe test discovery from Plan 01-02
provides:
  - Import-safe smoke-check CLI for local module discovery, CUDA, cache paths, Hugging Face package/access surface, and OCR availability
  - CPU-safe pytest coverage for smoke CLI listing, unknown-check handling, and heavy-import side effects
affects: [command-catalog, diagnostics, environment, phase-1-plan-04]

tech-stack:
  added: []
  patterns: [argparse CLI, importlib.util.find_spec package discovery, explicit opt-in diagnostics]

key-files:
  created:
    - scripts/smoke_environment.py
    - tests/test_smoke_environment.py
  modified: []

key-decisions:
  - "Keep smoke CLI module import-safe by using standard-library discovery and deferring torch import to the explicit CUDA check only."
  - "Report Hugging Face/cache credential presence without printing token or secret values."

patterns-established:
  - "Smoke checks default to listing/import-safe behavior; GPU, model, and OCR checks require explicit --check selection."
  - "Tests assert no torch, diffusers, transformers, paddleocr, vllm, or mlx_lm import side effects from the smoke module."

requirements-completed: [ENV-04, TEST-06, TEST-07]

duration: 3min
completed: 2026-05-04
---

# Phase 1 Plan 03: Import-Safe Smoke Environment Checks Summary

**Argparse smoke CLI with CPU-safe tests for listing imports, cache/model/OCR readiness, and heavy-import isolation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-04T13:30:26Z
- **Completed:** 2026-05-04T13:32:57Z
- **Tasks:** 2
- **Files modified:** 2 implementation/test files plus planning metadata

## Accomplishments

- Added `scripts/smoke_environment.py` with explicit checks for `imports`, `cuda`, `cache`, `model-access`, and `ocr`.
- Added CPU-safe pytest coverage proving check listing, `--list` output, unknown-check handling, and no heavy package import side effects.
- Verified default pytest remains CPU-safe and that model/OCR/cache checks can be invoked explicitly without model downloads or OCR construction.

## RED/GREEN Evidence

- **RED:** `uv run pytest tests/test_smoke_environment.py` failed with `ModuleNotFoundError: No module named 'scripts.smoke_environment'` after adding tests only.
- **GREEN:** `uv run pytest tests/test_smoke_environment.py && uv run python -m scripts.smoke_environment --list && uv run python -m scripts.smoke_environment --check imports` passed after implementation.
- **Full suite:** `uv run pytest` passed with 11 CPU-safe tests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write CPU-safe smoke CLI tests first** - `d7dd8d8` (`test`)
2. **Task 2: Implement explicit smoke-check CLI** - `502b8d4` (`feat`)

## Files Created/Modified

- `tests/test_smoke_environment.py` - CPU-safe smoke CLI tests using `capsys` and `sys.modules` assertions.
- `scripts/smoke_environment.py` - Import-safe smoke-check CLI with explicit diagnostics and secret-safe environment reporting.

## Decisions Made

- Kept `torch` import local to `_check_cuda`; all other smoke checks use `importlib.util.find_spec` rather than importing heavy packages or calling model-loading APIs.
- Reported Hugging Face credential/cache status as presence/path status only, avoiding token or secret value output.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used the installed uv binary from `/root/.local/bin` for verification**
- **Found during:** Task 1 (RED verification)
- **Issue:** The `uv` package was installed but `/root/.local/bin` was not on the shell `PATH`, so the required `uv run ...` command initially failed before exercising the tests.
- **Fix:** Exported `PATH="/root/.local/bin:$PATH"` for verification commands; no repository files were changed.
- **Files modified:** None
- **Verification:** Required RED and GREEN `uv run ...` commands executed successfully after exporting PATH.
- **Committed in:** Not applicable; environment-only fix.

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Verification could run as specified; no scope or code behavior changed.

## Issues Encountered

- `uv run ruff check scripts/smoke_environment.py tests/test_smoke_environment.py` could not run because `ruff` is not installed in the active uv environment. This was outside the plan's required verification and no lint changes were needed.

## Verification Results

- `uv run pytest tests/test_smoke_environment.py` — passed, 4 tests.
- `uv run python -m scripts.smoke_environment --list` — passed and printed all checks.
- `uv run python -m scripts.smoke_environment --check imports` — passed and discovered required lightweight/local modules.
- `uv run python -m scripts.smoke_environment --check model-access --allow-missing` — passed without model downloads.
- `uv run python -m scripts.smoke_environment --check ocr --allow-missing` — passed without constructing PaddleOCR; reported missing package as allowed.
- `uv run python -m scripts.smoke_environment --check cache` — passed and reported cache/runtime path status.
- `uv run pytest` — passed, 11 tests.

## Known Stubs

None - no placeholder or mock-data stubs were introduced.

## Threat Flags

None - the new CLI surfaces were covered by the plan threat model, and mitigations T-01-05/T-01-06 were implemented.

## User Setup Required

None - no external service configuration required. Optional Hugging Face credentials, CUDA, and OCR packages remain user/environment-specific smoke-check inputs.

## Next Phase Readiness

- Phase 1 Plan 04 can document `python -m scripts.smoke_environment --list`, `--check imports`, and opt-in diagnostic checks in the command catalog.
- Default automated tests remain isolated from CUDA/model/OCR diagnostics.

## Self-Check: PASSED

- Found `scripts/smoke_environment.py`.
- Found `tests/test_smoke_environment.py`.
- Found task commit `d7dd8d8`.
- Found task commit `502b8d4`.

---
*Phase: 01-execution-surface-and-pipeline-inventory*
*Completed: 2026-05-04*
