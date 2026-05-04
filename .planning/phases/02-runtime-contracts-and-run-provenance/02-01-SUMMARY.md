---
phase: 02-runtime-contracts-and-run-provenance
plan: 01
subsystem: runtime-config
tags: [pydantic, config-validation, path-policy, provenance, cpu-safe-tests]

requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: Python 3.11 uv/pytest/Ruff tooling, CPU-safe test discovery, and command-surface constraints
provides:
  - Shared `src.runtime.config_io` helpers for strict SFT, DPO, and masked-SFT JSON config validation
  - Pydantic v2-backed field-level validation with secret-safe runtime errors
  - Path policy checks for committed config paths and immutable config snapshots for later manifests
affects: [phase-2-artifact-contracts, phase-2-run-manifests, phase-2-trainer-wiring, phase-4-config-tests]

tech-stack:
  added: [pydantic]
  patterns: [strict-pydantic-models, dataclass-preserving-loaders, path-policy-validation, sorted-config-snapshots]

key-files:
  created:
    - src/runtime/__init__.py
    - src/runtime/config_io.py
    - tests/test_runtime_config_io.py
  modified:
    - pyproject.toml
    - uv.lock

key-decisions:
  - "Return the existing trainer-facing `SFTConfig`, `DPOConfig`, and `MaskedSFTConfig` dataclasses from shared runtime loaders so trainer wiring can remain behavior-preserving in Plan 02-04."
  - "Keep runtime config helpers import-safe and CPU-safe by validating JSON/path strings only and avoiding trainer, CUDA, model, OCR, or artifact loading."
  - "Use secret-safe `RuntimeConfigError` messages that identify the config path and failing field without echoing raw user-provided values."

patterns-established:
  - "Stage loaders validate via strict Pydantic models, reject unknown fields, and then convert to existing dataclasses."
  - "Path policy allows committed relative paths under configs/, data/, outputs/, and runs/ while rejecting traversal, home paths, and off-repo absolutes."
  - "Config snapshots are sorted JSON-serializable dictionaries with `schema_version` and `stage` metadata."

requirements-completed: [CFG-01, CFG-04, STR-02]

duration: 4min
completed: 2026-05-04
---

# Phase 2 Plan 01: Shared Runtime Config Loading Summary

**Pydantic-backed SFT, DPO, and masked-SFT config validation with dataclass-preserving loaders, path policy checks, and immutable snapshots.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-04T13:59:23Z
- **Completed:** 2026-05-04T14:03:49Z
- **Tasks:** 2
- **Files modified:** 5 task files plus this summary

## Accomplishments

- Added RED-first CPU-safe tests covering valid SFT/DPO/masked-SFT config loading, invalid values, missing/unknown fields, malformed JSON, path-policy failures, and snapshot immutability.
- Added `src.runtime.config_io` with strict Pydantic v2 models that validate existing JSON config families before trainer startup while returning existing training dataclasses.
- Added shared path-policy and snapshot helpers for later runtime manifests without wiring trainers or launching CUDA/model/OCR work.
- Added `pydantic>=2,<3` to the project dependency manifest and refreshed `uv.lock`.

## RED/GREEN Evidence

- **RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py -x` failed during collection with `ModuleNotFoundError: No module named 'src.runtime'` after adding tests only.
- **GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py` passed with 23 tests after implementation.
- **Full suite:** `PATH="/root/.local/bin:$PATH" uv run pytest` passed with 34 CPU-safe tests.
- **Lint:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime tests/test_runtime_config_io.py` passed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Specify config validation behavior before implementation** - `ba1e84f` (test)
2. **Task 2: Implement shared validated config loading** - `a7b1bd0` (feat)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_runtime_config_io.py` - CPU-safe tests for stage config loading, field-level runtime errors, path policy, and immutable snapshots.
- `src/runtime/__init__.py` - Public exports for runtime config helpers.
- `src/runtime/config_io.py` - Pydantic-backed loader, validation, dataclass conversion, path-policy, and snapshot implementation.
- `pyproject.toml` - Adds `pydantic>=2,<3` as a runtime dependency.
- `uv.lock` - Refreshes the dependency lock with Pydantic and transitive dependencies.

## Decisions Made

- Preserved existing trainer-facing dataclass return types rather than changing trainers or introducing new config objects at trainer boundaries.
- Required core stage fields such as data paths, output paths, training steps, seed, model ID, LoRA, and mixed precision, while preserving existing dataclass defaults for optional sampling/resume fields.
- Validated path strings without checking filesystem existence because artifact existence/preflight belongs to later Phase 2 plans.
- Treated the current FLUX.2 Klein base model ID as the supported per-stage model contract for this initial loader.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used web documentation fallback after Context7 CLI resolution failed**
- **Found during:** Task 2 (Implement shared validated config loading)
- **Issue:** The required documentation lookup fallback command `npx --yes ctx7@latest ...` failed due a local npm `ENOENT` error under `/root/.cursor-server/...`.
- **Fix:** Used Pydantic's official documentation via web fetch for v2 validators and `ConfigDict` behavior before implementing the Pydantic models.
- **Files modified:** None
- **Verification:** Runtime config tests and Ruff checks passed after implementation.
- **Committed in:** Not applicable; environment/documentation lookup issue only.

**2. [Rule 3 - Blocking] Fixed Ruff violations in new runtime config files**
- **Found during:** Task 2 verification
- **Issue:** The lint command found an overlong line in `src/runtime/config_io.py` and import-order/line-length issues in `tests/test_runtime_config_io.py`.
- **Fix:** Wrapped the long call/parameterization and applied Ruff's import ordering to the new test file.
- **Files modified:** `src/runtime/config_io.py`, `tests/test_runtime_config_io.py`
- **Verification:** `uv run --extra lint ruff check src/runtime tests/test_runtime_config_io.py` passed.
- **Committed in:** `a7b1bd0`

---

**Total deviations:** 2 auto-fixed/blocking (one environment-only documentation fallback, one lint cleanup).
**Impact on plan:** Both supported the required implementation and verification without changing scope or trainer behavior.

## Issues Encountered

- `gsd-sdk` was not available under local `node_modules` or `PATH`, so state/roadmap/requirements updates for this plan were applied manually instead of via SDK query handlers.
- The worktree had substantial unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from task commits.
- The Context7 CLI fallback failed because npm could not access a local Cursor server path; official Pydantic docs were fetched directly instead.

## Auth Gates

None.

## Known Stubs

None. Optional `None` defaults in runtime config models mirror existing trainer dataclass fields and do not block the plan goal.

## Threat Flags

None. The new local JSON config and filesystem path trust boundaries were covered by this plan's threat model and mitigated through strict validation, safe error formatting, and path policy checks.

## User Setup Required

None - no external service configuration required.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py -x` — RED failed as expected before implementation with missing `src.runtime`.
- `PATH="/root/.local/bin:$PATH" uv lock --check` — passed.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py` — passed, 23 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest` — passed, 34 CPU-safe tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime tests/test_runtime_config_io.py` — passed.

## Next Phase Readiness

- Plan 02-02 can build artifact contracts and preflight validators on top of the shared path policy.
- Plan 02-03 can use `resolve_config_snapshot` as the resolved-config payload for local run manifests.
- Plan 02-04 can wire trainer `load_config` functions to `load_stage_config` without changing trainer-facing dataclass types.

## Self-Check: PASSED

- Found `src/runtime/__init__.py`, `src/runtime/config_io.py`, `tests/test_runtime_config_io.py`, `pyproject.toml`, `uv.lock`, and this summary file.
- Found task commits `ba1e84f` and `a7b1bd0` in git history.
- Required verification commands passed.

---
*Phase: 02-runtime-contracts-and-run-provenance*
*Completed: 2026-05-04*
