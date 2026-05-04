---
phase: 02-runtime-contracts-and-run-provenance
plan: 04
subsystem: runtime-preflight
tags: [trainer-startup, config-validation, preflight-cli, artifacts, manifests, cpu-safe-tests]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Shared config validation, canonical paths, artifact validators, and run manifest helpers from plans 02-01 through 02-03.
provides:
  - Trainer `load_config` wiring through `src.runtime.config_io.load_stage_config` for SFT, DPO, and masked-SFT entry points.
  - CPU-safe `scripts/preflight_runtime.py` CLI for config, artifact, and manifest readiness checks before expensive stage execution.
  - JSON readiness reports with `stage`, `config`, `artifacts`, `manifest`, `blocking_errors`, and `warnings` fields.
affects: [phase-2-command-docs, phase-2-plan-05, trainer-startup, runtime-preflight]

tech-stack:
  added: []
  patterns: [validated-trainer-loaders, argparse-preflight-cli, aggregate-readiness-reports, cpu-safe-runtime-checks]

key-files:
  created:
    - scripts/preflight_runtime.py
    - tests/test_runtime_preflight.py
    - src/training/masked_sft_trainer.py
  modified:
    - src/training/sft_trainer.py
    - src/training/dpo_trainer.py

key-decisions:
  - "Keep trainer `main()` behavior unchanged after config loading; only replace per-trainer JSON/dataclass parsing with the shared runtime validation path."
  - "Keep preflight CLI CPU-safe by importing only runtime helpers and avoiding CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, and OCR/model construction."
  - "Emit aggregate readiness reports so users can fix config, artifact, and manifest blockers before launching long GPU jobs."

patterns-established:
  - "Trainer loader tests monkeypatch `src.runtime.config_io.load_stage_config`, requiring trainer code to delegate through the shared module path."
  - "Preflight supports Phase 2 stage aliases (`generate`, `score`, `sft`, `dpo`, `masked-sft`, `synthetic`, `evaluation`) while reporting the user-facing stage in JSON."
  - "Manifest inspection reports resume readiness without printing secret values or launching pipeline stages."

requirements-completed: [CFG-01, ART-01, RUN-04, STR-02]

duration: 8min
completed: 2026-05-04T14:39:11Z
---

# Phase 2 Plan 04: Trainer Runtime Preflight Summary

**Shared trainer config validation with a CPU-safe preflight CLI for configs, artifacts, manifests, and resume readiness.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-04T14:31:30Z
- **Completed:** 2026-05-04T14:39:11Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added RED-first tests for trainer `load_config` delegation, invalid-config behavior before heavy runtime imports, and preflight JSON reports for generation, scoring, SFT, DPO, masked-SFT, synthetic, and evaluation stages.
- Replaced direct trainer JSON/dataclass parsing in `sft_trainer.load_config`, `dpo_trainer.load_config`, and `masked_sft_trainer.load_config` with shared `load_stage_config` calls.
- Added `scripts/preflight_runtime.py`, an import-safe argparse CLI that validates optional config files, canonical or explicit artifact paths, and optional run manifests.
- Verified the new CLI reports blocking errors and warnings in machine-readable JSON without launching trainers, CUDA/model loading, OCR, or downloads.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py -x` failed as expected after adding tests only because `sft_trainer.load_config` still tried to open `configs/example.json` directly instead of delegating to `load_stage_config`.
- **Task 2 GREEN subset:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py` reached 4 passing trainer-loader tests after delegation, with remaining failures only for the missing preflight CLI.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py` passed with 13 tests after implementing `scripts/preflight_runtime.py`.
- **Full suite:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py && PATH="/root/.local/bin:$PATH" uv run pytest` passed with 62 CPU-safe tests.

## Task Commits

Each planned task was committed atomically where possible:

1. **Task 1: Specify trainer loader wiring and preflight behavior** - `7840313` (test)
2. **Task 2: Delegate trainer config loaders to shared validation** - `a56208c` (feat)
3. **Task 3: Implement CPU-safe runtime preflight CLI** - `60a3d50` (feat)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_runtime_preflight.py` - CPU-safe tests for trainer delegation, heavy-import isolation around invalid config handling, and preflight JSON behavior.
- `src/training/sft_trainer.py` - `load_config` now delegates to `config_io.load_stage_config("sft", path)`.
- `src/training/dpo_trainer.py` - `load_config` now delegates to `config_io.load_stage_config("dpo", path)`.
- `src/training/masked_sft_trainer.py` - Added/tracked masked-SFT trainer entry point with `load_config` delegating to `config_io.load_stage_config("masked_sft", path)`.
- `scripts/preflight_runtime.py` - CPU-safe CLI for config, artifact, and manifest readiness reports.

## Decisions Made

- Used module-level `from src.runtime import config_io` imports in trainer files so tests and future integrations can monkeypatch the shared validation path directly.
- Kept `--config` optional in preflight; configs are validated when provided, while artifact checks can still run against canonical defaults or explicit path flags.
- Kept preflight exit status simple: `0` when no blocking errors are present, `1` when config, artifact, or manifest blockers are reported.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Isolated trainer-loader tests from pre-existing reward import side effects**
- **Found during:** Task 1 RED verification
- **Issue:** Importing the current dirty `sft_trainer` pulled in `refl_trainer` and reward diagnostics, which attempted to import unavailable Paddle-related modules before exercising `load_config`.
- **Fix:** Scoped the test to loader wiring by monkeypatching a minimal `src.training.refl_trainer` module in `sys.modules`; production trainer behavior was not changed.
- **Files modified:** `tests/test_runtime_preflight.py`
- **Commit:** `7840313`

**2. [Rule 1 - Bug] Added CLI stage alias handling for `generate`**
- **Found during:** Task 3 GREEN verification
- **Issue:** `resolve_stage_paths` accepts canonical `generated`/`generation` aliases, while the plan-required CLI stage is `generate`.
- **Fix:** Added a preflight-only helper-stage alias map so the JSON report preserves `stage: generate` while validators receive `generated`.
- **Files modified:** `scripts/preflight_runtime.py`
- **Commit:** `60a3d50`

---

**Total deviations:** 2 auto-fixed (one test isolation blocker, one CLI alias bug).
**Impact on plan:** Both supported the required CPU-safe validation behavior without launching heavy runtime work.

## Issues Encountered

- `gsd-sdk` was not available under local `node_modules` or `PATH`, so state, roadmap, and requirement updates were applied manually instead of through SDK query handlers.
- The worktree had substantial unrelated pre-existing dirty and untracked files before execution; they were left untouched except for plan-targeted trainer/preflight/test files.
- `src/training/masked_sft_trainer.py` was already present as an untracked plan-target file; it was committed as part of Task 2 because the plan required masked-SFT loader wiring.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder text in task-created or task-modified files.

## Threat Flags

None. The plan threat model covered CLI path arguments, trainer config validation before model loading, and secret-safe preflight output; the implementation routes paths through existing runtime helpers and does not print token values.

## User Setup Required

None - no external service credentials, CUDA device, model cache, or OCR package is required for the preflight CLI tests.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py -x` — RED failed as expected before implementation with direct trainer file loading.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py` — passed, 13 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_preflight.py && PATH="/root/.local/bin:$PATH" uv run pytest` — passed, 62 CPU-safe tests.

## Next Phase Readiness

- Plan 02-05 can document `python -m scripts.preflight_runtime --stage ... --json` alongside existing manifest commands and Makefile aliases.
- Later trainer/refactor phases can rely on shared config validation being the trainer startup gate before model loading.

## Self-Check: PASSED

- Found created/modified task files: `tests/test_runtime_preflight.py`, `scripts/preflight_runtime.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, and this summary file.
- Found task commits: `7840313`, `a56208c`, and `60a3d50` in git history.
- Required verification commands passed.

---
*Phase: 02-runtime-contracts-and-run-provenance*
*Completed: 2026-05-04T14:39:11Z*
