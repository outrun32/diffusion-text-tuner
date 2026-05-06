---
phase: 07-moderate-structure-and-extension-cleanup
plan: 03
subsystem: scoring-pipeline-seam
tags: [scoring, rewards, cpu-safe-tests, import-safety, cli-wrapper, product-score]

requires:
  - phase: 06-reward-and-evaluation-validity
    provides: Canonical RewardResult/ProductScoreFormula, score CSV fields, and score sidecar contracts
  - phase: 07-moderate-structure-and-extension-cleanup
    provides: Structure and extension rules plus importable generation seam pattern
provides:
  - Importable `src.scoring.pipeline` seam for task collection, canonical row conversion, score sidecars, and scoring orchestration
  - Thin `scripts.score_images` CLI wrapper preserving `python -m scripts.score_images` behavior and compatibility re-exports
  - CPU-safe scoring seam tests covering metadata loading, import safety, canonical outputs, and CLI delegation
affects: [phase-7-structure-cleanup, scoring-outputs, reward-variants, extension-points]

tech-stack:
  added: []
  patterns: [thin-cli-wrapper, importable-pipeline-seam, lazy-heavy-scorer-construction, canonical-score-compatibility]

key-files:
  created:
    - src/scoring/__init__.py
    - src/scoring/pipeline.py
    - tests/test_scoring_pipeline_contracts.py
  modified:
    - scripts/score_images.py
    - tests/test_reward_wrapper_contracts.py

key-decisions:
  - "Keep canonical score helpers re-exported from scripts.score_images so existing imports remain compatible while new variants import src.scoring.pipeline."
  - "Keep Qwen/PaddleOCR/torchvision/PIL scorer construction inside run_scoring branches rather than at module or script import time."
  - "Treat the stale reward-wrapper import-safety assertion as a contract update because reward imports moved from the script wrapper into the pipeline seam."

patterns-established:
  - "ScoringConfig and ScoringTask are frozen dataclasses that make scoring orchestration reusable without CLI parsing."
  - "collect_scoring_tasks loads tiny text-embedding metadata with torch.load(..., map_location='cpu', weights_only=True) and preserves skip-with-warning behavior."
  - "scripts.score_images should parse args, build ScoringConfig, call run_scoring, and re-export compatibility helpers only."

requirements-completed: [STR-05, STR-06]

metrics:
  duration: 4min
  completed: 2026-05-06T15:54:00Z
  tasks: 3
  files: 6
---

# Phase 7 Plan 03: Importable Reward Scoring Pipeline Seam Summary

**Reward scoring now runs through an import-safe `src.scoring.pipeline` seam with canonical Phase 6 score outputs preserved behind the existing `python -m scripts.score_images` command.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-06T15:50:03Z
- **Completed:** 2026-05-06T15:54:00Z
- **Tasks:** 3
- **Files modified:** 6 including this summary

## Accomplishments

- Added `src/scoring/pipeline.py` exporting `CANONICAL_SCORE_COLUMNS`, `ScoringConfig`, `ScoringTask`, `collect_scoring_tasks`, `build_canonical_score_row`, `write_score_schema_sidecar`, and `run_scoring` for future reward/scoring variants.
- Preserved canonical Phase 6 score CSV fields, product-score computation, missing-evidence fields, schema sidecars, manifest links, resume behavior, shard output naming, and skip-with-warning task discovery.
- Replaced `scripts.score_images` with a thin parser/delegation wrapper that keeps `python -m scripts.score_images` and compatibility helper imports working.
- Added CPU-safe scoring seam tests using temporary directories and tiny `.pt` metadata files; no Qwen, PaddleOCR, CUDA, model weights, or generated images were loaded.
- Updated the reward-wrapper import-safety test to follow the new seam: reward model imports remain lazy, but now inside `src.scoring.pipeline.run_scoring` rather than inside the script wrapper.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_scoring_pipeline_contracts.py -x` failed as intended with `ModuleNotFoundError: No module named 'src.scoring'` after the scoring seam contract tests were written.
- **Task 2/3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_scoring_pipeline_contracts.py tests/test_evaluation_scoring_outputs.py tests/test_reward_wrapper_contracts.py -q` passed with 23 tests after implementing `src.scoring.pipeline`, thinning the script wrapper, and updating the import-safety contract.
- **Task 3 Ruff:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/scoring/pipeline.py scripts/score_images.py tests/test_scoring_pipeline_contracts.py` passed after formatting plan-owned files.

## Task Commits

1. **Task 1: Specify scoring seam contracts** - `dfc317f` (`test`)
2. **Tasks 2-3: Implement importable scoring pipeline module and thin CLI wrapper** - `37b7b54` (`feat`)

**Plan metadata:** recorded in final docs commit.

_Note: Tasks 2 and 3 were committed together because moving orchestration into `src.scoring.pipeline` and thinning `scripts.score_images` were interdependent for the verified green state._

## Files Created/Modified

- `src/scoring/__init__.py` - Re-exports the scoring seam API for `src.scoring` consumers.
- `src/scoring/pipeline.py` - Importable scoring implementation for canonical rows, task collection, sidecars, sharding/resume handling, scorer selection, CSV writing, and summary stats.
- `scripts/score_images.py` - Thin CLI wrapper that parses unchanged scoring arguments, builds `ScoringConfig`, delegates to `run_scoring`, returns a process code, and re-exports compatibility helpers.
- `tests/test_scoring_pipeline_contracts.py` - CPU-safe seam contract tests for canonical score rows, task discovery, skip warnings, sidecars, import safety, and script delegation.
- `tests/test_reward_wrapper_contracts.py` - Updated import-safety assertion to verify scorer imports moved to the pipeline execution branch.
- `.planning/phases/07-moderate-structure-and-extension-cleanup/07-03-SUMMARY.md` - This execution summary.

## Decisions Made

- Re-exported canonical score helpers from `scripts.score_images` so existing Phase 6 tests and any downstream imports remain compatible while new code can import `src.scoring.pipeline` directly.
- Kept top-level scoring pipeline imports limited to CPU-safe dependencies plus `torch` for metadata loading; Qwen, PaddleOCR, PIL image opening, and torchvision remain inside explicit scoring branches in `run_scoring`.
- Updated the reward wrapper import-safety test rather than preserving stale script-source expectations, because the plan intentionally moves reward imports out of `scripts.score_images` and into the importable pipeline seam.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale reward-wrapper import-safety assertion**
- **Found during:** Task 2 verification
- **Issue:** `tests/test_reward_wrapper_contracts.py` still expected scorer imports to appear inside `scripts.score_images.main`, but the plan-required thin wrapper moves those imports into `src.scoring.pipeline.run_scoring`.
- **Fix:** Updated the assertion to verify no reward imports exist in the script wrapper and that `run_scoring` keeps reward class imports inside scorer-selection branches.
- **Files modified:** `tests/test_reward_wrapper_contracts.py`
- **Verification:** Targeted scoring seam, Phase 6 scoring output, and reward wrapper tests passed afterward.
- **Committed in:** `37b7b54`

**2. [Rule 3 - Blocking] Fixed plan-owned Ruff issues**
- **Found during:** Task 3 verification
- **Issue:** Ruff flagged import ordering, line length, and an unused test import in plan-owned scoring files.
- **Fix:** Applied Ruff-compatible import ordering and wrapping without changing behavior.
- **Files modified:** `src/scoring/pipeline.py`, `scripts/score_images.py`, `tests/test_scoring_pipeline_contracts.py`
- **Verification:** Targeted pytest and Ruff commands passed afterward.
- **Committed in:** `37b7b54`

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 3 blocking verification fix)
**Impact on plan:** Both fixes were required by the planned seam move and verification targets. No generated artifacts, model/OCR/CUDA execution, or unrelated user worktree changes were introduced.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found only benign argparse/default string values and file-open arguments in plan-modified files; no TODO/FIXME placeholders, UI stubs, mock-only data flows, or unwired placeholders were introduced.

## Threat Flags

None. The plan threat model covered local `.pt` metadata loading, generated image paths flowing into explicit scorer execution, and scoring results written to CSV/sidecars. No network endpoints, auth paths, new trust-boundary schema changes, or unplanned file access surfaces were introduced.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_scoring_pipeline_contracts.py -x` — RED failed as intended on missing `src.scoring` before implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_scoring_pipeline_contracts.py tests/test_evaluation_scoring_outputs.py tests/test_reward_wrapper_contracts.py -q` — passed with 23 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/scoring/pipeline.py scripts/score_images.py tests/test_scoring_pipeline_contracts.py` — passed.

## TDD Gate Compliance

- RED gate commit exists: `dfc317f`.
- GREEN gate commit exists after RED: `37b7b54`.
- No separate refactor-only commit was needed; formatting was included in the GREEN implementation commit.

## Deferred Issues

- Unrelated pre-existing dirty and untracked files remain in training modules, configs, data roots, thesis docs, scripts, synthetic assets, loss helpers, and loss tests. They were present before execution and were left untouched/excluded from all Plan 07-03 commits.
- The GSD SDK CLI was unavailable in this checkout (`gsd-sdk: command not found` and no local `node_modules/@gsd-build/sdk` output), so planning state files were updated directly instead of through SDK query handlers.
- No GPU/model/OCR diagnostics were run, per plan and user constraints.

## Next Phase Readiness

- Future reward/scoring variants can import task collection, canonical row conversion, sidecar writing, and full scoring orchestration without editing `scripts.score_images`.
- Phase 7 Plan 04 can follow the same thin-script/importable-module pattern for synthetic dataset building while keeping heavy SynthTIGER/model work out of import-time paths.

## Self-Check: PASSED

- Found created/modified files: `src/scoring/__init__.py`, `src/scoring/pipeline.py`, `scripts/score_images.py`, `tests/test_scoring_pipeline_contracts.py`, `tests/test_reward_wrapper_contracts.py`, and this summary.
- Found task commits in git history: `dfc317f` and `37b7b54`.
- Required targeted pytest and Ruff verification commands passed.

---
*Phase: 07-moderate-structure-and-extension-cleanup*  
*Completed: 2026-05-06T15:54:00Z*
