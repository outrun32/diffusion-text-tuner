---
phase: 02-runtime-contracts-and-run-provenance
plan: 02
subsystem: runtime-artifacts
tags: [artifact-contracts, path-resolution, preflight-validation, git-safety, cpu-safe-tests]

requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: CPU-safe uv/pytest tooling, generated-artifact safety boundaries, and documented command surface
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Shared runtime config path policy and source-tree runtime package from Plan 02-01
provides:
  - Canonical runtime path resolution for prompts, generated outputs, scoring, training, synthesis, evaluation, plotting, and run manifests
  - CPU-safe artifact validators for prompt JSONL, scores CSV, generated images/latents/text embeddings, masked-SFT tensors, checkpoints/logs/eval outputs, and manifests
  - Generated-artifact git-safety classification aligned with `.gitignore`
  - Human-readable runtime contract documentation with schema/version metadata and local/SLURM guidance
affects: [phase-2-run-manifests, phase-2-preflight-cli, phase-3-data-contracts, phase-4-fixture-tests, phase-5-training-comparability]

tech-stack:
  added: []
  patterns: [dataclass-runtime-paths, aggregate-artifact-reports, weights-only-tensor-inspection, docs-drift-tests]

key-files:
  created:
    - src/runtime/paths.py
    - src/runtime/artifacts.py
    - docs/runtime_contracts.md
    - tests/test_runtime_artifacts.py
  modified:
    - src/runtime/__init__.py

key-decisions:
  - "Keep artifact validators CPU-safe and model-download-free by inspecting only JSONL, CSV, directory names, file presence, and tiny trusted local tensor dictionaries with `torch.load(..., map_location=\"cpu\", weights_only=True)`."
  - "Return aggregate `ArtifactReport` errors by default so users can fix all visible contract problems before expensive jobs, while allowing `require_ready=True` to raise `ArtifactValidationError` at blocking preflight gates."
  - "Classify generated runtime roots, checkpoints, logs, tensors, and generated images as non-committable by default, with narrow fixture exceptions for `experiments/assets/` and `tests/fixtures/`."

patterns-established:
  - "Runtime path contracts resolve canonical repository-relative roots (`data/`, `outputs/`, `runs/`, `configs/`) without personal absolute paths."
  - "Artifact validators model hidden filesystem/tensor contracts as explicit schema-versioned reports."
  - "Runtime contract docs are protected by a lightweight pytest assertion covering required headings and artifact families."

requirements-completed: [ART-01, ART-02, ART-03, ART-04, RUN-03, STR-02]

duration: 6min
completed: 2026-05-04
---

# Phase 2 Plan 02: Runtime Artifact Contracts Summary

**Canonical runtime paths with CPU-safe artifact validators, schema metadata, and generated-artifact git-safety enforcement.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-04T14:05:39Z
- **Completed:** 2026-05-04T14:12:02Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added RED-first tests for ART-01 through ART-04 covering prompt JSONL line-number errors, scores CSV metadata, generated image/latent/text-embedding layouts, masked-SFT tensor contracts, git-safety behavior, and blocking preflight errors.
- Implemented `src.runtime.paths` with `RuntimePaths`, `resolve_stage_paths`, and `assert_artifact_git_safety` for canonical local/SLURM-compatible runtime roots.
- Implemented `src.runtime.artifacts` with `ArtifactReport`, `ArtifactValidationError`, and `validate_artifacts` for CPU-safe preflight validation across Phase 2 artifact families.
- Published `docs/runtime_contracts.md` with canonical paths, producers/consumers, schema/version metadata, preflight hooks, resume/inspect notes, local/SLURM guidance, and git-safety classifications.

## RED/GREEN Evidence

- **RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_artifacts.py -x` failed during collection with `ModuleNotFoundError: No module named 'src.runtime.artifacts'` after adding tests only.
- **GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_artifacts.py` passed with 8 tests after implementing `src.runtime.paths` and `src.runtime.artifacts`.
- **Docs drift:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_artifacts.py` passed with 9 tests after adding `docs/runtime_contracts.md` and the required-heading/artifact assertion.
- **Full suite:** `PATH="/root/.local/bin:$PATH" uv run pytest` passed with 43 CPU-safe tests.
- **Lint:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/paths.py src/runtime/artifacts.py src/runtime/__init__.py tests/test_runtime_artifacts.py` passed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Specify artifact validators and git-safety behavior** - `03fb07d` (test)
2. **Task 2: Implement path contracts and artifact validation helpers** - `374cf4b` (feat)
3. **Task 3: Document canonical runtime contracts** - `173361a` (docs)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_runtime_artifacts.py` - CPU-safe tests for path resolution, artifact validation, git-safety, blocking preflight behavior, and docs drift.
- `src/runtime/paths.py` - Canonical runtime roots/stage path maps and generated-artifact git-safety classifier.
- `src/runtime/artifacts.py` - Schema-versioned aggregate artifact reports and validators for prompt, score, generated, masked-SFT, training, checkpoint, log, eval, and manifest contracts.
- `src/runtime/__init__.py` - Public exports for artifact and path helpers.
- `docs/runtime_contracts.md` - User-facing runtime contract tables and local/SLURM/git-safety guidance.

## Decisions Made

- Kept path helpers independent of machine-specific configuration; callers can pass `root=...` for local or SLURM workspaces while defaults stay under canonical repository-relative roots.
- Treated `.pt` inspection as a trusted-local-output validation boundary, using PyTorch `weights_only=True` and documenting that downloaded third-party pickle artifacts require separate review.
- Chose aggregate reports over immediate exceptions for normal validation so preflight commands can show all contract failures at once; only `require_ready=True` raises for expensive-stage gates.
- Added a docs drift test rather than relying on manual documentation review for required artifact family coverage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used web documentation fallback after Context7 CLI resolution failed**
- **Found during:** Task 2 (Implement path contracts and artifact validation helpers)
- **Issue:** The required documentation lookup fallback command `npx --yes ctx7@latest ...` failed with a local npm `ENOENT` error under `/root/.cursor-server/...`.
- **Fix:** Fetched official PyTorch `torch.load` documentation directly and confirmed `weights_only=True` and `map_location="cpu"` behavior before implementing tensor validators.
- **Files modified:** None
- **Verification:** Artifact tests, full pytest, and Ruff checks passed after implementation.
- **Committed in:** Not applicable; environment/documentation lookup issue only.

**2. [Rule 1 - Bug] Adjusted blocking preflight error context to include path key names**
- **Found during:** Task 2 GREEN verification
- **Issue:** The `ArtifactValidationError` for SFT preflight identified the missing scores path but did not include the `scores_csv` key expected by the contract test.
- **Fix:** Added `scores_csv:` context to blocking training-input errors before raising.
- **Files modified:** `src/runtime/artifacts.py`
- **Verification:** `uv run pytest tests/test_runtime_artifacts.py` passed.
- **Committed in:** `374cf4b`

**3. [Rule 3 - Blocking] Fixed Ruff violations in new runtime artifact files**
- **Found during:** Task 2 verification
- **Issue:** Ruff reported import ordering, Python 3.11 typing import style, and line-length violations in new runtime files.
- **Fix:** Applied Ruff-compatible imports and wrapped long lines.
- **Files modified:** `src/runtime/artifacts.py`, `src/runtime/paths.py`
- **Verification:** `uv run --extra lint ruff check src/runtime/paths.py src/runtime/artifacts.py src/runtime/__init__.py tests/test_runtime_artifacts.py` passed.
- **Committed in:** `374cf4b`

---

**Total deviations:** 3 auto-fixed (1 documentation lookup blocker, 1 implementation bug, 1 lint blocker).
**Impact on plan:** All fixes supported the planned CPU-safe validation contracts without changing scope or adding generated artifacts.

## Issues Encountered

- `gsd-sdk` was not available under local `node_modules` or `PATH`, so state/roadmap/requirements updates for this plan were applied manually instead of through SDK query handlers.
- The worktree had substantial unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from task commits.
- The Context7 CLI fallback could not run due an environment-local npm path error; official PyTorch docs were fetched directly instead.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no placeholder/TODO/FIXME or intentionally empty UI data flows in files created or modified by this plan.

## Threat Flags

None. The new local artifact-to-validator and artifact-to-git trust boundaries were covered by this plan's threat model and mitigated through `weights_only=True` tensor inspection, aggregate preflight reporting, and generated-artifact git-safety classification.

## User Setup Required

None - no external service configuration required.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_artifacts.py -x` — RED failed as expected before implementation with missing `src.runtime.artifacts`.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_artifacts.py` — passed, 9 tests after docs task.
- `PATH="/root/.local/bin:$PATH" uv run pytest` — passed, 43 CPU-safe tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/paths.py src/runtime/artifacts.py src/runtime/__init__.py tests/test_runtime_artifacts.py` — passed.
- `git status --short` — contained no new `outputs/`, `runs/`, or generated `.pt` fixtures from this plan; unrelated pre-existing dirty files remained untouched.

## Next Phase Readiness

- Plan 02-03 can build run manifests on top of `resolve_stage_paths` and the documented `runs/<run_id>/manifest.json` contract.
- Plan 02-04 can wire trainer preflight CLI behavior to `validate_artifacts(..., require_ready=True)` without loading CUDA/model/OCR stacks.
- Phase 3 data-selection plans can extend the reserved selected-sample and preference-pair contracts with materialized schemas.

## Self-Check: PASSED

- Found `src/runtime/paths.py`, `src/runtime/artifacts.py`, `docs/runtime_contracts.md`, `tests/test_runtime_artifacts.py`, and this summary file.
- Found task commits `03fb07d`, `374cf4b`, and `173361a` in git history.
- Required verification commands passed.
- Stub-pattern scan found no blocking stubs; empty collection literals in validators are implementation accumulators, not placeholder UI/data stubs.

---
*Phase: 02-runtime-contracts-and-run-provenance*
*Completed: 2026-05-04*
