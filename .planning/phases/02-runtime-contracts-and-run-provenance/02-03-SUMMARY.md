---
phase: 02-runtime-contracts-and-run-provenance
plan: 03
subsystem: runtime-provenance
tags: [run-manifests, reproducibility, config-snapshots, cli, cpu-safe-tests]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Shared runtime config snapshots from Plan 02-01 and artifact schema metadata from Plan 02-02.
provides:
  - Local `runs/<run_id>/manifest.json` creation with immutable `config_snapshot.json` files.
  - CPU-safe reproducibility collectors for git, Python/platform/package, CUDA presence, cache presence, seeds, and model IDs/revisions.
  - Manifest load/update/inspection helpers that preserve prior provenance while appending notes and merging metrics.
  - Import-safe `scripts.run_manifest` CLI for `init`, `inspect`, `note`, and `metrics` operations.
affects: [phase-2-preflight-cli, phase-2-runtime-docs, phase-5-run-comparison, thesis-provenance]

tech-stack:
  added: []
  patterns: [sorted-json-manifests, immutable-config-snapshots, secret-presence-only-env-metadata, temp-root-cli-tests]

key-files:
  created:
    - src/runtime/manifests.py
    - src/runtime/reproducibility.py
    - scripts/run_manifest.py
    - tests/test_runtime_manifests.py
  modified:
    - src/runtime/__init__.py

key-decisions:
  - "Keep run manifests local and file-backed under ignored `runs/` roots, with tests using pytest temporary directories rather than committed runtime artifacts."
  - "Serialize secret-related environment variables as boolean presence only, and serialize cache paths as presence flags instead of private machine paths."
  - "Back the manifest CLI directly with `src.runtime.manifests` so command behavior remains CPU-safe and import-safe before GPU/model/OCR stages launch."

patterns-established:
  - "Manifest JSON and config snapshots are written atomically with sorted keys and stable schema versions."
  - "Updates append timestamped notes and merge metrics without changing command, git, environment, or config provenance fields."
  - "CLI tests invoke `main(argv)` directly and assert stdout/stderr behavior without launching pipeline stages."

requirements-completed: [RUN-01, RUN-03, RUN-04, CFG-03, STR-02]

duration: 17min
completed: 2026-05-04T14:28:55Z
---

# Phase 2 Plan 03: Local Run Manifest Provenance Summary

**Local run manifests with immutable config snapshots, secret-safe reproducibility metadata, and a CPU-only CLI for init/inspect/notes/metrics.**

## Performance

- **Duration:** 17 min
- **Started:** 2026-05-04T14:12:02Z
- **Completed:** 2026-05-04T14:28:55Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added RED-first tests covering complete manifest schema fields, temporary run directory creation, config snapshot writing, secret-safe environment summaries, manifest load/update immutability, human-readable inspection summaries, and CLI behavior.
- Implemented `src.runtime.reproducibility` with CPU-safe git state, Python/platform/package summaries, CUDA presence metadata, cache presence metadata, secret environment presence flags, seeds, and model revision extraction.
- Implemented `src.runtime.manifests` with `RunManifest`, `create_run_manifest`, `load_run_manifest`, `update_run_manifest`, and inspection helpers that write deterministic sorted JSON.
- Added `scripts/run_manifest.py` with `init`, `inspect`, `note`, and `metrics` subcommands backed by manifest helpers and direct `main(argv)` tests.
- Exported manifest and reproducibility helpers from `src.runtime.__init__` while preserving existing config/artifact/path APIs from earlier Phase 2 plans.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py -x` failed during collection with `ImportError: cannot import name 'manifests' from 'src.runtime'` after adding tests only.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py` passed with 5 manifest tests after implementing manifest helpers, reproducibility collectors, exports, and the initial CLI backing.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py::test_run_manifest_cli_reports_invalid_metrics_payloads -x` failed because a missing metrics file raised uncaught `FileNotFoundError`.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py` passed with 6 tests after converting missing metrics files into nonzero CLI responses.
- **Full suite:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py && PATH="/root/.local/bin:$PATH" uv run pytest` passed with 49 CPU-safe tests.
- **Lint:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/manifests.py src/runtime/reproducibility.py src/runtime/__init__.py scripts/run_manifest.py tests/test_runtime_manifests.py` passed.

## Task Commits

Each planned task was committed atomically where possible:

1. **Task 1: Specify run manifest schema and CLI behavior** - `6751b52` (test)
2. **Task 2: Implement run manifests and reproducibility collectors** - `eee7167` (feat)
3. **Task 3 RED: Add CLI metrics error coverage** - `432448e` (test)
4. **Task 3 GREEN: Report CLI metrics file errors** - `71cf1ab` (fix)

Task 2 also introduced the initial import-safe CLI backing because the Task 1 RED contract included CLI behavior in the same test file; the Task 3 commits then tightened CLI error handling with an additional RED/GREEN cycle.

## Files Created/Modified

- `tests/test_runtime_manifests.py` - CPU-safe tests for manifest schema, immutable updates, secret-safe environment summaries, CLI init/inspect/note/metrics operations, and CLI error paths.
- `src/runtime/reproducibility.py` - Git/environment/package/CUDA/cache/seed/model metadata collectors that do not serialize secret values.
- `src/runtime/manifests.py` - Run manifest dataclass, creation/loading/updating helpers, config snapshot writing, summary formatting, and run-root validation.
- `scripts/run_manifest.py` - Argparse CLI for initializing, inspecting, annotating, and updating local manifest metrics.
- `src/runtime/__init__.py` - Public exports for manifest and reproducibility helpers.

## Decisions Made

- Preserved the existing `load_stage_config`, `resolve_config_snapshot`, `resolve_stage_paths`, and `validate_artifacts` APIs and layered manifests on top of them.
- Used `config_snapshot.json` next to `manifest.json` in each run directory and embedded the same snapshot in the manifest for convenient inspection.
- Recorded generated outputs as path/metadata dictionaries only; tests create temporary run directories and do not commit anything under real `runs/`.
- Kept environment metadata secret-safe by recording token/key/password variable presence only and cache variable presence only.
- Kept the CLI limited to provenance file management; it does not launch training, generation, model loading, OCR, CUDA work, or other expensive stages.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Implemented initial CLI backing during Task 2**
- **Found during:** Task 2 GREEN verification
- **Issue:** The Task 1 RED test contract intentionally covered `scripts.run_manifest` behavior in the same test file that Task 2 was required to pass, so helper-only implementation would leave the plan-level tests red.
- **Fix:** Added an import-safe CPU-only `scripts/run_manifest.py` backed by `src.runtime.manifests` during Task 2, then used Task 3 for additional CLI-specific RED/GREEN error handling.
- **Files modified:** `scripts/run_manifest.py`, `tests/test_runtime_manifests.py`
- **Commit:** `eee7167`

**2. [Rule 3 - Blocking] Fixed Ruff violations in new manifest files**
- **Found during:** Task 2 verification
- **Issue:** Ruff flagged `Path.cwd()` in default arguments and an unused test import.
- **Fix:** Moved `Path.cwd()` resolution inside function bodies and removed the unused import.
- **Files modified:** `src/runtime/manifests.py`, `src/runtime/reproducibility.py`, `tests/test_runtime_manifests.py`
- **Commit:** `eee7167`

**3. [Rule 1 - Bug] Converted missing metrics files into CLI errors**
- **Found during:** Task 3 RED CLI metrics coverage
- **Issue:** `run_manifest metrics --file missing.json` raised an uncaught `FileNotFoundError` instead of returning a nonzero CLI status with clear stderr.
- **Fix:** Caught `OSError` while reading metrics files and re-raised `ManifestError` so `main(argv)` returns `2` with a concise error message.
- **Files modified:** `scripts/run_manifest.py`, `tests/test_runtime_manifests.py`
- **Commits:** `432448e`, `71cf1ab`

---

**Total deviations:** 3 auto-fixed (one task sequencing blocker, one lint blocker, one CLI error-handling bug).
**Impact on plan:** All fixes supported the planned CPU-safe manifest/CLI behavior and did not alter existing runtime config or artifact APIs.

## Issues Encountered

- `gsd-sdk` was not available under local `node_modules` or `PATH`, so state, roadmap, and requirement updates were applied manually instead of through SDK query handlers.
- The worktree had substantial unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from task and metadata commits.
- Full pytest collected `tests/test_losses.py` from unrelated pre-existing untracked work and it passed; it was not staged or committed by this plan.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder text in the files created or modified by this plan. Empty dict/list defaults are normal manifest data structures and not UI/data-source stubs.

## Threat Flags

None. The new local environment/git-to-manifest and CLI-args-to-filesystem trust boundaries were covered by this plan's threat model and mitigated by secret-presence-only serialization, cache-presence-only serialization, append-only timestamped notes, immutable provenance preservation, and run-root validation for `runs` directories.

## User Setup Required

None - no external service credentials or model/cache access are required for manifest helpers or CLI tests.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py -x` — RED failed as expected before implementation with missing `src.runtime.manifests` import.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py` — passed with 6 tests after implementation and CLI error fix.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py && PATH="/root/.local/bin:$PATH" uv run pytest` — passed with 49 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/manifests.py src/runtime/reproducibility.py src/runtime/__init__.py scripts/run_manifest.py tests/test_runtime_manifests.py` — passed.
- `git status --short` — no generated `runs/` artifacts from this plan; unrelated pre-existing dirty and untracked files remain untouched.

## Next Phase Readiness

- Plan 02-04 can wire trainer preflight and manifest creation around these helpers without changing trainer config dataclasses.
- Plan 02-05 can document `python -m scripts.run_manifest init/inspect/note/metrics` and add Makefile aliases if desired.
- Later run-comparison work can build on stable `manifest.json` and `config_snapshot.json` conventions.

## Self-Check: PASSED

- Found created/modified files: `src/runtime/manifests.py`, `src/runtime/reproducibility.py`, `scripts/run_manifest.py`, `tests/test_runtime_manifests.py`, `src/runtime/__init__.py`, this summary, `.planning/STATE.md`, `.planning/ROADMAP.md`, and `.planning/REQUIREMENTS.md`.
- Found task commits: `6751b52`, `eee7167`, `432448e`, and `71cf1ab`.
- Required verification commands passed, and no generated `runs/` artifacts from this plan were committed.

---
*Phase: 02-runtime-contracts-and-run-provenance*
*Completed: 2026-05-04T14:28:55Z*
