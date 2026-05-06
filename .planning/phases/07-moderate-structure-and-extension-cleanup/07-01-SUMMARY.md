---
phase: 07-moderate-structure-and-extension-cleanup
plan: 01
subsystem: structure-extension-docs
tags: [structure, extension-points, docs-drift-tests, scripts, artifact-safety, cpu-safe]

requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: Pipeline inventory, command catalog, and generated-artifact safety boundaries.
  - phase: 06-reward-and-evaluation-validity
    provides: Stable reward/evaluation command and thesis-output docs that Phase 7 structure guidance preserves.
provides:
  - Canonical Phase 7 structure and extension guide for source, scripts, cluster launchers, configs, diagnostics, experiments, generated outputs, tests, and thesis artifacts.
  - Script-family navigation that separates supported thin wrappers from diagnostics, cluster jobs, synthesis helpers, plotting helpers, and historical scripts.
  - CPU-safe docs drift tests guarding STR-01/STR-06 structure, extension, README-link, and generated-artifact safety wording.
affects: [phase-7-structure-cleanup, future-pipeline-seams, script-navigation, thesis-provenance]

tech-stack:
  added: []
  patterns: [docs-drift-tests, canonical-repository-homes, thin-wrapper-boundaries, generated-artifact-safety]

key-files:
  created:
    - docs/structure_and_extension.md
    - scripts/README.md
    - tests/test_structure_extension_docs.py
  modified:
    - README.md

key-decisions:
  - "Use `docs/structure_and_extension.md` as the canonical Phase 7 navigation contract before moving implementation code behind importable seams."
  - "Keep script guidance behavior-preserving: supported commands remain wrappers, diagnostics stay opt-in, and generated runtime outputs remain ignored/private by default."
  - "Guard STR-01/STR-06 wording with exact-string CPU-safe pytest tests so later cleanup cannot silently drop structure or extension guidance."

patterns-established:
  - "Repository-home docs classify each path as source, wrapper, diagnostic, config, test, generated runtime output, or thesis evidence."
  - "Script README classifies supported wrappers separately from manual diagnostics, cluster jobs, synthesis helpers, plotting helpers, and historical scripts."
  - "README front-door links Phase 7 structure/extension guidance alongside pipeline inventory, commands, and runtime contracts."

requirements-completed: [STR-01, STR-06]

metrics:
  duration: 20min
  completed: 2026-05-06T15:42:58Z
  tasks: 2
  files: 4
---

# Phase 7 Plan 01: Structure and Extension Navigation Summary

**Tested Phase 7 repository-home and script-family navigation with extension rules for future experiments, trainers, reward variants, datasets, pipelines, plots, and thesis-output steps.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-05-06T15:23:00Z
- **Completed:** 2026-05-06T15:42:58Z
- **Tasks:** 2
- **Files modified:** 4 task files plus this summary and planning metadata

## Accomplishments

- Added RED-first `tests/test_structure_extension_docs.py` with CPU-safe docs drift checks for required repository homes, script family classifications, extension rules, README discoverability, and generated-artifact safety wording.
- Published `docs/structure_and_extension.md` as the canonical Phase 7 structure and extension contract covering `src/`, `scripts/`, `scripts/cluster/`, `scripts/synth/`, `scripts/thesis_plots/`, `configs/`, `configs/experiments/`, `tests/`, `experiments/`, ignored runtime roots, and thesis evidence artifacts.
- Added `scripts/README.md` to classify supported thin wrappers, manual diagnostics, cluster jobs, synthesis helpers, plotting helpers, and historical/experiment scripts while warning that diagnostics and generated outputs can contain private prompt text, scores, paths, or run metadata.
- Linked the new Phase 7 structure/extension guide from the README front door.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -x` failed as intended because `docs/structure_and_extension.md` did not exist yet.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -q` passed with 4 CPU-safe docs drift tests after publishing the guide, scripts README, and README link.
- **Final verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -q && git status --short` passed the focused suite and showed only unrelated pre-existing dirty/untracked files remaining.

## Task Commits

Each task was committed atomically:

1. **Task 1: Specify structure and script-navigation drift tests** - `4b3c490` (`test`)
2. **Task 2: Publish the repository structure and script-home contract** - `7da3068` (`docs`)

**Plan metadata:** recorded in the final docs commit after this summary is created.

## Files Created/Modified

- `tests/test_structure_extension_docs.py` - CPU-safe docs drift tests for Phase 7 structure homes, extension rules, scripts navigation, README links, and artifact-safety wording.
- `docs/structure_and_extension.md` - Canonical structure and extension guide for repository homes, generated runtime output boundaries, and future extension rules.
- `scripts/README.md` - Script-family navigation guide that separates supported wrappers from diagnostics, cluster jobs, synthesis helpers, plotting helpers, and historical scripts.
- `README.md` - Front-door link to the Phase 7 structure/extension guide.

## Decisions Made

- Kept Phase 7 Plan 01 as documentation/navigation only; no source modules, scripts, configs, generated outputs, or runtime behavior were moved.
- Used exact-string docs drift tests because STR-01/STR-06 are navigation contracts and later cleanup should fail visibly if critical headings or artifact-safety wording disappear.
- Treated generated thesis outputs, contact sheets, reports, and private run outputs as runtime artifacts by default, matching the plan threat model and existing project constraints.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Matched structure guide wording to the docs drift test contract**
- **Found during:** Task 2 (Publish the repository structure and script-home contract)
- **Issue:** The structure guide communicated the GPU/model/OCR/SLURM prerequisite guidance, but one test required the exact lowercase phrase `document CPU-safe verification and explicit GPU/model/OCR/SLURM prerequisites`.
- **Fix:** Reworded the extension rule to include the exact tested phrase while preserving the intended guidance.
- **Files modified:** `docs/structure_and_extension.md`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -q` passed with 4 tests.
- **Committed in:** `7da3068`

---

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** The fix tightened the documentation contract and did not expand scope, move code, or create generated artifacts.

## Issues Encountered

- The worktree contained unrelated pre-existing dirty and untracked files before execution, including training module edits, config variants, data roots, thesis docs/scripts, loss helpers, and loss tests. They were left untouched and excluded from Plan 07-01 commits.
- The GSD SDK CLI was unavailable in this checkout (`node_modules/@gsd-build/sdk` missing and no `gsd-sdk` on `PATH`), so planning state, roadmap, and requirement updates were applied directly rather than through SDK query handlers.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no `TODO`, `FIXME`, placeholder, coming-soon, not-available markers, hardcoded empty UI data, or unwired mock data in files created or modified by this plan.

## Threat Flags

None. The plan threat model covered docs-to-shell-command and docs-to-artifact-handling trust boundaries. The changes introduce no new network endpoints, auth paths, file-access implementations, schema changes, or runtime execution surfaces; they explicitly keep generated artifacts out of git and mark diagnostics/generated outputs as potentially private runtime surfaces.

## User Setup Required

None - no external service configuration, CUDA device, model cache, OCR package, or generated artifact is required for the docs tests.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -x` — RED failed as intended before docs implementation because `docs/structure_and_extension.md` was missing.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -q` — passed with 4 tests after docs implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py -q && git status --short` — passed and confirmed no plan-created generated artifacts were left uncommitted; only unrelated pre-existing dirty/untracked files remained.

## Deferred Issues

- Unrelated pre-existing dirty/untracked files remain in `src/training/`, root config variants, `configs/synth/`, generated-looking `data/` subtrees, `docs/thesis/`, scripts, `scripts/synth/`, `scripts/thesis_plots/`, and loss tests. They were present before Plan 07-01 execution and were not modified or committed.
- No full pytest suite was run because the plan and user constraints requested CPU-safe docs/navigation work only, and targeted docs drift verification was sufficient for this documentation-only plan.

## TDD Gate Compliance

- RED gate commit exists: `4b3c490`.
- GREEN/docs commit exists after RED: `7da3068`.
- No refactor-only commit was needed.

## Next Phase Readiness

- Phase 7 follow-up plans can move generation, scoring, synthesis, and plotting code behind importable modules while preserving the documented homes and thin-wrapper boundaries.
- The README and `scripts/README.md` now point users to the structure/extension contract before adding new experiments, trainers, reward variants, datasets, pipelines, plots, or thesis-output steps.

## Self-Check: PASSED

- Found created/modified files: `tests/test_structure_extension_docs.py`, `docs/structure_and_extension.md`, `scripts/README.md`, `README.md`, and this summary.
- Found task commits in git history: `4b3c490` and `7da3068`.
- Required focused pytest verification passed with 4 CPU-safe docs drift tests.

---
*Phase: 07-moderate-structure-and-extension-cleanup*  
*Completed: 2026-05-06T15:42:58Z*
