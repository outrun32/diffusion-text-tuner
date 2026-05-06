---
phase: 06-reward-and-evaluation-validity
plan: 07
subsystem: reward-evaluation-command-surface
tags: [commands, docs, makefile, docs-drift-tests, reward-evaluation, cpu-safe-tests]

requires:
  - phase: 06-reward-and-evaluation-validity
    provides: Canonical reward interface, held-out evaluation harness, gold/slice diagnostics, score sidecars, reward diagnostics, and thesis-output bundles from Plans 06-01 through 06-06.
provides:
  - Docs drift tests for the integrated Phase 6 reward/evaluation command surface.
  - Makefile aliases for held-out plan materialization, score validation, reward diagnostics, gold checks, thesis bundles, and focused Phase 6 tests.
  - README and guide links that distinguish CPU-safe checks from explicit GPU/model/OCR evaluation jobs.
affects: [phase-6-reward-evaluation-validity, phase-7-structure-cleanup, command-catalog, thesis-evidence-workflow]

tech-stack:
  added: []
  patterns: [docs-drift-command-tests, dry-run-make-aliases, explicit-runtime-artifact-boundaries, lazy-test-imports]

key-files:
  created:
    - tests/test_evaluation_command_docs.py
  modified:
    - Makefile
    - docs/commands.md
    - README.md
    - docs/reward_evaluation.md
    - docs/evaluation_harness.md
    - docs/evaluation_diagnostics.md
    - docs/thesis_outputs.md
    - tests/test_evaluation_scoring_outputs.py
    - tests/test_thesis_outputs.py

key-decisions:
  - "Use Makefile aliases as dry-run discoverability for Phase 6 commands while keeping actual generation/scoring/model/OCR execution explicit."
  - "Publish score validation through the existing CPU-safe `validate_artifacts('evaluation_scores', ...)` contract instead of adding a new CLI."
  - "Keep generated score files, diagnostics, contact sheets, thesis bundles, plots, images, tensors, checkpoints, logs, and run outputs documented as runtime artifacts."

patterns-established:
  - "Phase command docs are guarded by exact-string tests across docs/commands.md, README.md, related guides, and `make -n` aliases."
  - "Focused Phase 6 verification is one pytest file-selection command that remains CPU-safe and model-download-free."
  - "Tests that import heavy-adjacent runtime helpers defer those imports until individual tests execute so import-safety checks stay order-independent."

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08, RUN-05, STR-03]

metrics:
  duration: 7min 02s
  completed: 2026-05-06T15:16:01Z
  tasks: 3
  files: 10
---

# Phase 6 Plan 07: Reward/Evaluation Command Surface Summary

**Integrated Phase 6 reward/evaluation commands are now discoverable through tested docs, README links, and Makefile aliases while preserving CPU-safe defaults and explicit GPU/model/OCR boundaries.**

## Performance

- **Duration:** 7 min 02 sec
- **Started:** 2026-05-06T15:08:59Z
- **Completed:** 2026-05-06T15:16:01Z
- **Tasks:** 3
- **Files modified:** 10 task files plus this summary and planning metadata

## Accomplishments

- Added `tests/test_evaluation_command_docs.py` with docs drift checks for exact Phase 6 commands, README links, related guide cross-links, runtime artifact boundaries, and `make -n` alias output.
- Added Makefile aliases: `phase6-heldout-plan`, `phase6-score-validation`, `phase6-reward-diagnostics`, `phase6-gold-diagnostics`, `phase6-thesis-outputs`, and `phase6-evaluation-tests`.
- Expanded `docs/commands.md` with exact CPU-safe and explicit runtime commands for held-out evaluation plans, score/product sidecar validation, reward diagnostics, gold checks, thesis bundles, and focused Phase 6 tests.
- Linked Phase 6 reward/evaluation docs from the README and cross-linked the four Phase 6 guides so users can move between reward contracts, held-out harnesses, diagnostics, thesis outputs, and command catalog entries.
- Stabilized the focused Phase 6 CPU-safe suite by deferring heavy-adjacent imports in existing tests until the individual test that needs them runs.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py -x` failed as intended on missing Phase 6 command-catalog strings before docs/Makefile updates.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py -q && make -n phase6-heldout-plan phase6-reward-diagnostics phase6-thesis-outputs` passed with 4 docs drift tests and dry-run alias output after command docs and Makefile aliases were added.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py tests/test_evaluation_reward_interface.py tests/test_heldout_evaluation_harness.py tests/test_evaluation_slices_gold.py tests/test_evaluation_scoring_outputs.py tests/test_reward_diagnostics.py tests/test_thesis_outputs.py -q` passed with 49 focused CPU-safe tests after README/guide links and import-order stabilization.

## Task Commits

1. **Task 1: Specify Phase 6 command documentation drift tests** - `15be70b` (`test`)
2. **Task 2: Add Makefile aliases and command docs** - `c6c308e` (`docs`)
3. **Task 3: Link Phase 6 docs from README and verify focused suite** - `86e24b3` (`docs`)

**Plan metadata:** committed separately after summary creation.

## Files Created/Modified

- `tests/test_evaluation_command_docs.py` - CPU-safe docs drift tests for Phase 6 command catalog, README links, guide cross-links, and Makefile dry-run aliases.
- `Makefile` - Phase 6 alias variables and aliases for plan materialization, score validation, reward diagnostics, gold checks, thesis outputs, and focused tests.
- `docs/commands.md` - Integrated Phase 6 command section with exact commands, CPU-safe verification, explicit runtime scoring boundaries, and generated-artifact safety notes.
- `README.md` - Front-door Phase 6 reward/evaluation validity guidance and alias list.
- `docs/reward_evaluation.md` - Links to related Phase 6 harness, diagnostics, thesis-output, and command docs.
- `docs/evaluation_harness.md` - Links to related Phase 6 reward, diagnostics, thesis-output, and command docs.
- `docs/evaluation_diagnostics.md` - Links to related Phase 6 guides and expands runtime artifact wording for thesis bundles/plots.
- `docs/thesis_outputs.md` - Links to related Phase 6 reward, harness, diagnostics, and command docs.
- `tests/test_evaluation_scoring_outputs.py` - Deferred artifact-validator import to avoid collection-time heavy optional module loading during focused suite import-safety checks.
- `tests/test_thesis_outputs.py` - Deferred thesis-output implementation imports to keep import-safety checks order-independent during collection.

## Decisions Made

- Used exact-string docs drift tests instead of loose prose assertions so command drift is visible when Phase 6 CLIs or aliases change.
- Kept score validation as a Python one-liner through `src.runtime.artifacts.validate_artifacts` because the existing artifact contract already provides a CPU-safe sidecar/score-file check.
- Added Makefile aliases for discoverability and `make -n` testing; expensive generation/scoring commands remain explicit runtime commands, not default automation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stabilized focused suite import-safety checks**
- **Found during:** Task 3 (Link Phase 6 docs from README and verify focused suite)
- **Issue:** The focused Phase 6 pytest command failed because collection-time imports in `tests/test_evaluation_scoring_outputs.py` and `tests/test_thesis_outputs.py` loaded `torch` before import-safety tests executed, making otherwise CPU-safe modules appear to load heavy optional dependencies.
- **Fix:** Deferred the artifact-validator and thesis-output implementation imports into the tests that actually need them, preserving behavior while making import-safety checks order-independent.
- **Files modified:** `tests/test_evaluation_scoring_outputs.py`, `tests/test_thesis_outputs.py`
- **Verification:** The focused Phase 6 suite passed with 49 tests afterward.
- **Committed in:** `86e24b3`

---

**Total deviations:** 1 auto-fixed blocking verification issue.
**Impact on plan:** The fix was test-only, CPU-safe, and directly required to run the plan's focused verification command. No generated artifacts, model/OCR/CUDA work, or runtime behavior changes were introduced.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no `TODO`, `FIXME`, placeholder, coming-soon, not-available markers, hardcoded empty UI data, or unwired mock data in files created or modified by this plan.

## Threat Flags

None. The plan threat model covered docs/Makefile shell guidance, README evidence trust, and runtime artifact documentation. No new network endpoints, auth paths, file-access implementations, schema migrations, CUDA/model/OCR execution, or unplanned trust boundaries were introduced.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py -x` — RED failed as intended before docs/Makefile updates.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py -q && make -n phase6-heldout-plan phase6-reward-diagnostics phase6-thesis-outputs` — passed with 4 tests and dry-run alias output.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py tests/test_evaluation_reward_interface.py tests/test_heldout_evaluation_harness.py tests/test_evaluation_slices_gold.py tests/test_evaluation_scoring_outputs.py tests/test_reward_diagnostics.py tests/test_thesis_outputs.py -q` — passed with 49 tests.
- `make -n phase6-heldout-plan phase6-reward-diagnostics phase6-thesis-outputs` — printed the expected held-out plan, reward diagnostics, and thesis-output commands without executing them.

## Deferred Issues

- Unrelated pre-existing dirty and untracked files remain in training modules, configs, data roots, thesis docs/scripts, loss helpers, and loss tests. They were present before execution and were left untouched/excluded from all Plan 06-07 commits.
- The GSD SDK CLI was unavailable in this checkout (`gsd-sdk: command not found` and no local `node_modules/@gsd-build/sdk`), so planning state files were updated directly instead of through SDK query handlers.
- No FLUX/Qwen/PaddleOCR/CUDA/model-weight diagnostics were run, per plan and user constraints.

## TDD Gate Compliance

- RED gate commit exists: `15be70b`.
- GREEN/docs commits exist after RED: `c6c308e` and `86e24b3`.
- No refactor-only commit was needed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 6 command docs are complete and test-guarded across docs, README, and Makefile aliases.
- Phase 7 can consume a stable command surface while planning moderate structure and extension cleanup.

## Self-Check: PASSED

- Found created/modified files: `tests/test_evaluation_command_docs.py`, `docs/commands.md`, `README.md`, `Makefile`, `docs/reward_evaluation.md`, `docs/evaluation_harness.md`, `docs/evaluation_diagnostics.md`, `docs/thesis_outputs.md`, and this summary.
- Found task commits in git history: `15be70b`, `c6c308e`, and `86e24b3`.
- Required focused pytest and Makefile dry-run verification commands passed.

---
*Phase: 06-reward-and-evaluation-validity*  
*Completed: 2026-05-06T15:16:01Z*
