---
phase: 07-moderate-structure-and-extension-cleanup
plan: 06
subsystem: extension-registry-command-surface
tags: [extension-points, structure, docs, makefile, cpu-safe-tests, tdd]

requires:
  - phase: 07-moderate-structure-and-extension-cleanup
    provides: Phase 7 structure guide and importable generation/scoring/synthesis/plotting seams.
  - phase: 06-reward-and-evaluation-validity
    provides: Evaluation, diagnostics, thesis-output, and score-output docs referenced by the registry.
provides:
  - Import-safe `src.toolkit.extension_points` registry for prompt generation, image generation, scoring, synthesis, training, evaluation, plotting, run comparison, diagnostics, and thesis outputs.
  - Final Phase 7 structure guide table and extension checklist mirrored from the registry.
  - CPU-safe `phase7-structure-tests` Makefile alias and command docs.
affects: [phase-7-extension-points, command-surface, structure-docs, future-pipeline-extension]

tech-stack:
  added: []
  patterns: [frozen-dataclass-registry, docs-drift-tests, thin-command-alias, import-safety-tests]

key-files:
  created:
    - src/toolkit/__init__.py
    - src/toolkit/extension_points.py
    - tests/test_extension_points_docs.py
  modified:
    - docs/structure_and_extension.md
    - docs/commands.md
    - README.md
    - Makefile

key-decisions:
  - "Keep the extension registry descriptive only: no dynamic plugin loader, command execution, or runtime artifact validation at import time."
  - "Use `phase7-structure-tests` as the focused CPU-safe verification alias for Phase 7 registry/docs/importable seam drift."
  - "Mirror registry entries in `docs/structure_and_extension.md` so users can navigate extension points from docs or Python without hidden assumptions."

requirements-completed: [STR-01, STR-05, STR-06]

metrics:
  duration: 4m08s
  completed: 2026-05-06T16:15:26Z
  tasks: 3
  files: 7
---

# Phase 7 Plan 06: Extension Registry and Command Surface Summary

**Import-safe Phase 7 extension-point registry with a mirrored structure guide, complete future-pipeline checklist, and one focused CPU-safe verification alias.**

## Performance

- **Duration:** 4 min 08 sec
- **Started:** 2026-05-06T16:11:18Z
- **Completed:** 2026-05-06T16:15:26Z
- **Tasks:** 3
- **Files modified:** 7 plan files plus this summary and planning metadata

## Accomplishments

- Added RED-first `tests/test_extension_points_docs.py` covering exact registry stage names, CPU-safe importability, `get_extension_point("scoring")`, docs/README/Makefile command drift, registry-doc mirroring, and the extension checklist.
- Added `src.toolkit.extension_points` with a frozen `ExtensionPoint` dataclass, immutable registry tuple, `list_extension_points()`, and `get_extension_point(name)`.
- Registered prompt generation, image generation, scoring, synthesis, training, evaluation, plotting, run comparison, diagnostics, and thesis outputs with implementation modules, wrappers/commands, config homes, docs paths, test targets, and generated-artifact notes.
- Updated `docs/structure_and_extension.md` with the final registry mirror table and explicit extension checklist for config, artifact/manifest contract, importable module, thin CLI wrapper, CPU-safe tests, command docs, and generated-artifact safety.
- Added and documented `make phase7-structure-tests`, preserving Phase 1-6 command surfaces while exposing focused Phase 7 verification through `docs/commands.md` and README.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_extension_points_docs.py -x` failed as intended with `ModuleNotFoundError: No module named 'src.toolkit'`.
- **Task 2 GREEN (registry subset):** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_extension_points_docs.py::test_extension_registry_lists_all_supported_extension_points tests/test_extension_points_docs.py::test_get_extension_point_returns_scoring_and_rejects_unknown_names tests/test_extension_points_docs.py::test_registered_modules_import_without_new_heavy_optional_stacks -q` passed with 3 registry/import tests.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py tests/test_generation_pipeline_contracts.py tests/test_scoring_pipeline_contracts.py tests/test_synthesis_pipeline_contracts.py tests/test_plotting_pipeline_contracts.py tests/test_extension_points_docs.py -q && make -n phase7-structure-tests && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/toolkit/extension_points.py tests/test_extension_points_docs.py` passed after docs/Makefile updates and lint fixes.

## Task Commits

1. **Task 1: Specify extension registry and command docs drift tests** - `f5fa6af` (`test`)
2. **Task 2: Implement extension-point registry** - `9185076` (`feat`)
3. **Task 3: Publish final Phase 7 command surface and extension checklist** - `66b7e15` (`docs`)

## Files Created/Modified

- `src/toolkit/__init__.py` - Public toolkit exports for the extension registry API.
- `src/toolkit/extension_points.py` - Standard-library-only descriptive registry and lookup helpers.
- `tests/test_extension_points_docs.py` - CPU-safe registry/docs/command drift and import-safety tests.
- `docs/structure_and_extension.md` - Final Phase 7 extension-point table, checklist, and `phase7-structure-tests` guidance.
- `docs/commands.md` - Phase 7 CPU-safe verification command documentation.
- `README.md` - Front-door mention of the Phase 7 structure/extension guide and verification alias.
- `Makefile` - `phase7-structure-tests` alias using the focused Phase 7 pytest selection.

## Decisions Made

- Kept `src.toolkit.extension_points` as a registry/index only, not a plugin loader, so importing it cannot execute commands, traverse files, or initialize optional ML/OCR stacks.
- Used exact stage names in tests and registry order for discoverability: prompt generation, image generation, scoring, synthesis, training, evaluation, plotting, run comparison, diagnostics, and thesis outputs.
- Documented generated-artifact safety per extension point because registry/docs guidance crosses into artifact-handling decisions for future users.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Narrowed the import-safety test to the plan's heavy-stack scope**
- **Found during:** Task 2 verification
- **Issue:** The initial test treated newly imported `PIL` modules as heavy-stack failures, but the plan's import-safety requirement was to avoid FLUX/Qwen/PaddleOCR/CUDA/model-weight stacks; Phase 6 thesis-output docs already allow PIL only for optional contact sheets.
- **Fix:** Removed `PIL` from the heavy optional module set while continuing to guard `diffusers`, `transformers`, `paddle`, `paddleocr`, `torchvision`, `synthtiger`, `mlx_lm`, and Qwen utility imports.
- **Files modified:** `tests/test_extension_points_docs.py`
- **Committed in:** `9185076`

**2. [Rule 3 - Blocking] Fixed plan-owned Ruff failures**
- **Found during:** Task 3 verification
- **Issue:** Ruff flagged long registry string literals and a blank-line import-format issue in the new tests.
- **Fix:** Wrapped registry strings with implicit concatenation and adjusted the test import block without changing registry values.
- **Files modified:** `src/toolkit/extension_points.py`, `tests/test_extension_points_docs.py`
- **Committed in:** `66b7e15`

---

**Total deviations:** 2 auto-fixed issues.
**Impact on plan:** Both fixes were limited to plan-owned tests/registry formatting and preserved the CPU-safe docs/registry/command-surface scope.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO, FIXME, placeholder, coming-soon, not-available markers, hardcoded empty UI data, or unwired mock data in plan-created/modified files.

## Threat Flags

None. The plan threat model covered registry/docs→implementation drift, Makefile/docs→shell command exposure, and extension guidance→artifact handling. The registry is descriptive only and introduces no network endpoints, auth paths, schema migrations, or runtime execution beyond the documented CPU-safe Makefile alias.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_extension_points_docs.py -x` — RED failed as intended on missing `src.toolkit` before implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_extension_points_docs.py::test_extension_registry_lists_all_supported_extension_points tests/test_extension_points_docs.py::test_get_extension_point_returns_scoring_and_rejects_unknown_names tests/test_extension_points_docs.py::test_registered_modules_import_without_new_heavy_optional_stacks -q` — passed, 3 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py tests/test_generation_pipeline_contracts.py tests/test_scoring_pipeline_contracts.py tests/test_synthesis_pipeline_contracts.py tests/test_plotting_pipeline_contracts.py tests/test_extension_points_docs.py -q` — passed, 41 tests.
- `make -n phase7-structure-tests` — printed the expected focused pytest selection without running heavy jobs.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/toolkit/extension_points.py tests/test_extension_points_docs.py` — passed.

## Deferred Issues

- Unrelated pre-existing dirty and untracked files remain in training modules, config variants, generated-looking data roots, thesis docs/scripts, synthetic helpers, loss helpers, and loss tests. They were present before execution and were left untouched/excluded from all Plan 07-06 commits.
- The GSD SDK CLI was unavailable in this checkout (`node_modules/@gsd-build/sdk` missing and no `gsd-sdk` on `PATH`), so planning state files were updated directly rather than through SDK query handlers.

## TDD Gate Compliance

- RED gate commit exists: `f5fa6af`.
- GREEN registry commit exists after RED: `9185076`.
- Final docs/command-surface commit exists after GREEN: `66b7e15`.

## Self-Check: PASSED

- Found created/modified files: `src/toolkit/__init__.py`, `src/toolkit/extension_points.py`, `tests/test_extension_points_docs.py`, `docs/structure_and_extension.md`, `docs/commands.md`, `README.md`, `Makefile`, and this summary.
- Found task commits in git history: `f5fa6af`, `9185076`, and `66b7e15`.
- Required focused pytest, Makefile dry-run, and Ruff verification commands passed.

---
*Phase: 07-moderate-structure-and-extension-cleanup*  
*Completed: 2026-05-06T16:15:26Z*
