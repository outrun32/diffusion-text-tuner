---
phase: 07-moderate-structure-and-extension-cleanup
plan: 02
subsystem: generation
tags: [python, cli, generation, flux, tdd, cpu-safe-tests]

requires:
  - phase: 07-moderate-structure-and-extension-cleanup
    provides: Phase 7 structure homes and extension rules from Plan 01
provides:
  - Import-safe `src.generation.pipeline` seam for image generation
  - Thin `scripts.generate_images` CLI wrapper preserving existing arguments
  - CPU-safe contract tests for generation defaults, paths, seeds, imports, prompt slicing, and CLI delegation
affects: [phase-7, generation, scripts, extension-points]

tech-stack:
  added: []
  patterns:
    - Frozen dataclass config/path contracts behind a thin CLI wrapper
    - Heavy FLUX, torchvision, and training utility imports localized inside `run_generation`

key-files:
  created:
    - src/generation/__init__.py
    - src/generation/pipeline.py
    - tests/test_generation_pipeline_contracts.py
  modified:
    - scripts/generate_images.py

key-decisions:
  - "Keep `scripts.generate_images` as the public entry point while moving generation orchestration into `src.generation.pipeline`."
  - "Keep FLUX, torchvision, and `src.training.flux2_utils` imports inside `run_generation` so CPU-safe tests can import the seam without model/CUDA side effects."

patterns-established:
  - "Generation scripts should parse CLI arguments, build an importable config dataclass, and delegate to source modules."
  - "Generation pipeline helpers should keep path resolution, seed planning, and prompt loading pure/testable."

requirements-completed: [STR-05, STR-06]

duration: 3m20s
completed: 2026-05-06
---

# Phase 7 Plan 02: Importable Generation Pipeline Summary

**FLUX image generation now runs through an import-safe `src.generation.pipeline` seam while `python -m scripts.generate_images` keeps the existing CLI contract.**

## Performance

- **Duration:** 3m20s
- **Started:** 2026-05-06T15:44:59Z
- **Completed:** 2026-05-06T15:48:19Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added CPU-safe contract tests for CLI defaults, prompt slicing, output path planning, deterministic generation seed formula, import safety, and wrapper delegation.
- Created `GenerationConfig`, `GenerationPaths`, `load_prompt_records`, `resolve_generation_paths`, `plan_generation_seed`, and `run_generation` in `src.generation.pipeline`.
- Replaced the body of `scripts.generate_images` with a thin parser/delegator while preserving argument names/defaults and module invocation behavior.

## Task Commits

1. **Task 1: Specify generation module contracts** - `6229441` (test)
2. **Task 2: Implement import-safe generation pipeline module** - `8019a3d` (feat)
3. **Task 3: Thin the generation CLI wrapper** - `8019a3d` (feat)

_Note: Tasks 2 and 3 share one GREEN commit because the contract tests covered both the pipeline seam and wrapper delegation, and full verification required both to be present._

## Files Created/Modified

- `src/generation/__init__.py` - Exports the import-safe generation seam symbols.
- `src/generation/pipeline.py` - Contains frozen generation config/path dataclasses, prompt loading, path resolution, seed planning, and the moved FLUX generation loop.
- `scripts/generate_images.py` - Parses the existing CLI arguments and delegates to `run_generation(GenerationConfig(...))`.
- `tests/test_generation_pipeline_contracts.py` - CPU-safe tests covering generation contracts and wrapper delegation without loading FLUX/CUDA/model weights.

## Decisions Made

- Kept the legacy CLI command and flags intact, including `--save_latents` and `--save_png` defaulting to `True`, to preserve existing `python -m scripts.generate_images` behavior.
- Kept all heavy imports (`diffusers`, `torchvision`, and `src.training.flux2_utils`) inside `run_generation` so importing `src.generation.pipeline` stays CPU-safe.
- Used `Path`-based config/path helpers for new reusable seams while keeping output artifact names, manifest keys, skip-if-existing logic, and seed formula unchanged.

## Verification

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_generation_pipeline_contracts.py -x` — RED verified before production code (`ModuleNotFoundError: No module named 'src.generation'`).
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_generation_pipeline_contracts.py -q` — PASS, 8 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/generation/pipeline.py scripts/generate_images.py tests/test_generation_pipeline_contracts.py` — PASS.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None. Stub-pattern scan only found intentional `None` defaults, local empty test dictionaries, and empty list initialization in prompt loading.

## Threat Flags

None. The plan threat model already covered prompt JSONL parsing, CLI path filesystem writes, expensive generation controls, and manifest information disclosure.

## Issues Encountered

- `gsd-sdk` was not available in this worktree, so state/roadmap/requirements updates were applied manually instead of via SDK query helpers.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 07-03 can follow the same thin-wrapper/import-safe-module pattern for reward scoring.
- Generated images, tensors, checkpoints, logs, and model artifacts were not created or committed.

## Self-Check: PASSED

- Found `src/generation/__init__.py`.
- Found `src/generation/pipeline.py`.
- Found `scripts/generate_images.py`.
- Found `tests/test_generation_pipeline_contracts.py`.
- Found `.planning/phases/07-moderate-structure-and-extension-cleanup/07-02-SUMMARY.md`.
- Found task commits `6229441` and `8019a3d` in git history.

---
*Phase: 07-moderate-structure-and-extension-cleanup*
*Completed: 2026-05-06*
