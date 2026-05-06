---
phase: 07-moderate-structure-and-extension-cleanup
plan: 04
subsystem: synthesis-dataset-builder
tags: [synthesis, synthetic-data, cli-wrapper, import-safe, cpu-safe-tests]

requires:
  - phase: 07-moderate-structure-and-extension-cleanup
    provides: Phase 7 structure rules and prior generation/scoring CLI seam pattern
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Synthetic masked-SFT quality contracts and generated-artifact safety guidance
provides:
  - Import-safe `src.synthesis.dataset_builder` module for synthetic dataset build phases
  - Thin `scripts/synth/build_dataset.py` CLI wrapper preserving existing flags and defaults
  - CPU-safe synthesis contract tests for collation, schemas, orchestration, import safety, and CLI delegation
affects: [phase-7-extension-registry, synthesis-dataset-variants, masked-sft-data-preparation]

tech-stack:
  added: []
  patterns: [thin-cli-wrapper, import-safe-pipeline-seam, lazy-heavy-imports, dataclass-build-config]

key-files:
  created:
    - src/synthesis/__init__.py
    - src/synthesis/dataset_builder.py
    - tests/test_synthesis_pipeline_contracts.py
  modified:
    - scripts/synth/build_dataset.py

key-decisions:
  - "Keep `scripts/synth/build_dataset.py` as a compatibility wrapper that parses existing flags, re-exports phase functions, and delegates to `src.synthesis.dataset_builder.build_dataset`."
  - "Move Torch, Pillow, NumPy, Diffusers, text-encoder, and mask-latent imports inside explicitly gated GPU/model phase functions so importing synthesis contracts remains CPU-safe."
  - "Preserve the historical `clean` behavior: runtime directories are removed only when rendering is not skipped, so `--skip-render --clean` does not delete reused raw data."

patterns-established:
  - "Synthesis variants should import and compose `src.synthesis.dataset_builder` phases instead of adding hidden logic to CLI scripts."
  - "Generated synthetic images, masks, latents, text embeddings, and indexes stay under caller-provided runtime paths and remain non-committable artifacts."

requirements-completed: [STR-05, STR-06]

duration: 5min
completed: 2026-05-06T16:00:46Z
---

# Phase 7 Plan 04: Synthetic Dataset Builder Seam Summary

**Import-safe synthetic dataset builder module with a preserved `python -m scripts.synth.build_dataset` command and reusable phase-level extension seams.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-06T15:56:15Z
- **Completed:** 2026-05-06T16:00:46Z
- **Tasks:** 3
- **Files modified:** 4 task files plus this summary and planning metadata

## Accomplishments

- Added `src.synthesis.dataset_builder` with `SynthesisBuildConfig`, pure collation/fan-out/index phases, gated latent/text encoding phases, and `build_dataset(config)` orchestration.
- Preserved the existing synthesis command surface by replacing `scripts/synth/build_dataset.py` with CLI parsing plus delegation, while re-exporting phase functions for script-path compatibility.
- Added CPU-safe contract tests using temporary metadata and text fixtures; tests avoid SynthTIGER, FLUX, CUDA, OCR, image loading, text encoders, and model weights.
- Verified heavy dependencies are lazy-loaded only inside explicit `bake_latents_phase` and `encode_text_phase` calls.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthesis_pipeline_contracts.py -x` failed with `ModuleNotFoundError: No module named 'src.synthesis'` after adding initial synthesis contract tests.
- **Task 2 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthesis_pipeline_contracts.py -x` failed with `ModuleNotFoundError: No module named 'src.synthesis'` after adding orchestration and gated-phase contracts.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthesis_pipeline_contracts.py tests/test_synthetic_quality.py -q` passed after adding `src.synthesis.dataset_builder` and public exports.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthesis_pipeline_contracts.py -x` failed because the historical script had no `build_dataset` delegation attribute for monkeypatching.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthesis_pipeline_contracts.py tests/test_synthetic_quality.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/synthesis/dataset_builder.py scripts/synth/build_dataset.py tests/test_synthesis_pipeline_contracts.py` passed after thinning the wrapper and fixing plan-owned lint issues.

## Task Commits

1. **Task 1 RED: Synthesis module contracts** - `f9b48cd` (`test`)
2. **Task 2 RED: Builder orchestration contracts** - `c48cfc7` (`test`)
3. **Task 2 GREEN: Importable synthesis dataset builder** - `d15e2d6` (`feat`)
4. **Task 3 RED: Thin CLI wrapper contracts** - `0cc6da2` (`test`)
5. **Task 3 GREEN: Thin CLI wrapper** - `45b91c8` (`feat`)

## Files Created/Modified

- `src/synthesis/__init__.py` - Public import-safe exports for the synthesis dataset builder seam.
- `src/synthesis/dataset_builder.py` - Reusable synthetic dataset phases, `SynthesisBuildConfig`, lazy heavy imports, and `build_dataset(config)` orchestration.
- `scripts/synth/build_dataset.py` - Thin compatibility CLI wrapper for existing flags, defaults, direct execution, module execution, and phase re-exports.
- `tests/test_synthesis_pipeline_contracts.py` - CPU-safe tests for metadata collation, schema writers, fan-out, config defaults, import safety, phase order, gates, CLI delegation, compatibility exports, and wrapper thinness.

## Decisions Made

- Used a frozen dataclass for `SynthesisBuildConfig` so CLI defaults are explicit and reusable by future dataset variants.
- Preserved `subprocess.run(..., check=True)` in `render_phase`, explicit runner/template/config arguments, and existing phase order to avoid changing rendering failure behavior.
- Kept generated indexes and tensor/image outputs under caller-provided runtime directories and did not add any generated data to git.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed plan-owned test and Ruff issues during Task 3 verification**
- **Found during:** Task 3 (Thin the synthesis CLI wrapper)
- **Issue:** The new compatibility re-export test checked `build_dataset` identity after monkeypatching it, and Ruff reported a `Sequence` import modernization plus long assertion lines.
- **Fix:** Moved the re-export identity assertions before monkeypatching, imported `Sequence` from `collections.abc`, and wrapped long test assertions.
- **Files modified:** `tests/test_synthesis_pipeline_contracts.py`, `scripts/synth/build_dataset.py`
- **Verification:** Final synthesis tests and Ruff command passed.
- **Committed in:** `45b91c8`

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking)
**Impact on plan:** The fix was limited to plan-owned tests and wrapper lint compliance; no scope expansion or behavior change beyond the requested seam extraction.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan only found type annotations, file-open modes, and expected local empty-list initializers used to accumulate records; no TODO/FIXME/placeholder/coming-soon/not-available markers or UI-facing empty stubs were introduced.

## Threat Flags

None. The plan threat model covered the new synthesis surfaces: local SynthTIGER subprocess invocation still uses explicit runner/template/config arguments with `check=True`; generated indexes remain under caller-provided runtime paths; and GPU/model work remains gated by `bake_latents` and `encode_text`.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthesis_pipeline_contracts.py tests/test_synthetic_quality.py -q` — passed, 14 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/synthesis/dataset_builder.py scripts/synth/build_dataset.py tests/test_synthesis_pipeline_contracts.py` — passed.
- CPU-safe import contract asserted that importing `src.synthesis.dataset_builder` does not import `diffusers`, `numpy`, `PIL`, `torch`, `src.training.flux2_utils`, or `src.training.losses`.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`, so STATE/ROADMAP updates were prepared manually.
- The worktree contained unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 7 Plan 05 can follow the same import-safe module plus thin wrapper pattern for plotting.
- Phase 7 Plan 06 can include `src.synthesis.dataset_builder` in the final extension-point registry as the synthesis/dataset-builder seam.
- Future synthesis variants can reuse `render_phase`, `collate_records`, `fan_out`, schema writers, latent baking, text encoding, and `build_dataset` without editing unrelated scripts.

## Self-Check: PASSED

- Found all created/modified task files: `src/synthesis/__init__.py`, `src/synthesis/dataset_builder.py`, `scripts/synth/build_dataset.py`, `tests/test_synthesis_pipeline_contracts.py`, and this summary.
- Found task commits `f9b48cd`, `c48cfc7`, `d15e2d6`, `0cc6da2`, and `45b91c8` in git history.
- Required verification commands passed, default synthesis contract tests remain CPU-safe, and no generated images/tensors/checkpoints/logs were committed.

---
*Phase: 07-moderate-structure-and-extension-cleanup*
*Completed: 2026-05-06T16:00:46Z*
