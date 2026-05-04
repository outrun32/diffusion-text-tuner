---
phase: 02-runtime-contracts-and-run-provenance
plan: 05
subsystem: runtime-docs-command-surface
tags: [runtime-contracts, config-organization, manifests, preflight, makefile, docs-tests]

requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: CPU-safe command surface, Makefile aliases, and generated-artifact safety guidance.
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Shared config validation, runtime artifact contracts, local run manifests, and CPU-safe preflight CLI from plans 02-01 through 02-04.
provides:
  - Config-family organization and naming contract under `configs/experiments/`.
  - Runtime manifest and preflight command catalog entries in `docs/commands.md` and `README.md`.
  - Makefile aliases for CPU-safe preflight and manifest dry-run review.
  - CPU-safe docs drift tests for runtime command-surface strings.
affects: [phase-3-data-curriculum, phase-4-characterization-tests, phase-5-run-comparability, thesis-provenance]

tech-stack:
  added: []
  patterns: [docs-drift-tests, cpu-safe-preflight-aliases, compatibility-root-configs, family-scoped-experiment-configs]

key-files:
  created:
    - configs/experiments/README.md
    - configs/experiments/sft/README.md
    - configs/experiments/dpo/README.md
    - configs/experiments/masked_sft/README.md
    - configs/experiments/reward/README.md
    - configs/experiments/evaluation/README.md
    - configs/experiments/synthesis/README.md
    - tests/test_runtime_docs.py
  modified:
    - docs/commands.md
    - README.md
    - Makefile

key-decisions:
  - "Keep `configs/sft.json`, `configs/dpo.json`, `configs/masked_sft.json`, and current root variants as runnable compatibility entry points while placing new comparison variants under `configs/experiments/` family directories."
  - "Expose runtime helpers as CPU-safe validation/provenance commands only; preflight reports readiness and never launches generation, scoring, training, synthesis, evaluation, CUDA, model downloads, or OCR."
  - "Use Makefile aliases for quick dry-run review while preserving Phase 1 setup, test, lint, format, and smoke aliases unchanged."

patterns-established:
  - "New experiment config names follow `{stage}_{reward_or_data}_{purpose}.json` in family directories."
  - "Runtime docs tests assert important command-surface strings so future doc edits do not silently drop manifest/preflight guidance."
  - "README front-door workflow points users to runtime contracts and config organization before expensive jobs."

requirements-completed: [CFG-02, ART-02, ART-04, RUN-01, RUN-03, RUN-04]

duration: 3min
completed: 2026-05-04T14:44:08Z
---

# Phase 2 Plan 05: Runtime Command Surface and Config Organization Summary

**Discoverable runtime manifests, CPU-safe preflight commands, and family-scoped experiment config contracts for comparison-grade research runs.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-04T14:41:39Z
- **Completed:** 2026-05-04T14:44:08Z
- **Tasks:** 3
- **Files modified:** 11 task files plus this summary and planning metadata

## Accomplishments

- Added `configs/experiments/` family documentation for SFT, DPO, masked-SFT, reward/scoring, synthesis, evaluation, and ablation naming while preserving existing root config compatibility paths.
- Added RED-first `tests/test_runtime_docs.py` checks that guard runtime contract docs, Makefile aliases, README links, manifest examples, preflight examples, and generated-artifact safety wording.
- Published manifest and preflight command examples in `docs/commands.md`, linked runtime contracts and config organization from `README.md`, and added CPU-safe Makefile aliases for preflight and manifest review.
- Verified the full CPU-safe pytest suite and Makefile dry-run command surface without launching GPU/model/OCR work.

## RED/GREEN Evidence

- **Task 2 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_docs.py -x` failed as expected after adding tests only because `docs/commands.md`, `Makefile`, and `README.md` had not yet published the runtime command surface.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_docs.py` passed with 3 tests after publishing docs and Makefile aliases.
- **Full verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_docs.py && make -n preflight-sft preflight-dpo preflight-masked-sft manifest-init-sft manifest-inspect && PATH="/root/.local/bin:$PATH" uv run pytest` passed with 65 CPU-safe tests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define config family organization and naming rules** - `f7c6e4a` (docs)
2. **Task 2: Add docs tests for runtime command surface** - `a485716` (test)
3. **Task 3: Publish manifest and preflight commands in docs and Makefile** - `5bfe394` (docs)

**Plan metadata:** recorded in the final docs commit after this summary is created.

## Files Created/Modified

- `configs/experiments/README.md` - Config family, compatibility, naming, metadata, manifest, preflight, and artifact-safety contract.
- `configs/experiments/sft/README.md` - SFT variant naming and metadata expectations.
- `configs/experiments/dpo/README.md` - DPO variant naming, preference-source, and manifest expectations.
- `configs/experiments/masked_sft/README.md` - Masked-SFT synthetic-data, mask, loss, and manifest expectations.
- `configs/experiments/reward/README.md` - Reward/scoring config contract for VLM, OCR, product-score, and calibration variants.
- `configs/experiments/evaluation/README.md` - Held-out evaluation, checkpoint comparison, and thesis-output config contract.
- `configs/experiments/synthesis/README.md` - Synthetic prompt/rendering/font/background config contract.
- `tests/test_runtime_docs.py` - CPU-safe string/drift tests for docs, README, and Makefile runtime command surface.
- `docs/commands.md` - Runtime contracts section with manifest init/inspect/note/metrics and generation/scoring/SFT/DPO/masked-SFT/synthetic/evaluation preflight examples.
- `README.md` - Execution Surface links and preflight/manifest-before-expensive-work guidance.
- `Makefile` - CPU-safe preflight aliases and manifest aliases while preserving Phase 1 aliases.

## Decisions Made

- Preserved existing root configs and documented commands as compatibility entry points rather than moving or deleting root config files.
- Added a dedicated synthesis family README because the plan objective required users to find synthesis config rules even though the task file list omitted that family-specific README.
- Used `RUN_MANIFEST ?= runs/example/manifest.json` for `manifest-inspect` so the alias is dry-run friendly by default while docs/tests still show the canonical `runs/<run_id>/manifest.json` placeholder.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added synthesis family documentation**
- **Found during:** Task 1 (Define config family organization and naming rules)
- **Issue:** The action required SFT, DPO, masked-SFT, reward/scoring, synthesis, evaluation, and ablation naming contracts, but the explicit task file list omitted a synthesis family README.
- **Fix:** Added `configs/experiments/synthesis/README.md` and linked synthesis rules from the top-level config organization contract.
- **Files modified:** `configs/experiments/README.md`, `configs/experiments/synthesis/README.md`
- **Verification:** Task 1 file/string assertion passed, and docs review confirmed synthesis is documented.
- **Committed in:** `f7c6e4a`

**2. [Rule 1 - Bug] Matched README guidance to the docs drift test contract**
- **Found during:** Task 3 verification
- **Issue:** The README communicated the preflight-before-expensive-work guidance, but the new drift test expected the exact lowercase phrase `before long-running GPU/model work`.
- **Fix:** Reworded the README sentence to include the exact phrase while preserving the intended user guidance.
- **Files modified:** `README.md`
- **Verification:** `uv run pytest tests/test_runtime_docs.py` passed.
- **Committed in:** `5bfe394`

---

**Total deviations:** 2 auto-fixed (1 missing critical docs contract, 1 documentation test mismatch).
**Impact on plan:** Both fixes tightened the required runtime documentation surface without changing behavior or launching heavy work.

## Issues Encountered

- `gsd-sdk` was not available under local `node_modules` or on `PATH`, so state, roadmap, and requirement updates were applied manually instead of through SDK query handlers.
- The worktree contained unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from task commits.
- The full pytest suite collected unrelated pre-existing `tests/test_losses.py` from the dirty worktree and it passed; it was not staged or committed by this plan.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder text or empty UI/data-source stubs in files created or modified by this plan.

## Threat Flags

None. The plan threat model covered documentation-to-execution and config-to-artifact trust boundaries; the changes use generic `runs/<run_id>` paths, keep aliases CPU-safe, avoid tokens/personal paths, and document manifest/config metadata for traceability.

## User Setup Required

None - no external service credentials, CUDA device, model cache, OCR package, or generated artifact is required for the docs tests or Makefile dry-run checks.

## Verification Results

- `python -c "from pathlib import Path; required=[Path('configs/experiments/README.md'),Path('configs/experiments/sft/README.md'),Path('configs/experiments/dpo/README.md'),Path('configs/experiments/masked_sft/README.md'),Path('configs/experiments/reward/README.md'),Path('configs/experiments/evaluation/README.md')]; missing=[str(p) for p in required if not p.exists()]; assert not missing, missing; text=Path('configs/experiments/README.md').read_text(); assert 'configs/sft.json' in text and 'configs/dpo.json' in text and 'configs/masked_sft.json' in text; assert '{stage}_{reward_or_data}_{purpose}.json' in text"` — passed.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_docs.py -x` — RED failed as expected before Task 3 updates.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_docs.py` — passed with 3 tests after Task 3.
- `make -n preflight-sft preflight-dpo preflight-masked-sft manifest-init-sft manifest-inspect` — passed and printed CPU-safe/dry-run command lines.
- `PATH="/root/.local/bin:$PATH" uv run pytest` — passed with 65 CPU-safe tests.

## Next Phase Readiness

- Phase 2 now exposes shared config validation, runtime paths/artifact contracts, run manifests, preflight checks, config-family docs, command docs, Makefile aliases, and README links.
- Phase 3 can add data curriculum and dataset quality configs under the documented `configs/experiments/` family structure and use manifests/preflight before expensive generation or synthesis work.

## Self-Check: PASSED

- Found created/modified task files: `configs/experiments/README.md`, `configs/experiments/sft/README.md`, `configs/experiments/dpo/README.md`, `configs/experiments/masked_sft/README.md`, `configs/experiments/reward/README.md`, `configs/experiments/evaluation/README.md`, `configs/experiments/synthesis/README.md`, `tests/test_runtime_docs.py`, `docs/commands.md`, `README.md`, and `Makefile`.
- Found task commits `f7c6e4a`, `a485716`, and `5bfe394` in git history.
- Stub scan found no blocking stubs, and required verification commands passed.

---
*Phase: 02-runtime-contracts-and-run-provenance*
*Completed: 2026-05-04T14:44:08Z*
