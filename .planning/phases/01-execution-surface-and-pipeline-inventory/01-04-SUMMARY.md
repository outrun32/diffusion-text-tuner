---
phase: 01-execution-surface-and-pipeline-inventory
plan: 04
subsystem: docs-testing-tooling
tags: [pytest, ruff, uv, makefile, diagnostics, slurm, documentation]

requires:
  - phase: 01-execution-surface-and-pipeline-inventory
    provides: Pipeline inventory, uv/pyproject tooling, and import-safe smoke checks from plans 01-01 through 01-03.
provides:
  - Guarded manual gradient diagnostics outside pytest-style discovery.
  - Standard command catalog for setup, CPU-safe tests, smokes, local pipelines, SLURM variants, diagnostics, and artifact safety.
  - Makefile aliases for common CPU-safe and explicit smoke commands.
  - README front-door links to the pipeline inventory and command catalog.
affects: [phase-2-runtime-contracts, phase-4-cpu-safe-tests, phase-5-training-comparability, slurm, diagnostics]

tech-stack:
  added: [Makefile command aliases]
  patterns: [guarded diagnostic main functions, CPU-safe default command surface, opt-in smoke and diagnostic commands]

key-files:
  created:
    - scripts/diagnose_gradient_flow.py
    - scripts/diagnose_grad_magnitude.py
    - docs/commands.md
    - Makefile
  modified:
    - README.md
    - docs/pipeline_inventory.md
    - scripts/test_gradient_flow.py
    - scripts/test_grad_magnitude.py

key-decisions:
  - "Keep default pytest discovery CPU-safe by renaming manual CUDA/model diagnostics and guarding all diagnostic work behind explicit main() entry points."
  - "Use docs/commands.md plus Makefile aliases as the standard command surface while preserving existing local and SLURM entry points."
  - "Document GPU, model-access, OCR, SLURM, and gradient diagnostics as opt-in workflows rather than default automation."

patterns-established:
  - "Manual diagnostics use diagnose_*.py names and if __name__ == \"__main__\": raise SystemExit(main()) guards."
  - "Command documentation separates CPU-safe defaults from smoke checks, local pipelines, SLURM variants, and manual diagnostics."
  - "Generated artifact guidance names ignored runtime roots before users run expensive jobs."

requirements-completed: [INV-01, INV-02, INV-03, INV-04, ENV-03, ENV-04, ENV-05, ENV-06, TEST-06, TEST-07]

duration: 4min
completed: 2026-05-04
---

# Phase 1 Plan 04: Command Surface and Diagnostic Separation Summary

**CPU-safe command catalog with Makefile aliases and guarded opt-in gradient diagnostics for local and SLURM research workflows.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-04T13:34:39Z
- **Completed:** 2026-05-04T13:38:11Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Renamed `scripts/test_gradient_flow.py` and `scripts/test_grad_magnitude.py` to `scripts/diagnose_gradient_flow.py` and `scripts/diagnose_grad_magnitude.py`, with heavy CUDA/model work moved into guarded `main()` functions.
- Added `docs/commands.md` with setup, CPU-safe pytest/Ruff commands, explicit smoke checks, local pipeline commands, SLURM variants, manual diagnostics, and generated-artifact safety guidance.
- Added a root `Makefile` with short aliases for setup, tests, linting, format checks, and opt-in smoke checks.
- Updated `README.md` with an `Execution Surface` section linking the pipeline inventory and command catalog, including CPU-safe versus opt-in diagnostic guidance.
- Updated `docs/pipeline_inventory.md` so user-facing diagnostic paths match the new guarded filenames.

## Task Commits

Each planned task was committed atomically where possible:

1. **Task 1: Rename manual gradient diagnostics and guard execution** - `ea645c9` (refactor)
2. **Task 2: Add command catalog, Makefile aliases, and README links** - `8e4a8ff` (docs)
3. **Deviation fix: Refresh diagnostic inventory paths** - `61897ab` (fix)

## Files Created/Modified

- `scripts/diagnose_gradient_flow.py` - Guarded manual ReFL gradient-flow diagnostic.
- `scripts/diagnose_grad_magnitude.py` - Guarded manual gradient-magnitude diagnostic for ReFL steps.
- `scripts/test_gradient_flow.py` - Removed old pytest-style diagnostic path.
- `scripts/test_grad_magnitude.py` - Removed old pytest-style diagnostic path.
- `docs/commands.md` - Standard command catalog for setup, safe automation, smokes, local/SLURM flows, diagnostics, and git safety.
- `Makefile` - Aliases for setup, pytest, Ruff checks, and smoke checks.
- `README.md` - Front-door execution-surface section linking inventory and command catalog.
- `docs/pipeline_inventory.md` - Manual diagnostic table updated for renamed diagnostic scripts.

## Verification Results

- `python -c "from pathlib import Path; assert Path('scripts/diagnose_gradient_flow.py').exists(); assert Path('scripts/diagnose_grad_magnitude.py').exists(); assert not Path('scripts/test_gradient_flow.py').exists(); assert not Path('scripts/test_grad_magnitude.py').exists(); text1=Path('scripts/diagnose_gradient_flow.py').read_text(); text2=Path('scripts/diagnose_grad_magnitude.py').read_text(); assert 'def main(' in text1 and 'if __name__ == \"__main__\"' in text1; assert 'def main(' in text2 and 'if __name__ == \"__main__\"' in text2" && uv run pytest --collect-only` — passed with 11 collected tests under `tests/` only.
- `python -c "from pathlib import Path; docs=Path('docs/commands.md').read_text(); make=Path('Makefile').read_text(); readme=Path('README.md').read_text(); required_docs=['## Setup','## CPU-safe default commands','## Smoke checks','## Local pipeline commands','## SLURM command variants','## Manual diagnostics','uv run pytest','uv run ruff check .','python -m scripts.smoke_environment --check imports','sbatch --array=0-15 scripts/cluster/generate_images.sbatch','sbatch scripts/cluster/sft.sbatch']; missing=[s for s in required_docs if s not in docs]; assert not missing, missing; targets=['setup','test','lint','format','smoke-imports','smoke-cuda','smoke-model-access','smoke-ocr','smoke-cache']; missing_targets=[t for t in targets if f'{t}:' not in make]; assert not missing_targets, missing_targets; required_readme=['## Execution Surface','docs/pipeline_inventory.md','docs/commands.md','CPU-safe','opt-in']; missing_readme=[s for s in required_readme if s not in readme]; assert not missing_readme, missing_readme" && make -n test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr smoke-cache` — passed; dry-run printed expected commands.
- `uv run pytest` — passed with 11 tests.
- `python -c "from pathlib import Path; text=Path('docs/pipeline_inventory.md').read_text(); assert 'scripts/diagnose_gradient_flow.py' in text; assert 'scripts/diagnose_grad_magnitude.py' in text; assert 'scripts/test_gradient_flow.py' not in text; assert 'scripts/test_grad_magnitude.py' not in text" && uv run pytest --collect-only` — passed.

## Decisions Made

- Kept the diagnostic logic behavior-preserving, but moved all CUDA/model work behind `main()` guards so imports are safe.
- Preserved existing local and SLURM entry points rather than introducing new wrappers beyond Makefile aliases.
- Kept expensive CUDA/model/OCR checks opt-in and documented separately from the default pytest/Ruff command surface.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used the installed uv binary from `/root/.local/bin` for verification**
- **Found during:** Task 1 verification
- **Issue:** `uv` was installed but not present on the shell `PATH`, causing the required `uv run pytest --collect-only` verification command to fail with `uv: command not found`.
- **Fix:** Verified with `PATH="/root/.local/bin:$PATH"` so the required uv commands could run without modifying project files.
- **Files modified:** None
- **Verification:** `uv run pytest --collect-only` and `uv run pytest` both passed with the adjusted PATH.
- **Committed in:** Not applicable; environment-only fix.

**2. [Rule 1 - Bug] Updated stale diagnostic paths in the pipeline inventory**
- **Found during:** Post-task documentation review
- **Issue:** After renaming diagnostics, `docs/pipeline_inventory.md` still pointed users to deleted `scripts/test_*.py` paths.
- **Fix:** Updated the manual diagnostics table to reference `scripts/diagnose_gradient_flow.py` and `scripts/diagnose_grad_magnitude.py`.
- **Files modified:** `docs/pipeline_inventory.md`
- **Verification:** Path assertion plus `uv run pytest --collect-only` passed.
- **Committed in:** `61897ab`

---

**Total deviations:** 2 auto-fixed (1 blocking environment issue, 1 documentation bug)
**Impact on plan:** Both fixes supported the plan goal without changing pipeline behavior or running real GPU/model/OCR diagnostics.

## Issues Encountered

- The shell could not find `uv` until `/root/.local/bin` was added to `PATH` for verification commands.
- The existing unrelated dirty worktree files were left untouched and were not staged or committed.

## Auth Gates

None.

## Known Stubs

None. The stub-pattern scan only found the intentional empty negative prompt string in `scripts/diagnose_grad_magnitude.py`.

## Threat Flags

None. The changed files document CLI execution surfaces and generated-artifact safety; no new network endpoint, authentication path, file deserialization path, or schema trust boundary was introduced.

## User Setup Required

None - no external service configuration required by this plan. Users still need optional GPU/model/OCR credentials and caches only when choosing documented opt-in commands.

## Next Phase Readiness

- Phase 1 now has a complete inventory, dependency/tooling contract, import-safe smoke checks, command catalog, and diagnostic separation.
- Phase 2 can build runtime contracts and run provenance on top of the documented command surface.

## Self-Check: PASSED

- Found created/modified files: `scripts/diagnose_gradient_flow.py`, `scripts/diagnose_grad_magnitude.py`, `docs/commands.md`, `Makefile`, `README.md`, `docs/pipeline_inventory.md`, and this summary.
- Found task/deviation commits: `ea645c9`, `8e4a8ff`, and `61897ab`.

---
*Phase: 01-execution-surface-and-pipeline-inventory*
*Completed: 2026-05-04*
