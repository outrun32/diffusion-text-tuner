---
phase: 06-reward-and-evaluation-validity
plan: 02
subsystem: held-out-evaluation-harness
tags: [evaluation, heldout, manifests, cpu-safe, cli, docs, tdd]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Local run manifest schema and provenance conventions.
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Controlled training comparison expectations and manifest-linked evidence.
  - phase: 06-reward-and-evaluation-validity
    provides: Canonical reward score metadata and missing-evidence conventions from Plan 06-01.
provides:
  - CPU-safe held-out evaluation config and target validation contract.
  - Deterministic held-out evaluation JSON plan and Markdown report materialization.
  - Thin `python -m scripts.run_heldout_evaluation` CLI for plan-only validation/report writes.
  - Documentation for fixed prompts, seeds, inference settings, target manifests, local/SLURM templates, and artifact safety.
affects: [phase-6-evaluation-validity, heldout-evaluation, thesis-run-traceability, phase-6-command-docs]

tech-stack:
  added: []
  patterns: [standard-library-plan-builder, materialize-only-cli, manifest-linked-reports, traversal-safe-output-validation]

key-files:
  created:
    - src/evaluation/heldout.py
    - scripts/run_heldout_evaluation.py
    - tests/test_heldout_evaluation_harness.py
    - docs/evaluation_harness.md
  modified:
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
    - .planning/STATE.md

key-decisions:
  - "Keep the held-out harness materialize-only: it validates configs and writes reports but never runs FLUX generation, Qwen/PaddleOCR scoring, CUDA, or model weights."
  - "Require every target to link a source run manifest and require exactly one baseline plus at least one trained LoRA target for comparison-grade plans."
  - "Reject traversal and home expansion in writable evaluation paths so config-provided command strings cannot redirect planned outputs outside reviewed runtime roots."

patterns-established:
  - "Held-out plans use `heldout-evaluation-config/v1` input and `heldout-evaluation-plan/v1` output schemas with fixed prompts, seeds, inference settings, target manifests, and planned commands."
  - "CLI tests call `main(argv)` directly and use only pytest `tmp_path` JSON/JSONL fixtures, keeping default verification CPU-safe and artifact-free."
  - "Docs drift tests guard exported names, config field names, CLI flags, materialize-only behavior, SLURM templates, and generated-artifact safety language."

requirements-completed: [EVAL-03, RUN-05]

metrics:
  duration: 6min 57s
  completed: 2026-05-06T14:37:35Z
  tasks: 3
  files: 8
---

# Phase 6 Plan 02: Held-out Evaluation Harness Summary

**CPU-safe held-out evaluation plans now bind fixed prompts, fixed seeds, inference settings, baseline/trained LoRA targets, output paths, and source run manifests before expensive generation or scoring runs.**

## Performance

- **Duration:** 6 min 57 sec
- **Started:** 2026-05-06T14:30:38Z
- **Completed:** 2026-05-06T14:37:35Z
- **Tasks:** 3
- **Files modified:** 4 task files plus this summary and planning metadata

## Accomplishments

- Added `tests/test_heldout_evaluation_harness.py` with CPU-safe pytest fixtures for valid held-out configs, missing fixed prompt/seed/settings validation, CLI report materialization, unsafe output path rejection, and docs drift.
- Added `src/evaluation/heldout.py`, exporting `HeldoutEvaluationConfig`, `EvaluationTarget`, `build_evaluation_plan`, and `write_evaluation_plan` with deterministic JSON/Markdown output and no generation/scoring execution.
- Added `scripts/run_heldout_evaluation.py` as a thin CLI supporting `--config`, `--output-plan`, and `--markdown-summary`, returning nonzero for invalid configs.
- Published `docs/evaluation_harness.md` with exact config fields, manifest linkage, local and SLURM command templates, Phase 5 prerequisites, and generated-artifact safety guidance.
- Updated planning metadata so Phase 6 now records Plan 02 and requirements `EVAL-03`/`RUN-05` as complete.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -x` failed as expected with `ModuleNotFoundError: No module named 'src.evaluation.heldout'` after the initial config/target contract tests were written.
- **Task 2 RED:** The expanded CLI/materialization and output-path safety tests continued to fail on missing `src.evaluation.heldout` before implementation.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -q` passed with 12 tests after adding the plan builder and CLI.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -x` failed with `FileNotFoundError: docs/evaluation_harness.md` after docs drift assertions were added.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/heldout.py scripts/run_heldout_evaluation.py tests/test_heldout_evaluation_harness.py` passed with 13 tests and Ruff clean after docs and plan-owned lint fixes.

## Task Commits

1. **Task 1: Specify held-out evaluation config contracts** - `46ca045` (`test`)
2. **Task 2 RED: Add held-out CLI materialization tests** - `33c3cdf` (`test`)
3. **Task 2 RED: Require safe held-out output paths** - `c62e734` (`test`)
4. **Task 2 GREEN: Implement held-out plan builder and CLI** - `2188b5a` (`feat`)
5. **Task 3 RED: Add held-out evaluation docs drift check** - `03fa163` (`test`)
6. **Task 3 GREEN: Document held-out evaluation workflow** - `e7d32bc` (`docs`)

**Plan metadata:** committed separately after summary creation.

## Files Created/Modified

- `src/evaluation/heldout.py` - Pure held-out config/target validator, deterministic plan builder, Markdown renderer, atomic report writer, and planned command constructor.
- `scripts/run_heldout_evaluation.py` - Thin argparse CLI for validating and materializing held-out evaluation reports without running generation/scoring.
- `tests/test_heldout_evaluation_harness.py` - CPU-safe TDD coverage using `tmp_path` JSON/JSONL fixtures only.
- `docs/evaluation_harness.md` - User guide covering config schema, local/SLURM templates, Phase 5 prerequisites, manifest linkage, and artifact safety.
- `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md` - Updated Phase 6 Plan 02 and requirement progress.

## Decisions Made

- Used a standard-library-only held-out module instead of importing generation, reward, torch/CUDA, OCR, or image-processing code into the planning path.
- Validated source run manifests shallowly by reading local JSON and checking `run-manifest/v1`, `run_id`, and `command`, preserving RUN-05 links without inspecting generated artifacts.
- Planned generation/scoring commands with `argv`, shell-quoted `command`, and `status: planned-not-run` so reports are reproducible but execution remains explicit.
- Required output paths to stay under `output_root` and reject `..`/`~` for writable paths as the plan threat-model mitigation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed docs drift and Ruff issues in plan-owned files**
- **Found during:** Task 3 verification
- **Issue:** The docs drift test required exact contract phrases, and Ruff flagged line length/import issues in the newly created plan-owned Python/test files.
- **Fix:** Added exact docs language and wrapped/cleaned plan-owned Python/test code while preserving behavior.
- **Files modified:** `docs/evaluation_harness.md`, `src/evaluation/heldout.py`, `scripts/run_heldout_evaluation.py`, `tests/test_heldout_evaluation_harness.py`
- **Verification:** Targeted pytest plus Ruff command passed.
- **Committed in:** `e7d32bc`

---

**Total deviations:** 1 auto-fixed blocking verification issue.
**Impact on plan:** The fix was limited to plan-owned docs/lint cleanup and did not add new runtime scope or heavy dependencies.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found only normal optional defaults (`None`), empty local accumulators, and test assertions for invalid empty seed lists; no placeholder data or unwired mock behavior prevents the plan goal.

## Threat Flags

None. The new local config-to-command and manifest-to-report trust boundaries were covered by the plan threat model, and the implementation added validation for required keys, source manifests, `output_root` containment, and traversal/home-expansion rejection. No network endpoints, auth paths, schema migrations, or unplanned file-access surfaces were introduced.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -x` — RED failed as expected on missing `src.evaluation.heldout` before implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -q` — passed with 12 tests after implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -x` — RED failed as expected on missing `docs/evaluation_harness.md` before documentation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_heldout_evaluation_harness.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/heldout.py scripts/run_heldout_evaluation.py tests/test_heldout_evaluation_harness.py` — passed with 13 tests and Ruff clean.

## Deferred Issues

- Unrelated pre-existing dirty and untracked files remain in training modules, configs, data roots, thesis docs, scripts, loss helpers, and loss tests. They were present before execution and were left untouched/excluded from all Plan 06-02 commits.
- The GSD SDK CLI was unavailable in this checkout (`gsd-sdk: command not found` and no local `node_modules/@gsd-build/sdk`), so planning state files were updated directly instead of through SDK query handlers.
- No GPU/model/OCR diagnostics were run, per plan and user constraints.

## TDD Gate Compliance

- RED gate commits exist: `46ca045`, `33c3cdf`, `c62e734`, and `03fa163`.
- GREEN gate commits exist after RED commits: `2188b5a` for the harness/CLI and `e7d32bc` for docs.
- No separate refactor-only commit was needed; formatting and lint cleanup were included in Task 3 GREEN.

## Next Phase Readiness

- Phase 6 Plan 03 can build Russian text difficulty slices and gold diagnostic contracts on top of the held-out plan schema, target manifest links, fixed prompts, and score output paths.
- Phase 6 Plan 04 can later wire actual scoring outputs to the planned `score_output_path` values while preserving the materialize-only default harness.

## Self-Check: PASSED

- Found created/modified files: `src/evaluation/heldout.py`, `scripts/run_heldout_evaluation.py`, `tests/test_heldout_evaluation_harness.py`, `docs/evaluation_harness.md`, and this summary.
- Found task commits in git history: `46ca045`, `33c3cdf`, `c62e734`, `2188b5a`, `03fa163`, and `e7d32bc`.
- Required targeted pytest and Ruff verification commands passed.

---
*Phase: 06-reward-and-evaluation-validity*  
*Completed: 2026-05-06T14:37:35Z*
