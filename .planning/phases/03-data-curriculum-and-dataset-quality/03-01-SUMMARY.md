---
phase: 03-data-curriculum-and-dataset-quality
plan: 01
subsystem: prompt-curriculum
tags: [prompt-generation, curriculum, config-validation, cli, cpu-safe-tests]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: CPU-safe config/path/test conventions and generated-artifact safety boundaries
provides:
  - Frozen prompt curriculum config contracts in `src.data_quality.curriculum`
  - Committed simple, full, and curriculum prompt-generation config files
  - Config-driven `src.prompt_pipeline.generate --config` CLI with stage provenance
  - CPU-safe prompt curriculum tests and user documentation
affects: [phase-3-prompt-validation, phase-3-dataset-manifests, phase-3-runtime-docs]

tech-stack:
  added: []
  patterns: [frozen-dataclasses, config-first-cli, deterministic-stage-allocation, cpu-safe-validation]

key-files:
  created:
    - src/data_quality/__init__.py
    - src/data_quality/curriculum.py
    - configs/prompts/simple.json
    - configs/prompts/full.json
    - configs/prompts/curriculum.json
    - tests/test_prompt_curriculum.py
    - docs/data_curriculum.md
  modified:
    - src/prompt_pipeline/generate.py

key-decisions:
  - "Keep prompt curriculum validation dependency-light by using frozen dataclasses and existing prompt constants instead of model/runtime imports."
  - "Preserve legacy flag-only prompt generation while making config-driven simple/full/curriculum modes the explicit Phase 3 contract."
  - "Add `prompt_mode`, `curriculum_stage`, and `curriculum_family` only for config-driven generation so existing record shape remains compatible for legacy calls."

requirements-completed: [DATA-01, DATA-03]

duration: 8min
completed: 2026-05-04T15:18:51Z
---

# Phase 3 Plan 01: Prompt Curriculum Config Summary

**Explicit simple/full/curriculum prompt-generation configs with validated stage contracts and CLI provenance tags.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-04T15:11:17Z
- **Completed:** 2026-05-04T15:18:51Z
- **Tasks:** 3
- **Files modified:** 8 task files plus this summary and planning metadata

## Accomplishments

- Added `src.data_quality.curriculum` with frozen `PromptGenerationConfig`, `GenerationSettings`, and `CurriculumStage` dataclasses plus `load_prompt_generation_config` validation.
- Added deterministic stage sample allocation, safe repo-relative output path checks, bounded sample/batch validation, script/content/case/language validation, and mappings to existing prompt generator constants.
- Added `configs/prompts/simple.json`, `configs/prompts/full.json`, and `configs/prompts/curriculum.json` as committed source contracts for explicit prompt modes.
- Extended `src.prompt_pipeline.generate` with `--config`, early config errors before LLM imports, legacy flag-only compatibility, and config-stage provenance tags in generated records.
- Documented the config-driven workflow, local commands, stage provenance fields, and generated prompt artifact safety in `docs/data_curriculum.md`.

## RED/GREEN Evidence

- **Task 1 RED:** `uv run pytest tests/test_prompt_curriculum.py -q` failed with `ModuleNotFoundError: No module named 'src.data_quality.curriculum'` after adding config-contract tests only.
- **Task 1 GREEN:** `uv run pytest tests/test_prompt_curriculum.py -q` passed after adding `src/data_quality/curriculum.py` and exports.
- **Task 2 RED:** `uv run pytest tests/test_prompt_curriculum.py -q` failed because `configs/prompts/simple.json` did not exist after adding committed-config tests.
- **Task 2 GREEN:** `uv run pytest tests/test_prompt_curriculum.py -q` passed after adding simple, full, and curriculum config files.
- **Task 3 RED:** `uv run pytest tests/test_prompt_curriculum.py -q` failed because `generate.main(argv)` and `generate_dataset(..., prompt_config=...)` were not implemented.
- **Task 3 GREEN:** `uv run pytest tests/test_prompt_curriculum.py -q` passed after CLI/config wiring and record provenance tags were implemented.

## Task Commits

1. **Task 1 RED: Prompt curriculum contract tests** - `6d0b627` (`test`)
2. **Task 1 GREEN: Prompt curriculum config models** - `e98cbdb` (`feat`)
3. **Task 2 RED: Committed config file tests** - `b85754f` (`test`)
4. **Task 2 GREEN: Simple/full/curriculum configs** - `036a26a` (`feat`)
5. **Task 3 RED: Prompt CLI config wiring tests** - `1d55b00` (`test`)
6. **Task 3 GREEN: CLI config wiring and docs** - `c142d10` (`feat`)

## Files Created/Modified

- `src/data_quality/__init__.py` - Public exports for Phase 3 data-quality curriculum helpers.
- `src/data_quality/curriculum.py` - CPU-safe prompt-generation config validation, stage models, deterministic allocation, and prompt-generator constant mapping.
- `configs/prompts/simple.json` - Quick no-LLM letters/short-words prompt config.
- `configs/prompts/full.json` - Explicit broad prompt-generation config matching the existing full-distribution behavior.
- `configs/prompts/curriculum.json` - Named DATA-01 curriculum stages for single letters, short words, phrases, digits, punctuation, mixed case, multiline, style, and scene cases.
- `src/prompt_pipeline/generate.py` - Adds `--config`, early config loading/errors, stage-aware generation metadata, and legacy compatibility.
- `tests/test_prompt_curriculum.py` - CPU-safe RED/GREEN tests for config validation, committed config files, CLI behavior, and stage provenance.
- `docs/data_curriculum.md` - User-facing config-driven prompt generation workflow and artifact safety notes.

## Decisions Made

- Used frozen dataclasses rather than adding a new validation dependency because Phase 3 only needs a lightweight source contract and Phase 2 already established the CPU-safe validation style.
- Kept output paths repo-relative and under generated roots (`data/`, `outputs/`, or `runs/`) to satisfy the plan threat model without blocking existing generated prompt workflows.
- Left existing flag-only CLI defaults intact and applied config values only when `--config` is supplied; `--no-llm` remains an override so users can force CPU-safe generation from any config.
- Tagged generated records with curriculum provenance only for config-driven runs to preserve the historical prompt JSONL shape for legacy invocations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed Ruff import and line-length issues in new prompt curriculum files**
- **Found during:** Task 3 verification
- **Issue:** Ruff reported import ordering, line length, blind exception assertion, and `zip(strict=...)` issues across the new curriculum/test/CLI changes.
- **Fix:** Applied Ruff-compatible import ordering, wrapped long lines, asserted `FrozenInstanceError`, and made the LLM batch `zip` strict.
- **Files modified:** `src/data_quality/curriculum.py`, `src/prompt_pipeline/generate.py`, `tests/test_prompt_curriculum.py`
- **Commit:** `c142d10`

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no TODO/FIXME/placeholder/coming-soon text in files created or modified by this plan.

## Threat Flags

None. The plan threat model already covered user config to prompt CLI and prompt CLI to filesystem trust boundaries; mitigations were implemented through schema validation, path policy checks, bounded counts/batch sizes, and early errors before LLM imports.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_prompt_curriculum.py -q` — passed, 13 tests.
- `PATH="/root/.local/bin:$PATH" uv run python -m src.prompt_pipeline.generate --help` — passed and shows `--config` without importing or constructing heavy LLM/model/OCR runtimes.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/data_quality src/prompt_pipeline/generate.py tests/test_prompt_curriculum.py` — passed.
- `PATH="/root/.local/bin:$PATH" uv run pytest -q` — passed, 81 CPU-safe tests.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirements updates were therefore prepared manually.
- The worktree had unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## Next Plan Readiness

- Plan 03-02 can consume config-stage provenance fields (`prompt_mode`, `curriculum_stage`, `curriculum_family`) when validating prompt dataset quality and generating dataset manifests.
- Plan 03-06 can link `docs/data_curriculum.md` and add command aliases once the remaining Phase 3 implementation contracts exist.

## Self-Check: PASSED

- Found all created/modified task files: `src/data_quality/__init__.py`, `src/data_quality/curriculum.py`, `configs/prompts/simple.json`, `configs/prompts/full.json`, `configs/prompts/curriculum.json`, `src/prompt_pipeline/generate.py`, `tests/test_prompt_curriculum.py`, `docs/data_curriculum.md`, and this summary.
- Found task commits `6d0b627`, `e98cbdb`, `b85754f`, `036a26a`, `1d55b00`, and `c142d10` in git history.
- Required verification commands passed, and no generated prompt JSONL outputs were committed.

---
*Phase: 03-data-curriculum-and-dataset-quality*
*Completed: 2026-05-04T15:18:51Z*
