---
phase: 04-cpu-safe-characterization-tests
plan: 04
subsystem: prompt-generation-characterization
tags: [cpu-safe-tests, prompt-generation, determinism, curriculum, no-llm]

requires:
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Config-driven prompt curriculum files and prompt record provenance fields
  - phase: 04-cpu-safe-characterization-tests
    provides: Prior CPU-safe characterization test posture and import isolation fixes
provides:
  - Fixed-seed characterization tests for committed prompt config allocations and stage metadata
  - Deterministic generation-plan tests for config-driven and legacy prompt generation
  - Stage-family text-policy tests for letters, digits, punctuation, mixed case, and multiline text
  - No-LLM prompt-generation provenance tests using lightweight fakes instead of model backends
affects: [phase-4-characterization-tests, phase-5-trainer-comparability, prompt-data-generation]

tech-stack:
  added: []
  patterns: [pytest-fakes, fixed-seed-random-characterization, no-llm-import-safety]

key-files:
  created:
    - tests/test_prompt_generation_determinism.py
  modified: []

key-decisions:
  - "Keep prompt determinism coverage test-only because existing curriculum and generation code already satisfied the characterized contracts."
  - "Use lightweight fake prompt components for generated-record tests so no LLM, vLLM, MLX, Qwen, FLUX, CUDA, or model backend is imported."

patterns-established:
  - "Prompt generation determinism tests compare fixed plan signatures instead of generated runtime artifacts."
  - "Generated prompt record tests monkeypatch only local prompt component classes and write output under pytest tmp_path."

requirements-completed: [TEST-04]

metrics:
  duration: 4min
  completed: 2026-05-05T18:26:54Z
  tasks: 2
  files: 2
---

# Phase 4 Plan 04: Fixed-Seed Prompt Generation Determinism Summary

**CPU-safe fixed-seed tests now lock prompt curriculum allocation, generation-plan metadata, stage-family text policies, config-driven record provenance, and no-LLM import safety.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T18:22:39Z
- **Completed:** 2026-05-05T18:26:54Z
- **Tasks:** 2
- **Files modified:** 2 including this summary

## Accomplishments

- Added `tests/test_prompt_generation_determinism.py` with 325 lines of CPU-safe prompt determinism characterization.
- Verified committed `simple`, `full`, and `curriculum` prompt configs load deterministically and produce stable stage names, families, and sample allocations.
- Characterized `_build_generation_plan` under fixed seeds, materially different seeds, config-stage metadata, explicit fallback metadata, and legacy unconfigured metadata.
- Characterized `_apply_stage_text_policy` for single-letter, digit, punctuation, mixed-case, and multiline stage families under seeded RNGs.
- Used lightweight `TextGenerator`, `StyleGenerator`, `ScenePool`, and `Assembler` fakes to verify config-driven generated records include `prompt_mode`, `curriculum_stage`, and `curriculum_family` without invoking real model/data backends.
- Verified `--no-llm` config CLI behavior avoids importing `LLMClient`, `transformers`, `vllm`, `mlx_lm`, `diffusers`, or `torch` as newly loaded modules.

## RED/GREEN Evidence

- **Task 1 characterization:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_prompt_generation_determinism.py -x` passed with 4 tests after the allocation/plan-metadata tests were added. Existing implementation already satisfied the characterized contracts; no production code was changed.
- **Task 2 characterization:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_prompt_generation_determinism.py tests/test_prompt_curriculum.py -q` passed with 20 tests after adding text-policy, provenance-record, and no-LLM import-safety tests. Existing implementation already satisfied the characterized contracts; no production code was changed.

## Task Commits

1. **Task 1: Characterize curriculum allocation and stage metadata determinism** - `f0e8aff` (`test`)
2. **Task 2: Characterize stage-family text policies and config-driven record provenance** - `4dd4ce5` (`test`)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `tests/test_prompt_generation_determinism.py` - CPU-safe characterization tests for fixed-seed prompt config allocation, generation-plan signatures, fallback metadata, stage-family policies, generated record provenance, and no-LLM import safety.
- `.planning/phases/04-cpu-safe-characterization-tests/04-04-SUMMARY.md` - This execution summary.

## Decisions Made

- Kept this plan test-only because the existing Phase 3 prompt curriculum and generation implementation already met the determinism and provenance contracts under characterization.
- Used deterministic signatures of observable plan fields (`stage_name`, `stage_family`, `content_type`, `tier`, `case`, `lang`) rather than asserting private RNG internals.
- Wrote generated-record tests against `tmp_path` outputs and fake local components so default tests remain CPU-safe and model-backend-free.

## Deviations from Plan

None - plan executed exactly as written. No production fixes were required.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found only local test dictionaries/defaults, not placeholder data flows or unimplemented behavior.

## Threat Flags

None. The plan threat model covered prompt config to generation-plan and fake generator component to record-output boundaries; tests use committed configs, pytest `tmp_path`, fixed seeds, and no model/backend imports.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_prompt_generation_determinism.py -x` — passed, 4 Task 1 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_prompt_generation_determinism.py tests/test_prompt_curriculum.py -q` — passed, 20 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check tests/test_prompt_generation_determinism.py` — passed.

## Issues Encountered

- Initial uncommitted test expectations for curriculum allocation and seeded single-letter output were corrected before Task 1 was committed; the committed tests reflect the actual deterministic contracts.
- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirements updates were prepared manually.
- The worktree contains unrelated pre-existing dirty and untracked files in training, scripts, configs, docs, and data directories. They were left untouched and excluded from all commits.

## Deferred Issues

- No GPU/model/OCR diagnostics were run, per plan constraints.
- Reward wrapper import-safety remains deferred to Phase 4 Plan 05 as documented in Plan 04-03.

## Next Phase Readiness

- Plan 04-05 can add fake/mock reward-wrapper tests with prompt determinism now covered by fixed-seed CPU-safe tests.
- Phase 5 prompt/data comparability work can rely on characterization coverage for explicit configs, stage provenance fields, and no-LLM generation paths.

## Self-Check: PASSED

- Found `tests/test_prompt_generation_determinism.py` and this summary file.
- Found task commits `f0e8aff` and `4dd4ce5` in git history.
- Required verification commands passed, and no generated prompt JSONL outputs were committed.

---
*Phase: 04-cpu-safe-characterization-tests*  
*Completed: 2026-05-05T18:26:54Z*
