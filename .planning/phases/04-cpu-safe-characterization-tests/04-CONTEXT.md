# Phase 4 Context: CPU-Safe Characterization Tests

**Phase:** 04 - CPU-Safe Characterization Tests  
**Prepared:** 2026-05-05  
**Mode:** Standard plan-phase, no optional flags

## Workflow Gate Results

1. **ROADMAP validation:** Phase 4 exists in `.planning/ROADMAP.md`, is the current unstarted phase, and lists requirements `TEST-01`, `TEST-02`, `TEST-03`, `TEST-04`, `TEST-05`, and `TRN-01`.
2. **Prior phases:** Phase 1, Phase 2, and Phase 3 are verified complete. Phase 4 can build on `pyproject.toml`, strict pytest markers, `src.runtime`, Phase 3 data-quality helpers, and the current 120-test CPU-safe suite.
3. **SDK fallback:** `gsd-sdk` is unavailable in this workspace; planning validation is performed manually against required frontmatter, task XML fields, requirement coverage, source audit, threat models, and wave/file-conflict checks.
4. **No discuss-phase CONTEXT:** There is no separate Phase 4 `/gsd-discuss-phase` artifact. Locked scope is derived from ROADMAP Phase 4, REQUIREMENTS.md, STATE.md risks, and prior verification notes.

## Locked Phase Scope

- Implement `TEST-01` through `TEST-05` and `TRN-01` exactly as mapped to Phase 4.
- Preserve Phase 4 goal: users can verify fragile behavior with fast, deterministic tests before reward, trainer, prompt, dataset, or runtime code is moved.
- Add CPU-safe tests for config parsing/validation and artifact path/shape contracts using tiny fixtures.
- Add CPU-safe tests for dataset loading, SFT/DPO/masked-SFT collators, pair/sample selection, and resolution bucket behavior using tiny fixtures.
- Add deterministic tests for masked losses, scheduler helpers, and DPO objective sign/beta/winner-loser semantics.
- Add deterministic prompt-generation tests under fixed seeds and explicit prompt configs without loading LLM, vLLM, MLX, FLUX, CUDA, Qwen, PaddleOCR, or SynthTIGER.
- Add reward-wrapper tests using fakes/mocks; default tests must not load Qwen, PaddleOCR, or external model weights.
- Keep optional slow/GPU/OCR/model/integration/manual checks explicit and outside default pytest execution.

## Existing Constraints To Preserve

- Default automated tests stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, SynthTIGER, or external model downloads.
- Generated images, tensors, checkpoints, logs, large prompt datasets, and runtime roots stay out of git unless they are intentionally tiny fixtures under `tests/fixtures/`.
- Preserve existing runnable training, scoring, prompt-generation, synthetic, and runtime CLIs while adding characterization coverage around their importable contracts.
- Prefer focused helper extraction over expanding large trainer modules.
- Treat DPO objective sign/beta scaling and winner/loser semantics as research-critical and explicitly tested before Phase 5 trainer comparability work.

## Phase Dependency Inputs

- Phase 2 provides strict config loading in `src.runtime.config_io`, artifact validation in `src.runtime.artifacts`, canonical paths in `src.runtime.paths`, local manifests, and CPU-safe preflight commands.
- Phase 3 provides prompt curriculum configs, prompt validation/manifests, synthetic quality helpers, materialized training selections, and source comparison tools.
- Current high-risk code paths include `src/training/dataset.py`, `src/training/losses.py`, `src/training/flux2_utils.py`, `src/training/dpo_trainer.py`, `src/training/refl_trainer.py`, `src/training/rewards.py`, and `src/prompt_pipeline/generate.py`.
- Current reward module imports Paddle-related internals at module import time; Phase 4 must make reward wrapper tests CPU-safe without requiring the optional OCR stack.

## Planning Decision Summary

- Create six executable plans across two waves.
- Wave 1 contains independent characterization slices: runtime config/artifacts, dataset/collator contracts, training objective math, prompt determinism, and reward wrapper fake tests.
- Wave 2 publishes the characterization command/docs surface after Wave 1 test files and helper contracts exist.
- All production-code changes are test-driven and narrowly scoped to make characterization possible without changing intended training/reward semantics.
