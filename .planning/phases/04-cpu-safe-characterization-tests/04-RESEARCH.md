# Phase 4 Research: CPU-Safe Characterization Tests

**Phase:** 04 - CPU-Safe Characterization Tests  
**Researched:** 2026-05-05  
**Discovery level:** Level 1 quick verification plus project-specific code investigation. No new external libraries are required; this phase extends established pytest/PyTorch patterns.

## Sources Reviewed

- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/research/SUMMARY.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/TESTING.md`
- `.planning/codebase/CONVENTIONS.md`
- `.planning/codebase/CONCERNS.md`
- Phase 2 summaries for runtime config, artifacts, manifests, preflight, and docs
- Phase 3 context, research, patterns, validation, verification, and implemented file list
- `src/training/dataset.py`, `src/training/losses.py`, `src/training/flux2_utils.py`, `src/training/dpo_trainer.py`, `src/training/sft_trainer.py`, `src/training/refl_trainer.py`, `src/training/rewards.py`
- `src/prompt_pipeline/generate.py`, `src/data_quality/curriculum.py`, `src/data_quality/prompt_validation.py`
- `scripts/score_images.py`, `pyproject.toml`, existing tests under `tests/`

## Findings

### Runtime config and artifact tests

Phase 2 already introduced `tests/test_runtime_config_io.py`, `tests/test_runtime_artifacts.py`, and `tests/test_runtime_preflight.py`. Phase 4 should not duplicate them verbatim. The missing characterization layer is coverage that asserts committed training configs and representative tiny artifact layouts remain parseable/validated through the real public helpers before later refactors.

Recommended coverage: load `configs/sft.json`, `configs/dpo.json`, `configs/masked_sft.json`, selected root variants where compatible, and prompt configs; assert config validation errors remain field-specific and secret-safe; assert artifact validators handle tiny prompt/scores/generated/masked-SFT layouts without model loading.

### Dataset, collator, and selection contracts

`src/training/dataset.py` defines `SFTDataset`, `DPODataset`, `MaskedSFTDataset`, `ResolutionBucketSampler`, and collators. These are central to Phase 5 trainer comparability and currently rely on filesystem `.pt`, CSV, and shapes contracts. Tests should create tiny local tensor dictionaries with `torch.save` under `tmp_path`, use `weights_only=True` loading through the real dataset classes, and assert shape/padding/order behavior.

Phase 3 added `src/training/selection.py` and tests for materialized selected samples and DPO pairs. Phase 4 should cover the boundary between materialized selections and existing constructor behavior without forcing trainers to consume materialized artifacts yet.

### Objective math

`src/training/losses.py` already has tests for masked flow-matching loss and mask downsampling. Gaps remain around FLUX latent geometry utilities, scheduler helper behavior, and DPO objective sign/beta semantics.

`src/training/dpo_trainer.py` currently defines `compute_sigma`, `time_dependent_beta`, and `compute_dpo_loss` in the trainer module. This module imports Accelerate, PEFT, and trainer dependencies at import time, making pure math tests more coupled than needed. The safe pattern is to extract the pure DPO schedule/log-ratio/loss math into a small import-safe module and have `dpo_trainer.py` delegate to it.

### Prompt determinism

Phase 3 made prompt generation config-driven. The determinism-sensitive pieces are stage allocation, generation plan ordering, stage-family text policies, and seed-driven output records. Tests can avoid LLM/model loading by using `--no-llm`, explicit prompt configs, and monkeypatched/fake lightweight generator components where needed.

### Reward wrapper tests

`src/training/rewards.py` contains useful pure reward helpers (`_normalize_homoglyphs`, `_char_error_rate`, `_ctc_entropy_stats`) and wrapper classes (`QwenYesProbReward`, `OcrCerEntropyReward`). However, the OCR monkey patch imports `paddlex` internals at module import time. That violates Phase 4's default CPU-safe test posture in environments without OCR extras.

Recommended change: make `src.training.rewards` import-safe by moving Paddle/PaddleX imports and CTC monkey-patching behind OCR wrapper initialization. Then add fake-based tests for yes/no token aggregation, `score_batch` delegation, OCR CER/entropy formula behavior, and homoglyph normalization without loading Qwen or PaddleOCR.

## Recommended Architecture

- Keep all default tests under `tests/` and run them with `PATH="/root/.local/bin:$PATH" uv run pytest`.
- Use `tmp_path` fixtures for all artifact/tensor/image/CSV/JSONL data.
- Add small characterization test modules instead of a monolithic test file.
- Extract only pure math/import-safety helpers required for testability; do not reorganize trainer or reward packages beyond the narrow contracts under test.
- Keep docs updates in a final dependent plan so command guidance reflects the implemented characterization surface.

## Risks and Mitigations

| Risk | Mitigation in plans |
|------|---------------------|
| Default pytest loads optional heavy OCR/model stacks | Make reward module import-safe and test with fakes only. |
| DPO tests accidentally bless the wrong sign | Specify numeric examples where better winner policy loss lowers objective and `accuracy` reflects the configured negative beta convention. |
| Tests depend on generated artifacts | Use `tmp_path` and tiny tensors/images only. |
| Characterization tests become brittle implementation checks | Test observable contracts: loaded shapes, sorted samples/pairs, loss/logit direction, deterministic output, and import safety. |
| Later refactors ignore Phase 4 tests | Publish command/docs surface and docs drift tests in Wave 2. |

## Research Conclusion

Phase 4 needs no new external services or dependencies. The work is a focused test-driven characterization pass over existing runtime, dataset, prompt, reward, and trainer-objective boundaries so Phase 5 and Phase 6 can refactor and compare behavior with deterministic CPU-safe evidence.
