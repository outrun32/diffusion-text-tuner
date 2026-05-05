# Phase 4 Implementation Patterns

## Test Placement

- Add focused pytest files under `tests/`:
  - `tests/test_characterization_config_artifacts.py`
  - `tests/test_training_dataset_contracts.py`
  - `tests/test_training_objective_math.py`
  - `tests/test_prompt_generation_determinism.py`
  - `tests/test_reward_wrapper_contracts.py`
  - `tests/test_characterization_docs.py`
- Keep all fixtures tiny and generated inside `tmp_path` unless a committed fixture is intentionally needed under `tests/fixtures/`.
- Do not add tests outside the configured `tests/` discovery root.

## CPU-Safe Fixture Pattern

- Use `torch.save({"latent": tensor}, path)` and `torch.save({"prompt_embeds": tensor}, path)` for local tensor contracts.
- Use `torch.load(..., map_location="cpu", weights_only=True)` through production code paths; do not introduce pickle-backed custom objects.
- Use tiny PIL images only in `tmp_path` for image/report checks.
- Use JSONL/CSV text fixtures written in each test for prompt, score, and manifest contracts.

## TDD Pattern

- For production-code changes, write the characterization test first and run the targeted pytest file to confirm RED for the intended missing behavior.
- Implement the smallest code change that satisfies the contract.
- Re-run the targeted pytest file, then the relevant prior suite, then full `uv run pytest` when feasible.
- Keep RED/GREEN evidence in each plan summary.

## DPO Objective Pattern

- Put pure schedule/log-ratio/loss helpers in `src/training/dpo_objective.py` so tests do not need to import full trainer dependencies.
- Have `src/training/dpo_trainer.py` import and delegate to the pure helpers; preserve public trainer function names and training-loop behavior.
- Numeric tests must cover:
  - `compute_sigma` monotonicity and boundary behavior.
  - `time_dependent_beta` is non-positive and scales with `beta_conf` and timestep.
  - Winner-better-than-loser examples produce the expected objective direction under the existing negative beta convention.
  - Swapping winner/loser changes logits/accuracy accordingly.

## Reward Wrapper Pattern

- `src.training.rewards` must be importable without `paddleocr`, `paddlex`, Qwen, CUDA, or model weights installed.
- Optional OCR/Paddle imports and CTC monkey-patching belong inside `OcrCerEntropyReward.__init__` or an internal lazy setup helper.
- Tests should use fakes/mocks and `object.__new__` where needed; they must not call real `QwenYesProbReward.__init__` or real `OcrCerEntropyReward.__init__` unless optional dependencies are explicitly mocked.

## Documentation Pattern

- Update `docs/commands.md`, `docs/runtime_contracts.md`, `README.md`, and `Makefile` only after Wave 1 tests exist.
- Document Phase 4 characterization commands as CPU-safe and default-pytest-compatible.
- Keep slow/GPU/OCR/model/integration/manual markers explicit and opt-in.
