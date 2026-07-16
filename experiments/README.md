# Historical experiments

Files in this directory are opt-in research probes, not pytest tests and not supported production
commands. They may download OCR/VLM weights, use large amounts of memory, or depend on backend APIs
that change faster than the toolkit contracts.

- `ocr_reward_tests/` contains small Qwen, PaddleOCR, and TrOCR calibration probes over
  `experiments/assets/`.
- `legacy/profile_refl_step.py` is the original CUDA profiler. Launch it through
  `uv run python -m scripts.profile_step`; importing the wrapper does not load models.

The directory is excluded from default pytest and Ruff discovery. A probe becomes supported only
after reusable logic moves under `src/`, its dependencies are pinned, paths are repository-relative,
and a separate opt-in smoke command exists.

Do not add private customer images, chat exports, editor databases, model tokens, or large generated
outputs here. Images under `experiments/assets/` are reviewed fixtures for the historical OCR/VLM
notes and should stay small enough for source control.
