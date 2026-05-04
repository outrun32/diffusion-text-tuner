# Phase 3 Implementation Patterns

## Module Placement

- Use `src/data_quality/` for CPU-safe data-quality logic:
  - `curriculum.py` for prompt curriculum config dataclasses/Pydantic models and stage expansion.
  - `prompt_validation.py` for prompt JSONL metrics and threshold checks.
  - `manifests.py` for dataset manifest creation/loading/hash helpers.
  - `synthetic_quality.py` for masked-SFT metadata/image/mask quality reports and optional OCR result ingestion.
  - `source_comparison.py` for generated-vs-synthetic comparison reports.
- Keep existing producer entry points intact:
  - `src/prompt_pipeline/generate.py` remains the prompt-generation CLI.
  - `scripts/synth/build_dataset.py` remains the synthetic dataset builder.
  - `src.training.dataset` remains importable for existing trainers.
- Use thin CLI wrappers under `scripts/`:
  - `scripts/validate_prompt_dataset.py`
  - `scripts/inspect_synthetic_dataset.py`
  - `scripts/materialize_training_data.py`
  - `scripts/compare_data_sources.py`

## Config and Artifact Patterns

- Config files should live under `configs/prompts/` or `configs/experiments/<stage>/` and use repository-relative paths only.
- Dataset manifests should use a stable schema string such as `dataset-manifest/v1` and include:
  - dataset kind (`prompt`, `synthetic`, `selected_samples`, `preference_pairs`, `comparison`)
  - created timestamp
  - git commit/dirty state via `src.runtime.reproducibility.collect_git_state`
  - config path and content hash
  - seed strategy
  - source artifact paths and hashes
  - model IDs/revisions when a model-producing stage is involved
  - output counts and filtering stats
- Quality reports should be JSON and deterministic in key ordering so tests can compare fields.

## CPU-Safe Test Pattern

- Tests should use `tmp_path` for generated JSONL/CSV/image/tensor fixtures.
- Do not import `diffusers`, `transformers`, `paddleocr`, `vllm`, `mlx_lm`, or SynthTIGER in default tests.
- Use PIL-created tiny images/masks for synthetic quality tests.
- Use CSV/JSONL fixtures for prompt validation and selection materialization.
- Keep tests under `tests/` and run with `uv run pytest tests/test_*.py`.

## Validation Rules To Encode

- Prompt validation checks:
  - malformed JSONL lines
  - required fields (`id`, `prompt`, `target_text`, `content_type`, `style`, `lang` where required by mode)
  - target length min/max
  - allowed script/character policy
  - rare Cyrillic character coverage
  - duplicate target or duplicate `(target_text, content_type)` rate
  - content type distribution against expected ranges
  - style distribution for font/color/effect/size
  - malformed LLM artifacts such as empty strings, surrounding quotes-only outputs, instruction prose, or too many repeated tokens
- Synthetic quality checks:
  - missing raw image/mask/meta/index files
  - mask area fraction and bbox thresholds
  - contrast between text region and nearby background
  - character, font, and resolution distributions
  - optional OCR result match/CER thresholds from a supplied OCR file
  - contact sheet generation from sampled accepted/rejected examples
- Selection materialization checks:
  - SFT modes: threshold, top-k/top-1 per prompt, score column, source score paths
  - DPO modes: best-vs-worst, margin threshold, winner/loser semantics, ambiguity filter
  - schema fields for every selected sample/pair include IDs, versions, scores, mode, source hashes, and manifest link

## Documentation Pattern

- Update `docs/runtime_contracts.md` with any new artifact families and schema names.
- Update `docs/commands.md`, `Makefile`, and `README.md` only after CLIs exist.
- Document that reports/manifests may reference generated artifacts but generated artifacts remain non-committable.
