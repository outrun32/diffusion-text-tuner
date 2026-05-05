.PHONY: setup test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr smoke-cache preflight-generate preflight-score preflight-sft preflight-dpo preflight-masked-sft manifest-init-sft manifest-inspect phase3-generate-prompts phase3-validate-prompts phase3-inspect-synthetic phase3-materialize-sft phase3-materialize-dpo phase3-compare-sources characterization-test characterization-runtime characterization-datasets characterization-objectives characterization-prompts characterization-rewards compare-training-runs

RUN_MANIFEST ?= runs/example/manifest.json
PROMPT_CONFIG ?= configs/prompts/curriculum.json
PROMPTS_JSONL ?= data/prompts/curriculum.jsonl
PROMPT_QUALITY_REPORT ?= runs/prompt-quality/prompt-quality.json
PROMPT_DATASET_MANIFEST ?= runs/prompt-quality/dataset-manifest.json
SYNTHETIC_DATA_DIR ?= data/synth_cyrillic/masked_sft
SYNTHETIC_RAW_DIR ?= data/synth_cyrillic/raw
SYNTHETIC_QUALITY_REPORT ?= runs/synthetic-quality/synthetic-quality.json
SYNTHETIC_DATASET_MANIFEST ?= runs/synthetic-quality/dataset-manifest.json
SYNTHETIC_CONTACT_SHEET ?= runs/synthetic-quality/contact-sheet.png
GENERATED_SCORES_CSV ?= outputs/generated/scores.csv
GENERATED_OUTPUT_DIR ?= outputs/generated
SELECTED_SAMPLES_MANIFEST ?= outputs/generated/selected_samples.manifest.json
PREFERENCE_PAIRS_MANIFEST ?= outputs/generated/preference_pairs.manifest.json
DATA_SOURCE_COMPARISON ?= runs/comparisons/generated-vs-synthetic.json
DATA_SOURCE_COMPARISON_MD ?= runs/comparisons/generated-vs-synthetic.md
LEFT_MANIFEST ?= runs/a/manifest.json
RIGHT_MANIFEST ?= runs/b/manifest.json
TRAINING_RUN_COMPARISON ?= runs/comparisons/training-run-comparison.md

setup:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run --extra lint ruff check scripts/smoke_environment.py tests

format:
	uv run --extra lint ruff format --check scripts/smoke_environment.py tests

smoke-imports:
	uv run python -m scripts.smoke_environment --check imports

smoke-cuda:
	uv run python -m scripts.smoke_environment --check cuda --allow-missing

smoke-model-access:
	uv run python -m scripts.smoke_environment --check model-access --allow-missing

smoke-ocr:
	uv run python -m scripts.smoke_environment --check ocr --allow-missing

smoke-cache:
	uv run python -m scripts.smoke_environment --check cache --allow-missing

preflight-generate:
	uv run python -m scripts.preflight_runtime --stage generate --prompts data/prompts_simple.jsonl --output-dir outputs/generated --json

preflight-score:
	uv run python -m scripts.preflight_runtime --stage score --images-dir outputs/generated/images --text-embeds-dir outputs/generated/text_embeds --scores-csv outputs/generated/scores.csv --json

preflight-sft:
	uv run python -m scripts.preflight_runtime --stage sft --config configs/sft.json --json

preflight-dpo:
	uv run python -m scripts.preflight_runtime --stage dpo --config configs/dpo.json --json

preflight-masked-sft:
	uv run python -m scripts.preflight_runtime --stage masked-sft --config configs/masked_sft.json --json

manifest-init-sft:
	uv run python -m scripts.run_manifest init --stage sft --config configs/sft.json --command "accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/sft.json"

# Placeholder form for docs/tests: python -m scripts.run_manifest inspect runs/<run_id>/manifest.json
manifest-inspect:
	uv run python -m scripts.run_manifest inspect $(RUN_MANIFEST)

phase3-generate-prompts:
	uv run python -m src.prompt_pipeline.generate --config $(PROMPT_CONFIG) --no-llm

phase3-validate-prompts:
	uv run python scripts/validate_prompt_dataset.py --input $(PROMPTS_JSONL) --report $(PROMPT_QUALITY_REPORT) --manifest $(PROMPT_DATASET_MANIFEST) --config $(PROMPT_CONFIG)

phase3-inspect-synthetic:
	uv run python scripts/inspect_synthetic_dataset.py --data-dir $(SYNTHETIC_DATA_DIR) --raw-dir $(SYNTHETIC_RAW_DIR) --report $(SYNTHETIC_QUALITY_REPORT) --manifest $(SYNTHETIC_DATASET_MANIFEST) --contact-sheet $(SYNTHETIC_CONTACT_SHEET)

phase3-materialize-sft:
	uv run python scripts/materialize_training_data.py --kind sft --scores-csv $(GENERATED_SCORES_CSV) --output-dir $(GENERATED_OUTPUT_DIR) --manifest $(SELECTED_SAMPLES_MANIFEST)

phase3-materialize-dpo:
	uv run python scripts/materialize_training_data.py --kind dpo --scores-csv $(GENERATED_SCORES_CSV) --output-dir $(GENERATED_OUTPUT_DIR) --manifest $(PREFERENCE_PAIRS_MANIFEST)

phase3-compare-sources:
	uv run python scripts/compare_data_sources.py --generated-prompt-quality-report $(PROMPT_QUALITY_REPORT) --selected-samples $(GENERATED_OUTPUT_DIR)/selected_samples.jsonl --preference-pairs $(GENERATED_OUTPUT_DIR)/preference_pairs.jsonl --generated-dataset-manifest $(SELECTED_SAMPLES_MANIFEST) --synthetic-quality-report $(SYNTHETIC_QUALITY_REPORT) --synthetic-manifest $(SYNTHETIC_DATASET_MANIFEST) --output-report $(DATA_SOURCE_COMPARISON) --markdown-summary $(DATA_SOURCE_COMPARISON_MD)

characterization-test:
	uv run pytest tests/test_characterization_config_artifacts.py tests/test_training_dataset_contracts.py tests/test_training_objective_math.py tests/test_prompt_generation_determinism.py tests/test_reward_wrapper_contracts.py tests/test_characterization_docs.py

characterization-runtime:
	uv run pytest tests/test_characterization_config_artifacts.py

characterization-datasets:
	uv run pytest tests/test_training_dataset_contracts.py

characterization-objectives:
	uv run pytest tests/test_training_objective_math.py

characterization-prompts:
	uv run pytest tests/test_prompt_generation_determinism.py

characterization-rewards:
	uv run pytest tests/test_reward_wrapper_contracts.py

compare-training-runs:
	uv run python -m scripts.compare_training_runs --left-manifest $(LEFT_MANIFEST) --right-manifest $(RIGHT_MANIFEST) --markdown --output $(TRAINING_RUN_COMPARISON)
