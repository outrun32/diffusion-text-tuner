.PHONY: setup test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr smoke-cache preflight-generate preflight-score preflight-sft preflight-dpo preflight-masked-sft manifest-init-sft manifest-inspect

RUN_MANIFEST ?= runs/example/manifest.json

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
