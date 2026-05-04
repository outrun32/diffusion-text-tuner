.PHONY: setup test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr smoke-cache

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
