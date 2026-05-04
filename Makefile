.PHONY: setup test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr smoke-cache

setup:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

smoke-imports:
	python -m scripts.smoke_environment --check imports

smoke-cuda:
	python -m scripts.smoke_environment --check cuda --allow-missing

smoke-model-access:
	python -m scripts.smoke_environment --check model-access --allow-missing

smoke-ocr:
	python -m scripts.smoke_environment --check ocr --allow-missing

smoke-cache:
	python -m scripts.smoke_environment --check cache --allow-missing
