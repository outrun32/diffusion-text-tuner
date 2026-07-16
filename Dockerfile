# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS quality-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/workspace/.venv/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --yes --no-install-recommends git make shellcheck \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

FROM quality-base AS source

COPY . .

RUN test ! -e .env \
    && test ! -d .vscode \
    && test ! -d .copilot \
    && test ! -e experiments/assets/bad_text.png \
    && test ! -e data/prompts_simple.jsonl \
    && test -f reports/final/evidence_manifest.json \
    && test -f docs/project-page/assets/product_bias.webp

FROM quality-base AS dependencies

COPY pyproject.toml uv.lock README.md .python-version ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev \
        --extra test --extra lint \
        --prune torch --no-emit-project --no-hashes \
        --output-file /tmp/requirements.txt \
    && uv venv .venv \
    && uv pip install --python .venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0" \
    && uv pip install --python .venv/bin/python \
        --requirements /tmp/requirements.txt

FROM source AS quality

COPY --from=dependencies /workspace/.venv /workspace/.venv

RUN uv pip install --python .venv/bin/python --no-deps --editable .

ENV UV_NO_SYNC=1
RUN make check

CMD ["make", "check"]
