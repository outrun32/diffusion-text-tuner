# Stack Research: Reproducible Python/PyTorch Diffusion Research Toolkit

**Project:** Diffusion Text Tuner  
**Domain:** brownfield thesis ML research toolkit for multilingual text rendering with diffusion fine-tuning  
**Question:** What is the standard 2025 stack/tooling pattern for a reproducible Python/PyTorch diffusion research toolkit?  
**Researched:** 2026-05-04  
**Overall confidence:** HIGH for tooling pattern; MEDIUM for exact ML package version pins until validated on the target CUDA/SLURM machines.

## Executive Recommendation

Use a **single `pyproject.toml` plus committed `uv.lock`** as the environment source of truth, with Python 3.11 retained because it is already the project baseline. Use **uv** for project sync/locking, **Ruff** for linting and formatting, **pytest** for fast automated tests, **Pydantic v2 + pydantic-settings** for typed config validation and environment overrides, and **local run manifests** for experiment metadata. Keep Conda only as an optional cluster compatibility wrapper if the SLURM environment requires CUDA modules or non-Python libraries; do not keep `pip install ...` README prose as the canonical dependency definition.

The brownfield priority is not to redesign the model stack. Keep the existing foundations: PyTorch, Diffusers, Transformers, Accelerate, PEFT, bitsandbytes, PaddleOCR/Qwen reward paths, vLLM/MLX optional prompt backends, SynthTIGER, and SLURM launch scripts. The reproducibility work should wrap these dependencies with lockfiles, grouped installs, config validation, smoke tests, and run metadata.

## Standard 2025 Pattern

| Concern | Recommendation | Confidence | Rationale |
|---|---|---:|---|
| Dependency source of truth | `pyproject.toml` with PEP 621 metadata, dependency groups, optional extras, and a committed `uv.lock` | HIGH | uv supports project dependencies, optional dependencies, dependency groups, PyTorch indexes, locking, and syncing. Python packaging docs identify `pyproject.toml` as the common config surface for dependencies and tools. |
| Environment creation | `uv sync` into `.venv`; optional `uv export` or cluster setup wrapper only when required | HIGH | Python virtual environments are disposable and should be recreated from manifests; uv makes recreate/sync fast and lock-aware. |
| GPU backend | Pin one known-good PyTorch/CUDA wheel family through uv PyTorch index configuration | MEDIUM | uv officially supports PyTorch accelerator indexes and explicit indexes. Exact CUDA tag must be chosen against cluster drivers and existing README CUDA 12.1 assumptions. |
| Dependency grouping | Base = core import/runtime; extras = optional capabilities; groups = local development/test/lint docs | HIGH | uv distinguishes `project.dependencies`, `project.optional-dependencies`, and `[dependency-groups]`; this matches a research repo with optional OCR/synthesis/vLLM/MLX paths. |
| Config validation | Pydantic v2 models for experiment configs; pydantic-settings for env/secrets/runtime overrides | HIGH | Pydantic is current, type-hinted, fast, emits JSON Schema, and pydantic-settings supports env, dotenv, CLI, nested env vars, and secrets. |
| Experiment config format | Keep existing JSON first; validate/load through Pydantic; generate JSON Schema; consider YAML/Hydra later only if config composition becomes painful | HIGH | Brownfield configs are already JSON. Pydantic validation gives safety without a disruptive Hydra migration. |
| Lint/format | Ruff only, configured in `pyproject.toml` | HIGH | Ruff is a fast linter and formatter with pyproject support and can replace Flake8, Black, isort, pyupgrade, and related plugins for this repo's needs. |
| Tests | pytest configured in `pyproject.toml`; mark `slow`, `gpu`, `model`, `ocr`, `integration`; default command runs CPU-only lightweight tests | HIGH | pytest supports pyproject configuration and custom markers; strict markers prevent accidental broad discovery. |
| Type checking | Add basedpyright or mypy as optional strictness, but do not block the first reproducibility phase on full ML typing | MEDIUM | Useful for config/data contracts, but ML code often requires gradual typing. Start with Pydantic schemas and Ruff. |
| Run metadata | Local `runs/<run_id>/manifest.json` plus frozen config snapshot, command, git metadata, lock hash, environment summary, artifacts index, and notes | HIGH | Matches project decision to defer MLflow/W&B while preserving rerun and comparison data. |
| External tracking | Defer MLflow/W&B until local manifests prove insufficient | HIGH | Project explicitly scopes immediate tracking to simple local manifests. External services add setup, credentials, and thesis overhead. |

## Recommended Stack

### Core Environment and Packaging

| Technology | Recommended Use | Version Strategy | Confidence | Why |
|---|---|---|---:|---|
| Python | Runtime language | `requires-python = ">=3.11,<3.12"` initially | HIGH | Existing repo documents Python 3.11. Narrowing to 3.11 reduces ML wheel ambiguity while reproducing thesis runs. |
| uv | Package manager, resolver, lockfile, environment sync | Pin only in docs/CI bootstrap; commit `uv.lock` | HIGH | uv supports dependency groups, optional dependencies, explicit PyTorch indexes, lock/sync, and fast recreation. |
| `pyproject.toml` | Project metadata, dependency groups, tool config | Commit as primary manifest | HIGH | Standard packaging/config surface; avoids scattered README-only dependencies. |
| `.python-version` | Local interpreter hint | `3.11` | HIGH | Keeps uv/editor/CI aligned with the brownfield runtime. |
| `.venv` | Disposable local environment | ignored | HIGH | Virtual environments are inherently non-portable; recreate from lockfile. |

### Core ML Runtime

| Package | Group | Version Strategy | Confidence | Why |
|---|---|---|---:|---|
| `torch`, `torchvision` | base or `gpu` extra depending install strategy | Pin exact known-good versions after smoke test; configure PyTorch index | MEDIUM | Core training/generation dependency. Exact versions depend on CUDA driver and FLUX/Accelerate compatibility. |
| `diffusers` | base | Pin exact known-good version | MEDIUM | FLUX.2 Klein pipeline APIs are high-risk if unpinned. |
| `transformers` | base | Pin exact known-good version | MEDIUM | Qwen chat/template/model APIs move quickly. |
| `accelerate` | base | Pin exact known-good version | HIGH | Existing training entry points use Accelerate and configs already exist. |
| `peft` | base | Pin exact known-good version | HIGH | LoRA adapter logic is central to SFT/DPO/masked-SFT. |
| `bitsandbytes` | `vlm`/`reward` extra or base if Qwen reward is standard | Pin exact known-good version with platform marker if needed | MEDIUM | 4-bit Qwen reward path depends on it; wheel compatibility can be fragile. |
| `datasets` | data extra or base | Pin major/minor | HIGH | Used for HF dataset download paths. |
| `safetensors`, `huggingface-hub` | base | Pin compatible versions | HIGH | Common dependency for HF model access and safe model artifact IO. |

### Optional Capability Groups

Use **extras** for installable runtime capabilities and **dependency groups** for local developer workflows.

| Capability | Install Selector | Include | Confidence | Rationale |
|---|---|---|---:|---|
| OCR/reward scoring | `--extra ocr` or `--extra reward` | `paddleocr`, `paddlepaddle`/GPU-specific Paddle package if needed | MEDIUM | Paddle packages can be platform/CUDA-sensitive; keep optional so CPU tests do not require them. |
| Qwen/VLM reward | `--extra vlm` or `--extra reward` | `bitsandbytes`, image/VLM helpers if not already transitive | MEDIUM | Heavy GPU dependency; separate from base test installs where possible. |
| Synthetic data | `--extra synth` | `synthtiger`, image/font/layout deps | MEDIUM | SynthTIGER may have native/system-font assumptions; isolate. |
| vLLM backend | `--extra vllm` | `vllm` | MEDIUM | vLLM has strict CUDA/PyTorch compatibility; should not be installed by default. |
| Apple local LLM backend | `--extra mlx` | `mlx-lm` with Darwin/arm64 marker | HIGH | MLX is only relevant on Apple Silicon paths. |
| notebooks/analysis | `--group analysis` | `ipykernel`, `pandas`, `matplotlib` if needed | MEDIUM | Useful for thesis analysis, but avoid forcing it into training environment. |
| tests | `--group test` | `pytest`, `pytest-cov`, optionally `pytest-xdist` | HIGH | Formalizes lightweight automated test gate. |
| lint/format | `--group lint` | `ruff` | HIGH | Single fast tool is sufficient initially. |
| dev aggregate | `--group dev` | include `lint`, `test`, optionally `analysis` | HIGH | One command for contributors. |

## Prescriptive Manifest Shape

Create `pyproject.toml` in the repository root. Keep it conservative and brownfield-compatible:

```toml
[project]
name = "diffusion-text-tuner"
version = "0.1.0"
description = "Thesis toolkit for diffusion-based multilingual text rendering experiments"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  # Pin exact versions after a known-good local/SLURM smoke test.
  "torch",
  "torchvision",
  "diffusers",
  "transformers",
  "accelerate",
  "peft",
  "datasets",
  "safetensors",
  "huggingface-hub",
  "pillow",
  "tqdm",
  "numpy",
  "pydantic>=2,<3",
  "pydantic-settings>=2,<3",
]

[project.optional-dependencies]
ocr = [
  "paddleocr",
  # Add the exact Paddle runtime package that matches the target CUDA/CPU environment.
]
reward = [
  "bitsandbytes",
  "paddleocr",
]
synth = [
  "synthtiger",
]
vllm = [
  "vllm",
]
mlx = [
  "mlx-lm; sys_platform == 'darwin' and platform_machine == 'arm64'",
]

[dependency-groups]
lint = ["ruff"]
test = ["pytest", "pytest-cov"]
dev = [
  { include-group = "lint" },
  { include-group = "test" },
]
analysis = ["ipykernel", "pandas", "matplotlib"]

[tool.uv]
# Keep default sync lightweight; install heavy optional stacks explicitly.
default-groups = ["dev"]

# Example only: choose the CUDA index that matches the validated cluster runtime.
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

**Important:** the PyTorch index above is illustrative. The implementation phase should run a GPU smoke test on the actual workstation/SLURM cluster and then pin exact `torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `peft`, `bitsandbytes`, `paddleocr`, and `vllm` versions. If the existing cluster is still tied to CUDA 12.1, use the matching PyTorch wheel/index/version combination instead of upgrading solely for tooling modernity.

## Install Commands to Standardize

Add these to README or a small `Makefile`/`justfile` after the manifest exists:

```bash
# Create/sync the default dev environment from lockfile
uv sync

# Sync GPU training/generation core with dev tools
uv sync --group dev

# Add OCR/reward capabilities only when needed
uv sync --extra ocr --extra reward --group dev

# Add synthetic dataset generation dependencies
uv sync --extra synth --group dev

# Add vLLM prompt backend only on compatible Linux CUDA machines
uv sync --extra vllm --group dev

# Run commands through the locked environment
uv run pytest
uv run ruff check .
uv run ruff format .
uv run accelerate env
```

For SLURM, keep `scripts/cluster/setup_env.sh` as a thin wrapper around the same lockfile, not a second dependency specification. It may load CUDA modules, set cache directories, and run `uv sync --frozen --extra ...`; it should not contain unpinned ad hoc `pip install` lines except as a temporary workaround documented in the manifest comments.

## Lint and Format Tooling

Use Ruff as the only initial lint/format tool:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100
extend-exclude = [
  "outputs",
  "runs",
  "data",
  ".venv",
]

[tool.ruff.lint]
select = [
  "E", "F", "I", "UP", "B", "SIM", "RUF",
]
ignore = [
  # Allow gradual cleanup in brownfield trainer scripts.
  "E501",
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

Rationale: Ruff provides pyproject configuration, linting, import sorting, pyupgrade-style checks, and formatting in one fast tool. This avoids introducing Black + isort + Flake8 + plugin sprawl into an ML thesis repo that currently has no tool config.

## Test Tooling

Use pytest with strict markers and make CPU-only tests the default. Expensive CUDA/model/OCR diagnostics must not be collected by default.

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
addopts = [
  "-ra",
  "--strict-markers",
]
markers = [
  "slow: long-running tests excluded from default local checks",
  "gpu: requires CUDA-capable GPU",
  "model: downloads or loads large model weights",
  "ocr: requires PaddleOCR/Paddle runtime",
  "integration: crosses filesystem/process boundaries",
]
```

Recommended commands:

```bash
# Default: fast CPU tests only
uv run pytest -m "not slow and not gpu and not model and not ocr"

# GPU smoke tests after environment changes
uv run pytest -m "gpu and not slow"

# Explicit expensive diagnostics only
uv run pytest -m "slow or model or ocr"
```

Immediate brownfield cleanup: rename or guard current expensive `test_*.py` diagnostics outside `tests/` so future pytest discovery cannot accidentally load FLUX/Qwen/Paddle models.

## Config Validation Pattern

Do **not** migrate the repo to Hydra in the first reproducibility milestone. Keep existing `configs/*.json`, but require all entry points to load them through Pydantic schemas.

Recommended structure:

```text
src/config/
  __init__.py
  base.py          # shared paths, seed, dtype, device, output/run settings
  training.py      # SFT/DPO/masked-SFT schemas
  generation.py    # FLUX generation schema
  scoring.py       # OCR/VLM reward schema
  synth.py         # SynthTIGER/dataset schema
  io.py            # load_json_config(), dump_schema(), snapshot_config()
```

Schema rules:

- Use `BaseModel` with `extra="forbid"` for experiment config files so misspelled keys fail fast.
- Use `Path` fields for inputs/outputs, but resolve paths relative to repo root or config file directory explicitly.
- Validate enumerations such as trainer type, dtype, scheduler, reward backend, and model IDs.
- Validate artifact requirements before expensive work starts: prompt files, latents, embeddings, masks, `shapes.csv`, checkpoint paths.
- Generate JSON Schema into `docs/config-schema/` or `.planning`-adjacent docs if useful for thesis documentation.
- Use `pydantic-settings` only for runtime/environment overlays: cache roots, Hugging Face token presence, SLURM job IDs, device visibility, debug flags. Do not bury experiment-defining hyperparameters in `.env`.

Example loader contract:

```python
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

class RunMetadataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    output_dir: Path
    run_name: str | None = None
    notes: str | None = None
```

Why Pydantic over Hydra now: Hydra is powerful for config composition and sweeps, but this repo first needs validation, schema clarity, and stable behavior for existing JSON configs. Pydantic gives that with low migration risk. Reconsider Hydra/OmegaConf later if experiment matrix composition becomes the main bottleneck.

## Run Metadata Standard

Implement a local manifest writer before adding external trackers. Every generation/scoring/training/evaluation command should create or update:

```text
runs/<YYYYMMDD-HHMMSS>-<slug>/
  manifest.json
  config.snapshot.json
  command.txt
  notes.md
  artifacts.json
  logs/
```

Minimum `manifest.json` fields:

| Field | Required | Why |
|---|---:|---|
| `run_id`, `run_name`, `created_at_utc` | yes | Stable identity for thesis comparisons. |
| `entrypoint`, `argv`, `cwd` | yes | Reconstructs invocation. |
| `git_commit`, `git_branch`, `git_dirty` | yes | Ties outputs to code state. |
| `python_version`, `platform`, `hostname` | yes | Captures environment context. |
| `uv_lock_hash` or `dependency_manifest_hash` | yes | Detects environment drift. |
| `torch_version`, `cuda_available`, `cuda_version`, `gpu_names` | yes for GPU commands | Critical for diffusion run reproducibility. |
| `accelerate_config_path`, `accelerate_env_summary` | training yes | Captures distributed/mixed-precision topology. |
| `config_path`, `config_snapshot_path`, `config_hash` | yes | Prevents later config edits from changing run meaning. |
| `input_artifacts`, `output_artifacts` | yes | Makes data lineage explicit without committing large files. |
| `hf_model_ids_or_paths` | model commands yes | Records gated/external model dependency. |
| `seed_values` | generation/training yes | Enables deterministic rerun attempts. |
| `status`, `started_at`, `finished_at`, `exception_summary` | yes | Supports failed-run diagnosis. |

Do not commit `runs/` contents except tiny examples/fixtures. The manifest format belongs in source control; actual run manifests remain ignored unless intentionally sanitized for documentation.

## Preflight and Smoke Commands

Add a lightweight preflight command after the manifest exists:

```bash
uv run python -m src.tools.preflight --config configs/sft.json --mode train
uv run python -m src.tools.preflight --config configs/masked_sft.json --mode masked-sft
uv run python -m src.tools.preflight --config configs/eval_suite.json --mode eval
```

Preflight should validate:

- Python version is 3.11.
- Required extras appear installed for the selected mode.
- CUDA visibility and bf16 support if training/generation requires GPU.
- Hugging Face token/access or local model path availability for FLUX/Qwen IDs.
- Accelerate config file exists and is parseable.
- Input artifact paths exist and expected manifest columns are present.
- Output/run directories are writable and ignored by git.

Add a GPU smoke test separate from unit tests:

```bash
uv run python -m src.tools.smoke_gpu --check torch --check diffusers --check accelerate
```

This should import key packages, print versions, allocate a tiny CUDA tensor, and optionally load only lightweight components. It should not download FLUX/Qwen by default.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Now | Confidence |
|---|---|---|---|---:|
| Package manager | uv | pip + requirements files | Requirements files can work, but dependency groups, lock/sync, PyTorch index config, and pyproject integration are cleaner in uv. | HIGH |
| Package manager | uv | Poetry | Poetry is capable, but uv has strong PyTorch/index guidance, speed, dependency groups, and low ceremony for research repos. | HIGH |
| Environment | uv `.venv` | Conda-only `environment.yml` | Conda can be useful for CUDA/system packages, but Python deps should still have a lockfile; conda-only often drifts across channels. | MEDIUM |
| Config | Pydantic + existing JSON | Hydra/OmegaConf now | Hydra is useful for sweeps/composition, but disruptive for brownfield JSON configs; validation is the immediate gap. | HIGH |
| Lint/format | Ruff | Black + isort + Flake8 | More moving parts for similar initial value. Ruff covers the needed baseline. | HIGH |
| Tests | pytest | unittest/custom scripts | pytest markers/config/fixtures are better for separating fast unit tests from expensive GPU diagnostics. | HIGH |
| Tracking | Local manifests | MLflow/W&B | External tracking is intentionally out of immediate scope and introduces credential/service overhead. | HIGH |
| Data versioning | Documented artifact manifests | DVC/Git LFS immediately | Helpful later, but first formalize local artifact paths and manifests; do not commit large artifacts. | MEDIUM |

## Roadmap Implications

1. **Manifest and lockfile first**
   - Add `pyproject.toml`, `.python-version`, `uv.lock`, Ruff config, pytest config.
   - Validate base import/test workflow before touching trainer internals.

2. **Environment groups second**
   - Split OCR, reward, synth, vLLM, and MLX dependencies into explicit extras.
   - Update README/cluster scripts to use `uv sync --frozen` selectors.

3. **Config validation third**
   - Introduce Pydantic schemas around existing JSON configs.
   - Add tests for config parsing and invalid/misspelled fields.

4. **Run metadata fourth**
   - Add run manifest writer and wire it into generation, scoring, training, and evaluation entry points.
   - Capture config snapshots and environment summaries.

5. **Smoke/preflight fifth**
   - Add CPU preflight tests and optional GPU smoke commands.
   - Use these before trainer/refactor work.

## Notable Risks and Mitigations

| Risk | Severity | Mitigation | Confidence |
|---|---:|---|---:|
| Exact PyTorch/CUDA/Paddle/vLLM wheels conflict on cluster | High | Choose one validated CUDA target, pin exact versions, keep vLLM/Paddle optional, run GPU smoke tests under SLURM. | HIGH |
| `uv lock` tries to resolve incompatible optional extras together | Medium | Use `tool.uv.conflicts` for mutually exclusive CPU/CUDA or backend extras; avoid installing all extras by default. | HIGH |
| Existing scripts rely on undeclared transitive dependencies | Medium | Add import smoke tests and preflight; migrate missing imports into the right base/extra group. | HIGH |
| Expensive diagnostics accidentally run during pytest | High | Register markers, strict markers, default testpaths only `tests`, rename/guard diagnostics outside `tests`. | HIGH |
| Config schema rejects legacy config quirks | Medium | Start with schemas that mirror current configs; add migration warnings before enforcing stricter semantics. | MEDIUM |
| Run manifests capture sensitive prompt text | Medium | Keep `runs/` ignored, document sanitization before sharing, store artifact paths/hashes rather than full private data where possible. | HIGH |

## Sources

| Source | Confidence | Notes |
|---|---:|---|
| uv dependency docs: https://docs.astral.sh/uv/concepts/projects/dependencies/ | HIGH | Documents `project.dependencies`, optional dependencies, dependency groups, `tool.uv.sources`, explicit indexes, and group behavior. |
| uv project config docs: https://docs.astral.sh/uv/concepts/projects/config/ | HIGH | Documents Python requirements, package behavior, build isolation, conflicts, and environment constraints. |
| uv PyTorch integration: https://docs.astral.sh/uv/guides/integration/pytorch/ | HIGH | Documents PyTorch accelerator indexes, explicit indexes, extra-based CPU/CUDA selectors, and `--torch-backend` for `uv pip`. |
| Python Packaging User Guide, pyproject: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ | HIGH | Confirms pyproject as dependency/tool config surface and PEP 621 metadata. |
| Python `venv` docs: https://docs.python.org/3/library/venv.html | HIGH | Confirms virtual environments are disposable and should be recreated from manifests. |
| Ruff docs: https://docs.astral.sh/ruff/ | HIGH | Confirms Ruff is a Python linter/formatter with pyproject support and can replace Flake8/Black/isort-style tooling. |
| pytest config docs: https://docs.pytest.org/en/stable/reference/customize.html | HIGH | Documents pytest pyproject support and root/config behavior. |
| pytest marker docs: https://docs.pytest.org/en/stable/how-to/mark.html | HIGH | Documents custom markers and strict marker validation. |
| Pydantic docs: https://docs.pydantic.dev/latest/ | HIGH | Documents Pydantic v2 validation, JSON Schema, strict/lax modes, type hints, and ecosystem. |
| pydantic-settings docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/ | HIGH | Documents env, dotenv, CLI, nested env vars, and secrets settings sources. |
| Hugging Face Accelerate install/config docs: https://huggingface.co/docs/accelerate/en/basic_tutorials/install | HIGH | Documents `accelerate config` and `accelerate env` for runtime topology/environment reporting. |

## Gaps Requiring Implementation-Time Validation

- Exact CUDA backend and PyTorch wheel target for the thesis machines. Existing project notes mention CUDA 12.1; current uv docs show newer CUDA index examples. Validate before pinning.
- Exact compatible versions of FLUX.2 Klein `diffusers`, Qwen `transformers`, PEFT, bitsandbytes, PaddleOCR/Paddle, and vLLM. These must be smoke-tested together rather than inferred.
- Whether SynthTIGER installs cleanly from PyPI in Python 3.11 on the target Linux environment or needs a Git/path/source pin.
- Whether the cluster allows uv-managed `.venv` environments directly or needs a Conda/module wrapper for system CUDA libraries and compilers.
- Whether future experiment sweeps justify Hydra/OmegaConf. Current recommendation is to defer until Pydantic-validated JSON becomes insufficient.
