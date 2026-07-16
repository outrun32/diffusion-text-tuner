# Diffusion Text Tuner

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Reward-filtered alignment of FLUX.2 Klein for Russian and Cyrillic text rendering.

[Project page](https://outrun32.github.io/diffusion-text-tuner/project-page/) ·
[Prompt dataset](https://huggingface.co/datasets/Outrun32/cyrillic-prompts-15k) ·
[Public evidence](reports/final/README.md)

## Why this project exists

The work started with a client request: make an open-source image model generate supplied names and
short phrases inside images. English was the first target, but the available open models could not
render Russian text consistently. Even when the composition looked right, individual Cyrillic
letters were substituted, mixed with Latin homoglyphs, or dropped.

Training a LoRA with ordinary image-similarity MSE was a poor fit for this failure mode. Public
Russian text-image data was scarce, and learning nearly the whole Cyrillic alphabet from image
reconstruction alone would require far more clean data. The project therefore treats text rendering
as an alignment problem: generate several candidates, score the exact requested text, and train on
the candidates that the reward can defend.

The resulting bachelor thesis studies three routes around FLUX.2 Klein Base 4B:

- reward-filtered generated-output self-training with LoRA;
- DPO-style preference refinement over best/worst candidate pairs;
- synthetic masked-SFT tooling for explicit text-region supervision.

## Main finding

For the reported Russian benchmark, Product SFT had the lowest strict and normalized CER among the
three surviving aggregate rows. It reduced normalized character error rate from `0.859` to `0.126`.
DPO improved exact-match and VLM/product scores, but its normalized CER (`0.168`) remained worse than
Product SFT.

These are historical defense aggregates. The original per-sample score files, run manifests, and
checkpoint hashes are not in this repository, so the table cannot be recomputed from the checkout.
The exact status is recorded in [reports/final/README.md](reports/final/README.md).

| Run | Strict CER ↓ | Normalized CER ↓ | Strict exact ↑ | Normalized exact ↑ | OCR ↑ | VLM ↑ | Product ↑ | Script mix ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.984 | 0.859 | 33.3% | 41.7% | 0.560 | 0.658 | 0.392 | 23.3% |
| Product SFT | **0.238** | **0.126** | 40.8% | 50.0% | 0.635 | 0.681 | 0.444 | **16.7%** |
| Product DPO | 0.287 | 0.168 | **44.2%** | **52.5%** | **0.639** | **0.704** | **0.470** | 17.5% |

The interpretation is deliberately narrow. Product SFT is the recorded defense demo checkpoint with
the lowest CER; this table does not establish that product filtering wins for every prompt
distribution or language.

## Method

```text
prompt dataset
    ↓
several base-model candidates per prompt
    ↓
VLM exact-text score + OCR CER/entropy score
    ├── high product score ──→ LoRA self-training
    └── best/worst pair ─────→ DPO-style refinement
```

The SFT objective is standard flow-matching MSE on selected generated samples:

```text
L_SFT = ||v_θ(x_σ, t, c) - (ε - x₀)||²
```

The DPO objective uses the difference between policy and frozen-reference flow-matching errors. It
is a diffusion surrogate, not language-model DPO. Deterministic tests check its sign, margin, beta
schedule, and gradient direction.

## Reward definitions

The historical Product reward is now named `thesis_vlm_ocr_product_v1`:

```text
R_product = R_VLM × R_OCR
```

`scripts.score_images --product_formula thesis` reproduces that definition. Files created with
`--scorer vlm` use VLM as their primary `score`; OCR files use OCR; `--scorer both` uses the product.

A later five-component geometric formula also exists for diagnostics. It mixes VLM, OCR, CER
quality, entropy quality, and exact match. It must be selected explicitly with
`--product_formula diagnostic` and must not be compared to the thesis Product column as if they were
the same metric.

## Platform support

MLX is used for local text-prompt generation, not for FLUX diffusion training. The current
repository has no MLX implementation of the FLUX transformer, VAE training path, SFT, DPO, or ReFL.
MPS availability on a Mac does not change that boundary.

| Workflow | Apple Silicon Mac | Linux + CUDA |
| --- | --- | --- |
| Tests, lint, manifests, dataset checks | Supported | Supported |
| Algorithmic prompt generation | Supported | Supported |
| LLM-assisted prompt generation | MLX (`mlx-lm`) | Transformers or vLLM |
| Recorded-score analysis and plots | Supported | Supported |
| PaddleOCR scoring | CPU, optional environment | CPU or GPU |
| FLUX image generation / latent baking | Not supported | Supported |
| SFT, DPO, masked-SFT, ReFL | Not supported | Supported |
| PyTorch VLM reward | Not supported | CUDA only |

Runtime preflight now fails early on a Mac instead of starting a CUDA-only stage and crashing during
model loading.

## Setup on Apple Silicon

Python 3.11 is pinned by `.python-version` and `uv.lock`.

```bash
brew install uv shellcheck gitleaks actionlint
uv python install 3.11
uv sync --frozen --group dev --extra lint --extra mlx --extra plotting --extra analysis
make mac-check
```

Safe local commands:

```bash
# Generate a deterministic prompt sample without a model.
uv run python -m src.prompt_pipeline.generate \
  --config configs/prompts/simple.json \
  --no-llm

# Generate prompts with an MLX language model.
uv run python -m src.prompt_pipeline.generate \
  --n 100 \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --backend mlx \
  --seed 42 \
  --output runs/prompts/mlx-sample.jsonl

# Validate all CPU-safe behavior.
make check
```

The optional CPU quality image runs the same repository checks in Linux without claiming GPU
support:

```bash
make container-check
```

This image is for lint, tests, manifests, and evidence verification. FLUX generation and training
still require a separately provisioned Linux/CUDA host.

The MLX backend uses `mlx_lm.sample_utils.make_sampler`; the seed is passed into MLX so prompt runs
can be repeated with the same config.

The optional vLLM prompt backend has its own Linux environment because its pinned PyTorch stack is
incompatible with the trainer/reward extras. The exact command is documented in
[docs/commands.md](docs/commands.md); it is not part of the Mac setup.

## CUDA execution

The following clean-run path requires a Linux/CUDA machine. Create the manifest and run preflight
before long-running GPU/model work. The sequence downloads the immutable 15,000-row prompt source,
generates and scores candidates, freezes the Product selection, and only then starts SFT. The shell
variables retain the timestamped run-manifest paths printed by the manifest CLI.

```bash
uv sync --frozen --group dev --extra gpu --extra reward

uv run python -m scripts.download_dataset \
  --repo Outrun32/cyrillic-prompts-15k \
  --revision ecd8b2da9820b35afc65e2d56eaf37a662c37976 \
  --data-file data/train-00000-of-00001.parquet \
  --output data/prompts_simple.jsonl \
  --manifest runs/datasets/cyrillic-prompts-15k.manifest.json

GENERATION_RUN_DIR="$(uv run python -m scripts.run_manifest init \
  --stage generate \
  --run-root runs/generation \
  --command 'uv run python -m scripts.generate_images --prompts data/prompts_simple.jsonl --output_dir outputs/generated --model_id black-forest-labs/FLUX.2-klein-base-4B --model_revision a3b4f4849157f664bdbc776fd7453c2783562f4d --versions_per_prompt 3 --num_inference_steps 50 --guidance_scale 4.0 --resolution 512 --seed 42 --device cuda --manifest_path outputs/generated/manifest.json')"
GENERATION_RUN_MANIFEST="${GENERATION_RUN_DIR}/manifest.json"

uv run python -m scripts.preflight_runtime \
  --stage generate \
  --prompts data/prompts_simple.jsonl \
  --output-dir outputs/generated \
  --manifest "${GENERATION_RUN_MANIFEST}" \
  --json

uv run python -m scripts.generate_images \
  --prompts data/prompts_simple.jsonl \
  --output_dir outputs/generated \
  --model_id black-forest-labs/FLUX.2-klein-base-4B \
  --model_revision a3b4f4849157f664bdbc776fd7453c2783562f4d \
  --versions_per_prompt 3 \
  --num_inference_steps 50 \
  --guidance_scale 4.0 \
  --resolution 512 \
  --seed 42 \
  --device cuda \
  --manifest_path outputs/generated/manifest.json \
  --run_manifest_path "${GENERATION_RUN_MANIFEST}"

SCORING_RUN_DIR="$(uv run python -m scripts.run_manifest init \
  --stage score \
  --run-root runs/scoring \
  --command 'uv run python -m scripts.score_images --images_dir outputs/generated/images --text_embeds_dir outputs/generated/text_embeds --output_csv outputs/generated/scores_product.csv --scorer both --vlm_model_id Qwen/Qwen3.5-9B --vlm_model_revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --ocr_device cpu --product_formula thesis --source_manifest outputs/generated/manifest.json')"
SCORING_RUN_MANIFEST="${SCORING_RUN_DIR}/manifest.json"

uv run python -m scripts.preflight_runtime \
  --stage score \
  --images-dir outputs/generated/images \
  --text-embeds-dir outputs/generated/text_embeds \
  --scores-csv outputs/generated/scores_product.csv \
  --scorer both \
  --ocr-device cpu \
  --manifest "${SCORING_RUN_MANIFEST}" \
  --json

uv run python -m scripts.score_images \
  --images_dir outputs/generated/images \
  --text_embeds_dir outputs/generated/text_embeds \
  --output_csv outputs/generated/scores_product.csv \
  --scorer both \
  --vlm_model_id Qwen/Qwen3.5-9B \
  --vlm_model_revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --ocr_device cpu \
  --product_formula thesis \
  --manifest_path "${SCORING_RUN_MANIFEST}" \
  --source_manifest outputs/generated/manifest.json \
  --source_manifest "${GENERATION_RUN_MANIFEST}" \
  --source_manifest "${SCORING_RUN_MANIFEST}"

uv run python -m scripts.materialize_training_data \
  --kind sft \
  --scores-csv outputs/generated/scores_product.csv \
  --output outputs/generated/selected_samples.jsonl \
  --manifest outputs/generated/selected_samples.manifest.json \
  --mode threshold \
  --score-column score \
  --threshold 0.3

SFT_RUN_DIR="$(uv run python -m scripts.run_manifest init \
  --stage sft \
  --config configs/experiments/sft/sft_product_rerun_v2.json \
  --run-root runs/sft \
  --command 'uv run accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/experiments/sft/sft_product_rerun_v2.json')"
SFT_RUN_MANIFEST="${SFT_RUN_DIR}/manifest.json"

uv run python -m scripts.preflight_runtime \
  --stage sft \
  --config configs/experiments/sft/sft_product_rerun_v2.json \
  --manifest "${SFT_RUN_MANIFEST}" \
  --json

uv run accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  -m src.training.sft_trainer \
  --config configs/experiments/sft/sft_product_rerun_v2.json
```

On a SLURM cluster, run `scripts/cluster/setup_env.sh` on the networked login node. It downloads the
FLUX and Qwen snapshots at the revisions in `reports/final/current_model_sources.json`; the compute
jobs run with Hugging Face offline mode enabled. SFT and DPO launchers accept an explicit
`CONFIG_PATH`, and reject configs that omit `model_revision`.

The scorer writes `scores_product.schema.json`, while materialization records the exact score-file
hash in every selected row and in `selected_samples.manifest.json`. The SFT config rejects a
selection whose hash, threshold, mode, or score column no longer matches. The DPO and masked-SFT
trainers use the same preflight/manifest pattern; command variants are in
[docs/commands.md](docs/commands.md) and [docs/runtime_contracts.md](docs/runtime_contracts.md).

## Data and evaluation

The public prompt dataset contains 15,000 rows. Its pinned revision and parquet hash are recorded in
[prompt_dataset_source.manifest.json](reports/final/prompt_dataset_source.manifest.json).
Current model commits for future reruns are recorded separately in
[current_model_sources.json](reports/final/current_model_sources.json); the historical run revisions
remain unknown.

The old 120-prompt defense set contained duplicate and training-overlapping target strings. The new
[benchmark_prompts_v2.jsonl](reports/final/benchmark_prompts_v2.jsonl) has:

- 120 unique target strings;
- six 20-prompt difficulty slices;
- no exact target overlap with the pinned training prompt pool;
- a committed SHA-256 and source manifest.

It has no model scores yet. A valid comparison requires the original checkpoints or a new multi-seed
CUDA run. Per-seed score files can be joined with `scripts.aggregate_heldout_scores`.

The reported selection-length shift (`15 → 8` characters) is preserved separately in
[historical_selection_bias.json](reports/final/historical_selection_bias.json). It is a transcription
of an aggregate from the defense materials; the missing selection rows prevent recomputation.

## CPU-safe quality gates

```bash
make test
make lint
make format
make characterization-test
make compare-training-runs \
  LEFT_MANIFEST=runs/baseline/manifest.json \
  RIGHT_MANIFEST=runs/candidate/manifest.json
make evidence-verify
make security-current
make dependency-audit
make history-audit  # expected to fail until the old handoff export is removed from Git history
```

`compare-training-runs` has no meaningful zero-argument default: set both `LEFT_MANIFEST` and
`RIGHT_MANIFEST` to the runs being compared. It checks whether those runs are controlled enough to
interpret. The focused
characterization suite covers config/artifact contracts, dataset selection, objective math, prompt
determinism, and reward wrappers. GPU, model, OCR, integration, and manual diagnostics stay outside
default pytest discovery.

## Documentation map

- Runtime and experiment provenance: [docs/runtime_contracts.md](docs/runtime_contracts.md) and
  [configs/README.md](configs/README.md).
- Controlled run comparison: [docs/training_comparability.md](docs/training_comparability.md).
- Prompt curriculum and data checks: [docs/data_curriculum.md](docs/data_curriculum.md),
  [docs/dataset_quality.md](docs/dataset_quality.md),
  [docs/synthetic_quality.md](docs/synthetic_quality.md),
  [docs/data_selection.md](docs/data_selection.md), and
  [docs/data_source_comparison.md](docs/data_source_comparison.md).
- Reward and evaluation validity: [docs/reward_evaluation.md](docs/reward_evaluation.md),
  [docs/evaluation_harness.md](docs/evaluation_harness.md),
  [docs/evaluation_diagnostics.md](docs/evaluation_diagnostics.md), and
  [docs/thesis_outputs.md](docs/thesis_outputs.md).
- Repository boundaries and extension rules:
  [docs/structure_and_extension.md](docs/structure_and_extension.md).
- Supported wrappers and historical diagnostics: [scripts/README.md](scripts/README.md).
Generated artifacts, including reports, images, tensors, contact sheets, selections, comparisons,
checkpoints, logs, and
private run manifests remain out of git under `outputs/`, `runs/`, or ignored generated `data/`
subtrees. The reviewed files under `reports/final/`, `docs/project-page/assets/`, and `tests/fixtures/`
are deliberate small exceptions.

## Repository layout

```text
configs/                 experiment, prompt, accelerator, and evaluation configs
data/                    small source resources; generated datasets are ignored
docs/                    method, runtime, evaluation, and project-page documentation
experiments/             opt-in historical OCR/VLM probes, excluded from default tests
reports/final/           public evidence with hashes and explicit evidence status
scripts/                 supported CLI wrappers plus documented manual diagnostics
src/                     importable generation, scoring, training, runtime, and evaluation code
tests/                   CPU-safe default tests and tiny fixtures
```

## Personal contribution and AI tooling

This was an individual bachelor thesis. I was responsible for the problem statement, prompt/data
pipeline, reward design, FLUX LoRA training paths, experiments, evaluation, failure analysis, and
defense. AI coding agents were later used to inspect the repository, add tests, and clean up
documentation and runtime contracts. I reviewed those changes; the reported experimental decisions
and limitations remain my responsibility.

Some cleanup commits carry a machine-local `root` Git identity. They should not be read as a second
human contributor.

## Limitations

- The recorded evidence covers Russian/Cyrillic, not multilingual rendering in general.
- Product filtering favors shorter and easier samples.
- OCR and Qwen were used during selection and evaluation, so they are not independent judges.
- The historical table has one generated image per prompt and no confidence intervals.
- The public Product SFT/DPO checkpoints and raw benchmark rows are currently unavailable.
- DPO pairs were generated by the base model while the policy started from SFT, creating an
  off-policy preference set.

## Citation

```bibtex
@thesis{saparov2026cyrillic,
  title  = {Developing a Diffusion-based Training Toolkit for Multilingual Text Rendering},
  author = {Saparov, Iakov},
  school = {Innopolis University},
  year   = {2026}
}
```

## License

Code and documentation in this repository are released under the
[Apache License 2.0](LICENSE). Model weights and third-party datasets remain subject to their own
upstream licenses and access terms.
