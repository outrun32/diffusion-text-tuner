# Command Catalog

This catalog is the standard execution surface for the Diffusion Text Tuner thesis toolkit. Default automation is CPU-safe; GPU, model-access, OCR, SLURM, and manual diagnostics are explicit opt-in commands.

## Setup

Use Python 3.11 and the committed uv project files for reproducible local tooling:

```bash
uv sync --group dev
```

Install optional dependency groups only for the workflows you intend to run:

```bash
uv sync --group dev --extra gpu
uv sync --group dev --extra ocr --extra reward
uv sync --group dev --extra synthesis
uv sync --group dev --extra plotting --extra analysis
```

For cluster environments that still use the historical conda bootstrap, run:

```bash
bash scripts/cluster/setup_env.sh
```

## CPU-safe default commands

These commands are safe defaults for local automation and broad verification. They do not intentionally launch CUDA, FLUX, Qwen, PaddleOCR, OCR, or SLURM work.

```bash
uv run pytest
uv run --extra lint ruff check scripts/smoke_environment.py tests
uv run --extra lint ruff format --check scripts/smoke_environment.py tests
```

The Ruff commands intentionally check the Phase 1 CPU-safe automation surface first. Broader repository lint cleanup is deferred until later structure/refactor phases so existing research scripts are not reformatted before their behavior is characterized.

## Smoke checks

Smoke checks are explicit preflight commands. The import check should remain lightweight; CUDA, model access, OCR, and cache checks are opt-in diagnostics before long jobs.

```bash
uv run python -m scripts.smoke_environment --check imports
uv run python -m scripts.smoke_environment --check cuda --allow-missing
uv run python -m scripts.smoke_environment --check model-access --allow-missing
uv run python -m scripts.smoke_environment --check ocr --allow-missing
uv run python -m scripts.smoke_environment --check cache --allow-missing
```

## Phase 4 CPU-safe characterization tests

Phase 4 publishes CPU-safe characterization commands that lock fragile behavior before
reward, trainer, prompt, dataset, or runtime code is moved. These tests are part of the
default pytest posture: default pytest does not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, or SynthTIGER,
and it does not download external model weights. Optional
slow/GPU/OCR/model/integration/manual diagnostics remain opt-in through explicit marker
commands such as `pytest -m slow`, `pytest -m gpu`, `pytest -m ocr`, `pytest -m model`,
`pytest -m integration`, and `pytest -m manual`.

Run the full Phase 4 characterization surface with either pytest or the Makefile alias:

```bash
uv run pytest tests/test_characterization_config_artifacts.py tests/test_training_dataset_contracts.py tests/test_training_objective_math.py tests/test_prompt_generation_determinism.py tests/test_reward_wrapper_contracts.py tests/test_characterization_docs.py
make characterization-test
```

Focused CPU-safe characterization groups are available when working on a narrower
boundary:

| Focus | Pytest command | Makefile alias |
|-------|----------------|----------------|
| config/artifact characterization | `uv run pytest tests/test_characterization_config_artifacts.py` | `make characterization-runtime` |
| dataset and collator characterization | `uv run pytest tests/test_training_dataset_contracts.py` | `make characterization-datasets` |
| objective math and DPO characterization | `uv run pytest tests/test_training_objective_math.py` | `make characterization-objectives` |
| prompt determinism characterization | `uv run pytest tests/test_prompt_generation_determinism.py` | `make characterization-prompts` |
| reward wrapper fake characterization | `uv run pytest tests/test_reward_wrapper_contracts.py` | `make characterization-rewards` |

Characterization fixtures must stay tiny and local to pytest. Prefer `tmp_path` for JSONL,
CSV, image, and tensor fixtures; inspect trusted local tensors only with
`torch.load(..., map_location="cpu", weights_only=True)`. generated artifacts and private prompts remain out of git unless they are intentionally tiny reviewed fixtures under an
allowed fixture/documentation path.

## Phase 5 training comparability

Phase 5 publishes CPU-safe comparison commands for controlled baseline, SFT, DPO,
masked-SFT, combined, and curriculum approach reviews. These commands read local
manifest/config JSON only; they do not launch training, CUDA, FLUX, Qwen,
PaddleOCR, OCR, model downloads, or generated image/tensor/checkpoint loading.
Write comparison outputs under ignored runtime roots such as `runs/comparisons/`.

Materialize comparison-grade selections before launching SFT or DPO runs:

```bash
uv run python scripts/materialize_training_data.py --kind sft --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --manifest outputs/generated/selected_samples.manifest.json
uv run python scripts/materialize_training_data.py --kind dpo --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --manifest outputs/generated/preference_pairs.manifest.json
```

Compare run manifests for config, data source, reward, seed, inference, metric,
artifact, and secret-safe environment differences:

```bash
python -m scripts.compare_run_manifests --left runs/<a>/manifest.json --right runs/<b>/manifest.json
```

Check controlled training comparability before interpreting approach differences:

```bash
python -m scripts.check_training_comparability --left-manifest runs/<a>/manifest.json --right-manifest runs/<b>/manifest.json
```

Generate one integrated Markdown report that includes both the manifest diff and
blocking/warning comparability mismatches:

```bash
python -m scripts.compare_training_runs --left-manifest runs/<a>/manifest.json --right-manifest runs/<b>/manifest.json --markdown --output runs/comparisons/training-run-comparison.md
make compare-training-runs LEFT_MANIFEST=runs/<a>/manifest.json RIGHT_MANIFEST=runs/<b>/manifest.json
```

Use the integrated `compare-training-runs` alias when reviewing baseline vs SFT,
baseline vs DPO, SFT vs masked-SFT, combined training, or curriculum ablations.
Blocking mismatches mean the runs are not controlled enough for thesis-grade
claims unless the mismatch is intentionally documented; warnings identify step,
metric, or artifact evidence that needs interpretation notes.

## Phase 6 reward/evaluation validity

Phase 6 publishes the reward/evaluation command surface for held-out plan materialization, canonical score files and product sidecars, reward diagnostics, gold diagnostic checks, thesis output bundles, and CPU-safe Phase 6 verification. GPU/model/OCR jobs remain explicit: the default docs and Makefile checks either inspect local metadata, validate score sidecars, materialize plans, or run pytest selections; they do not generate images, initialize Qwen/PaddleOCR, load CUDA, or load model weights.

Read the Phase 6 guides before using results as thesis evidence: [`docs/reward_evaluation.md`](reward_evaluation.md), [`docs/evaluation_harness.md`](evaluation_harness.md), [`docs/evaluation_diagnostics.md`](evaluation_diagnostics.md), and [`docs/thesis_outputs.md`](thesis_outputs.md).

### CPU-safe Phase 6 verification

Run the focused CPU-safe Phase 6 suite, including docs drift coverage, with:

```bash
PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_command_docs.py tests/test_evaluation_reward_interface.py tests/test_heldout_evaluation_harness.py tests/test_evaluation_slices_gold.py tests/test_evaluation_scoring_outputs.py tests/test_reward_diagnostics.py tests/test_thesis_outputs.py -q
make phase6-evaluation-tests
```

### Held-out plan materialization

Materialize a held-out comparison plan and Markdown review without running image generation or reward scoring:

```bash
python -m scripts.run_heldout_evaluation --config configs/experiments/evaluation/heldout_product_vs_baseline.json --output-plan runs/evaluation/heldout-001/plan.json --markdown-summary runs/evaluation/heldout-001/plan.md
make phase6-heldout-plan HELDOUT_EVAL_CONFIG=configs/experiments/evaluation/heldout_product_vs_baseline.json HELDOUT_EVAL_PLAN=runs/evaluation/heldout-001/plan.json HELDOUT_EVAL_PLAN_MD=runs/evaluation/heldout-001/plan.md
```

The plan records fixed prompts, fixed seeds, inference settings, baseline/trained target manifests, planned generation commands, and planned scoring commands with `status: planned-not-run`. Users launch any expensive generation/scoring work separately after reviewing the plan.

### Score validation and product sidecars

Canonical Phase 6 score files use `phase6-score-file/v1` or `phase6-score-jsonl/v1` rows plus sibling `.schema.json` sidecars with product formula, scorer versions, thresholds, and manifest links. A complete scoring command is explicit runtime work:

```bash
python -m scripts.score_images --images_dir outputs/generated/images --text_embeds_dir outputs/generated/text_embeds --output_csv outputs/generated/scores.csv --scorer both --manifest_path runs/scoring/manifest.json --source_manifest runs/generation/manifest.json --source_manifest runs/scoring/manifest.json
```

Validate recorded score rows and sidecars CPU-safely before diagnostics or thesis output generation:

```bash
python -c "from src.runtime.artifacts import validate_artifacts; report = validate_artifacts('evaluation_scores', {'scores_csv': 'outputs/generated/scores.csv'}); raise SystemExit(0 if report.ok else 1)"
make phase6-score-validation EVAL_SCORES_CSV=outputs/generated/scores.csv
```

### Reward diagnostics and gold checks

Analyze recorded score outputs against optional gold labels without invoking OCR/VLM/model code:

```bash
python scripts/analyze_reward_diagnostics.py --scores runs/eval/baseline/scores.csv --gold tests/fixtures/evaluation/gold_diagnostic.jsonl --output-report runs/eval/baseline/reward_diagnostics.json --markdown-summary runs/eval/baseline/reward_diagnostics.md --positive-threshold 0.80 --negative-threshold 0.50
make phase6-reward-diagnostics REWARD_DIAGNOSTIC_SCORES=runs/eval/baseline/scores.csv GOLD_DIAGNOSTIC_JSONL=tests/fixtures/evaluation/gold_diagnostic.jsonl REWARD_DIAGNOSTIC_REPORT=runs/eval/baseline/reward_diagnostics.json REWARD_DIAGNOSTIC_MD=runs/eval/baseline/reward_diagnostics.md
```

Run a focused gold diagnostic contract check when editing benchmark labels or prediction joins:

```bash
python -c "from src.evaluation.gold_benchmark import evaluate_gold_predictions; report = evaluate_gold_predictions('tests/fixtures/evaluation/gold_diagnostic.jsonl', []); raise SystemExit(0 if report['missing_prediction_count'] >= 0 else 1)"
make phase6-gold-diagnostics GOLD_DIAGNOSTIC_JSONL=tests/fixtures/evaluation/gold_diagnostic.jsonl
```

### Thesis output bundles

Build thesis-ready tables, SVG plots, bundle JSON, Markdown summaries, and optional bounded contact sheets from recorded manifests/reports only:

```bash
python scripts/build_thesis_outputs.py --config configs/thesis/eval_bundle.json --output-bundle outputs/thesis/eval_bundle/bundle.json --markdown-summary outputs/thesis/eval_bundle/bundle.md
make phase6-thesis-outputs THESIS_OUTPUT_CONFIG=configs/thesis/eval_bundle.json THESIS_OUTPUT_BUNDLE=outputs/thesis/eval_bundle/bundle.json THESIS_OUTPUT_MD=outputs/thesis/eval_bundle/bundle.md
```

generated score files, diagnostics, contact sheets, thesis bundles, plots, images, tensors, checkpoints, logs, and run outputs remain runtime artifacts. Keep them under ignored roots such as `runs/`, `outputs/`, or generated `data/` subtrees unless a future plan intentionally adds tiny reviewed fixtures.

The Makefile aliases above preserve the Phase 1-5 command surfaces and add dry-run discoverability for Phase 6:

```bash
make -n phase6-heldout-plan phase6-score-validation phase6-reward-diagnostics phase6-gold-diagnostics phase6-thesis-outputs phase6-evaluation-tests
```

## Phase 7 structure and extension seams

Phase 7 publishes the final CPU-safe structure and extension command surface.
Use the focused alias when editing repository-home docs, `src.toolkit.extension_points`,
or the importable generation, scoring, synthesis, plotting, run-comparison,
diagnostic, evaluation, and thesis-output seams:

```bash
PATH="/root/.local/bin:$PATH" uv run pytest tests/test_structure_extension_docs.py tests/test_generation_pipeline_contracts.py tests/test_scoring_pipeline_contracts.py tests/test_synthesis_pipeline_contracts.py tests/test_plotting_pipeline_contracts.py tests/test_extension_points_docs.py -q
make phase7-structure-tests
```

The `phase7-structure-tests` alias is CPU-safe: it runs docs/registry/command
drift tests and import-safety seam tests only. It does not launch FLUX, Qwen,
PaddleOCR, CUDA, SynthTIGER, model downloads, generation, scoring, training,
OCR, SLURM, or generated-artifact production.

## Runtime contracts

Read [`docs/runtime_contracts.md`](runtime_contracts.md) and [`configs/experiments/README.md`](../configs/experiments/README.md) before launching long-running generation, scoring, training, synthesis, or evaluation jobs. The runtime contract helpers are CPU-safe gates: they validate configs, manifests, and local artifact layout, but they do not launch CUDA, FLUX, Qwen, PaddleOCR, OCR, or SLURM work.

Use `scripts.run_manifest` to create and inspect local provenance under `runs/<run_id>/manifest.json`:

```bash
python -m scripts.run_manifest init --stage generate --command "python -m scripts.generate_images --prompts data/prompts_simple.jsonl --output_dir outputs/generated"
python -m scripts.run_manifest init --stage sft --config configs/sft.json --command "accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/sft.json"
python -m scripts.run_manifest init --stage evaluation --command "python -m src.evaluation.generate_baseline --prompts data/prompts_simple.jsonl --output-dir outputs/baseline"
python -m scripts.run_manifest inspect runs/<run_id>/manifest.json
python -m scripts.run_manifest note runs/<run_id>/manifest.json "Prepared inputs and verified runtime contracts before launch"
python -m scripts.run_manifest metrics runs/<run_id>/manifest.json --json '{"loss": 0.123}'
```

Training-stage manifests require `--config` so the runtime can validate and snapshot SFT/DPO/masked-SFT JSON. Non-training stages such as generation, scoring, synthesis, and evaluation can be initialized before a stage-specific config exists; pass `--config` when you have one and the raw JSON will be snapshotted for provenance.

Use `scripts.preflight_runtime` to validate readiness before expensive stages. These examples are safe to run locally; missing generated artifacts are reported as blockers instead of triggering model work.

```bash
python -m scripts.preflight_runtime --stage generate --prompts data/prompts_simple.jsonl --output-dir outputs/generated --json
python -m scripts.preflight_runtime --stage score --images-dir outputs/generated/images --text-embeds-dir outputs/generated/text_embeds --scores-csv outputs/generated/scores.csv --json
python -m scripts.preflight_runtime --stage sft --config configs/sft.json --manifest runs/<run_id>/manifest.json --json
python -m scripts.preflight_runtime --stage dpo --config configs/dpo.json --manifest runs/<run_id>/manifest.json --json
python -m scripts.preflight_runtime --stage masked-sft --config configs/masked_sft.json --manifest runs/<run_id>/manifest.json --json
python -m scripts.preflight_runtime --stage synthetic --output-dir data/synth_cyrillic/masked_sft --json
python -m scripts.preflight_runtime --stage evaluation --output-dir outputs/evaluation --manifest runs/<run_id>/manifest.json --json
```

The same surface is available through Makefile aliases for dry-run review:

```bash
make -n preflight-sft preflight-dpo preflight-masked-sft manifest-init-sft manifest-inspect
```

Generated artifacts remain non-committable by default. Keep generated images, tensors, scores, checkpoints, logs, and private manifests under ignored roots such as `outputs/`, `runs/`, and generated `data/` subtrees unless they are intentionally tiny reviewed fixtures.

## Phase 3 data curriculum and quality

Phase 3 adds a CPU-safe command surface for prompt curriculum configs, prompt dataset
validation, synthetic masked-SFT inspection, materialized training selections, and
generated-vs-synthetic source comparison. These commands are local and SLURM-compatible:
run them from a checkout or job workspace, keep paths repository-relative, and write outputs to
ignored runtime roots such as `runs/`, `outputs/`, or generated `data/` subtrees.
OCR/model-heavy checks are opt-in; default report, manifest, selection, and comparison commands do
not launch FLUX, Qwen, PaddleOCR, CUDA, SynthTIGER, OCR, or model inference.

Generate prompt datasets from explicit configs instead of monkey-patching prompt constants:

```bash
python -m src.prompt_pipeline.generate --config configs/prompts/simple.json --no-llm
python -m src.prompt_pipeline.generate --config configs/prompts/full.json
python -m src.prompt_pipeline.generate --config configs/prompts/curriculum.json
```

Validate prompt JSONL quality and write a `dataset-manifest/v1` manifest:

```bash
uv run python scripts/validate_prompt_dataset.py \
    --input data/prompts/curriculum.jsonl \
    --report runs/prompt-quality/prompt-quality.json \
    --manifest runs/prompt-quality/dataset-manifest.json \
    --config configs/prompts/curriculum.json \
    --strict-warnings
```

Inspect synthetic masked-SFT data with PIL/CSV/JSON only. Pass precomputed OCR evidence only when
you have produced it through a separate opt-in OCR diagnostic or reward workflow:

```bash
uv run python scripts/inspect_synthetic_dataset.py \
    --data-dir data/synth_cyrillic/masked_sft \
    --raw-dir data/synth_cyrillic/raw \
    --report runs/synthetic-quality/synthetic-quality.json \
    --manifest runs/synthetic-quality/dataset-manifest.json \
    --contact-sheet runs/synthetic-quality/contact-sheet.png
```

Materialize reward-filtered SFT samples and DPO preference pairs before comparison-grade training:

```bash
uv run python scripts/materialize_training_data.py --kind sft \
    --scores-csv outputs/generated/scores.csv \
    --output-dir outputs/generated \
    --manifest outputs/generated/selected_samples.manifest.json

uv run python scripts/materialize_training_data.py --kind dpo \
    --scores-csv outputs/generated/scores.csv \
    --output-dir outputs/generated \
    --manifest outputs/generated/preference_pairs.manifest.json
```

Compare generated reward-filtered evidence against synthetic masked-SFT evidence without opening
generated images or tensors:

```bash
uv run python scripts/compare_data_sources.py \
    --generated-prompt-quality-report runs/prompt-quality/prompt-quality.json \
    --selected-samples outputs/generated/selected_samples.jsonl \
    --preference-pairs outputs/generated/preference_pairs.jsonl \
    --generated-dataset-manifest outputs/generated/selected_samples.manifest.json \
    --synthetic-quality-report runs/synthetic-quality/synthetic-quality.json \
    --synthetic-manifest runs/synthetic-quality/dataset-manifest.json \
    --output-report runs/comparisons/generated-vs-synthetic.json \
    --markdown-summary runs/comparisons/generated-vs-synthetic.md
```

Generated artifacts remain private by default: generated reports, images, tensors, contact sheets, selections, and comparisons are runtime artifacts. Keep them out of git unless they are intentionally tiny reviewed fixtures or documentation assets.

## Local pipeline commands

Run these from the repository root after installing the appropriate optional dependencies and preparing required model/cache access. Commands write generated artifacts under ignored runtime roots such as `data/`, `outputs/`, and `runs/` unless you choose different paths.

### Prompt generation

```bash
python -m src.prompt_pipeline.generate \
    --n 1000 \
    --output data/prompts_simple.jsonl \
    --seed 42 \
    --no-llm
```

### Image generation

```bash
python -m scripts.generate_images \
    --prompts data/prompts_simple.jsonl \
    --output_dir outputs/generated \
    --versions_per_prompt 3 \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --resolution 512 \
    --seed 42
```

### Reward scoring

```bash
python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv \
    --scorer vlm \
    --resume
```

### SFT training

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.sft_trainer \
    --config configs/sft.json
```

### DPO training

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.dpo_trainer \
    --config configs/dpo.json
```

### Masked-SFT training

```bash
python -m src.training.masked_sft_trainer \
    --config configs/masked_sft.json
```

### Synthetic data

```bash
python -m scripts.synth.build_dataset \
    --num 1000 \
    --config configs/synth/cyrillic.yaml \
    --raw-dir data/synth_cyrillic/raw \
    --out-masked data/synth_cyrillic/masked_sft \
    --out-anyword data/synth_cyrillic/anyword \
    --seed 42
```

Bake latents and text embeddings only when GPU/model dependencies are available:

```bash
python -m scripts.synth.build_dataset \
    --num 1000 \
    --skip-render \
    --bake-latents \
    --encode-text \
    --device cuda
```

### Evaluation

```bash
python -m src.evaluation.generate_baseline \
    --prompts data/prompts_simple.jsonl \
    --output-dir outputs/baseline \
    --num-samples 500 \
    --seed 42

python -m src.evaluation.evaluate_rewards \
    --metadata outputs/baseline/metadata.jsonl \
    --output outputs/eval/rewards.jsonl \
    --reward vlm ocr
```

### Thesis plotting

```bash
python scripts/thesis_plots/plot_sft_losses.py
python scripts/thesis_plots/plot_dpo_metrics.py
```

## SLURM command variants

The cluster launchers mirror the local generation, scoring, SFT, and DPO stages while writing logs under ignored `runs/` paths. Review each `.sbatch` file for node constraints, prompt counts, config paths, and cache assumptions before submission.

```bash
bash scripts/cluster/setup_env.sh

sbatch --array=0-15 scripts/cluster/generate_images.sbatch

sbatch --array=0-7 scripts/cluster/score_images.sbatch
bash scripts/cluster/merge_scores.sh

sbatch scripts/cluster/sft.sbatch
sbatch scripts/cluster/dpo.sbatch
```

Current SLURM coverage is explicit rather than implied:

| Flow | SLURM status |
|------|--------------|
| Image generation | Wrapped by `scripts/cluster/generate_images.sbatch` |
| Reward scoring | Wrapped by `scripts/cluster/score_images.sbatch` plus `scripts/cluster/merge_scores.sh` |
| SFT training | Wrapped by `scripts/cluster/sft.sbatch` |
| DPO training | Wrapped by `scripts/cluster/dpo.sbatch` |
| Masked-SFT training | Local command documented; no committed SLURM wrapper yet |
| Synthetic data | Local command documented; no committed SLURM wrapper yet |
| Evaluation | Local command documented; no committed SLURM wrapper yet |
| Thesis plotting | Local command documented; no committed SLURM wrapper yet |

## Manual diagnostics

Manual diagnostics are not part of default pytest discovery. Run them only in an environment with the required CUDA, model, cache, and generated-artifact prerequisites.

```bash
python scripts/diagnose_gradient_flow.py
python scripts/diagnose_grad_magnitude.py
```

OCR/VLM experiments under `experiments/ocr_reward_tests/test_*.py` remain research scripts rather than CPU-safe tests. Do not add them to default test discovery.

## Generated artifacts and git safety

Keep generated outputs, runs, logs, tensors, checkpoints, and generated images out of git. In particular, do not commit artifacts from `outputs/`, `runs/`, generated `data/synth_cyrillic/` payloads, model caches, `.pt` tensors, checkpoint weights, or private prompt/output logs.

Before committing source or documentation changes, inspect the working tree and stage only intentional files:

```bash
git status --short
```

Generated artifacts may contain prompt text, target text, model outputs, local paths, and reward scores. Treat them as environment-specific research outputs unless they are intentionally tiny fixtures or documentation assets.
