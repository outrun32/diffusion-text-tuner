# Structure and Extension Guide

This guide defines where reusable modules, wrappers, launchers, configs, diagnostics, experiments,
runtime outputs, tests, and thesis evidence belong in the public repository.

Use it with [`docs/pipeline_inventory.md`](pipeline_inventory.md),
[`docs/commands.md`](commands.md), and [`docs/runtime_contracts.md`](runtime_contracts.md).
Existing commands remain discoverable, generated artifacts stay out of Git, and extensions start
from stable homes instead of hidden one-off assumptions.

Run `make check` after changing repository boundaries or the `src.toolkit.extension_points`
registry. The default suite verifies import safety and documentation links without launching
GPU/model/OCR jobs or writing generated artifacts.

## Canonical repository homes

| Home | Classification | What belongs here | What does not belong here |
| --- | --- | --- | --- |
| Reusable source modules: `src/` | Source | Importable implementation modules for prompt generation, runtime contracts, data quality, training, scoring/reward, evaluation, plotting, and future pipeline seams. Prefer small focused helpers that can be imported by tests and thin wrappers without launching CUDA, model downloads, OCR, or SLURM work at import time. | One-off shell orchestration, generated outputs, private run data, checkpoint weights, ad-hoc notebooks, or model-cache paths. |
| Thin CLI wrappers: `scripts/` | Wrapper | Command-line wrappers that parse arguments, call importable modules, configure logging, and write explicit runtime outputs. Supported wrappers should be listed in `docs/commands.md` and stay thin enough that behavior can be tested through `src/` modules. | Large reusable implementations that should live under `src/`, generated images/tensors/scores, private logs, or hidden hardcoded run assumptions. |
| Cluster launchers: `scripts/cluster/` | Wrapper | SLURM `.sbatch` jobs, cluster setup helpers, merge helpers, and local-vs-cluster launch variants. These launchers may invoke GPU/model/OCR jobs, but they should point at documented configs and ignored runtime outputs. | Importable Python implementation logic, local-only diagnostics without cluster context, or generated cluster logs. |
| Synthesis helpers: `scripts/synth/` | Wrapper / synthesis helper | Synthetic masked-SFT rendering/build wrappers, SynthTIGER templates, fixture builders, background/font fetch helpers, and compatibility entry points that will progressively delegate to importable `src.synthesis` modules. | Generated synthetic images, masks, latents, text embeddings, large downloaded backgrounds/fonts, or private synthetic output reports. |
| Thesis plotting helpers: `scripts.plot_metrics` and `scripts.build_thesis_outputs` | Wrapper / plotting helper | Plotting CLIs consume recorded manifests, metrics, score files, diagnostics, or thesis bundle inputs and write figures under ignored output roots. Shared plotting logic lives in `src.plotting`. | Manually edited figures that cannot be traced to recorded inputs, raw generated plots committed by default, or plotting code embedded in trainer scripts. |
| Configuration contracts: `configs/` | Config | Root compatibility configs for supported flows, Accelerate topology YAML, prompt configs, thesis/evaluation configs, and family directories. Keep values reviewable, repository-relative, and free of secrets or personal absolute paths. | Runtime snapshots, generated manifests, private tokens, model caches, checkpoints, or run-specific outputs. |
| Experiment config variants: `configs/experiments/` | Config | New comparison-grade SFT, DPO, masked-SFT, reward/scoring, synthesis, evaluation, and ablation variants using the documented family naming contract. | Historical one-off outputs, generated reports, run manifests, or private environment-specific overrides. |
| CPU-safe tests: `tests/` | Test | Default pytest coverage, characterization tests, docs drift tests, tiny fixtures, and CPU-safe contract tests. Tests should not import CUDA/model/OCR stacks at collection time. | Manual diagnostics, GPU/OCR/model jobs, large generated fixtures, or expensive integration checks that should be explicit opt-in commands. |
| Historical experiments: `experiments/` | Diagnostic / experiment | One-off research probes, OCR/VLM reward experiments, and tiny reviewed assets that document historical reasoning without becoming default automation. | Supported toolkit entry points, default pytest tests, generated large artifacts, or source modules required by production wrappers. |
| Generated runtime outputs: ignored `outputs/`, `runs/`, and generated `data/` subtrees | Generated runtime output | Images, tensors, checkpoints, logs, reports, contact sheets, score files, local run manifests, synthetic payloads, private prompt/output records, and stage outputs produced by local or SLURM jobs. | Source code, canonical docs, committed configs, default tests, or anything that should be required to import the project. |
| Thesis evidence artifacts: recorded manifests, reports, plots, contact sheets, and bundles | Thesis evidence | Reproducible evidence produced from recorded runs and traced through manifests/configs/seeds/rewards. Keep generated thesis outputs under ignored roots such as `outputs/thesis/` or `runs/`, except for tiny reviewed fixtures. | Static/manual numbers without provenance, private full outputs committed by default, or claims that cannot be tied back to exact runs and configs. |

Across every repository home, generated images, tensors, checkpoints, logs, reports, contact sheets,
thesis bundles, and private run outputs remain runtime artifacts. Do not commit generated runtime
outputs unless they are intentionally tiny reviewed fixtures or documentation assets.

## Supported wrappers, diagnostics, and generated-output boundaries

- Supported wrappers are commands documented in `docs/commands.md` or `docs/pipeline_inventory.md`; they should call importable implementation modules wherever practical and write outputs only to explicit paths.
- Manual diagnostics are opt-in investigation tools. Keep them outside default pytest discovery and document GPU/model/OCR/generated-artifact prerequisites when they are relevant.
- Historical experiments preserve research context but should not become hidden dependencies for supported flows. Promote reusable logic into `src/` first, then leave the experiment as a thin historical probe or remove it in an explicit cleanup change.
- Generated runtime outputs are not source. They may contain prompt text, target text, model outputs, reward scores, local paths, run metadata, or thesis draft evidence, so keep them in ignored runtime roots and stage only source, docs, configs, tests, and approved tiny fixtures.

## Extension rules before adding new work

Before adding a new experiment, trainer, reward variant, dataset, pipeline, plot, or thesis-output step, use these rules:

1. Start from the nearest existing documented command in `docs/commands.md` and the inventory in `docs/pipeline_inventory.md`.
2. Put reusable behavior in an importable module under `src/`; prefer importable implementation modules under `src/` with thin wrappers in `scripts/`.
3. Keep CLI wrappers responsible for argument parsing, logging setup, and delegating to importable code. Avoid burying reusable behavior in shell or script-only globals.
4. Add or update CPU-safe tests under `tests/` before relying on behavior for refactors or thesis claims.
5. Add config variants under `configs/experiments/` when the work changes an experimental choice, reward family, data source, evaluation specification, or ablation rather than editing root compatibility configs in place.
6. For every public entry point, document CPU-safe verification and explicit GPU/model/OCR/SLURM prerequisites in `docs/commands.md`, a focused guide, or the wrapper help text.
7. Write generated artifacts to ignored `outputs/`, `runs/`, or generated `data/` roots and record provenance through manifests or sidecars before using them as thesis evidence.

| Extension type | Home for reusable logic | Wrapper/config home | Required navigation update |
| --- | --- | --- | --- |
| New experiment | Promote reusable pieces to `src/`; keep historical probes in `experiments/` only when they are opt-in. | Add reviewed config variants under `configs/experiments/` and document commands if they become repeatable. | Update `docs/pipeline_inventory.md` if it changes supported or historical tracks. |
| New trainer | `src/training/` for config/runtime/objective/data helpers and trainer orchestration. | Root compatibility config only when it is a supported default; otherwise use `configs/experiments/<family>/`. | Update training comparability and command docs before interpreting runs. |
| New reward variant | Shared reward/scoring/evaluation logic under `src/training/`, `src/scoring/`, or `src/evaluation/`. | Thin scorer wrapper in `scripts/` and reward config under `configs/experiments/reward/`. | Update reward/evaluation docs and score sidecar expectations if output schema changes. |
| New dataset or data pipeline | Importable dataset, quality, manifest, or synthesis helpers under `src/data_quality/`, `src/training/`, or future `src/synthesis/`. | Explicit CLI wrapper plus config under `configs/experiments/synthesis/` or the relevant data family. | Update dataset quality/runtime docs and keep generated datasets out of git. |
| New generation, scoring, synthesis, evaluation, or plotting pipeline | Importable pipeline module under `src/generation/`, `src/scoring/`, `src/synthesis/`, `src/evaluation/`, or `src/plotting/`. | Thin wrapper under `scripts/` or `scripts/synth/`; cluster variant under `scripts/cluster/` when needed. | Add CPU-safe tests and exact command docs before considering the pipeline supported. |
| New thesis-output step | `src/evaluation/` or `src/plotting/` for reproducible table/plot/bundle builders. | Wrapper under `scripts/`; reviewed config under `configs/experiments/evaluation/` when the required evidence exists. | Link outputs to manifests/reports and keep thesis bundles under ignored runtime roots. |

## Extension-point registry

The importable registry in `src/toolkit/extension_points.py` exposes `list_extension_points()` and `get_extension_point(name)`. The table below mirrors the registry entries so users can navigate prompt generation, image generation, scoring, synthesis, training, evaluation, plotting, run comparison, diagnostics, and thesis outputs from one index.

| Stage | Purpose | Implementation module | Thin script / command | Config home | Docs path | Test target | Generated-artifact notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prompt generation | Create multilingual/Cyrillic text-rendering prompt JSONL datasets from config-driven curricula. | `src.prompt_pipeline.generate` | `uv run python -m src.prompt_pipeline.generate` | `configs/prompts/` | `docs/data_curriculum.md` | `tests/test_prompt_curriculum.py` | Prompt JSONL outputs are generated data; keep full datasets under ignored data/runs roots unless intentionally tiny fixtures. |
| image generation | Generate FLUX images, latents, and text embeddings through an importable orchestration seam. | `src.generation.pipeline` | `scripts/generate_images.py` | `configs/experiments/evaluation/` for held-out suites; direct generation uses runtime arguments. | `docs/commands.md` | `tests/test_generation_pipeline_contracts.py` | Generated images, latents, embeddings, metadata, and manifests stay in ignored outputs/ or runs/ roots. |
| scoring | Score generated images and write canonical reward rows, sidecars, and scorer provenance. | `src.scoring.pipeline` | `scripts/score_images.py` | `configs/experiments/reward/` | `docs/reward_evaluation.md` | `tests/test_scoring_pipeline_contracts.py` | Score CSV/JSONL files and sidecars may contain prompt text, local paths, and reward evidence; keep runtime score outputs ignored by default. |
| synthesis | Build synthetic masked-SFT datasets through reusable render, collation, latent, and text-encoding steps. | `src.synthesis.dataset_builder` | `scripts/synth/build_dataset.py` | `configs/experiments/synthesis/` | `docs/synthetic_quality.md` | `tests/test_synthesis_pipeline_contracts.py` | Synthetic images, masks, latents, embeddings, indexes, and quality reports remain generated artifacts under ignored data/ or runs/ roots. |
| training | Extend SFT, DPO, masked-SFT, runtime metadata, objective helpers, sampling, schedulers, and checkpointing without hiding trainer choices. | `src.training.runtime` | `src.training.sft_trainer; src.training.dpo_trainer; src.training.masked_sft_trainer` (full `uv run` commands are in `docs/commands.md`) | `configs/experiments/` (`sft/`, `dpo/`, and `masked_sft/`) | `docs/training_comparability.md` | `tests/test_training_shared_utilities.py` | Checkpoints, logs, sample images, tensors, and run manifests are runtime artifacts; record provenance but do not commit generated outputs. |
| evaluation | Materialize held-out evaluation plans and validate recorded reward/evaluation evidence without launching expensive work by default. | `src.evaluation.heldout` | `scripts/run_heldout_evaluation.py` | `configs/experiments/evaluation/` | `docs/evaluation_harness.md` | `tests/test_heldout_evaluation_harness.py` | Held-out plans, generated eval outputs, scores, and manifests belong under ignored runs/ or outputs/ roots. |
| plotting | Load, summarize, smooth, and plot training metrics through import-safe helpers with lazy plotting backends. | `src.plotting.training_metrics` | `scripts/plot_metrics.py` | `runtime metric CSV paths and configs/experiments/ metadata` | `docs/commands.md` | `tests/test_plotting_pipeline_contracts.py` | Training curves and thesis plots are generated artifacts; keep figures under ignored run/output roots unless reviewed as tiny docs assets. |
| run comparison | Compare local run manifests and controlled training settings before interpreting experiments as thesis evidence. | `src.runtime.manifest_diff` | `scripts/compare_training_runs.py` | `runs/<run_id>/manifest.json and configs/experiments/` | `docs/training_comparability.md` | `tests/test_runtime_manifest_diff.py` | Comparison Markdown/JSON reports should be written under ignored runs/comparisons/ unless intentionally tiny reviewed evidence. |
| diagnostics | Inspect reward disagreements, gold slices, gradients, and manual checks through explicit opt-in diagnostics outside default heavy jobs. | `src.evaluation.diagnostics` | `scripts/analyze_reward_diagnostics.py and scripts/diagnose_*.py` | `configs/experiments/evaluation/ and explicit diagnostic arguments` | `docs/evaluation_diagnostics.md` | `tests/test_reward_diagnostics.py` | Diagnostic reports, contact sheets, local paths, prompt text, scores, and logs are runtime/private artifacts by default. |
| thesis outputs | Build thesis-ready tables, SVG plots, contact sheets, bundles, and Markdown summaries from recorded manifests and reports. | `src.evaluation.thesis_outputs` | `scripts/build_thesis_outputs.py` | `configs/experiments/evaluation/`; pass a runtime evidence-bundle config to the CLI | `docs/thesis_outputs.md` | `tests/test_thesis_outputs.py` | Thesis tables, plots, bundles, and contact sheets stay in ignored `outputs/thesis/` or `runs/` roots unless a tiny fixture is reviewed. |

## Extension checklist

Use this checklist before adding future experiments or pipelines:

1. **config** — Add a reviewed root compatibility config or a variant under the appropriate `configs/experiments/` family instead of editing hidden constants.
2. **artifact/manifest contract** — Define the input/output schema, sidecar, run manifest, and provenance fields before relying on generated evidence.
3. **importable module** — Put reusable implementation logic under `src/` and keep import-time behavior CPU-safe.
4. **thin CLI wrapper** — Keep `scripts/`, `scripts/synth/`, or cluster wrappers focused on argument parsing, logging, and delegation.
5. **CPU-safe tests** — Add or update focused tests under `tests/` that avoid FLUX, Qwen, PaddleOCR, CUDA, model downloads, SynthTIGER, and generated artifacts by default.
6. **command docs** — Document exact local, SLURM, or Makefile commands when the change affects public entry points.
7. **generated-artifact safety** — Write images, tensors, checkpoints, logs, score files, plots, thesis bundles, contact sheets, and private run outputs only to ignored runtime roots unless a tiny fixture is reviewed.

## Review checklist

- Does the change preserve existing public commands and file/artifact layouts?
- Is reusable behavior importable and CPU-safe at import time?
- Is any expensive CUDA/model/OCR/SLURM work explicit rather than default automation?
- Are generated outputs written to ignored runtime roots and excluded from commits?
- Are new extension points documented before future users copy the pattern?
