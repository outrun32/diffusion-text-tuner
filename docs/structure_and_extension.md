# Structure and Extension Guide

This Phase 7 guide is the canonical navigation contract for moderate brownfield cleanup. It explains where reusable modules, wrappers, launchers, configs, diagnostics, experiments, runtime outputs, tests, and thesis evidence belong before later plans move implementation code behind importable seams.

Use this guide together with [`docs/pipeline_inventory.md`](pipeline_inventory.md), [`docs/commands.md`](commands.md), and [`docs/runtime_contracts.md`](runtime_contracts.md). The intent is behavior-preserving structure cleanup: existing commands remain discoverable, generated artifacts stay out of git, and future extension work starts from stable homes instead of one-off hidden assumptions.

## Canonical repository homes

| Home | Classification | What belongs here | What does not belong here |
| --- | --- | --- | --- |
| Reusable source modules: `src/` | Source | Importable implementation modules for prompt generation, runtime contracts, data quality, training, scoring/reward, evaluation, plotting, and future pipeline seams. Prefer small focused helpers that can be imported by tests and thin wrappers without launching CUDA, model downloads, OCR, or SLURM work at import time. | One-off shell orchestration, generated outputs, private run data, checkpoint weights, ad-hoc notebooks, or model-cache paths. |
| Thin CLI wrappers: `scripts/` | Wrapper | Command-line wrappers that parse arguments, call importable modules, configure logging, and write explicit runtime outputs. Supported wrappers should be listed in `docs/commands.md` and stay thin enough that behavior can be tested through `src/` modules. | Large reusable implementations that should live under `src/`, generated images/tensors/scores, private logs, or hidden hardcoded run assumptions. |
| Cluster launchers: `scripts/cluster/` | Wrapper | SLURM `.sbatch` jobs, cluster setup helpers, merge helpers, and local-vs-cluster launch variants. These launchers may invoke GPU/model/OCR jobs, but they should point at documented configs and ignored runtime outputs. | Importable Python implementation logic, local-only diagnostics without cluster context, or generated cluster logs. |
| Synthesis helpers: `scripts/synth/` | Wrapper / synthesis helper | Synthetic masked-SFT rendering/build wrappers, SynthTIGER templates, fixture builders, background/font fetch helpers, and compatibility entry points that will progressively delegate to importable `src.synthesis` modules. | Generated synthetic images, masks, latents, text embeddings, large downloaded backgrounds/fonts, or private synthetic output reports. |
| Thesis plotting helpers: `scripts/thesis_plots/` | Wrapper / plotting helper | Plotting CLIs that consume recorded manifests, metrics, score files, diagnostics, or thesis bundle inputs and write figures under ignored output roots. Future shared plotting logic should move to `src.plotting`. | Manually edited final figures that cannot be traced to recorded inputs, raw generated plots committed by default, or plotting code embedded in unrelated trainer scripts. |
| Configuration contracts: `configs/` | Config | Root compatibility configs for supported flows, Accelerate topology YAML, prompt configs, thesis/evaluation configs, and family directories. Keep values reviewable, repository-relative, and free of secrets or personal absolute paths. | Runtime snapshots, generated manifests, private tokens, model caches, checkpoints, or run-specific outputs. |
| Experiment config variants: `configs/experiments/` | Config | New comparison-grade SFT, DPO, masked-SFT, reward/scoring, synthesis, evaluation, and ablation variants using the documented family naming contract. | Historical one-off outputs, generated reports, run manifests, or private environment-specific overrides. |
| CPU-safe tests: `tests/` | Test | Default pytest coverage, characterization tests, docs drift tests, tiny fixtures, and CPU-safe contract tests. Tests should not import CUDA/model/OCR stacks at collection time. | Manual diagnostics, GPU/OCR/model jobs, large generated fixtures, or expensive integration checks that should be explicit opt-in commands. |
| Historical experiments: `experiments/` | Diagnostic / experiment | One-off research probes, OCR/VLM reward experiments, and tiny reviewed assets that document historical reasoning without becoming default automation. | Supported toolkit entry points, default pytest tests, generated large artifacts, or source modules required by production wrappers. |
| Generated runtime outputs: ignored `outputs/`, `runs/`, and generated `data/` subtrees | Generated runtime output | Images, tensors, checkpoints, logs, reports, contact sheets, score files, local run manifests, synthetic payloads, private prompt/output records, and stage outputs produced by local or SLURM jobs. | Source code, canonical docs, committed configs, default tests, or anything that should be required to import the project. |
| Thesis evidence artifacts: recorded manifests, reports, plots, contact sheets, and bundles | Thesis evidence | Reproducible evidence produced from recorded runs and traced through manifests/configs/seeds/rewards. Keep generated thesis outputs under ignored roots such as `outputs/thesis/` or `runs/` unless a future plan intentionally adds a tiny reviewed fixture. | Static/manual numbers without provenance, private full outputs committed by default, or claims that cannot be tied back to exact runs and configs. |

generated images, tensors, checkpoints, logs, reports, contact sheets, thesis bundles, and private run outputs remain runtime artifacts. Do not commit generated runtime outputs unless they are intentionally tiny reviewed fixtures or documentation assets.

## Supported wrappers, diagnostics, and generated-output boundaries

- Supported wrappers are commands documented in `docs/commands.md` or `docs/pipeline_inventory.md`; they should call importable implementation modules wherever practical and write outputs only to explicit paths.
- Manual diagnostics are opt-in investigation tools. Keep them outside default pytest discovery and document GPU/model/OCR/generated-artifact prerequisites when they are relevant.
- Historical experiments preserve research context but should not become hidden dependencies for supported flows. Promote reusable logic into `src/` first, then leave the experiment as a thin historical probe or retire it in a future explicit cleanup plan.
- Generated runtime outputs are not source. They may contain prompt text, target text, model outputs, reward scores, local paths, run metadata, or thesis draft evidence, so keep them in ignored runtime roots and stage only source, docs, configs, tests, and approved tiny fixtures.

## Extension rules before adding new work

Before adding a new experiment, trainer, reward variant, dataset, pipeline, plot, or thesis-output step, use these rules:

1. Start from the nearest existing documented command in `docs/commands.md` and the inventory in `docs/pipeline_inventory.md`.
2. Put reusable behavior in an importable module under `src/`; prefer importable implementation modules under `src/` with thin wrappers in `scripts/`.
3. Keep CLI wrappers responsible for argument parsing, logging setup, and delegating to importable code. Avoid burying reusable behavior in shell or script-only globals.
4. Add or update CPU-safe tests under `tests/` before relying on behavior for refactors or thesis claims.
5. Add config variants under `configs/experiments/` when the work changes an experimental choice, reward family, data source, evaluation plan, or ablation rather than editing root compatibility configs in place.
6. document CPU-safe verification and explicit GPU/model/OCR/SLURM prerequisites in `docs/commands.md`, a focused guide, or the wrapper help text.
7. Write generated artifacts to ignored `outputs/`, `runs/`, or generated `data/` roots and record provenance through manifests or sidecars before using them as thesis evidence.

| Extension type | Home for reusable logic | Wrapper/config home | Required navigation update |
| --- | --- | --- | --- |
| New experiment | Promote reusable pieces to `src/`; keep historical probes in `experiments/` only when they are opt-in. | Add reviewed config variants under `configs/experiments/` and document commands if they become repeatable. | Update `docs/pipeline_inventory.md` if it changes supported or historical tracks. |
| New trainer | `src/training/` for config/runtime/objective/data helpers and trainer orchestration. | Root compatibility config only when it is a supported default; otherwise use `configs/experiments/<family>/`. | Update training comparability and command docs before interpreting runs. |
| New reward variant | Shared reward/scoring/evaluation logic under `src/training/`, `src/scoring/`, or `src/evaluation/` as the Phase 7 seams evolve. | Thin scorer wrapper in `scripts/` and reward config under `configs/experiments/reward/`. | Update reward/evaluation docs and score sidecar expectations if output schema changes. |
| New dataset or data pipeline | Importable dataset, quality, manifest, or synthesis helpers under `src/data_quality/`, `src/training/`, or future `src/synthesis/`. | Explicit CLI wrapper plus config under `configs/experiments/synthesis/` or the relevant data family. | Update dataset quality/runtime docs and keep generated datasets out of git. |
| New generation, scoring, synthesis, evaluation, or plotting pipeline | Importable pipeline module under `src/generation/`, `src/scoring/`, `src/synthesis/`, `src/evaluation/`, or `src/plotting/` as those seams are introduced. | Thin wrapper under `scripts/`, `scripts/synth/`, or `scripts/thesis_plots/`; cluster variant under `scripts/cluster/` when needed. | Add CPU-safe tests and exact command docs before considering the pipeline supported. |
| New thesis-output step | `src/evaluation/` or future `src/plotting/` for reproducible table/plot/bundle builders. | Wrapper under `scripts/` or `scripts/thesis_plots/`; config under `configs/thesis/` or `configs/experiments/evaluation/`. | Link outputs to manifests/reports and keep thesis bundles under ignored runtime roots. |

## Review checklist

- Does the change preserve existing public commands and file/artifact layouts?
- Is reusable behavior importable and CPU-safe at import time?
- Is any expensive CUDA/model/OCR/SLURM work explicit rather than default automation?
- Are generated outputs written to ignored runtime roots and excluded from commits?
- Are new extension points documented before future users copy the pattern?
