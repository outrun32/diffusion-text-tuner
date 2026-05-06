# Script Navigation

This directory contains command-line entry points and compatibility helpers for the diffusion text-tuning thesis toolkit. Start with [`docs/structure_and_extension.md`](../docs/structure_and_extension.md) for the repository-wide structure contract, then use this file to decide whether a script is a supported wrapper, diagnostic, cluster launcher, synthesis helper, plotting helper, or historical experiment.

## Supported wrappers versus diagnostics

| Family | Classification | Examples | Boundary |
| --- | --- | --- | --- |
| Thin CLI wrappers | Supported wrappers backed by importable modules | `python -m scripts.generate_images`, `python -m scripts.score_images`, `python -m scripts.run_manifest`, `python -m scripts.preflight_runtime`, `python -m scripts.compare_run_manifests`, `python -m scripts.check_training_comparability`, `python -m scripts.compare_training_runs`, `python -m scripts.run_heldout_evaluation` | Wrappers should parse CLI arguments, configure logging, call importable `src/` modules where available, and write explicit runtime outputs. |
| Manual diagnostics | Diagnostics | `scripts/diagnose_gradient_flow.py`, `scripts/diagnose_grad_magnitude.py`, diagnostic one-liners in `docs/commands.md` | Run manually in prepared environments only. They may inspect generated tensors, CUDA/model state, reward evidence, or local paths and must stay outside default pytest discovery. |
| Cluster jobs | Cluster jobs | `scripts/cluster/*.sbatch`, `scripts/cluster/setup_env.sh`, `scripts/cluster/merge_scores.sh` | SLURM launchers mirror local supported commands and write logs/artifacts under ignored runtime roots. Review node/cache/config assumptions before submission. |
| Synthesis helpers | Synthesis helpers | `scripts/synth/build_dataset.py`, templates, fixture builders, background/font helpers, word samplers | Synthetic data wrappers may invoke rendering/model work explicitly. Generated synthetic images, masks, tensors, reports, and contacts sheets belong under ignored generated `data/`, `outputs/`, or `runs/` roots. |
| Plotting helpers | Plotting helpers | `scripts/thesis_plots/plot_sft_losses.py`, `scripts/thesis_plots/plot_dpo_metrics.py`, thesis-output builders documented in `docs/thesis_outputs.md` | Plotting helpers should consume recorded manifests/reports/metrics and produce reproducible figures under ignored runtime roots unless a future plan adds tiny reviewed docs assets. |
| Historical/experiment scripts | Historical/experiment scripts | Baseline helpers, older one-off shell launchers, and research-specific probes not listed as default supported commands | Treat as opt-in research history. Promote reusable behavior into `src/` and add docs/tests before making a historical script part of the supported command surface. |

## Generated runtime outputs are not source

diagnostics and generated outputs are local/runtime surfaces that can contain private prompt text, scores, paths, or run metadata. Keep generated images, tensors, checkpoints, logs, score files, reports, contact sheets, thesis bundles, synthetic payloads, and private run outputs under ignored roots such as `outputs/`, `runs/`, or generated `data/` subtrees. Do not stage them with script changes unless they are intentionally tiny reviewed fixtures or documentation assets.

## Adding or changing scripts

1. Prefer reusable implementation modules under `src/` and keep the script as a thin CLI wrapper.
2. Add CPU-safe tests for reusable behavior before refactoring or moving command logic.
3. Document supported commands in `docs/commands.md`; document repository-home changes in `docs/structure_and_extension.md`.
4. Mark GPU, model, OCR, SLURM, and generated-artifact prerequisites explicitly.
5. Keep compatibility wrappers in place when a public command is moved behind a new importable module.
