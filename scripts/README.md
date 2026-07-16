# Script Navigation

This directory contains command-line entry points and compatibility helpers for the diffusion
text-tuning thesis toolkit. Start with
[`docs/structure_and_extension.md`](../docs/structure_and_extension.md) for repository boundaries,
then use this file to distinguish supported wrappers from diagnostics, cluster launchers, synthesis
helpers, plotting helpers, and historical experiments.

## Supported wrappers versus diagnostics

| Family | Classification | Examples | Boundary |
| --- | --- | --- | --- |
| Thin CLI wrappers | Supported wrappers backed by importable modules | `uv run python -m scripts.generate_images`, `uv run python -m scripts.score_images`, `uv run python -m scripts.run_manifest`, `uv run python -m scripts.preflight_runtime`, `uv run python -m scripts.compare_run_manifests`, `uv run python -m scripts.check_training_comparability`, `uv run python -m scripts.compare_training_runs`, `uv run python -m scripts.run_heldout_evaluation` | Wrappers should parse CLI arguments, configure logging, call importable `src/` modules where available, and write explicit runtime outputs. |
| Manual diagnostics | Diagnostics | `scripts/diagnose_gradient_flow.py`, `scripts/diagnose_grad_magnitude.py`, `uv run python -m scripts.profile_step`, and diagnostic one-liners in `docs/commands.md` | Run manually in prepared environments only. They may inspect generated tensors, CUDA/model state, or reward evidence and stay outside default pytest discovery. |
| Cluster jobs | Cluster jobs | `scripts/cluster/*.sbatch`, `scripts/cluster/setup_env.sh`, `scripts/cluster/merge_scores.sh` | SLURM launchers mirror local supported commands and write logs/artifacts under ignored runtime roots. Review node/cache/config assumptions before submission. |
| Synthesis helpers | Synthesis helpers | `scripts/synth/build_dataset.py`, templates, fixture builders, background/font helpers, word samplers | Synthetic data wrappers may invoke rendering/model work explicitly. Generated synthetic images, masks, tensors, reports, and contacts sheets belong under ignored generated `data/`, `outputs/`, or `runs/` roots. |
| Plotting helpers | Plotting helpers | `uv run python -m scripts.plot_metrics`, `uv run python -m scripts.build_thesis_outputs`, and `docs/thesis_outputs.md` | Plotting consumes recorded manifests/reports/metrics and writes ignored runtime outputs unless a tiny reviewed evidence file is intentional. |
| Historical/experiment scripts | Historical/experiment scripts | Baseline helpers, older one-off shell launchers, and research-specific probes not listed as default supported commands | Treat as opt-in research history. Promote reusable behavior into `src/` and add docs/tests before making a historical script part of the supported command surface. |

## Explicit utility inventory

The less frequent entry points below are intentional; none should be inferred to be a default
training command merely because it lives under `scripts/`.

| Entry point | Status | Purpose / owner |
| --- | --- | --- |
| `scripts/download_dataset.py` | Supported CPU/network utility | Downloads the pinned prompt dataset and writes its source manifest. |
| `scripts/build_evidence_manifest.py` | Supported CPU utility | Builds or verifies the committed public-evidence hash index used by `make evidence-verify`. |
| `scripts/final_benchmark.py` | Compatibility wrapper | Delegates to `src.evaluation.final_benchmark`; the installed command is `dtt-final-benchmark`. |
| `scripts/aggregate_heldout_scores.py` | Supported CPU utility | Aggregates canonical multi-seed score files after generation/scoring jobs finish. |
| `scripts/merge_score_shards.py` | Supported CPU utility | Validates and merges completed score shards with matching task snapshots. |
| `scripts/precompute_text_embeddings.py` | Explicit CUDA preparation utility | Precomputes FLUX text embeddings; it is not a Mac/MLX workflow. |
| `scripts/generate_simple_dataset.py` | Historical compatibility helper | Older prompt-dataset generator retained for thesis reproducibility; new prompt runs use `src.prompt_pipeline.generate`. |
| `scripts/eval_quantized_vs_unquantized.py` | Manual model diagnostic | Compares reward backends in an explicitly prepared model environment; excluded from default tests. |
| `scripts/audit_git_history.py` | Security gate | Rejects reachable handoff/editor artifacts and oversized history blobs. |
| `scripts/prepare_history_cleanup.py` | Security maintenance utility | Creates and verifies a disposable filtered mirror; never pushes automatically. |
| `scripts/run_baseline.sh` and `scripts/run_refl.sh` | Historical shell launchers | Preserve earlier experiment commands; inspect their CUDA/model assumptions before use. |

## Generated runtime outputs are not source

diagnostics and generated outputs are local/runtime surfaces that can contain private prompt text, scores, paths, or run metadata. Keep generated images, tensors, checkpoints, logs, score files, reports, contact sheets, thesis bundles, synthetic payloads, and private run outputs under ignored roots such as `outputs/`, `runs/`, or generated `data/` subtrees. Do not stage them with script changes unless they are intentionally tiny reviewed fixtures or documentation assets.

## Adding or changing scripts

1. Prefer reusable implementation modules under `src/` and keep the script as a thin CLI wrapper.
2. Add CPU-safe tests for reusable behavior before refactoring or moving command logic.
3. Document supported commands in `docs/commands.md`; document repository-home changes in `docs/structure_and_extension.md`.
4. Mark GPU, model, OCR, SLURM, and generated-artifact prerequisites explicitly.
5. Keep compatibility wrappers in place when a public command is moved behind a new importable module.
