# SFT Configs

Place new supervised fine-tuning variants in this directory. Keep `configs/sft.json` and existing root variants runnable for compatibility with documented local and SLURM commands.

## Naming

Use `{stage}_{reward_or_data}_{purpose}.json` with `stage=sft`.

Examples:

- `sft_vlm_top1.json`
- `sft_ocr_threshold_ablation.json`
- `sft_product_weighted_samples.json`

## Required Contract

SFT configs should identify:

- `schema_version`, `stage: sft`, and `experiment_name`.
- Base `model_id` and optional `model_revision`.
- Reward/data source used to select samples (`vlm`, `ocr`, `product`, or another documented selector).
- `latents_dir`, `text_embeds_dir`, `scores_csv`, sample-selection thresholds/modes, and `seed`.
- LoRA, optimizer, schedule, precision, checkpoint, and `output_dir` settings.
- Manifest flow: create/inspect `runs/<run_id>/manifest.json`, then preflight with the SFT config and artifact paths before launching `src.training.sft_trainer`.

## Comparison Choice Fields

Use these fields to make SFT sample-selection choices explicit before training and in config snapshots:

| Field | Purpose | Expected values or notes |
|-------|---------|--------------------------|
| `selection_mode` | Names the SFT sample-selection contract. | `threshold`, `top_k_per_prompt`, `score_weighted`, or `hard_positive`; default is `threshold`. |
| `selected_samples_path` | Optional materialized selected-samples JSONL produced before comparison-grade runs. | Repository-relative runtime path such as `outputs/generated/selected_samples.jsonl`, or `null` for legacy CSV selection. |
| `score_column` | Score column read from the scores CSV or materialized selection source. | Default `score`; change only when the scoring artifact documents another column. |
| `score_threshold` | Minimum score included by threshold-style selection. | Float in `[0, 1]`; default root config behavior is preserved. |
| `hard_negative_threshold` | Prompt-level low-score cutoff used by `hard_positive` selection. | Float in `[0, 1]`; records the rejected hard-negative boundary. |
| `sample_weighting` | Names the sample-weight interpretation for snapshots and manifests. | `uniform` for unweighted modes or `score_normalized` for score-weighted selections. |

Generated checkpoints, samples, logs, and tensors remain non-committable runtime artifacts.
