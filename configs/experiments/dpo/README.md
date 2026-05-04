# DPO Configs

Place new preference-optimization variants in this directory. Keep `configs/dpo.json` and existing root DPO variants runnable for compatibility.

## Naming

Use `{stage}_{reward_or_data}_{purpose}.json` with `stage=dpo`.

Examples:

- `dpo_vlm_best_vs_worst.json`
- `dpo_ocr_margin_pairs.json`
- `dpo_product_ambiguity_filter_ablation.json`

## Required Contract

DPO configs should identify:

- `schema_version`, `stage: dpo`, and `experiment_name`.
- Base `model_id`, optional `model_revision`, and any `sft_lora_path` or reference checkpoint.
- Reward source, pair-construction mode, score thresholds, margins, `scores_csv`, `latents_dir`, `text_embeds_dir`, and `seed`.
- `beta`, optimizer/schedule settings, LoRA settings, checkpoint cadence, precision, and `output_dir`.
- Manifest expectations for `runs/<run_id>/manifest.json`, immutable config snapshots, metrics, notes, and resume inspection.

Materialized preference-pair artifacts should be stored under ignored runtime roots and documented before comparison-grade runs.
