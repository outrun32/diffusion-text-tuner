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

Generated checkpoints, samples, logs, and tensors remain non-committable runtime artifacts.
