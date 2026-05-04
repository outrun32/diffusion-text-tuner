# Masked-SFT Configs

Place new synthetic masked reconstruction variants in this directory. Keep `configs/masked_sft.json` and current root masked-SFT variants runnable for compatibility.

## Naming

Use `{stage}_{reward_or_data}_{purpose}.json` with `stage=masked_sft`.

Examples:

- `masked_sft_synthetic_strict_masks.json`
- `masked_sft_cyrillic_resume_ablation.json`
- `masked_sft_anyword_clean.json`

## Required Contract

Masked-SFT configs should identify:

- `schema_version`, `stage: masked_sft`, and `experiment_name`.
- Base `model_id`, optional `model_revision`, synthetic dataset version, font/background sources, and `seed`.
- `data_dir`, mask/latent/text-embedding expectations, validation suite, bucket/resolution settings, and masked/global loss weights.
- LoRA target/rank choices, optimizer/schedule settings, checkpoint cadence, precision, sample settings, and `output_dir`.
- Manifest expectations for generated synthetic data, config snapshots, run notes, metrics, and `runs/<run_id>/manifest.json`.

Generated synthetic images, masks, tensors, checkpoints, and logs remain ignored runtime artifacts unless intentionally tiny fixtures are reviewed.
