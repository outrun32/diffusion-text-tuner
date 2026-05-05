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

## Comparison Choice Fields

Use these fields to make masked-SFT loss, LoRA, dataset, and evaluation-suite choices explicit before training and in config snapshots:

| Field | Purpose | Expected values or notes |
|-------|---------|--------------------------|
| `masked_lambda` | Weight for masked-region flow-matching loss in `L = lambda * L_masked + (1 - lambda) * L_global`. | Float in `[0, 1]`; records the masked/global loss tradeoff. |
| `lora.attn_r` | LoRA rank for attention projection target modules. | Non-negative integer; pair with `lora.attn_alpha` and `lora.attn_modules`. |
| `lora.joint_attn_r` | LoRA rank for joint-attention/additional projection target modules. | Non-negative integer; pair with `lora.joint_attn_alpha` and `lora.joint_attn_modules`. |
| `data_dir` | Synthetic masked-SFT dataset root. | Repository-relative runtime path such as `data/synth_cyrillic/masked_sft`. |
| `eval_suite_path` | Optional JSON evaluation-suite reference sampled during validation. | Repository-relative config path such as `configs/eval_suite.json`, or `null` when disabled. |
| `validation_interval` | Step interval for validation loss and evaluation-suite sampling. | Positive integer. |
| `eval_suite_n_per_step` | Number of evaluation-suite items sampled at each validation step. | Positive integer. |

Generated synthetic images, masks, tensors, checkpoints, and logs remain ignored runtime artifacts unless intentionally tiny fixtures are reviewed.
