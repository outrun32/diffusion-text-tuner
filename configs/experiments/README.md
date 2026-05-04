# Experiment Config Organization

This directory is the stable home for new experiment variants. Existing root configs remain runnable compatibility entry points and must not be moved by normal experiment work:

- `configs/sft.json`
- `configs/dpo.json`
- `configs/masked_sft.json`
- current root variants such as `configs/sft_vlm.json`, `configs/dpo_ocr.json`, and `configs/masked_sft_clean.json`

Use root configs for backwards-compatible local and SLURM commands that are already documented. Add new research variants under the family directories here so config discovery, manifests, and preflight checks stay predictable.

## Families

| Family | Directory | Stage value | Purpose |
|--------|-----------|-------------|---------|
| SFT | `configs/experiments/sft/` | `sft` | Reward-filtered supervised fine-tuning variants. |
| DPO | `configs/experiments/dpo/` | `dpo` | Preference optimization variants and pair-construction experiments. |
| Masked-SFT | `configs/experiments/masked_sft/` | `masked_sft` | Synthetic masked reconstruction and AnyWord-style training variants. |
| Reward/scoring | `configs/experiments/reward/` | `score` or `reward` | OCR, VLM, product-score, and calibration variants. |
| Synthesis | `configs/experiments/synthesis/` | `synthetic` | Synthetic prompt/rendering/font/background dataset generation variants. |
| Evaluation | `configs/experiments/evaluation/` | `evaluation` | Held-out prompt suites, scoring reports, and checkpoint comparison variants. |

Ablations stay in the closest owning family directory. Prefix or suffix them with `ablation` rather than creating a disconnected top-level folder.

## Naming Rules

New JSON configs should use:

```text
{stage}_{reward_or_data}_{purpose}.json
```

Examples:

- `sft_vlm_top1.json`
- `sft_product_threshold_ablation.json`
- `dpo_ocr_margin_pairs.json`
- `masked_sft_synthetic_strict_masks.json`
- `reward_product_calibration.json`
- `synthetic_cyrillic_font_ablation.json`
- `evaluation_vlm_heldout_ru.json`

Rules:

1. `stage` matches the runtime stage (`sft`, `dpo`, `masked_sft`, `reward`, `synthetic`, `evaluation`).
2. `reward_or_data` names the reward source, data source, or evaluation suite (`vlm`, `ocr`, `product`, `synthetic`, `cyrillic`, `heldout`).
3. `purpose` names the scientific comparison (`top1`, `margin_pairs`, `strict_masks`, `font_ablation`, `checkpoint_eval`).
4. Use lowercase snake_case and avoid personal names, absolute paths, run IDs, or machine-specific details.
5. If a config is a temporary local run edit, keep it outside git or record it only through `runs/<run_id>/config_snapshot.json`.

## Required Metadata

Every new family config should include or be convertible to manifest metadata with these fields:

- `schema_version`: config schema identifier, for example `runtime-config/v1` or the family-specific version.
- `stage`: one of `sft`, `dpo`, `masked_sft`, `reward`, `synthetic`, or `evaluation`.
- `experiment_name`: stable human-readable experiment slug.
- `model_id` and, when supported by the underlying tool, `model_revision` for base diffusion, reward, OCR/VLM, or evaluation models.
- `seed` and any additional data split, sampling, prompt, or generation seeds.
- Inputs: prompt JSONL, generated image roots, latent/text embedding directories, scores CSV, synthetic data roots, checkpoints, or evaluation suite paths.
- Outputs: ignored `outputs/` or `runs/` paths for generated images, tensors, scores, checkpoints, logs, metrics, and reports.
- Manifest expectations: create a run manifest with `python -m scripts.run_manifest init`, inspect it before launch, and pass `runs/<run_id>/manifest.json` to preflight/resume commands when available.

Root compatibility configs may not yet contain all metadata fields directly. For comparison-grade runs, capture the resolved root config snapshot in a manifest so missing compatibility fields can be tracked without breaking existing commands.

## Runtime Contract Expectations

- Use relative repository paths accepted by `src.runtime.config_io` path policy; do not commit `/home/...`, `/Users/...`, `~/...`, or off-repo absolute paths.
- Run `python -m scripts.preflight_runtime --stage ...` before long-running generation, scoring, training, synthesis, or evaluation work.
- Keep generated artifacts under ignored roots such as `outputs/`, `runs/`, and generated `data/` subtrees. Do not commit generated images, tensors, checkpoints, logs, or private output manifests.
- Document any new artifact layout in `docs/runtime_contracts.md` before relying on it for thesis evidence.
