# Reward and Scoring Configs

Place reward/scoring variants here, including VLM, OCR, product-score, calibration, and ablation configs.

## Naming

Use `{stage}_{reward_or_data}_{purpose}.json` with `stage=reward` for scorer setup or `stage=score` for scoring-run configs when matching runtime CLI terminology.

Examples:

- `reward_vlm_qwen_calibration.json`
- `reward_ocr_entropy_ablation.json`
- `reward_product_formula_v1.json`

## Required Contract

Reward/scoring configs should identify:

- `schema_version`, `stage`, and `experiment_name`.
- Reward model IDs/revisions, OCR engine/version assumptions, scorer thresholds, component weights, and calibration data.
- Inputs such as generated images, text embeddings, prompt metadata, target text, and prior score files.
- Outputs such as scores CSV/JSONL, schema metadata, reports, and manifest paths under ignored runtime roots.
- `seed` when sampling examples, bootstrapping calibration sets, or selecting diagnostics.
- Manifest expectations so thesis plots and trainer selections can trace back to exact scorer settings.

Do not commit private generated score files or model outputs from `outputs/`; commit only reviewed tiny fixtures or documentation assets.
