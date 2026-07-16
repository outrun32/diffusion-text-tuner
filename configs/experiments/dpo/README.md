# DPO Configs

Place new preference-optimization variants in this directory. Keep `configs/dpo.json` and existing root DPO variants runnable for compatibility.

Files ending in `_final.json` are historical defense config records. Files ending in
`_fixed_safe.json` encode the corrected DPO objective/pair choices for a future rerun after the sign
audit; they are not evidence that a replacement run completed.

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

## Comparison Choice Fields

Use these fields to make DPO pair construction and objective choices explicit before training and in config snapshots:

| Field | Purpose | Expected values or notes |
|-------|---------|--------------------------|
| `pair_construction_mode` | Names the DPO preference-pair construction contract. | `best_vs_worst`, `all_separated_pairs`, `margin_weighted`, or `ambiguity_filtered`; default is `best_vs_worst`. |
| `preference_pairs_path` | Optional materialized preference-pairs JSONL produced before comparison-grade runs. | Repository-relative runtime path such as `outputs/generated/preference_pairs.jsonl`, or `null` for legacy CSV pair construction. |
| `score_column` | Score column read from the scores CSV or pair source. | Default `score`; change only when the scoring artifact documents another column. |
| `score_threshold` | Minimum winning score required for pair construction. | Float in `[0, 1]`; preserves current root config semantics by default. |
| `score_diff_min` | Minimum score margin between winner and loser. | Positive float; documents strict winner-over-loser evidence. |
| `ambiguity_margin` | Best-vs-second-best ambiguity filter for ambiguity-aware modes. | Non-negative float; `0.0` disables extra ambiguity filtering. |
| `pair_weighting` | Names the pair-weight interpretation for snapshots and manifests. | `uniform` for unweighted modes or `margin_normalized` for margin-weighted pairs. |
| `beta` | DPO objective scaling value used by the trainer. | Positive float; current objective sign/beta behavior is characterized by CPU-safe tests. |

Materialized preference-pair artifacts should be stored under ignored runtime roots and documented before comparison-grade runs.
