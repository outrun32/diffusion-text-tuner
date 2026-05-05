# Data Selection Artifacts

Phase 3 Plan 03-04 materializes reward-filtered training selections as versioned metadata artifacts before SFT or DPO training. The artifacts are inspectable JSONL files derived from `outputs/generated/scores.csv` and are intended to be written under ignored runtime roots such as `outputs/generated/` or a run-specific `runs/<run_id>/` directory.

Do not commit generated images or tensors. Selection JSONL/manifest files under runtime roots are also generated artifacts by default; commit only tiny fixtures or documentation examples after review.

## Why materialize selections?

Historically, `SFTDataset` and `DPODataset` selected examples inside dataset constructors. That preserves a runnable training path, but it hides which rows and preference labels were used. Materialized artifacts make the training subset reproducible by recording thresholds, score columns, source hashes, counts, schema versions, and explicit DPO winner/loser semantics.

These contracts extend the selected-sample and preference-pair rows reserved in `docs/runtime_contracts.md`.

## SFT: `selected_samples.jsonl`

Default command:

```bash
uv run python scripts/materialize_training_data.py --kind sft \
  --scores-csv outputs/generated/scores.csv \
  --output-dir outputs/generated \
  --threshold 0.3 \
  --manifest outputs/generated/selected_samples.manifest.json
```

The default equivalence is the current `SFTDataset` constructor behavior: every row where `score >= score_threshold` is selected. Use `--score-column` to select by scorer-specific columns such as `score_ocr`.

### SFT selection modes

Use exact mode names with `--mode`; the CLI forwards the string without aliases so manifests can be compared by name.

| Mode | Selection behavior | Example |
|------|--------------------|---------|
| `threshold` | Select every row where `score_column >= --threshold`. This is the default equivalence for current SFT CSV loading. | `uv run python scripts/materialize_training_data.py --kind sft --mode threshold --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.3` |
| `top_k_per_prompt` | For each prompt, select the highest-scoring eligible rows, ordered score-descending then version-ascending, up to `--top-k-per-prompt`. | `uv run python scripts/materialize_training_data.py --kind sft --mode top_k_per_prompt --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.3 --top-k-per-prompt 1` |
| `score_weighted` | Select threshold-passing rows and add `sample_weight = selected_score / max_selected_score`, rounded to 12 decimals. | `uv run python scripts/materialize_training_data.py --kind sft --mode score_weighted --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.3` |
| `hard_positive` | Select rows above `--threshold` only for prompts that also have at least one version below `--hard-negative-threshold`. | `uv run python scripts/materialize_training_data.py --kind sft --mode hard_positive --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.7 --hard-negative-threshold 0.2` |

Each JSONL row uses schema `selected-samples/v1` and includes:

| Field | Meaning |
|-------|---------|
| `schema_version` | Always `selected-samples/v1`. |
| `sample_id` | Stable ID such as `sft:p1:v2:score`. |
| `prompt_id`, `version`, `target_text` | Source generated prompt/version metadata. |
| `selected_score`, `score_column` | Score value and CSV column used for selection. |
| `selection_mode` | One of `threshold`, `top_k_per_prompt`, `score_weighted`, or `hard_positive`. |
| `source_scores_path`, `source_scores_sha256` | Source CSV provenance. |
| `manifest_path` | Optional summary/manifest JSON path. |

## DPO: `preference_pairs.jsonl`

Default command:

```bash
uv run python scripts/materialize_training_data.py --kind dpo \
  --scores-csv outputs/generated/scores.csv \
  --output-dir outputs/generated \
  --threshold 0.5 \
  --margin 0.1 \
  --manifest outputs/generated/preference_pairs.manifest.json
```

The default equivalence is the current `DPODataset` constructor behavior: group rows by prompt, choose the best and worst scored versions, require the winner to meet `--threshold`, and require the winner/loser score gap to meet `--margin`. Pairs with equal scores or a gap at or below `--ambiguity-margin` are rejected so winner and loser labels are never silently inverted.

### DPO pair-construction modes

Use exact mode names with `--mode`; every mode enforces strict winner-over-loser semantics and rejects equal-score pairs.

| Mode | Pair behavior | Example |
|------|---------------|---------|
| `best_vs_worst` | Emit the best-scored version against the worst-scored version per prompt when the winner passes `--threshold` and the gap passes `--margin`. This is the default equivalence for current DPO CSV loading. | `uv run python scripts/materialize_training_data.py --kind dpo --mode best_vs_worst --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.5 --margin 0.1` |
| `all_separated_pairs` | Emit every same-prompt winner/loser pair where the winner passes `--threshold` and `winner_score - loser_score >= --margin`. | `uv run python scripts/materialize_training_data.py --kind dpo --mode all_separated_pairs --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.5 --margin 0.1` |
| `margin_weighted` | Emit best-vs-worst pairs and add `pair_weight = margin / max_margin`, rounded to 12 decimals. | `uv run python scripts/materialize_training_data.py --kind dpo --mode margin_weighted --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.5 --margin 0.1` |
| `ambiguity_filtered` | Emit best-vs-worst pairs only when the best-vs-second-best margin is greater than `--ambiguity-margin`, in addition to threshold and margin checks. | `uv run python scripts/materialize_training_data.py --kind dpo --mode ambiguity_filtered --scores-csv outputs/generated/scores.csv --output-dir outputs/generated --threshold 0.5 --margin 0.1 --ambiguity-margin 0.05` |

Each JSONL row uses schema `preference-pairs/v1` and includes:

| Field | Meaning |
|-------|---------|
| `schema_version` | Always `preference-pairs/v1`. |
| `pair_id` | Stable ID such as `dpo:p1:w2:l1:score`. |
| `prompt_id`, `target_text` | Prompt metadata. |
| `winner_version`, `loser_version` | Explicit preference labels. |
| `winner_score`, `loser_score`, `margin` | Numeric score evidence for label direction. |
| `score_column`, `pair_construction_mode` | Score source and exact construction strategy: `best_vs_worst`, `all_separated_pairs`, `margin_weighted`, or `ambiguity_filtered`. |
| `source_scores_path`, `source_scores_sha256` | Source CSV provenance. |
| `manifest_path` | Optional summary/manifest JSON path. |

## Summary manifests

When `--manifest` is supplied, the CLI writes a deterministic JSON summary next to the JSONL artifact. The summary records schema version, selection or pair mode, threshold/margin settings, score column, source score path/hash, output path, counts, and filtering stats.

For SFT, filtering stats include selected rows and below-threshold rows, plus mode-specific counters such as `unselected_by_top_k` or `prompts_without_hard_negative`. For DPO, filtering stats include selected pairs, prompts with insufficient versions, winners below threshold, and ambiguous pairs below the required margin; all-separated mode also records rejected equal-score and below-margin candidate pairs.

## Connecting to training configs

This plan does not change trainer loaders. Existing SFT and DPO configs still point at `scores_csv` and use the in-constructor selection defaults. Use these materialized artifacts before comparison-grade training runs to review and version the exact selection; later trainer/config plans can optionally consume these JSONL paths directly while preserving current behavior.

Recommended workflow:

1. Generate and score images so `outputs/generated/scores.csv` exists.
2. Run the SFT or DPO materialization command above.
3. Inspect the JSONL rows and summary manifest counts.
4. Record artifact paths in the run manifest or experiment notes before launching training.

Generated selection artifacts are metadata, not images/tensors/checkpoints. They should still live under ignored runtime roots unless deliberately promoted as tiny fixtures.
