# Reward Evaluation Contract

`src.evaluation.reward_interface` defines the CPU-safe contract for reward records, Product scores,
score sidecars, diagnostics, and
thesis reports. The module is import-safe: it does not import Qwen,
PaddleOCR, OCR engines, CUDA, vLLM, MLX, Diffusers, Transformers, PIL, torch,
or model weights.

This document describes evidence contracts only. It does not claim that any
reward model proves visual text-rendering quality without held-out evaluation,
diagnostics, and thesis validation.

Related guides: [`docs/evaluation_harness.md`](evaluation_harness.md),
[`docs/evaluation_diagnostics.md`](evaluation_diagnostics.md),
[`docs/thesis_outputs.md`](thesis_outputs.md), and the command catalog in
[`docs/commands.md`](commands.md).

## Canonical record: `RewardResult`

`RewardResult` represents one scored generated sample or evaluation output.
`RewardResult.to_row()` returns deterministic CSV/JSON-safe fields that can be
shared by scoring, training selection, evaluation diagnostics, and reports.

Canonical fields:

| Field | Meaning |
| --- | --- |
| `sample_id` | Stable prompt/sample identifier. This replaces ad hoc `id` naming in canonical evaluation artifacts. |
| `version` | Generated candidate/checkpoint/version number for the same `sample_id`. |
| `target_text` | Expected text the image should render. |
| `score` | Primary scalar score for the row, normally the product score when available. |
| `score_vlm` | Qwen/VLM yes-probability evidence, normalized to `[0, 1]` when present. |
| `score_ocr` | OCR-derived reward evidence, normalized to `[0, 1]` when present. |
| `cer` | Character error rate from detected text against `target_text`; lower is better. |
| `entropy` | OCR CTC uncertainty/entropy evidence; lower is better. |
| `exact_text_match` | Exact target/detected text match evidence as `True`/`False` or `1.0`/`0.0`. |
| `text_metrics` | JSON object for text metrics such as detected text and character accuracy. |
| `scorer_metadata` | JSON object for scorer names, revisions, configurations, and scorer versions. |
| `thresholds` | JSON object containing threshold names and values used for pass/fail flags. |
| `missing_components` | Comma-separated evidence names that were missing, `None`, non-finite, or invalid. |
| `manifest_path` | Link to the run/evaluation manifest that produced or contextualized this row. |

`scorer_metadata` should store scorer identity such as `qwen@revision`,
`paddleocr@revision`, prompt template/version, and OCR settings. It must not
store secrets, raw cache directories, private local environment values, or model
weights.

Training/offline reward dictionaries should be converted through
`src.training.rewards.build_training_reward_result` before they are used as
comparison evidence. Evaluation code reuses the file-path reward adapters in
`src.training.rewards` (`EvaluationQwenYesProbReward` and
`PaddleOCRAccuracyReward`) instead of maintaining separate evaluator-local Qwen
or OCR scorer classes. This keeps scoring, training, evaluation, diagnostics,
and thesis reports on one canonical `RewardResult` / product-score contract.

## Product formula: `ProductScoreFormula`

The thesis metric and the later diagnostic metric are separate formulas.
`scripts.score_images --product_formula thesis` uses the reported thesis definition:

```python
thesis_product_formula(
    scorer_versions={"vlm": "qwen@revision", "ocr": "paddleocr@revision"}
)
# name: thesis_vlm_ocr_product_v1
# aggregation: weighted_product
# weights: {"score_vlm": 1.0, "score_ocr": 1.0}
# require_all: True
```

Its score is exactly `score_vlm * score_ocr`; a missing component produces score `0` and
`formula_complete=false`.

`--product_formula diagnostic` selects the later five-component geometric metric:

```python
ProductScoreFormula(
    name="vlm_ocr_cer_entropy_exact_product_v1",
    weights={
        "score_vlm": 0.35,
        "score_ocr": 0.25,
        "cer_quality": 0.20,
        "entropy_quality": 0.10,
        "exact_text_match": 0.10,
    },
    thresholds={"score_vlm_min": 0.7, "score_ocr_min": 0.6, "cer_max": 0.2},
    scorer_versions={"vlm": "qwen@revision", "ocr": "paddleocr@revision"},
    entropy_scale=1.0,
    aggregation="weighted_geometric_mean",
    require_all=False,
)
```

`compute_product_score` transforms raw evidence into normalized formula terms:

| Term | Source evidence | Transformation |
| --- | --- | --- |
| `score_vlm` | `score_vlm` | Finite numeric value clamped to `[0, 1]`. |
| `score_ocr` | `score_ocr` | Finite numeric value clamped to `[0, 1]`. |
| `cer_quality` | `cer` | `1.0 - cer`, with `cer` clamped to `[0, 1]`. |
| `entropy_quality` | `entropy` | `exp(-entropy_scale * entropy)` for finite non-negative entropy. |
| `exact_text_match` | `exact_text_match` | `1.0` for true/exact, `0.0` for false/mismatch. |

For the diagnostic formula, the score is a weighted geometric mean over available normalized terms:

```text
score = exp(sum(weight_i * ln(term_i)) / sum(available_weight_i))
```

When all configured formula terms are present and valid, the denominator is the
full configured weight sum. When evidence is missing, the numeric score is
computed from available terms only, and missing evidence is recorded explicitly
so downstream comparisons can reject or flag incomplete rows.

`compute_product_score` returns:

| Field | Meaning |
| --- | --- |
| `score` | Numeric product score from available normalized terms. |
| `components` | Deterministic normalized component dictionary. |
| `missing_components` | Evidence names missing from the row, such as `score_ocr` or `cer`. |
| `threshold_flags` | Boolean pass/fail flags for formula thresholds such as `score_vlm_min` and `cer_max`. |
| `formula` | The `ProductScoreFormula` used for the computation. |
| `formula_complete` | `True` only when all positive-weight formula components were present and valid. |

Missing VLM/OCR evidence must never be silently treated as comparable. The diagnostic formula may
carry a numeric partial score for inspection; the thesis formula requires both terms.

## Score metadata: `build_score_metadata`

`build_score_metadata` returns sidecar metadata for generated score files and
reports:

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `reward-score-metadata/v1`. |
| `generated_at` | Caller-supplied or injected UTC timestamp. Tests should inject this for determinism. |
| `formula` | Formula name, aggregation, weights, thresholds, scorer versions, entropy scale, and `require_all`. |
| `source_manifest_paths` | Manifest links tying score rows back to exact runs/evaluation plans. |

The helper stores manifest links as strings and never opens local files, cache
paths, model directories, or environment variables. This keeps metadata creation
CPU-safe and secret-safe while preserving traceability to run manifests.

## Canonical score CSV/JSONL fields

Canonical scoring outputs extend the original `id`, `version`, `score`, and
`target_text` columns rather than removing them. New score CSV files written by
`uv run python -m scripts.score_images` use `phase6-score-file/v1`; JSONL records
written by `uv run python -m src.evaluation.evaluate_rewards` use
`phase6-score-jsonl/v1`. Both formats carry the same canonical evidence fields:

| Field | Meaning |
| --- | --- |
| `id` / `sample_id` | Stable generated sample identifier. `id` remains for SFT/DPO compatibility; `sample_id` is the canonical evaluation name. |
| `version` | Generated candidate or evaluation version. |
| `score` / `product_score` | `score` follows the selected scorer (`vlm`, `ocr`, or combined Product); `product_score` always stores the chosen product formula. |
| `target_text` | Expected rendered text. |
| `score_vlm` | VLM/Qwen yes-probability evidence when the VLM scorer ran. |
| `score_ocr` | OCR reward evidence when the OCR scorer ran. |
| `cer` | Character error rate, lower is better. |
| `entropy` | OCR confidence/entropy evidence, lower is better. |
| `ocr_detected` | OCR-detected text, empty when no OCR evidence exists. |
| `detection_status` | One of `detected_exact`, `detected_mismatch`, or `not_detected`. |
| `exact_text_match` | Boolean exact normalized text match between detected and target text. |
| `char_accuracy`, `char_matches`, `char_total` | Character-level text metrics used for diagnostics and slice reports. |
| `missing_components` | Comma-separated CSV value or JSON list naming absent VLM/OCR/CER/entropy/exact evidence. |
| `formula_complete` | `true` only when all positive-weight product formula components were available and finite. |
| `manifest_path` | Run or evaluation manifest link for traceability. |
| `text_metrics` | JSON object containing detected text, exact-match, character-level metrics, and detection status. |
| `scorer_metadata` | JSON object containing formula name and scorer versions. |
| `thresholds` | JSON object containing product-formula threshold pass/fail flags. |

Missing evidence remains explicit. For example, a VLM-only scoring run writes an
empty `score_ocr`, `cer`, and `entropy`, and records those evidence names in
`missing_components`; it must not be treated as comparable to complete VLM+OCR
product rows in diagnostics or thesis tables.

## Score sidecars and manifest links

Every canonical score output should have a sibling `.schema.json` sidecar. The
sidecar uses `schema_version="reward-score-metadata/v1"` and includes:

| Sidecar field | Meaning |
| --- | --- |
| `score_file_schema_version` | `phase6-score-file/v1` for CSV or `phase6-score-jsonl/v1` for JSONL. |
| `formula.name` | `thesis_vlm_ocr_product_v1` by default; `vlm_ocr_cer_entropy_exact_product_v1` only in diagnostic mode. |
| `formula.weights` | Formula-specific weights. Thesis mode has one unnormalized factor for VLM and OCR. |
| `formula.aggregation` | `weighted_product` for the thesis metric or `weighted_geometric_mean` for diagnostics. |
| `formula.thresholds` | Thresholds such as `score_vlm_min`, `score_ocr_min`, or `cer_max`. |
| `formula.scorer_versions` | Scorer identities/revisions such as Qwen model ID and OCR settings. |
| `source_manifest_paths` | Source run/evaluation manifests that produced or contextualized score rows. |
| `required_phase6_fields` | Canonical fields expected in rows for validator drift checks. |
| `source_manifest_sha256` | Hashes of source manifests that existed when scoring started. |
| `execution` | Shard index/count, discovered/expected/scored row counts, completion status, and final CSV SHA-256. |

`--resume` requires an existing sidecar with the same formula, primary score, scorer revisions,
source-manifest paths/hashes, and shard contract. It refuses to append rows under a new metric or
model revision. Existing rows are recomputed from their stored evidence and checked against current
sample identity/target text before reuse. Missing embeddings/target text and incomplete shard row
counts are blocking errors, not silent skips.

After the CSV header and after every persisted row, scoring atomically checkpoints the exact current
row count and CSV SHA-256 in the sidecar. An `in-progress` CSV without that matching checkpoint is
rejected rather than re-signed during resume.

The score script accepts manifest links with `--manifest_path` for each row and
one or more `--source_manifest` values for the sidecar:

```bash
uv run python -m scripts.score_images \
  --images_dir outputs/generated/images \
  --text_embeds_dir outputs/generated/text_embeds \
  --output_csv outputs/generated/scores.csv \
  --scorer both \
  --manifest_path runs/scoring/manifest.json \
  --source_manifest runs/generation/manifest.json \
  --source_manifest runs/scoring/manifest.json
```

The evaluation scoring path uses dash-style arguments for the same links:

```bash
uv run python -m src.evaluation.evaluate_rewards \
  --metadata outputs/baseline/metadata.jsonl \
  --output outputs/baseline/scores.jsonl \
  --reward all \
  --manifest-path runs/eval/manifest.json \
  --source-manifest runs/baseline/manifest.json
```

This JSONL evaluator never appends or silently skips missing images. Existing outputs require an
explicit `--overwrite`; partial `--start-idx` execution is rejected, and at least one real source
manifest is mandatory before model initialization.

Score provenance accepts only a strict `run-manifest/v1` or the current self-hashed
`generation-manifest/v4`; arbitrary JSON and legacy generation manifests are rejected.

## Score validation command examples

Score validation is CPU-safe and shallow. It reads CSV/JSONL rows and
their `.schema.json` sidecars, but it does not open generated images, tensors,
CUDA devices, Qwen, PaddleOCR, or model weights.

Python callers can validate a CSV score file with:

```python
from src.runtime.artifacts import validate_artifacts

report = validate_artifacts("evaluation_scores", {"scores_csv": "outputs/generated/scores.csv"})
if not report.ok:
    raise SystemExit(report.errors)
```

JSONL evaluation outputs use the same stage with `scores_jsonl`:

```python
report = validate_artifacts("evaluation_scores", {"scores_jsonl": "outputs/baseline/scores.jsonl"})
```

Use `require_ready=True` only at blocking preflight gates when missing score files
or missing sidecars should fail immediately.

Generated score files and `.schema.json` sidecars are runtime artifacts. Keep
them out of git, except for tiny fixtures required by CPU-safe tests or reviewed documentation
examples.

## Threshold semantics

Threshold names end in `_min` or `_max`:

- `score_vlm_min`: true when `score_vlm >= threshold`.
- `score_ocr_min`: true when `score_ocr >= threshold`.
- `cer_max`: true when raw `cer <= threshold`.
- `entropy_quality_min`: true when transformed `entropy_quality >= threshold`.
- `exact_text_match_min`: true when exact-match evidence meets the threshold.

If a threshold references missing or invalid evidence, the corresponding
`threshold_flags` value is `False`.

## Generated artifacts safety

Generated score files, score metadata sidecars, held-out evaluation outputs,
diagnostic reports, contact sheets, thesis tables/plots, checkpoints, logs,
tensors, generated images, and private run outputs remain runtime artifacts.
Keep them out of git, except for tiny fixtures required by CPU-safe tests or reviewed
documentation.

This contract is safe to import in tests and report builders, but actual Qwen,
PaddleOCR, OCR engine, CUDA, vLLM, MLX, or model-weight execution must remain in
explicit scoring/evaluation commands outside default pytest discovery.
