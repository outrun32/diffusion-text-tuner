# Reward Evaluation Contract

Phase 6 uses `src.evaluation.reward_interface` as the canonical CPU-safe
contract for reward records, product scores, score sidecars, diagnostics, and
thesis reports. The module is import-safe: it does not import Qwen,
PaddleOCR, OCR engines, CUDA, vLLM, MLX, Diffusers, Transformers, PIL, torch,
or model weights.

This document describes evidence contracts only. It does not claim that any
reward model proves visual text-rendering quality without held-out evaluation,
diagnostics, and thesis validation.

## Canonical record: `RewardResult`

`RewardResult` represents one scored generated sample or evaluation output.
`RewardResult.to_row()` returns deterministic CSV/JSON-safe fields that can be
shared by scoring, training selection, evaluation diagnostics, and reports.

Canonical fields:

| Field | Meaning |
| --- | --- |
| `sample_id` | Stable prompt/sample identifier. This replaces ad hoc `id` naming in new Phase 6 artifacts. |
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

## Product formula: `ProductScoreFormula`

`ProductScoreFormula` records the product-score formula name, component weights,
thresholds, scorer_versions, and entropy scaling:

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

The product score is a weighted geometric product over available normalized
terms:

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

Missing VLM/OCR evidence must never be silently treated as comparable. A row
with `missing_components=["score_ocr"]` may still carry a numeric score for
inspection, but reports and thesis tables should distinguish it from complete
VLM+OCR+CER+entropy+exact-text evidence.

## Score metadata: `build_score_metadata`

`build_score_metadata` returns sidecar metadata for generated score files and
reports:

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `reward-score-metadata/v1`. |
| `generated_at` | Caller-supplied or injected UTC timestamp. Tests should inject this for determinism. |
| `formula` | The formula name, weights, thresholds, scorer_versions, and `entropy_scale`. |
| `source_manifest_paths` | Manifest links tying score rows back to exact runs/evaluation plans. |

The helper stores manifest links as strings and never opens local files, cache
paths, model directories, or environment variables. This keeps metadata creation
CPU-safe and secret-safe while preserving traceability to run manifests.

## Canonical score CSV/JSONL fields

Phase 6 scoring outputs extend the original `id`, `version`, `score`, and
`target_text` columns rather than removing them. New score CSV files written by
`python -m scripts.score_images` use `phase6-score-file/v1`; JSONL records
written by `python -m src.evaluation.evaluate_rewards` use
`phase6-score-jsonl/v1`. Both formats carry the same canonical evidence fields:

| Field | Meaning |
| --- | --- |
| `id` / `sample_id` | Stable generated sample identifier. `id` remains for SFT/DPO compatibility; `sample_id` is the canonical Phase 6 name. |
| `version` | Generated candidate or evaluation version. |
| `score` / `product_score` | Primary score and explicit product-score value from `compute_product_score`. |
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
| `formula.name` | Product formula name, normally `vlm_ocr_cer_entropy_exact_product_v1`. |
| `formula.weights` | Product formula weights for VLM, OCR, CER quality, entropy quality, and exact match. |
| `formula.thresholds` | Thresholds such as `score_vlm_min`, `score_ocr_min`, or `cer_max`. |
| `formula.scorer_versions` | Scorer identities/revisions such as Qwen model ID and OCR settings. |
| `source_manifest_paths` | Source run/evaluation manifests that produced or contextualized score rows. |
| `required_phase6_fields` | Canonical fields expected in rows for validator drift checks. |

The score script accepts manifest links with `--manifest_path` for each row and
one or more `--source_manifest` values for the sidecar:

```bash
python -m scripts.score_images \
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
python -m src.evaluation.evaluate_rewards \
  --metadata outputs/baseline/metadata.jsonl \
  --output outputs/baseline/scores.jsonl \
  --reward all \
  --manifest-path runs/eval/manifest.json \
  --source-manifest runs/baseline/manifest.json
```

## Score validation command examples

Phase 6 score validation is CPU-safe and shallow. It reads CSV/JSONL rows and
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
them out of git unless a later plan intentionally adds tiny fixtures for
CPU-safe tests or documentation examples.

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
They should stay out of git unless a later plan intentionally creates tiny
fixtures for CPU-safe tests or documentation.

This contract is safe to import in tests and report builders, but actual Qwen,
PaddleOCR, OCR engine, CUDA, vLLM, MLX, or model-weight execution must remain in
explicit scoring/evaluation commands outside default pytest discovery.
