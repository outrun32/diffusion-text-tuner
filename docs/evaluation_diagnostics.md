# Evaluation Diagnostics: Difficulty Slices and Gold Benchmark

Phase 6 adds CPU-safe diagnostics for Russian text-rendering evaluation. The
contracts here are intentionally metadata-only: they inspect text fields,
prediction rows, and JSONL labels, but they do not load FLUX, Qwen, PaddleOCR,
CUDA, model weights, generated images, tensors, checkpoints, or logs.

## Russian text difficulty slices

Use `classify_text_slices(record)` to label one evaluation record by target text
and optional prompt metadata. Use `summarize_slices(records)` to count slice
coverage across a list of records and to surface records missing `target_text`.

Supported slice labels:

| Slice | Meaning |
| --- | --- |
| `rare_cyrillic` | `target_text` contains rare Cyrillic letters from the diagnostic set (`ё`, `ж`, `ц`, `щ`, `ъ`, case-insensitive). |
| `short_word` | Non-empty `target_text` has one short word and no longer phrase label. |
| `long_word` | Any punctuation-stripped word in `target_text` has at least 10 characters. |
| `multi_word_phrase` | `target_text` contains at least two whitespace-separated words. |
| `has_digits` | `target_text` contains one or more digits. |
| `has_punctuation` | `target_text` contains punctuation, including Cyrillic-relevant marks such as `№`, dashes, ellipsis, or guillemets. |
| `mixed_case` | `target_text` combines lowercase letters with uppercase words or internal uppercase letters. |
| `multiline` | `target_text` contains a newline or carriage return. |
| `font_or_style` | Metadata fields such as `font`, `font_family`, `font_style`, `style`, or `text_style` are present. |
| `scene_or_background` | Metadata fields such as `scene`, `background`, `background_type`, or `environment` are present. |

The slice classifier is deterministic and pure. It returns an empty set for
records with missing or blank `target_text`; `summarize_slices` reports those
records under `missing_target_text_records` instead of hiding them.

## Gold diagnostic benchmark validation

Use `load_gold_benchmark(path)` to load a small JSONL benchmark with schema
`gold-diagnostic-benchmark/v1`. The loader aggregates malformed line and
missing-field errors so users can fix all visible benchmark issues before using
the file as evidence. Each record must contain:

| Field | Required | Meaning |
| --- | --- | --- |
| `sample_id` | yes | Stable identifier used to join predictions to gold rows. |
| `target_text` | yes | Expected Russian or multilingual rendered text. |
| `image_path` | yes | Metadata path to the diagnostic image; tests use path strings only. |
| `expected_exact_match` | yes | Boolean expectation for exact text-match metrics. |
| `expected_ocr_detected` | yes | Boolean expectation for OCR detection metrics. |
| `human_label` | yes | Small diagnostic label such as `pass`, `fail`, or `needs-review`. |
| `notes` | optional | Free-text diagnostic note for reviewers. |

The committed `tests/fixtures/evaluation/gold_diagnostic.jsonl` is a
metadata-only fixture. It includes Cyrillic rare-letter, digit/punctuation,
mixed-case, and multiline examples. Its `image_path` values are references for
schema and documentation tests; no generated image binaries are committed.

Use `evaluate_gold_predictions(benchmark, predictions)` to join prediction rows
by `sample_id` and compute:

- `exact_agreement` against `expected_exact_match`;
- `ocr_detection_agreement` against `expected_ocr_detected`;
- `ocr_text_agreement` using normalized detected text and target text;
- `missing_prediction_count` and `missing_prediction_sample_ids`;
- per-slice records, missing predictions, and disagreement counts;
- markdown-ready `findings` that summarize missing evidence and disagreements.

Use `format_gold_report_markdown(report)` to render a concise diagnostic report.
Missing predictions/disagreements are explicit evidence, not hidden pass
conditions; in lower-case terms, missing predictions/disagreements are explicit evidence.
A clean report can support reward-validity confidence, but a report
with missing rows or disagreements is still useful thesis evidence because it
identifies where reward signals fail.

## Generated-artifact safety

Do not commit generated images, tensors, checkpoints, or logs for these
diagnostics. Keep runtime gold reports, scored predictions, contact sheets, and
private evaluation outputs under ignored runtime roots such as `runs/` or
`outputs/`. Only tiny reviewed fixtures like
`tests/fixtures/evaluation/gold_diagnostic.jsonl` should be committed.

These diagnostics are safeguards for reward validity. They are not a broad
human-evaluation taxonomy and should not be treated as final thesis scoring on
their own; later Phase 6 plans can combine them with recorded score outputs,
diagnostic reports, and thesis-ready tables.
