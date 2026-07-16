# Evaluation Diagnostics: Difficulty Slices, Gold Benchmark, and Reward Disagreement

These CPU-safe diagnostics inspect Russian text-rendering evaluation evidence. The contracts are
metadata-only: they inspect text fields,
prediction rows, recorded score outputs, and JSONL labels, but they do not load
FLUX, Qwen, PaddleOCR, CUDA, model weights, generated images, tensors,
checkpoints, or logs. In lower-case operational terms, these diagnostics do not run reward models; they consume outputs that scoring/evaluation jobs already
recorded.

Related guides: [`docs/reward_evaluation.md`](reward_evaluation.md),
[`docs/evaluation_harness.md`](evaluation_harness.md),
[`docs/thesis_outputs.md`](thesis_outputs.md), and the command catalog in
[`docs/commands.md`](commands.md).

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

## Reward disagreement diagnostics

Use `analyze_reward_disagreement(records, ...)` to inspect disagreement between
canonical score fields without invoking OCR/VLM/model code. The function accepts
recorded score outputs containing fields such as `score_vlm`, `score_ocr`,
`cer`, `exact_match`, `char_accuracy`, `product_score`, `missing_components`,
`target_text`, `detected_text`, and optional `image_path`. Use
`format_diagnostics_markdown(report)` to render the resulting dictionary as a
human-readable report.

The report includes:

- VLM-vs-OCR scatter/correlation summaries using rows where both `score_vlm`
  and `score_ocr` are present;
- missing evidence counts by component, so incomplete rows are not hidden;
- false-positive rows where a high product score conflicts with exact-match or
  gold diagnostic benchmark expectations;
- false-negative rows where a low product score conflicts with expected passing
  evidence;
- per-character confusion summaries comparing `target_text` with recorded OCR or
  detected text;
- per-slice disagreement counts using `classify_text_slices`, including rare
  Cyrillic, digits, punctuation, mixed case, multiline, font/style, and
  scene/background slices;
- optional bounded PIL contact-sheet metadata containing captions and source
  paths for selected false-positive and false-negative examples.

The gold diagnostic benchmark linkage is optional but recommended. Pass `--gold`
or `gold_records=` when hand-labeled expectations are available; otherwise the
diagnostic falls back to recorded exact-match fields when classifying
false-positive and false-negative rows.

### CLI examples

Generate deterministic JSON and Markdown from a recorded score CSV plus a gold
diagnostic JSONL file:

```bash
uv run python -m scripts.analyze_reward_diagnostics \
  --scores runs/eval/baseline/scores.csv \
  --gold tests/fixtures/evaluation/gold_diagnostic.jsonl \
  --output-report runs/eval/baseline/reward_diagnostics.json \
  --markdown-summary runs/eval/baseline/reward_diagnostics.md \
  --positive-threshold 0.80 \
  --negative-threshold 0.50
```

Add a bounded contact sheet for reviewable false-positive/false-negative rows:

```bash
uv run python -m scripts.analyze_reward_diagnostics \
  --scores runs/eval/trained/scores.jsonl \
  --gold runs/eval/gold_diagnostic.jsonl \
  --output-report runs/eval/trained/reward_diagnostics.json \
  --markdown-summary runs/eval/trained/reward_diagnostics.md \
  --contact-sheet runs/eval/trained/reward_disagreements.png \
  --contact-sheet-limit 24
```

The CLI returns nonzero for malformed score/gold inputs. It does not return
nonzero merely because disagreements, false positives, false negatives, missing
evidence, or per-character confusion rows were discovered; those findings are
the diagnostic output.

## Generated-artifact safety

Do not commit generated images, tensors, checkpoints, or logs for these
diagnostics. Keep runtime gold reports, scored predictions, contact sheets, and
private evaluation outputs under ignored runtime roots such as `runs/` or
`outputs/`. Only tiny reviewed fixtures like
`tests/fixtures/evaluation/gold_diagnostic.jsonl` should be committed.

Do not commit generated diagnostic reports or contact sheets from real runs.
Keep `reward_diagnostics.json`, `reward_diagnostics.md`, contact-sheet PNGs,
score CSV/JSONL files, held-out outputs, plots, thesis bundles, and private
benchmark labels under ignored runtime paths. Commit only tiny fixtures that a CPU-safe test or a
reviewed documentation example requires.

These diagnostics are safeguards for reward validity. They are not a broad
human-evaluation taxonomy and should not be treated as final thesis scoring on
their own; use them with recorded score outputs, diagnostic reports, and manifest-linked result
tables.
