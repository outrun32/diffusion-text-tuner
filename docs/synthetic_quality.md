# Synthetic Masked-SFT Quality Inspection

The CPU-safe synthetic dataset inspector checks outputs produced by
`scripts/synth/build_dataset.py`. It reads local CSV, JSONL, JSON metadata, images, and masks with
PIL only; the default path does not launch FLUX, Qwen, PaddleOCR, SynthTIGER, CUDA, OCR, or model
inference.

## Required input layout

The synthetic builder writes two related layouts. The quality CLI expects the masked-SFT directory as `--data-dir` and the raw SynthTIGER directory as `--raw-dir`.

```text
raw/imgs/{sid}.png
raw/masks/{sid}.png
raw/meta/{sid}.json
raw/index.jsonl
masked_sft/raw_imgs/{sid}.png
masked_sft/raw_masks/{sid}.png
masked_sft/index.csv
masked_sft/prompts.jsonl
masked_sft/shapes.csv
masked_sft/latents/{sid}.pt
masked_sft/text_embeds/{sid}.pt
```

`index.csv` provides sample IDs, resolution, target text, word counts, and captions. `prompts.jsonl` connects each sample ID to the prompt consumed by masked-SFT. `shapes.csv` records latent shapes when latents are baked. Raw `meta/{sid}.json` supplies annotations, bbox values, fonts, and renderer provenance when available.

## Local report command

Run inspection against existing build outputs without requiring OCR or model packages:

```bash
uv run python -m scripts.inspect_synthetic_dataset \
  --data-dir data/synth_cyrillic/masked_sft \
  --raw-dir data/synth_cyrillic/raw \
  --config <synthtiger-config.yaml> \
  --seed 42 \
  --template <synthtiger-template.py> \
  --runner <synthtiger-runner.py> \
  --model-id black-forest-labs/FLUX.2-klein-base-4B \
  --word-source data/ru_freq_50k.txt \
  --font-source data/fonts/font_list.txt \
  --background-source data/backgrounds/unsplash_meta/photos.tsv \
  --report runs/synthetic-quality/synthetic-quality.json \
  --manifest runs/synthetic-quality/dataset-manifest.json \
  --contact-sheet runs/synthetic-quality/contact-sheet.png \
  --min-mask-area-fraction 0.006 \
  --max-mask-area-fraction 0.30 \
  --min-bbox-height-fraction 0.02 \
  --min-contrast 15
```

The CLI returns `0` when no blocking missing-file or threshold rejection exists and `1` when filters reject samples. Threshold failures are counted by reason so users can adjust renderer, mask, bbox, or contrast settings before training.

## Quality report fields

Reports use `synthetic-quality/v1` and include:

- sample counts, accepted/rejected counts, missing file counters, and rejection reasons;
- mask area fraction summaries from `raw_masks`;
- bbox height fraction and bbox area fraction summaries from raw metadata annotations, falling back to mask bboxes when metadata is absent;
- foreground/background contrast estimates from PIL grayscale image pixels and masks;
- character coverage counts from target text;
- font coverage from raw metadata annotations when the renderer records font names;
- resolution distribution from index rows, metadata, shapes, or PIL image dimensions;
- optional OCR summary fields when `--ocr-results` is supplied;
- per-sample paths, target text, prompt, metric values, accepted status, and rejection reasons for manifest and contact-sheet handoff.

## Optional OCR handoff

OCR verification is optional. Default synthetic inspection never imports PaddleOCR. Produce OCR outputs with a separate opt-in diagnostic or reward pipeline, then hand the result file to the inspector:

```bash
uv run python -m scripts.inspect_synthetic_dataset \
  --data-dir data/synth_cyrillic/masked_sft \
  --raw-dir data/synth_cyrillic/raw \
  --report runs/synthetic-quality/synthetic-quality-with-ocr.json \
  --ocr-results runs/synthetic-quality/ocr-results.csv
```

OCR files may be CSV or JSONL with `id`, `target_text` (or `text`/`label`), and `ocr_text` (or `prediction`/`recognized_text`). The report records exact-match count, exact-match rate, and CER-style mean character error rate. OCR summaries are evidence inputs, not default test dependencies.

## Contact sheets

Use `--contact-sheet runs/synthetic-quality/contact-sheet.png` to render a PIL-only contact sheet from sampled rejected and accepted raw images. Labels include sample ID, accepted/rejected status, target text, and rejection reasons. Contact sheets are for visual inspection and may expose generated images and prompt text.

## Manifest provenance

When `--manifest` is provided, the CLI writes a `dataset-manifest/v1` manifest for synthetic data. The manifest records dataset paths, git state, build config hash/snapshot, `--seed` strategy, `--model-id` and optional `--model-revision`, safe source hashes for small CSV/JSONL/text inputs, referenced generated image/tensor paths, filtering stats, output counts, thresholds, report paths, and contact-sheet paths.

Use provenance flags to capture the synthetic builder inputs that matter for thesis reproducibility:

- `--config`: explicit external SynthTIGER build config; no default renderer config is committed.
- `--seed`: synthetic build seed.
- `--template`: SynthTIGER template path or identifier.
- `--runner`: builder/runner script path or identifier.
- `--model-id` / `--model-revision`: model used for latent or text-embedding baking, when applicable.
- `--word-source`: one or more word/frequency/source text files.
- `--font-source`: one or more font list or font provenance files.
- `--scene-source`: one or more scene metadata files.
- `--background-source`: one or more background manifest/provenance files.

The manifest lets training selection, comparisons, and thesis reports point back to exact synthetic
builder outputs, filter settings, report summaries, source hashes, and local run artifacts without
committing generated data.

## Threshold meanings

- `--min-mask-area-fraction` / `--max-mask-area-fraction`: reject samples whose positive mask pixels are too small or too large for stable masked-SFT.
- `--min-bbox-height-fraction`: reject samples whose rendered text bbox is too short relative to image height.
- `--min-bbox-area-fraction`: reject samples whose rendered text bbox area is too small.
- `--min-contrast`: reject samples whose estimated foreground/background contrast is too low.
- `--max-text-length`: reject samples whose target text is too long for the intended masked-SFT curriculum slice.
- `--max-samples`: bound CPU inspection for quick smoke checks or contact-sheet previews.

## Generated artifact safety

Generated reports, manifests, contact sheets, images, masks, tensors, and private OCR outputs are runtime artifacts. Keep them under ignored roots such as `runs/`, `outputs/`, or generated data roots like `data/synth_cyrillic/`. Do not commit generated synthetic artifacts unless they are intentionally tiny fixtures reviewed for tests or documentation.
