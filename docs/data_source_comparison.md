# Data Source Comparison

Phase 3 Plan 03-05 compares reward-filtered generated images against synthetic masked-SFT evidence before training comparisons. The goal is to make DATA-07 reproducible: users can see where generated-image SFT/DPO data and synthetic masked-SFT data are expected to help, where they may fail, and which artifacts support those conclusions.

The comparison is metadata-only and CPU-safe. It reads JSON/JSONL reports, selected-sample artifacts, preference-pair artifacts, and manifests. It does not inspect generated images, load tensors, import FLUX, Qwen, PaddleOCR, CUDA, or run OCR/model inference.

## Inputs

Use existing Phase 3 artifacts when they are available:

| Input | CLI flag | Evidence captured |
|-------|----------|-------------------|
| Prompt quality report | `--generated-prompt-quality-report` | Prompt counts, rare-character coverage, content distribution, and style distribution for generated-image prompts. |
| Selected SFT samples | `--selected-samples` | Reward-filtered generated-image rows from `selected_samples.jsonl`, including selected score summaries. |
| DPO preference pairs | `--preference-pairs` | Winner/loser evidence from `preference_pairs.jsonl`, including score margin summaries. |
| Generated dataset manifest | `--generated-dataset-manifest` | Source hashes, output counts, and provenance for generated reward-filtered data. |
| Synthetic quality report | `--synthetic-quality-report` | Synthetic sample counts, accepted/rejected counts, masks, contrast, OCR handoff summaries, fonts, resolutions, and character coverage. |
| Synthetic manifest | `--synthetic-manifest` | Source hashes, filtering stats, output counts, and provenance for synthetic masked-SFT data. |

All inputs are optional. Missing evidence is listed in `evidence_missing`; the tool reports unavailable metrics as `null` or empty maps rather than fabricating values.

## Command examples

Full comparison:

```bash
uv run python scripts/compare_data_sources.py \
  --generated-prompt-quality-report runs/prompt-quality/curriculum-report.json \
  --selected-samples outputs/generated/selected_samples.jsonl \
  --preference-pairs outputs/generated/preference_pairs.jsonl \
  --generated-dataset-manifest outputs/generated/selected_samples.manifest.json \
  --synthetic-quality-report runs/synthetic-quality/synthetic-quality.json \
  --synthetic-manifest runs/synthetic-quality/dataset-manifest.json \
  --output-report runs/comparisons/generated-vs-synthetic.json \
  --markdown-summary runs/comparisons/generated-vs-synthetic.md
```

Evidence-gap smoke check with only synthetic quality evidence:

```bash
uv run python scripts/compare_data_sources.py \
  --synthetic-quality-report runs/synthetic-quality/synthetic-quality.json \
  --output-report runs/comparisons/synthetic-only.json
```

Generated reports, Markdown summaries, selection artifacts, preference pairs, and manifests under `runs/` or `outputs/` are runtime artifacts. Do not commit generated images, tensors, private prompts, generated reports, or contact sheets unless they are intentionally tiny reviewed fixtures.

## Report schema

JSON reports use `data-source-comparison/v1` and include:

- `evidence_available` / `evidence_missing` for explicit proof boundaries;
- `counts` for generated prompt records, selected samples, preference pairs, synthetic samples, accepted samples, and rejected samples;
- `rare_character_coverage` with generated counts, synthetic counts, overlap, generated-only gaps, and synthetic-only gaps;
- `distribution_differences` for generated content/style evidence and synthetic font/resolution evidence;
- `generated_score_summary` for selected scores and DPO preference margins;
- `synthetic_mask_contrast_health` for mask area, contrast, rejection reasons, and optional OCR summary evidence;
- `expected_help` and `expected_failure` interpretation sections;
- `provenance` with paths, SHA-256 hashes, schema names, dataset kinds, and record counts for every parsed input.

## Interpretation guidance

Generated reward-filtered data is expected to help when the training question is alignment to actual FLUX outputs, score thresholds, reward preferences, and DPO winner/loser margins. It is the closest metadata evidence to the generated-image SFT/DPO training path.

Generated data can fail when the scorer admits reward/OCR false positives, when prompt distribution gaps are inherited from generation inputs, when rare-character coverage is weak, or when high reward scores reflect internal scoring behavior rather than human-visible rendered-text quality.

Synthetic masked-SFT data is expected to help controlled local reconstruction of text regions because it supplies masks, bbox/contrast evidence, font coverage, resolution mix, and renderer-controlled character coverage. It is useful for targeted rare-character coverage and mask-aware reconstruction experiments.

Synthetic data can fail when renderer outputs miss natural scene realism, domain complexity, noisy backgrounds, true camera artifacts, or FLUX-specific generation failures. Mask and contrast shortcuts can also overfit to synthetic conditions and not transfer to natural generated scenes.

## Thesis comparison caveats

Use this report as pre-training evidence for a thesis comparison, not as final proof that a method improves Russian or multilingual text rendering. In particular, training loss or DPO accuracy are internal signals until Phase 6 evaluation validates rendered-text quality on held-out prompts with OCR/VLM/product reward diagnostics and qualitative inspection.

When writing thesis notes, cite exact comparison reports, source manifests, score files, selection thresholds, margin filters, seeds, configs, and artifact hashes. If evidence is missing, state that limitation explicitly rather than inferring strengths or failures from absent data.
