# Public evidence bundle

This directory separates evidence that can be checked from aggregate numbers that survive only in
the thesis materials.

## What can be verified from this checkout

- `prompt_dataset_source.manifest.json` pins the public 15,000-row prompt dataset to Hugging Face
  revision `ecd8b2da9820b35afc65e2d56eaf37a662c37976`. It records the parquet and normalized JSONL
  SHA-256 hashes.
- `prompt_dataset_quality_v1.json` validates that exact normalized JSONL: 15,000 valid rows, no
  malformed records, 3.82% duplicate rows, complete configured rare-character coverage, and no
  blocking validation errors. The report includes the input SHA-256 and applied config thresholds.
  Rebuild it after downloading the pinned dataset with `make prompt-dataset-evidence`.
- `benchmark_prompts_v2.jsonl` contains 120 unique prompts across six difficulty slices. Its target
  strings are disjoint from the pinned prompt-training pool.
- `prompt_training_target_hashes_v1.json` is an exact, offline-checkable index of SHA-256 hashes for
  the normalized target strings in that pinned training pool; it contains no prompt or target text.
- `benchmark_prompts_v2.manifest.json` records the benchmark hash, slice counts, excluded overlaps,
  source-dataset provenance, and the hash of the target index used to recompute disjointness.
- `evidence_manifest.json` hashes every public artifact listed in this bundle and the six project-page
  figures.
- `current_model_sources.json` pins model revisions for a future rerun. It does not claim that the
  historical thesis jobs used those exact commits.
- `historical_selection_bias.json` transcribes the reported median target-length shift from 15 to 8
  characters. It marks the number as historical aggregate-only because the selection rows are gone.

## Historical results

`historical_benchmark_summary.csv` transcribes the final defense table for Base, Product SFT, and
Product DPO. These rows are aggregate-only evidence. The original per-sample score files, run
manifests, and checkpoint hashes are not present in the repository, so the table cannot be
independently recomputed from this checkout.

The reported Product score used the thesis formula `VLM × OCR`. The current scoring command now
records that formula as `thesis_vlm_ocr_product_v1`; the later five-component diagnostic formula has
a different name and must be selected explicitly.

The machine-readable selection-bias record preserves one other defense aggregate: median target
length changed from 15 characters before Product selection to 8 afterward. It cannot be recomputed
from this checkout and does not establish why the shift occurred.

Do not treat the historical table as a fresh benchmark of the new target-disjoint prompt set. A
comparison on `benchmark_prompts_v2.jsonl` requires the original Base, Product SFT, and Product DPO
artifacts on a Linux/CUDA host, followed by multi-seed scoring and aggregation.

## Missing artifacts

The following evidence must come from the original training storage or a new CUDA run:

- Product SFT and Product DPO LoRA checkpoints;
- immutable training and evaluation run manifests;
- per-sample Base/SFT/DPO OCR and VLM score files;
- training metrics tied to exact config snapshots;
- blind human-evaluation labels.

The figures under `docs/project-page/assets/` remain useful illustrations, but their source rows are
missing. The evidence manifest labels them as static defense assets rather than recomputable plots.
