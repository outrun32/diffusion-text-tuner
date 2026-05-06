# Thesis output bundles

Thesis output bundles turn recorded run evidence into thesis-ready tables, SVG plots, and contact sheets without rerunning generation, OCR, VLM scoring, CUDA, or model-weight code. The bundle is a provenance layer over existing artifacts: it reads local run manifests, score reports, diagnostic reports, and optional image paths, then writes deterministic runtime artifacts for figures and tables.

Use this workflow when thesis text needs a result table, visual score plot, or contact sheet that can be traced back to exact run manifests and recorded reports instead of manual/static numbers.

## Bundle config

The input config uses `thesis-output-config/v1` and is intentionally local-file based:

```json
{
  "schema_version": "thesis-output-config/v1",
  "source_manifests": ["runs/eval-sft/manifest.json", "runs/eval-dpo/manifest.json"],
  "score_reports": ["runs/eval-sft/scores.json"],
  "diagnostic_reports": ["runs/eval-sft/reward_diagnostics.json"],
  "output_dir": "outputs/thesis/eval_bundle",
  "table_specs": [
    {
      "name": "score_summary",
      "source": "runs/eval-sft/scores.json",
      "columns": ["sample_id", "run_id", "product_score", "exact_match"],
      "output_csv": "tables/score_summary.csv"
    }
  ],
  "svg_plot_specs": [
    {
      "name": "product_scores",
      "source": "runs/eval-sft/scores.json",
      "x": "sample_id",
      "y": "product_score",
      "output_svg": "plots/product_scores.svg",
      "title": "Product scores"
    }
  ],
  "contact_sheet_specs": [
    {
      "name": "false_rows",
      "output_image": "contact_sheets/false_rows.png",
      "limit": 12,
      "images": [
        {"sample_id": "ru-001", "path": "outputs/eval/images/ru-001.png", "caption": "FP ru-001"}
      ]
    }
  ]
}
```

Required fields:

- `source_manifests`: run manifest JSON files created by the runtime manifest helpers. Each manifest contributes run ID, stage, git state, config snapshot, input paths, output paths, and metrics.
- `score_reports`: score CSV/JSON/JSONL files or score report JSON files with a `records` list. These are the source rows for tables and SVG plots.
- `diagnostic_reports`: reward diagnostic JSON files, such as `reward-diagnostics/v1` outputs from disagreement analysis.
- `output_dir`: runtime output root for generated bundle artifacts.
- `table_specs`: deterministic CSV table definitions with `source`, `columns`, and `output_csv`.
- `svg_plot_specs`: simple deterministic SVG plot definitions with `source`, `x`, `y`, and `output_svg`.
- `contact_sheet_specs`: bounded PIL contact sheets with explicit image `path`, `sample_id`, and `caption` entries.

## Command

Build bundle JSON and Markdown summaries with the thin CLI:

```bash
PATH="/root/.local/bin:$PATH" uv run python scripts/build_thesis_outputs.py \
  --config configs/thesis/eval_bundle.json \
  --output-bundle outputs/thesis/eval_bundle/bundle.json \
  --markdown-summary outputs/thesis/eval_bundle/bundle.md
```

The command is CPU-safe. It imports `src.evaluation.thesis_outputs`, reads local JSON/CSV/JSONL files, writes text/SVG/PNG outputs, and never loads FLUX, Qwen, PaddleOCR, CUDA, tensors, checkpoints, or model weights.

## Readiness blocking errors

The bundle records readiness blocking errors whenever required provenance is absent or malformed. A non-ready bundle is not thesis-ready and the CLI exits nonzero. Blocking examples include:

- missing source manifest paths;
- malformed run manifests or missing config snapshots;
- missing score reports or diagnostic reports;
- score report JSON whose `records` field is not a list;
- table or SVG specs whose source report is missing;
- table specs with no columns.

Warnings are separate from readiness blocking errors. For example, a contact sheet can record a missing image path as a warning while preserving the source path in the bundle so the missing visual evidence is explicit.

## Outputs and provenance

The builder writes a `thesis-output-bundle/v1` JSON object and optional Markdown summary. The JSON includes:

- `source_manifests`: each manifest path, run ID, stage, git state, config snapshot reference, config snapshot content, inputs, outputs, and metrics;
- `evidence.score_reports`: each score report path, schema version, record count, and top-level keys;
- `evidence.diagnostic_reports`: each diagnostic report path, schema version, record count, and top-level keys;
- `tables`: output CSV paths, row counts, columns, and source report paths;
- `svg_plots`: output SVG paths, plotted fields, point counts, and source report paths;
- `contact_sheets`: output PNG paths, bounded entry counts, image source paths, captions, and limits;
- `readiness`: `ready`, warnings, and blocking readiness errors.

This means generated tables, SVG plots, contact sheets, bundle JSON, and Markdown remain runtime artifacts tied to recorded manifests and reports. They should not be edited by hand to change results; update the recorded score/report source and rebuild the bundle instead.

## Mapping thesis claims back to evidence

When using a bundle in the thesis:

1. Cite the bundle JSON path and the generated table/plot/contact-sheet path.
2. Use the Markdown source manifest table to identify exact run manifests and git commits.
3. Inspect each manifest's config snapshot to verify model IDs, seeds, prompts, scoring settings, and output artifact paths.
4. Use the score and diagnostic report paths to verify product scores, missing evidence, reward disagreement summaries, false rows, and contact-sheet source images.
5. Treat any readiness blocking errors as a hard stop before claiming a table or figure is thesis-ready.

Generated CSV tables, SVG plots, contact sheets, bundle JSON, and Markdown are runtime artifacts. Keep them under ignored output roots such as `outputs/` or `runs/` unless a future plan intentionally adds a tiny fixture or documentation asset.

## Safety notes

- The builder does not inspect tensors, checkpoints, generated latents, CUDA devices, OCR engines, or model weights.
- Contact sheets open only explicit local image paths with PIL and are bounded by each spec's `limit`.
- The bundle records paths and manifest metadata for traceability; do not place secrets in manifests or report paths.
- Missing reward/evaluation evidence is surfaced explicitly rather than treated as comparable or thesis-ready.
