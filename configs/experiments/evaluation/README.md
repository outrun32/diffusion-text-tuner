# Evaluation Configs

Place held-out evaluation, checkpoint comparison, and thesis-report variants here.

No runnable final comparison config is committed because the original Product SFT/DPO checkpoints
and source run manifests are unavailable. To materialize a held-out evaluation specification, pass
an explicit config whose manifest and checkpoint paths exist to
`uv run python -m scripts.run_heldout_evaluation`. The reviewed, target-disjoint prompt artifact
lives at `reports/final/benchmark_prompts_v2.jsonl`.

## Naming

Use `{stage}_{reward_or_data}_{purpose}.json` with `stage=evaluation`.

Examples:

- `evaluation_vlm_heldout_ru.json`
- `evaluation_product_checkpoint_comparison.json`
- `evaluation_ocr_rare_cyrillic_ablation.json`

## Required Contract

Evaluation configs should identify:

- `schema_version`, `stage: evaluation`, and `experiment_name`.
- Baseline and candidate model/checkpoint IDs, optional revisions, LoRA paths, and inference settings.
- Prompt/eval suite inputs, fixed seeds, sample counts, reward metrics, OCR/VLM/product scorer configs, and slicing dimensions.
- Output directories for generated samples, reports, metrics, plots, and contact sheets under ignored runtime roots.
- Manifest expectations linking evaluation outputs back to `runs/<run_id>/manifest.json`, config snapshots, and source checkpoints.

Final thesis figures should be generated from manifest-linked outputs, not from untracked manual numbers.
Historical aggregate-only figures are labeled in `reports/final/README.md` and must not be presented
as recomputable output.
