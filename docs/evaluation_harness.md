# Held-out Evaluation Harness

The held-out evaluation harness defines the comparison contract for baseline and
trained LoRA checkpoints before any expensive generation or reward scoring job is
launched. It is implemented by `src.evaluation.heldout` and exposes
`HeldoutEvaluationConfig`, `EvaluationTarget`, `build_evaluation_plan`, and
`write_evaluation_plan`.

The CLI is intentionally `materialize-only`: it validates config/report
contracts and writes JSON/Markdown plans, but it does not run FLUX, Qwen, PaddleOCR, CUDA, or model weights. Generation and scoring remain explicit local
or SLURM jobs that researchers launch after reviewing the materialized plan.

## Config schema

Use schema `heldout-evaluation-config/v1` for JSON configs:

```json
{
  "schema_version": "heldout-evaluation-config/v1",
  "fixed_prompts_path": "data/evaluation/heldout_prompts.jsonl",
  "fixed_seeds": [101, 202, 303],
  "inference_settings": {
    "model": "black-forest-labs/FLUX.2-klein-4B",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "scorer": "both"
  },
  "output_root": "runs/evaluation/heldout-001",
  "targets": [
    {
      "name": "baseline",
      "lora_checkpoint_path": null,
      "source_run_manifest_path": "runs/baseline/manifest.json",
      "generation_output_path": "runs/evaluation/heldout-001/baseline/generated",
      "score_output_path": "runs/evaluation/heldout-001/baseline/scores.csv",
      "notes": ["unadapted baseline checkpoint"]
    },
    {
      "name": "dpo-product-lora",
      "lora_checkpoint_path": "runs/dpo-product/checkpoints/final",
      "source_run_manifest_path": "runs/dpo-product/manifest.json",
      "generation_output_path": "runs/evaluation/heldout-001/dpo-product-lora/generated",
      "score_output_path": "runs/evaluation/heldout-001/dpo-product-lora/scores.csv",
      "notes": ["trained LoRA comparison target"]
    }
  ]
}
```

Required fields:

- `fixed_prompts_path`: JSONL file with stable held-out prompts. Each row must
  contain `prompt` and `target_text`.
- `fixed_seeds`: non-empty integer seed list shared by every target.
- `inference_settings`: fixed settings for model ID, resolution, steps,
  guidance, and optional scoring mode. These settings are copied into the
  `heldout-evaluation-plan/v1` report so later comparisons can prove they used
  the same controls.
- `output_root`: writable runtime root for held-out outputs. Traversal (`..`) and
  home expansion (`~`) are rejected for writable paths.
- `targets`: at least one `baseline` target with `lora_checkpoint_path: null`
  plus at least one trained target with a `lora_checkpoint_path`.
- `source_run_manifest_path`: manifest link for each target. This ties the
  comparison target to its Phase 5 training/generation provenance.
- `generation_output_path` and `score_output_path`: planned runtime artifact
  paths under `output_root`.
- `notes`: optional per-target context for thesis review.

## Materialize a local plan

```bash
python -m scripts.run_heldout_evaluation \
  --config configs/experiments/evaluation/heldout_product_vs_baseline.json \
  --output-plan runs/evaluation/heldout-001/plan.json \
  --markdown-summary runs/evaluation/heldout-001/plan.md
```

The JSON report uses schema `heldout-evaluation-plan/v1` and includes:

- `source_config_path`, `fixed_prompts_path`, `fixed_seeds`, and
  `inference_settings`.
- One entry per `EvaluationTarget`, including `lora_checkpoint_path`,
  `source_run_manifest_path`, `generation_output_path`, `score_output_path`, and
  `notes`.
- `planned_generation_commands` for every target/seed pair.
- `planned_scoring_commands` for every target.
- `manifest_links` summarizing all source run manifests.

Every planned command has `status: planned-not-run`. The harness only writes the
plan and summary; it does not execute image generation or reward scoring.

## SLURM-compatible template

Materialize the same plan on a login node or local CPU environment, then submit
the reviewed planned commands through your cluster wrapper:

```bash
python -m scripts.run_heldout_evaluation \
  --config configs/experiments/evaluation/heldout_product_vs_baseline.json \
  --output-plan runs/evaluation/heldout-001/plan.json \
  --markdown-summary runs/evaluation/heldout-001/plan.md

sbatch --job-name=heldout-generate --wrap="<planned generation command from plan.json>"
sbatch --job-name=heldout-score --wrap="<planned scoring command from plan.json>"
```

The default automated tests cover only JSON fixtures and temporary directories;
they do not make GPU/model/OCR work part of test discovery.

## Comparison prerequisites from Phase 5

Before trusting a held-out comparison, Phase 5 comparability evidence should be
available for the baseline and trained targets:

1. Source `manifest.json` files exist for every comparison target.
2. Training and generation manifests record controlled prompts, seeds, inference
   settings, model IDs, reward choices, and artifact paths.
3. Known differences between baseline, SFT, DPO, masked-SFT, combined, or
   curriculum runs are explicit rather than hidden in scripts.
4. Missing metrics or artifacts are surfaced as incomplete evidence, not treated
   as comparable proof.

## Generated-artifact safety

Held-out generation outputs, score files, generated images, tensors, checkpoints, logs, contact sheets, and private run artifacts belong under ignored
runtime roots such as `runs/` or `outputs/`. Do not commit generated images,
tensors, checkpoints, logs, or large evaluation outputs. Only tiny docs or test
fixtures should be committed.
