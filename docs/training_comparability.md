# Training Comparability Checks

Use the CPU-safe training comparability checker before treating baseline, SFT, DPO,
masked-SFT, combined, or curriculum run outputs as comparable evidence. The checker reads
local config JSON or run manifest JSON only; it does not load CUDA, FLUX, Qwen, PaddleOCR,
OCR, tensors, checkpoints, or generated images.

## Controlled fields

The report compares controlled fields that affect whether two training approaches can be
interpreted side by side:

| Group | Fields | Severity |
|-------|--------|----------|
| Training | `num_training_steps` | warning |
| Inference | `num_inference_steps`, `guidance_scale`, `prompt_embedding_padding` | blocking |
| Prompt | `seed`, `sample_prompt`, `sample_target_text` | blocking |
| Model | `model_id` | blocking |
| Data source | `latents_dir`, `text_embeds_dir`, `scores_csv`, `data_dir` | blocking |
| Reward | `score_column`, `reward_model`, `scorer` | blocking |
| Metrics | `metric_columns` | warning |
| Artifacts | `samples_dir` | warning |

Missing-vs-present fields are reported explicitly as `missing_left` or `missing_right`.
This is intentional: older configs may not contain fields such as `guidance_scale` or
`prompt_embedding_padding`, but absence is still an uncontrolled comparison choice.

## Training/inference mismatches

Training/inference mismatches are differences that can make visual samples or metrics
misleading even when the trainer objective appears unchanged. Examples include comparing a
DPO run sampled with more inference steps against an SFT run sampled with fewer steps,
using different prompt embedding padding behavior, or changing the prompt/seed while
claiming that the training objective caused the output difference.

Blocking mismatches should be resolved before thesis-grade comparisons unless the report is
being used only for exploratory debugging. Warning mismatches identify metric or artifact
availability differences that should be documented when interpreting results.

## Config comparison command

Compare validated SFT, DPO, or masked-SFT configs without launching training:

```bash
uv run python scripts/check_training_comparability.py \
    --left-config configs/sft.json \
    --left-stage sft \
    --right-config configs/dpo.json \
    --right-stage dpo \
    --markdown
```

The config mode delegates parsing and validation to `src.runtime.config_io.load_stage_config`
and compares immutable snapshots produced by `resolve_config_snapshot`. The command exits
`1` when blocking mismatches exist. Use `--allow-blocking` only when you want automation to
record the mismatch report without failing the current exploratory step:

```bash
uv run python scripts/check_training_comparability.py \
    --left-config configs/sft.json \
    --left-stage sft \
    --right-config configs/dpo.json \
    --right-stage dpo \
    --output runs/comparisons/sft-vs-dpo-comparability.json \
    --allow-blocking
```

## Manifest comparison command

Compare two local run manifests after they have captured config snapshots, metrics, and
artifact metadata:

```bash
uv run python scripts/check_training_comparability.py \
    --left-manifest runs/<baseline-run>/manifest.json \
    --right-manifest runs/<trained-run>/manifest.json \
    --markdown \
    --output runs/comparisons/baseline-vs-trained-comparability.md
```

Manifest mode delegates loading to `src.runtime.manifests.load_run_manifest` through
`compare_training_manifests`, then applies the same controlled-field comparison logic used
for config snapshots.

## Shared trainer seams

Trainer variants should grow through focused shared modules instead of adding more unrelated
responsibility to the large SFT, DPO, or masked-SFT trainer loops:

- `src.training.sampling` owns sampling interval decisions and eval-suite prompt item
  normalization that can be tested without model loading.
- `src.training.checkpointing` owns checkpoint interval decisions and standard checkpoint
  directory naming such as `checkpoints/step_000100`.
- `src.training.schedulers` re-exports shared schedule helpers, including the DPO-backed
  `compute_sigma` behavior that existing objective tests lock down.
- `src.training.runtime` owns config-snapshot-to-manifest input/output metadata extraction for
  training runs.

Do not add unrelated sampling, checkpointing, scheduler, or runtime code directly to large trainer modules.
Keep trainer-loop changes small, compatibility-focused, and backed by CPU-safe tests before wiring
them into GPU/model execution paths.

## Adding a trainer variant

Use this flow when adding a new SFT, DPO, masked-SFT, combined, or curriculum trainer variant:

1. Define explicit config fields for every objective, data-selection, scheduler, sampling,
   checkpointing, LoRA, and evaluation choice that affects comparability.
2. Materialize data artifacts first, such as selected samples, preference pairs, prompt suites,
   or synthetic dataset indexes, rather than hiding selection inside a trainer loop.
3. Validate the config through `load_stage_config` so path policy, model ID, precision, and
   choice fields are checked before expensive training begins.
4. Use shared sampling, checkpointing, scheduler, and runtime helpers from `src.training.sampling`,
   `src.training.checkpointing`, `src.training.schedulers`, and `src.training.runtime`.
5. create/compare run manifests so config snapshots, selected artifacts, output directories,
   metrics, and notes remain traceable across baseline and trained runs.
6. run CPU-safe tests for config validation, shared helper contracts, materialized artifacts,
   objective math, and comparability reports before launching CUDA/model/OCR jobs.

## Generated artifact safety

Reports are local runtime artifacts by default. Write generated comparability reports under
ignored roots such as `runs/` or `outputs/` unless a tiny reviewed fixture or documentation
example is intentionally committed. Do not commit generated images, tensors, checkpoints,
logs, private run manifests, or large comparison output dumps.
