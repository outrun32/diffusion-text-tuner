# Runtime Artifact Contracts

This document defines the Phase 2 and Phase 3 filesystem contracts that generation, scoring,
training, synthesis, data-quality, evaluation, and run-provenance helpers share. The contracts are
intentionally local and SLURM-compatible: paths are relative to the repository or to a job
workspace root, never to a personal absolute directory.

## Canonical Runtime Roots

| Root | Purpose | Git-safety classification | Notes |
|------|---------|---------------------------|-------|
| `data/` | Prompt resources and generated synthetic dataset roots | Mixed: source metadata may be committed; generated tensors/images remain non-committable | Use relative paths in configs and scripts. |
| `outputs/` | Generated images, latents, scores, samples, checkpoints, evaluation outputs, and plots | Non-committable runtime output | Ignored by `.gitignore`; safe for local and SLURM scratch/workspace output. |
| `runs/` | Run manifests, config snapshots, logs, and provenance metadata | Non-committable runtime output | Plan 02-03 owns manifest creation; this plan reserves the path contract. |
| `configs/` | Committed config inputs for local and SLURM commands | Committable source | Configs should use relative runtime paths and avoid home/off-repo absolutes. |
| `experiments/assets/` | Tiny documented image assets for OCR/VLM experiments | Allowed fixture/documentation exception | `.gitignore` explicitly allows PNG/JPG files in this directory. |
| `tests/fixtures/` | Future tiny test fixtures | Allowed fixture exception | Prefer `tmp_path` fixtures for tensors; do not commit generated binary artifacts unless intentionally tiny and reviewed. |

Use `src.runtime.paths.resolve_stage_paths(stage, root=...)` to inspect canonical defaults for a
local checkout, a SLURM job working directory, or an explicit run workspace. The helper returns
`RuntimePaths(schema-neutral path mapping)` with `data/`, `outputs/`, `runs/`, and `configs/`
anchored below the supplied root.

## Artifact Contract Matrix

| Artifact family | Canonical path | Producer | Consumer | Required fields / keys | Schema/version metadata | Preflight hook | Resume / inspect notes | Git-safety |
|-----------------|----------------|----------|----------|------------------------|-------------------------|----------------|------------------------|------------|
| prompts | `data/prompts_simple.jsonl` or stage-specific JSONL | `src.prompt_pipeline.generate`, `scripts/download_dataset.py`, synthetic builders | `scripts.generate_images`, validators, future manifests | JSONL object per line with required `prompt`; optional `target_text` | `runtime-artifacts/v1`; future prompt manifests may add `schema_version` | `validate_artifacts("prompts", {"prompts_jsonl": ...})` | Malformed JSON reports line numbers; empty lines are ignored | Generated large prompt JSONL files are ignored unless intentionally published. |
| images | `outputs/generated/images/{prompt_id}/v{version}.png` | `scripts.generate_images` | `scripts.score_images`, evaluation, contact sheets | Prompt directory ID and `v*.png` version naming | Covered by `runtime-artifacts/v1`; per-run details move to manifests | `validate_artifacts("generated", paths)` | Missing matching latent/image versions are reported before scoring/training | Non-committable except `experiments/assets/*.png`/`.jpg`. |
| latents | `outputs/generated/latents/{prompt_id}/v{version}.pt` | `scripts.generate_images` | `SFTDataset`, `DPODataset`, validators | Trusted local `.pt` dictionary with `latent` | Covered by `runtime-artifacts/v1`; tensor provenance belongs in manifests | `validate_artifacts("generated", paths)` or training preflight | Loaded only with `torch.load(..., map_location="cpu", weights_only=True)` for tiny/local validation | Non-committable generated tensor. |
| text embeddings | `outputs/generated/text_embeds/{prompt_id}.pt` | `scripts.generate_images` | scoring, SFT, DPO | Trusted local `.pt` dictionary with `prompt_embeds`, `target_text`, `prompt` | Covered by `runtime-artifacts/v1` | `validate_artifacts("generated", paths)` | Missing embeddings are reported by prompt ID | Non-committable generated tensor. |
| scores | `outputs/generated/scores.csv` | `scripts.score_images` | SFT/DPO sample and preference selection | CSV columns `id`, `version`, `score`, `target_text`; scorer-specific extra columns allowed | Optional `outputs/generated/scores.schema.json` with `schema_version` such as `scores/v1` | `validate_artifacts("scores", {"scores_csv": ...})` | `--resume` appends/skips in scorer; validators report bad columns/numeric values | Non-committable generated CSV under `outputs/`. |
| masks | `data/synth_cyrillic/masked_sft/raw_masks/{sample_id}.png` and latent masks in `.pt` | `scripts.synth.build_dataset` | `MaskedSFTDataset`, mask diagnostics, validators | Raw mask PNG plus trusted local latent dictionary key `mask_lat` | Covered by `runtime-artifacts/v1`; synthetic manifests later capture render config | `validate_artifacts("masked_sft", {"data_dir": ...})` | Mask tensors are checked with matching latent/text embedding sample IDs | Non-committable generated image/tensor. |
| synthetic index | `data/synth_cyrillic/masked_sft/index.csv`, `prompts.jsonl`, `shapes.csv` | `scripts.synth.build_dataset` | masked-SFT training, bucket sampler, validators | `shapes.csv` columns `id,H,W`; index rows identify sample text/caption | Covered by `runtime-artifacts/v1` | `validate_artifacts("masked_sft", {"data_dir": ...})` | `shapes.csv` is optional but recommended to avoid slow per-sample inspection | Generated dataset metadata; commit only if intentionally tiny/reviewed. |
| dataset manifests | `runs/prompt-quality/dataset-manifest.json`, `runs/synthetic-quality/dataset-manifest.json`, or selection sidecars under `outputs/generated/` | `scripts/validate_prompt_dataset.py`, `scripts/inspect_synthetic_dataset.py`, `scripts/materialize_training_data.py` | runtime docs, comparison reports, thesis notes, future training/eval gates | `schema_version`, `dataset_kind`, `dataset_paths`, `source_hashes`, `filtering_stats`, `output_counts` | `dataset-manifest/v1` | `validate_artifacts("dataset_manifest", {"dataset_manifest": ...})` | Manifests point to generated data and hashes, not committed generated payloads | Non-committable runtime output unless intentionally tiny fixtures. |
| prompt quality reports | `runs/prompt-quality/prompt-quality.json` | `scripts/validate_prompt_dataset.py` | prompt QA review, dataset manifests, source comparison reports | `schema_version`, `valid_records`, warnings/errors, coverage and distribution metrics | `prompt-quality/v1` | `validate_artifacts("prompt_quality_report", {"prompt_quality_report": ...})` | Use before generation/training so prompt quality blockers are visible early | Non-committable runtime output by default. |
| synthetic quality reports | `runs/synthetic-quality/synthetic-quality.json` | `scripts/inspect_synthetic_dataset.py` | masked-SFT review, dataset manifests, source comparison reports | `schema_version`, `sample_count`, accepted/rejected counts, mask/bbox/contrast, character/font/resolution coverage | `synthetic-quality/v1` | `validate_artifacts("synthetic_quality_report", {"synthetic_quality_report": ...})` | OCR evidence is optional and precomputed; default inspection is PIL/CSV/JSON only | Non-committable runtime output by default. |
| contact sheets | `runs/synthetic-quality/contact-sheet.png` | `scripts/inspect_synthetic_dataset.py --contact-sheet` | human visual inspection and thesis notes after review | PNG image grid with sample IDs, accepted/rejected status, target text, rejection reasons | Referenced from `dataset-manifest/v1`; no separate JSON schema | `assert_artifact_git_safety([contact_sheet])` | May expose generated images and prompt text; promote only reviewed docs assets | Non-committable runtime output outside allowed fixture/doc roots. |
| selected samples | `outputs/generated/selected_samples.jsonl` | `scripts/materialize_training_data.py --kind sft` | SFT configs/manifests, source comparison reports, analysis | JSONL rows with `schema_version`, `sample_id`, `prompt_id`, `version`, `selected_score`, `score_column`, source hash | `selected-samples/v1` | `validate_artifacts("selected_samples", {"selected_samples": ...})` | Inspect as materialized SFT sample selection for reproducibility | Non-committable under `outputs/` by default. |
| preference pairs | `outputs/generated/preference_pairs.jsonl` | `scripts/materialize_training_data.py --kind dpo` | DPO training, source comparison reports, manifest comparison | JSONL rows with `schema_version`, `pair_id`, `prompt_id`, `winner_version`, `loser_version`, winner/loser scores, margin | `preference-pairs/v1` | `validate_artifacts("preference_pairs", {"preference_pairs": ...})` | Prefer materialized pairs before comparison-grade DPO runs; winners must score above losers | Non-committable under `outputs/` by default. |
| source comparison reports | `runs/comparisons/generated-vs-synthetic.json` and optional Markdown summary | `scripts/compare_data_sources.py` | experiment planning, thesis notes, Phase 5/6 comparison setup | `schema_version`, `evidence_available`, `evidence_missing`, counts, distributions, expected help/failure, provenance | `data-source-comparison/v1` | `validate_artifacts("data_source_comparison", {"data_source_comparison": ...})` | Missing optional evidence remains explicit instead of fabricating metrics | Non-committable runtime output by default. |
| checkpoints | `outputs/{sft,dpo,masked_sft}/checkpoints/...` | trainers via PEFT/Accelerate | resume, sampling, evaluation | LoRA/checkpoint files such as `.safetensors`, `.pt`, `.bin` | Run manifests should record checkpoint step and schema in Plan 02-03 | `validate_artifacts("checkpoints", {"checkpoints_dir": ...})` | Resume paths should be relative or repo-contained per config path policy | Non-committable model artifact. |
| samples | `outputs/{sft,dpo,masked_sft,evaluation}/samples/...` | trainers and evaluation commands | visual inspection, thesis contact sheets | Image files plus optional per-sample metadata | Run/eval manifests should capture producer config | Stage-specific output validation and git-safety checks | Use contact sheets or copied docs assets for publication, not raw runtime roots | Non-committable generated images outside allowed docs/experiment assets. |
| logs | `runs/<run_id>/`, `runs/{sft,dpo,masked_sft}/`, `*.log` | local commands, SLURM, trainers | debugging, provenance, thesis notes | Text logs and structured run metadata | Run manifest `schema_version` starts in Plan 02-03 | `validate_artifacts("logs", {"logs_dir": ...})` plus `assert_artifact_git_safety` | SLURM stdout/stderr should stay in runtime/log roots | Non-committable and ignored by `*.log`, `runs/`. |
| eval outputs | `outputs/evaluation/` | `src.evaluation.*`, future eval harness | thesis plots, reward diagnostics | Scores CSV, generated samples, report files | Future `evaluation/v1`; current path reserved | `validate_artifacts("evaluation", {"outputs_dir": ..., "scores_csv": ...})` | Tie final claims back to run manifests and fixed prompt/eval configs | Non-committable generated outputs. |
| run manifests | `runs/<run_id>/manifest.json` | Plan 02-03 runtime manifest helper | all long-running stages, comparison tools | JSON object with command, git/config/env/artifact metadata | Manifest contains `schema_version` (planned) | `validate_artifacts("run_manifest", {"manifest_json": ...})` | `run_id` should be stable and not include personal absolute paths | Non-committable local provenance by default. |

## Local and SLURM Path Guidance

- Run local commands from the repository root, or pass a workspace root to
  `resolve_stage_paths(stage, root=Path.cwd())` before launching a stage.
- In SLURM jobs, set the working directory to the checkout or a job workspace and keep config paths
  relative, for example `outputs/generated`, `data/synth_cyrillic/masked_sft`, and `runs/<run_id>`.
- Do not write personal paths such as `/home/<user>/...`, `/Users/<name>/...`, or `~/...` into
  committed configs. Plan 02-01 `validate_path_policy` rejects home paths, traversal, and off-repo
  absolutes for committed config fields.
- It is fine for a job wrapper to bind a scratch/work directory at runtime; capture the resolved root
  in the run manifest instead of hardcoding it in source control.

## Git-Safety Classification

`assert_artifact_git_safety(paths)` reports generated artifacts as non-committable by default when
they are under `outputs/`, `runs/`, generated data roots such as `data/synth_cyrillic/`, checkpoint
directories, or have generated suffixes like `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `.log`,
`.png`, `.jpg`, or `.jpeg`.

Allowed exceptions are intentionally narrow:

- `experiments/assets/*.png` and `experiments/assets/*.jpg` for tiny documented OCR/VLM assets.
- `tests/fixtures/` for future tiny source-controlled fixtures, preferably text/CSV/JSON. Tensor
  fixtures should normally be created under pytest `tmp_path` instead of committed.

Generated binary artifacts should not be added to the repository during validation or test runs.
Phase 3 reports, manifests, contact sheets, selected samples, preference pairs, and source
comparison files are non-committable runtime output unless they are intentionally tiny reviewed
fixtures or documentation assets.

## Preflight Validator Hooks

| Stage | Recommended call | Blocks expensive work when |
|-------|------------------|----------------------------|
| prompt generation / prompts | `validate_artifacts("prompts", {"prompts_jsonl": path})` | Prompt JSONL is missing, malformed, or lacks required `prompt` fields. |
| image generation outputs | `validate_artifacts("generated", resolve_stage_paths("generated").paths)` | Images, latents, text embeddings, prompt IDs, or versions do not line up. |
| scoring outputs | `validate_artifacts("scores", {"scores_csv": path})` | Scores CSV is missing required columns or has invalid numeric fields. |
| SFT/DPO preflight | `validate_artifacts("sft", {**paths, "require_ready": True})` | Required `scores_csv`, `latents_dir`, or `text_embeds_dir` inputs are missing. |
| masked-SFT preflight | `validate_artifacts("masked_sft", {"data_dir": path})` | Latent/text embedding sample IDs or tensor keys do not match. |
| synthetic outputs | `validate_artifacts("synthetic", paths)` | Reserved for index/selection checks as materialized dataset contracts evolve. |
| dataset manifests | `validate_artifacts("dataset_manifest", {"dataset_manifest": path})` | Manifest JSON is missing, malformed, lacks `dataset_kind` / `dataset_paths`, or has the wrong `dataset-manifest/v1` schema. |
| prompt quality reports | `validate_artifacts("prompt_quality_report", {"prompt_quality_report": path})` | Prompt quality JSON is missing, malformed, lacks `valid_records`, or has the wrong `prompt-quality/v1` schema. |
| synthetic quality reports | `validate_artifacts("synthetic_quality_report", {"synthetic_quality_report": path})` | Synthetic quality JSON is missing, malformed, lacks `sample_count`, or has the wrong `synthetic-quality/v1` schema. |
| selected samples / preference pairs | `validate_artifacts("selected_samples" | "preference_pairs", paths)` | JSONL files are missing, malformed, lack `sample_id` / `selected_score` or `winner_version` / `loser_version`, or use the wrong schema. |
| source comparison reports | `validate_artifacts("data_source_comparison", {"data_source_comparison": path})` | Comparison JSON is missing, malformed, lacks `evidence_available` / `evidence_missing`, or has the wrong `data-source-comparison/v1` schema. |
| checkpoints/logs/eval outputs | `validate_artifacts("checkpoints" | "logs" | "evaluation", paths)` | Required resume/eval paths are missing when `require_ready=True`. |
| run manifests | `validate_artifacts("run_manifest", {"manifest_json": path})` | Manifest JSON is malformed or required manifest file is absent for ready checks. |

Validators aggregate errors in `ArtifactReport.errors` so users can fix all visible contract issues
before launching GPU/model/OCR-heavy commands. When a caller passes `require_ready=True`, blocking
errors raise `ArtifactValidationError` with the same precise path and field context.

## Tensor Trust Boundary

The validators only inspect trusted local outputs produced by this toolkit. `.pt` artifacts are
loaded with `torch.load(..., map_location="cpu", weights_only=True)` to avoid CUDA allocation and to
use PyTorch's restricted weights-only loading mode. Do not point these validators at untrusted
third-party pickle artifacts; treat downloaded tensors as a separate security review item.
