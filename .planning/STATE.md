---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: Phase 4 - CPU-Safe Characterization Tests
current_plan: 04-02-PLAN.md
status: phase-4-in-progress
last_updated: "2026-05-05T18:07:16Z"
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 6
  completed_plans: 1
  percent: 17
---

# Project State: Diffusion Text Tuner

**Initialized:** 2026-05-04  
**Last updated:** 2026-05-05 after Phase 4 Plan 01 execution

## Project Reference

**Project:** Diffusion Text Tuner  
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.  
**Project Type:** Brownfield thesis ML research toolkit  
**Granularity:** standard  
**Parallelization:** true  
**Verification Preference:** adaptive; use explicit verification on high-risk phases and avoid over-gating obvious inventory/documentation work.

## Current Position

**Current Phase:** Phase 4 - CPU-Safe Characterization Tests  
**Current Plan:** 04-02-PLAN.md  
**Status:** Phase 4 in progress; Plan 04-01 complete
**Progress:** [███░░░░░░░░░░░░░░░░░] 17% for Phase 4

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Execution Surface and Pipeline Inventory | Verified complete | 4/4 plans complete and phase verification passed 12/12 must-haves. |
| 2. Runtime Contracts and Run Provenance | Verified complete | 5/5 plans complete and phase verification passed 5/5 must-haves. Shared config validation, canonical paths, artifact validators, runtime contract docs, local manifests, trainer loader wiring, manifest CLI, runtime preflight CLI, config-family docs, and Makefile/README command surfaces are in place. |
| 3. Data Curriculum and Dataset Quality | Verified complete | 6/6 plans complete and phase verification passed 5/5 must-haves. Phase 3 now includes prompt curriculum configs, prompt dataset validation/manifests, synthetic masked-SFT quality reports/contact sheets/manifests, materialized SFT/DPO selection artifacts, generated-vs-synthetic source comparison reports, runtime contracts, command docs, README links, Makefile aliases, and docs tests. |
| 4. CPU-Safe Characterization Tests | In progress | 1/6 plans complete. Plan 04-01 added committed-config and tiny-artifact characterization tests. |
| 5. Training Objective and Pipeline Comparability | Not started | Explicit training modes, run diffs, controlled comparisons, shared training utilities. |
| 6. Reward and Evaluation Validity | Not started | Canonical rewards, held-out eval, diagnostic/gold checks, thesis outputs. |
| 7. Moderate Structure and Extension Cleanup | Not started | Safe file structure cleanup, importable modules, extension seams. |

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| v1 requirement coverage | 58/58 mapped; Phase 1, Phase 2, DATA-01 through DATA-07, TEST-01, and TEST-02 complete | 100% mapped; continue Phase 4 characterization plans |
| Roadmap phases planned | 7 total, Phase 3 has 6 executable plans | 6-8 standard-granularity phases |
| Default test posture | 128 CPU-safe pytest tests including smoke CLI, tensor-loss, runtime config validation, runtime artifact contracts, committed-config/tiny-artifact characterization, runtime manifest contracts, runtime docs checks, runtime preflight CLI behavior, prompt curriculum/config CLI tests, prompt dataset quality/manifest CLI tests, synthetic quality/manifest/contact-sheet CLI tests, training selection artifact/CLI tests, data source comparison/CLI/docs tests, and Phase 3 data-quality docs/runtime wiring tests; diagnostics are opt-in `diagnose_*.py` scripts | CPU-safe standard command |
| Reproducible environment | `.python-version`, `pyproject.toml`, and `uv.lock` committed in Phase 1 Plan 02 | Smoke-tested setup commands after Phase 1 |
| Run tracking | Local file-backed manifests with immutable config snapshots, secret-safe reproducibility metadata, trainer config-loader wiring, CPU-safe preflight reports, config-family docs, README/Makefile command aliases, prompt-side dataset manifests, synthetic quality dataset manifests, selection summary manifests, generated-vs-synthetic comparison reports, and Phase 3 runtime/docs command wiring | Extend characterization coverage during Phase 4 |

## Accumulated Context

### Decisions

- Treat the project as a brownfield thesis toolkit, not a hosted service or greenfield rewrite.
- Use moderate, behavior-preserving refactors rather than a big-bang package reorganization.
- Stabilize environment and command discovery before runtime contracts, tests, trainer/reward refactors, and evaluation claims.
- Use simple local run manifests first; defer MLflow, Weights & Biases, DVC, object storage, and plugin frameworks to v2 unless need becomes concrete.
- Keep generated images, tensors, checkpoints, logs, and large datasets out of git; only tiny fixtures and docs should be committed.
- Use `docs/pipeline_inventory.md` as the Phase 1 source of truth for supported entry points, non-default diagnostics, historical tracks, and artifact safety boundaries.
- Use `.python-version`, `pyproject.toml`, and `uv.lock` as the Python 3.11 dependency/tooling contract; keep default pytest discovery restricted to `tests/`.
- Keep heavyweight GPU, OCR, reward, synthesis, vLLM, and MLX stacks in optional dependency extras while using the uv dev group for CPU-safe pytest execution.
- Keep environment smoke checks explicit and import-safe: listing/import checks avoid CUDA/model/OCR loading, while CUDA/model-access/OCR diagnostics require explicit `--check` choices.
- Use `docs/commands.md` and `Makefile` as the standard command surface for setup, CPU-safe tests, Ruff checks, smoke checks, local pipelines, SLURM variants, manual diagnostics, and generated-artifact safety.
- Keep manual gradient diagnostics under guarded `scripts/diagnose_*.py` names rather than pytest-style `scripts/test_*.py` names.
- Return existing trainer-facing `SFTConfig`, `DPOConfig`, and `MaskedSFTConfig` dataclasses from shared runtime config loaders; trainer wiring is deferred to Phase 2 Plan 04.
- Keep `src.runtime.config_io` CPU/import-safe by validating JSON and path strings only, without artifact existence checks or CUDA/model/OCR work.
- Use secret-safe `RuntimeConfigError` messages with config path and field context but without echoing raw user-provided config values.
- Keep artifact validators CPU-safe and model-download-free by inspecting only JSONL, CSV, directory names, file presence, and tiny trusted local tensor dictionaries with `torch.load(..., map_location="cpu", weights_only=True)`.
- Return aggregate `ArtifactReport` errors by default so users can fix all visible contract problems before expensive jobs, while allowing `require_ready=True` to raise `ArtifactValidationError` at blocking preflight gates.
- Classify generated runtime roots, checkpoints, logs, tensors, and generated images as non-committable by default, with narrow fixture exceptions for `experiments/assets/` and `tests/fixtures/`.
- Keep run manifests local and file-backed under ignored `runs/` roots, with tests using pytest temporary directories rather than committed runtime artifacts.
- Serialize secret-related environment variables as boolean presence only, and serialize cache paths as presence flags instead of private machine paths.
- Back the manifest CLI directly with `src.runtime.manifests` so command behavior remains CPU-safe and import-safe before GPU/model/OCR stages launch.
- Use `scripts/preflight_runtime.py` as the CPU-safe preflight gate for config, artifact, and manifest readiness before generation, scoring, training, synthetic, or evaluation stages launch.
- Keep new experiment config variants under `configs/experiments/` using `{stage}_{reward_or_data}_{purpose}.json`, while preserving existing root configs as runnable compatibility entry points.
- Support run manifests for generation, scoring, synthesis, evaluation, SFT, DPO, and masked-SFT; non-training stages can be initialized without trainer configs, while training manifests still require validated config snapshots.
- Phase 3 will create six plans in three waves: Wave 1 prompt curriculum/configs, prompt validation/manifests, synthetic quality inspection, and materialized training selections; Wave 2 source comparison; Wave 3 runtime/docs command wiring.
- Phase 3 research found the existing project research summary's old Phase 3 test-harness framing insufficient for the current roadmap, so `.planning/phases/03-data-curriculum-and-dataset-quality/03-RESEARCH.md` is the Phase 3 planning research source.
- Use explicit prompt config files (`configs/prompts/simple.json`, `configs/prompts/full.json`, `configs/prompts/curriculum.json`) plus `python -m src.prompt_pipeline.generate --config ...` instead of editing prompt-generation constants for simple/full/curriculum modes.
- Config-driven prompt records include `prompt_mode`, `curriculum_stage`, and `curriculum_family` provenance fields; legacy flag-only generation remains compatible and does not add those fields.
- Use pure-Python prompt dataset validators for DATA-02 so malformed JSONL, required fields, length/script/rare-character/duplicate/distribution checks, and naturalness heuristics stay CPU-safe and deterministic.
- Use `dataset-manifest/v1` for prompt-side DATA-04 manifests, hashing safe small text sources while referencing generated binary tensors/images by path unless explicitly marked safe.
- Keep synthetic masked-SFT quality inspection CPU-safe by using PIL/CSV/JSON only and ingesting OCR verification solely from optional precomputed result files.
- Write synthetic quality reports, dataset manifests, and contact sheets only to explicit runtime output paths; generated images, tensors, reports, contact sheets, and OCR outputs remain non-committable by default.
- Reuse `dataset-manifest/v1` for synthetic provenance so later selection/comparison plans can trace filtering stats, source hashes, report paths, thresholds, and contact-sheet paths.
- Materialize reward-filtered training selections as JSONL metadata artifacts before comparison-grade runs, keeping current trainer loaders unchanged until a later optional loader-wiring plan.
- Treat DPO preference artifacts as invalid unless the winner score is strictly greater than the loser score and the configured margin/ambiguity filters pass.
- Compare generated reward-filtered data against synthetic masked-SFT data using metadata-only JSON/JSONL reports and manifests; missing optional evidence must remain explicit in comparison reports.
- Keep Phase 3 runtime validators shallow and CPU-safe: validate JSON/JSONL presence, `schema_version`, and required fields without inspecting generated images/tensors or invoking OCR/model stacks.
- Use `docs/commands.md`, `README.md`, and Makefile aliases as the standard discovery surface for Phase 3 prompt validation, synthetic inspection, selection materialization, and source comparison workflows.
- Treat generated Phase 3 reports, manifests, contact sheets, selected samples, preference pairs, and comparison outputs as non-committable runtime artifacts by default.
- Capture synthetic dataset provenance through `scripts/inspect_synthetic_dataset.py` flags for build config, seed, template/runner, model metadata, and word/font/scene/background sources before using synthetic data as thesis evidence.
- Use real committed SFT, DPO, and masked-SFT root configs as Phase 4 characterization fixtures for runtime config validation.
- Keep Phase 4 artifact characterization fixtures in pytest `tmp_path`, using only tiny trusted local tensor dictionaries inspected with `torch.load(..., map_location="cpu", weights_only=True)`.
- Preserve both mapping-based and keyword-based `require_ready` artifact readiness APIs so preflight callers can aggregate blocking errors before expensive stages.

### Important Caveats

- `REQUIREMENTS.md` listed coverage as 57 v1 requirements, but the actual v1 ID list contains 58 requirements. Traceability now maps all 58.
- Exact CUDA/PyTorch/Diffusers/Transformers/Accelerate/PEFT/PaddleOCR/vLLM/SynthTIGER versions must be smoke-tested on real local/SLURM environments.
- Reward/evaluation semantics are research-critical and should receive explicit verification before thesis claims.

### Known Risks to Preserve in Planning

- Expensive diagnostics named like tests can accidentally trigger CUDA/model downloads.
- Trainer modules combine many responsibilities; refactor only after characterization tests exist.
- Reward logic is duplicated across scoring, training, evaluation, and experiments, creating drift risk.
- Artifact/path/tensor contracts are now explicit for Phase 2 core families, but later data-selection/evaluation plans still need materialized selected-sample, preference-pair, and eval schemas.
- Hardcoded personal paths and unpinned model revisions can break reproducibility.

### Open Todos

- Execute Phase 4 Plan 02 dataset, collator, selection, and resolution-bucket characterization tests.
- Validate exact dependency pins and CUDA/module constraints on target machines with explicit smoke checks.
- Keep ROADMAP.md and REQUIREMENTS.md traceability synchronized after phase revisions.

### Blockers

- None for roadmap review.

## Session Continuity

**Next Recommended Action:** Execute Phase 4 Plan 02 dataset/collator/selection characterization tests.

**Files Created/Updated:**

- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/REQUIREMENTS.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-RESEARCH.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-VALIDATION.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-PATTERNS.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-01-PLAN.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-02-PLAN.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-03-PLAN.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-04-PLAN.md`
- `docs/pipeline_inventory.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-01-SUMMARY.md`
- `.python-version`
- `pyproject.toml`
- `uv.lock`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-02-SUMMARY.md`
- `scripts/smoke_environment.py`
- `tests/test_smoke_environment.py`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-03-SUMMARY.md`
- `scripts/diagnose_gradient_flow.py`
- `scripts/diagnose_grad_magnitude.py`
- `docs/commands.md`
- `Makefile`
- `README.md`
- `docs/pipeline_inventory.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/01-04-SUMMARY.md`
- `.planning/phases/01-execution-surface-and-pipeline-inventory/VERIFICATION.md`
- `tests/test_runtime_config_io.py`
- `src/runtime/__init__.py`
- `src/runtime/config_io.py`
- `pyproject.toml`
- `uv.lock`
- `.planning/phases/02-runtime-contracts-and-run-provenance/02-01-SUMMARY.md`
- `tests/test_runtime_artifacts.py`
- `src/runtime/paths.py`
- `src/runtime/artifacts.py`
- `docs/runtime_contracts.md`
- `.planning/phases/02-runtime-contracts-and-run-provenance/02-02-SUMMARY.md`
- `tests/test_runtime_manifests.py`
- `src/runtime/reproducibility.py`
- `src/runtime/manifests.py`
- `scripts/run_manifest.py`
- `.planning/phases/02-runtime-contracts-and-run-provenance/02-03-SUMMARY.md`
- `tests/test_runtime_preflight.py`
- `src/training/sft_trainer.py`
- `src/training/dpo_trainer.py`
- `src/training/masked_sft_trainer.py`
- `scripts/preflight_runtime.py`
- `.planning/phases/02-runtime-contracts-and-run-provenance/02-04-SUMMARY.md`
- `configs/experiments/README.md`
- `configs/experiments/sft/README.md`
- `configs/experiments/dpo/README.md`
- `configs/experiments/masked_sft/README.md`
- `configs/experiments/reward/README.md`
- `configs/experiments/evaluation/README.md`
- `configs/experiments/synthesis/README.md`
- `tests/test_runtime_docs.py`
- `.planning/phases/02-runtime-contracts-and-run-provenance/02-05-SUMMARY.md`
- `.planning/phases/02-runtime-contracts-and-run-provenance/VERIFICATION.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-CONTEXT.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-RESEARCH.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-PATTERNS.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-VALIDATION.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-01-PLAN.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-02-PLAN.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-03-PLAN.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-04-PLAN.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-05-PLAN.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-06-PLAN.md`
- `src/data_quality/__init__.py`
- `src/data_quality/curriculum.py`
- `configs/prompts/simple.json`
- `configs/prompts/full.json`
- `configs/prompts/curriculum.json`
- `src/prompt_pipeline/generate.py`
- `tests/test_prompt_curriculum.py`
- `docs/data_curriculum.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-01-SUMMARY.md`
- `src/data_quality/prompt_validation.py`
- `src/data_quality/manifests.py`
- `scripts/validate_prompt_dataset.py`
- `tests/test_prompt_dataset_quality.py`
- `docs/dataset_quality.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-02-SUMMARY.md`
- `src/data_quality/synthetic_quality.py`
- `scripts/inspect_synthetic_dataset.py`
- `tests/test_synthetic_quality.py`
- `docs/synthetic_quality.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-03-SUMMARY.md`
- `src/training/selection.py`
- `scripts/materialize_training_data.py`
- `tests/test_training_selection_artifacts.py`
- `docs/data_selection.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-04-SUMMARY.md`
- `src/data_quality/source_comparison.py`
- `scripts/compare_data_sources.py`
- `tests/test_data_source_comparison.py`
- `docs/data_source_comparison.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-05-SUMMARY.md`
- `tests/test_data_quality_docs.py`
- `src/runtime/paths.py`
- `src/runtime/artifacts.py`
- `docs/runtime_contracts.md`
- `docs/commands.md`
- `README.md`
- `Makefile`
- `.planning/phases/03-data-curriculum-and-dataset-quality/03-06-SUMMARY.md`
- `.planning/phases/03-data-curriculum-and-dataset-quality/VERIFICATION.md`
- `.planning/phases/04-cpu-safe-characterization-tests/04-CONTEXT.md`
- `.planning/phases/04-cpu-safe-characterization-tests/04-RESEARCH.md`
- `.planning/phases/04-cpu-safe-characterization-tests/04-PATTERNS.md`
- `.planning/phases/04-cpu-safe-characterization-tests/04-01-PLAN.md`
- `tests/test_characterization_config_artifacts.py`
- `configs/masked_sft.json`
- `src/runtime/artifacts.py`
- `.planning/phases/04-cpu-safe-characterization-tests/04-01-SUMMARY.md`

**Do Not Forget:** Commit approved planning artifacts only; leave unrelated worktree changes untouched.

---
*State initialized: 2026-05-04*
