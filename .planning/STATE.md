---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: Phase 6 - Reward and Evaluation Validity
current_plan: Phase 6 Plan 03 ready
status: phase-6-in-progress
last_updated: "2026-05-06T14:37:35Z"
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 7
  completed_plans: 2
  percent: 29
---

# Project State: Diffusion Text Tuner

**Initialized:** 2026-05-04  
**Last updated:** 2026-05-06 after Phase 6 Plan 02 execution

## Project Reference

**Project:** Diffusion Text Tuner  
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.  
**Project Type:** Brownfield thesis ML research toolkit  
**Granularity:** standard  
**Parallelization:** true  
**Verification Preference:** adaptive; use explicit verification on high-risk phases and avoid over-gating obvious inventory/documentation work.

## Current Position

**Current Phase:** Phase 6 - Reward and Evaluation Validity  
**Current Plan:** Phase 6 Plan 03 ready  
**Status:** Phase 6 in progress; Plans 01 and 02 complete
**Progress:** [██████░░░░░░░░░░░░░░] 29% for Phase 6 execution

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Execution Surface and Pipeline Inventory | Verified complete | 4/4 plans complete and phase verification passed 12/12 must-haves. |
| 2. Runtime Contracts and Run Provenance | Verified complete | 5/5 plans complete and phase verification passed 5/5 must-haves. Shared config validation, canonical paths, artifact validators, runtime contract docs, local manifests, trainer loader wiring, manifest CLI, runtime preflight CLI, config-family docs, and Makefile/README command surfaces are in place. |
| 3. Data Curriculum and Dataset Quality | Verified complete | 6/6 plans complete and phase verification passed 5/5 must-haves. Phase 3 now includes prompt curriculum configs, prompt dataset validation/manifests, synthetic masked-SFT quality reports/contact sheets/manifests, materialized SFT/DPO selection artifacts, generated-vs-synthetic source comparison reports, runtime contracts, command docs, README links, Makefile aliases, and docs tests. |
| 4. CPU-Safe Characterization Tests | Verified complete | 6/6 plans complete and phase verification passed 8/8 must-haves. Phase 4 includes committed-config/tiny-artifact characterization, dataset/collator/selection/resolution-bucket characterization, objective math/scheduler/latent-geometry/DPO sign-beta characterization, fixed-seed prompt-generation determinism/provenance/no-LLM import-safety tests, import-safe fake/mock reward wrapper tests, and published docs/Makefile aliases guarded by docs drift tests. |
| 5. Training Objective and Pipeline Comparability | Verified complete | 6/6 plans complete and phase verification passed 5/5 must-haves. Explicit SFT/DPO selection and pair-construction modes, CPU-safe run-manifest diff tooling, controlled training comparability checks, explicit config choice snapshots, import-safe shared training utilities, and integrated comparison command docs are implemented. |
| 6. Reward and Evaluation Validity | In progress | 2/7 plans complete. Canonical reward interface/product formula and held-out evaluation plan contract are implemented; slice/gold checks, canonical scoring outputs, diagnostics, thesis outputs, and command docs remain. |
| 7. Moderate Structure and Extension Cleanup | Not started | Safe file structure cleanup, importable modules, extension seams. |

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| v1 requirement coverage | 58/58 mapped; Phase 1 through Phase 5 requirements complete and Phase 6 now complete through EVAL-03, RUN-05, and STR-03 | 100% mapped; continue Phase 6 reward/evaluation validity work |
| Roadmap phases planned | 7 total, Phase 6 has 7 executable plans | 6-8 standard-granularity phases |
| Default test posture | 6 focused reward-interface/product-formula tests, 16 focused Phase 5 selection tests, 4 focused manifest diff tests, 7 focused training comparability tests, 9 focused shared training utility tests, 4 integrated comparison docs/CLI tests, and 36 focused runtime config/characterization tests plus the previously verified CPU-safe suite; diagnostics remain opt-in `diagnose_*.py` scripts | CPU-safe standard command |
| Reproducible environment | `.python-version`, `pyproject.toml`, and `uv.lock` committed in Phase 1 Plan 02 | Smoke-tested setup commands after Phase 1 |
| Run tracking | Local file-backed manifests with immutable config snapshots, secret-safe reproducibility metadata, trainer config-loader wiring, CPU-safe preflight reports, config-family docs, README/Makefile command aliases, prompt-side dataset manifests, synthetic quality dataset manifests, selection summary manifests, generated-vs-synthetic comparison reports, Phase 3 runtime/docs command wiring, CPU-safe run-manifest diff tooling, CPU-safe training comparability reports, explicit SFT/DPO/masked-SFT config choice snapshots, shared training runtime metadata helpers, integrated training-run comparison reports, reward score metadata helpers with formula/scorer/threshold/manifest links, and held-out evaluation plan reports linking fixed prompts/seeds/settings to target manifests | Extend evaluation traceability through slice/gold checks, diagnostics, and thesis bundles during remaining Phase 6 plans |

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
- Use pytest `tmp_path` CSV/JSONL/`.pt` fixtures to characterize SFT, DPO, masked-SFT, collator, selection, and resolution-bucket dataset contracts without generated runtime roots.
- Preserve trainer loaders while cross-checking materialized Phase 3 selection artifacts against dataset constructor semantics for threshold and strict winner/loser behavior.
- Extract DPO sigma, beta, and objective math into an import-safe `src.training.dpo_objective` helper while preserving trainer-facing delegation and the existing negative beta convention.
- Keep import-side-effect tests order-independent by restoring preloaded modules or asserting only newly imported heavy modules, so full CPU-safe pytest remains reliable after torch has already loaded.
- Use fixed-seed prompt generation signatures and lightweight fakes to characterize curriculum allocation, stage metadata, stage-family text policies, config-driven record provenance, and no-LLM import safety without loading model backends.
- Keep reward wrappers import-safe by lazy-loading Paddle/PaddleOCR/Transformers/Qwen dependencies and testing Qwen/OCR behavior with object-level fakes instead of model or OCR initialization.
- Keep scoring reward class imports inside scorer selection branches so importing scoring scripts during default pytest does not load optional model/OCR stacks.
- Publish Phase 4 characterization commands through exact pytest file selections and Makefile aliases, with docs drift tests guarding CPU-safe defaults and optional slow/GPU/OCR/model/integration/manual marker boundaries.
- Treat Phase 4 DPO tests as characterization of the current negative beta convention, not proof that the convention is scientifically correct; Phase 5 must decide and compare objective behavior explicitly.
- Preserve current default selection semantics by keeping `threshold` SFT and `best_vs_worst` DPO as default modes while recording exact mode names in selection artifacts.
- Treat `score_weighted` SFT and `margin_weighted` DPO weights as deterministic normalized metadata fields rounded to 12 decimals for comparison-grade artifact diffs.
- Enforce strict DPO winner-over-loser semantics across all pair-construction modes, rejecting equal-score candidates rather than emitting ambiguous preference labels.
- Keep run-manifest comparison CPU-safe by reading only local manifest/config JSON dictionaries, metrics, and output metadata without importing torch, diffusers, transformers, OCR, or model stacks.
- Preserve secret/cache privacy in diff output by comparing only env/cache presence booleans and omitting raw cache path metadata.
- Treat model ID, prompt/seed, inference settings, data-source paths, and reward/scorer differences as blocking training comparability mismatches, while surfacing training-step, metric, and artifact availability differences as warnings.
- Keep training comparability checks CPU-safe by comparing dictionaries, dataclasses, config snapshots, and run manifest metadata only; generated images, tensors, checkpoints, logs, CUDA, and model/OCR stacks are never loaded.
- Preserve `threshold` and `best_vs_worst` as backwards-compatible SFT/DPO config defaults while exposing explicit choice fields in dataclasses and snapshots for comparison-grade manifests.
- Validate materialized selected-sample and preference-pair config paths with the existing CPU-safe runtime path policy; do not perform generated artifact existence checks during config loading.
- Defer pre-existing dirty `src/training/config.py` Ruff line-length failures rather than touching unrelated user edits during Plan 05-04 execution.
- Keep shared trainer seams in import-safe modules (`src.training.sampling`, `src.training.checkpointing`, `src.training.schedulers`, and `src.training.runtime`) before compatibility wiring into large trainer loops.
- Publish integrated training-run comparison through `python -m scripts.compare_training_runs` and `make compare-training-runs`, composing manifest diffs and controlled comparability reports without launching training, CUDA, model, OCR, tensor, image, or checkpoint work.
- Treat Phase 5 comparability reports as metadata/control evidence, not proof of visual text-rendering quality; Phase 6 must validate reward and evaluation signals.
- Use `src.evaluation.reward_interface` as the canonical CPU-safe contract for reward rows, product-score formula metadata, thresholds, scorer versions, missing evidence, and manifest links.
- Compute product scores as a weighted geometric product over normalized VLM, OCR, CER-quality, entropy-quality, and exact-text terms while marking incomplete evidence with `missing_components` and `formula_complete`.
- Keep held-out evaluation harnesses materialize-only and CPU-safe: validate fixed prompts, seeds, inference settings, baseline/trained targets, writable output paths, and source run manifests before users launch explicit generation/scoring commands.
- Reject traversal and home expansion in held-out evaluation output paths so user-provided config values cannot redirect planned writes outside reviewed runtime roots.

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

- Continue Phase 6 reward and evaluation validity work with Wave 1 Plan 06-03.
- Validate exact dependency pins and CUDA/module constraints on target machines with explicit smoke checks.
- Keep ROADMAP.md and REQUIREMENTS.md traceability synchronized after phase revisions.

### Blockers

- None for roadmap review.

## Session Continuity

**Next Recommended Action:** Execute Phase 6 Wave 1 Plan 06-03.

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
- `tests/test_prompt_generation_determinism.py`
- `.planning/phases/04-cpu-safe-characterization-tests/04-04-SUMMARY.md`
- `tests/test_reward_wrapper_contracts.py`
- `src/training/rewards.py`
- `scripts/score_images.py`
- `.planning/phases/04-cpu-safe-characterization-tests/04-05-SUMMARY.md`
- `.planning/phases/04-cpu-safe-characterization-tests/04-06-SUMMARY.md`
- `.planning/phases/04-cpu-safe-characterization-tests/VERIFICATION.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-RESEARCH.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-01-PLAN.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-02-PLAN.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-03-PLAN.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-04-PLAN.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-05-PLAN.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-06-PLAN.md`
- `src/training/selection.py`
- `scripts/materialize_training_data.py`
- `tests/test_training_selection_artifacts.py`
- `docs/data_selection.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-01-SUMMARY.md`
- `tests/test_runtime_manifest_diff.py`
- `src/runtime/manifest_diff.py`
- `scripts/compare_run_manifests.py`
- `docs/runtime_contracts.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-02-SUMMARY.md`
- `tests/test_training_comparability.py`
- `src/training/comparability.py`
- `scripts/check_training_comparability.py`
- `docs/training_comparability.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-03-SUMMARY.md`
- `src/training/config.py`
- `src/runtime/config_io.py`
- `tests/test_runtime_config_io.py`
- `tests/test_characterization_config_artifacts.py`
- `configs/experiments/sft/README.md`
- `configs/experiments/dpo/README.md`
- `configs/experiments/masked_sft/README.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/deferred-items.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-04-SUMMARY.md`
- `tests/test_training_shared_utilities.py`
- `src/training/sampling.py`
- `src/training/checkpointing.py`
- `src/training/schedulers.py`
- `src/training/runtime.py`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-05-SUMMARY.md`
- `tests/test_training_comparison_docs.py`
- `scripts/compare_training_runs.py`
- `docs/commands.md`
- `README.md`
- `Makefile`
- `.planning/phases/05-training-objective-and-pipeline-comparability/05-06-SUMMARY.md`
- `.planning/phases/05-training-objective-and-pipeline-comparability/VERIFICATION.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-RESEARCH.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-01-PLAN.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-02-PLAN.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-03-PLAN.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-04-PLAN.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-05-PLAN.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-06-PLAN.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-07-PLAN.md`
- `src/evaluation/reward_interface.py`
- `tests/test_evaluation_reward_interface.py`
- `docs/reward_evaluation.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-01-SUMMARY.md`
- `src/evaluation/heldout.py`
- `scripts/run_heldout_evaluation.py`
- `tests/test_heldout_evaluation_harness.py`
- `docs/evaluation_harness.md`
- `.planning/phases/06-reward-and-evaluation-validity/06-02-SUMMARY.md`

**Do Not Forget:** Commit approved planning artifacts only; leave unrelated worktree changes untouched.

---
*State initialized: 2026-05-04*
