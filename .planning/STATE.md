---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: Phase 3 - Data Curriculum and Dataset Quality
current_plan: 03-01-PLAN.md
status: phase-3-planned
last_updated: "2026-05-04T15:30:00Z"
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 6
  completed_plans: 0
  percent: 0
---

# Project State: Diffusion Text Tuner

**Initialized:** 2026-05-04  
**Last updated:** 2026-05-04 after Phase 3 planning

## Project Reference

**Project:** Diffusion Text Tuner  
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.  
**Project Type:** Brownfield thesis ML research toolkit  
**Granularity:** standard  
**Parallelization:** true  
**Verification Preference:** adaptive; use explicit verification on high-risk phases and avoid over-gating obvious inventory/documentation work.

## Current Position

**Current Phase:** Phase 3 - Data Curriculum and Dataset Quality  
**Current Plan:** 03-01-PLAN.md  
**Status:** Phase 3 planned; ready to execute Wave 1
**Progress:** [░░░░░░░░░░░░░░░░░░░░] 0% for Phase 3

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Execution Surface and Pipeline Inventory | Verified complete | 4/4 plans complete and phase verification passed 12/12 must-haves. |
| 2. Runtime Contracts and Run Provenance | Verified complete | 5/5 plans complete and phase verification passed 5/5 must-haves. Shared config validation, canonical paths, artifact validators, runtime contract docs, local manifests, trainer loader wiring, manifest CLI, runtime preflight CLI, config-family docs, and Makefile/README command surfaces are in place. |
| 3. Data Curriculum and Dataset Quality | Planned | 6 plans across 3 waves covering prompt curricula, prompt validation, dataset manifests, synthetic quality inspection, selected sample/pair artifacts, generated-vs-synthetic comparison, and command/runtime docs wiring. |
| 4. CPU-Safe Characterization Tests | Not started | Behavior-locking tests before trainer/reward/pipeline refactors. |
| 5. Training Objective and Pipeline Comparability | Not started | Explicit training modes, run diffs, controlled comparisons, shared training utilities. |
| 6. Reward and Evaluation Validity | Not started | Canonical rewards, held-out eval, diagnostic/gold checks, thesis outputs. |
| 7. Moderate Structure and Extension Cleanup | Not started | Safe file structure cleanup, importable modules, extension seams. |

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| v1 requirement coverage | 58/58 mapped; Phase 1 plus all Phase 2 runtime requirements complete; DATA-01 through DATA-07 are planned in Phase 3 | 100% mapped; execute Phase 3 data requirements |
| Roadmap phases planned | 7 total, Phase 3 has 6 executable plans | 6-8 standard-granularity phases |
| Default test posture | 68 CPU-safe pytest tests including smoke CLI, tensor-loss, runtime config validation, runtime artifact contracts, runtime manifest contracts, runtime docs checks, and runtime preflight CLI behavior; diagnostics are opt-in `diagnose_*.py` scripts | CPU-safe standard command |
| Reproducible environment | `.python-version`, `pyproject.toml`, and `uv.lock` committed in Phase 1 Plan 02 | Smoke-tested setup commands after Phase 1 |
| Run tracking | Local file-backed manifests with immutable config snapshots, secret-safe reproducibility metadata, trainer config-loader wiring, CPU-safe preflight reports, config-family docs, README/Makefile command aliases, and Phase 3 data-manifest plans | Extend to prompt/synthetic/selection manifests during Phase 3 execution |

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

- Execute Phase 3 Wave 1 plans: 03-01, 03-02, 03-03, and 03-04.
- Validate exact dependency pins and CUDA/module constraints on target machines with explicit smoke checks.
- Keep ROADMAP.md and REQUIREMENTS.md traceability synchronized after phase revisions.

### Blockers

- None for roadmap review.

## Session Continuity

**Next Recommended Action:** Execute Phase 3 Wave 1 with `/gsd-execute-phase 3`.

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

**Do Not Forget:** Commit approved planning artifacts only; leave unrelated worktree changes untouched.

---
*State initialized: 2026-05-04*
