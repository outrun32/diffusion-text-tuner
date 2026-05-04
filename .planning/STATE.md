---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: Phase 2 - Runtime Contracts and Run Provenance
current_plan: 02-04-PLAN.md — trainer config loader wiring and CPU-safe preflight CLI
status: in-progress
last_updated: "2026-05-04T14:28:55Z"
progress:
  total_phases: 7
  completed_phases: 1
  total_plans: 5
  completed_plans: 3
  percent: 60
---

# Project State: Diffusion Text Tuner

**Initialized:** 2026-05-04  
**Last updated:** 2026-05-04 after Phase 2 Plan 03 execution

## Project Reference

**Project:** Diffusion Text Tuner  
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.  
**Project Type:** Brownfield thesis ML research toolkit  
**Granularity:** standard  
**Parallelization:** true  
**Verification Preference:** adaptive; use explicit verification on high-risk phases and avoid over-gating obvious inventory/documentation work.

## Current Position

**Current Phase:** Phase 2 - Runtime Contracts and Run Provenance  
**Current Plan:** 02-04-PLAN.md — trainer config loader wiring and CPU-safe preflight CLI  
**Status:** Phase 2 in progress
**Progress:** [████████████--------] 60%

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Execution Surface and Pipeline Inventory | Verified complete | 4/4 plans complete and phase verification passed 12/12 must-haves. |
| 2. Runtime Contracts and Run Provenance | In progress | 3/5 plans complete; shared config validation, canonical paths, artifact validators, runtime contract docs, local manifests, reproducibility metadata, and manifest CLI are in place. |
| 3. Data Curriculum and Dataset Quality | Not started | Prompt/synthetic curricula, validators, manifests, selected sample artifacts. |
| 4. CPU-Safe Characterization Tests | Not started | Behavior-locking tests before trainer/reward/pipeline refactors. |
| 5. Training Objective and Pipeline Comparability | Not started | Explicit training modes, run diffs, controlled comparisons, shared training utilities. |
| 6. Reward and Evaluation Validity | Not started | Canonical rewards, held-out eval, diagnostic/gold checks, thesis outputs. |
| 7. Moderate Structure and Extension Cleanup | Not started | Safe file structure cleanup, importable modules, extension seams. |

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| v1 requirement coverage | 58/58 mapped; Phase 1 plus CFG-01, CFG-03, CFG-04, ART-01, ART-02, ART-03, ART-04, RUN-01, RUN-03, RUN-04, and STR-02 complete | 100% mapped; complete remaining Phase 2 runtime requirements |
| Roadmap phases planned | 7 total, Phase 1 plan 04 complete | 6-8 standard-granularity phases |
| Default test posture | 49 CPU-safe pytest tests including smoke CLI, tensor-loss, runtime config validation, runtime artifact contracts, and runtime manifest contracts; diagnostics are opt-in `diagnose_*.py` scripts | CPU-safe standard command |
| Reproducible environment | `.python-version`, `pyproject.toml`, and `uv.lock` committed in Phase 1 Plan 02 | Smoke-tested setup commands after Phase 1 |
| Run tracking | Local file-backed manifests with immutable config snapshots and secret-safe reproducibility metadata | Trainer/preflight wiring and docs after Phase 2 |

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

- Execute Phase 2 Plan 04 for trainer config loader wiring and a CPU-safe preflight CLI.
- Validate exact dependency pins and CUDA/module constraints on target machines with explicit smoke checks.
- Keep ROADMAP.md and REQUIREMENTS.md traceability synchronized after phase revisions.

### Blockers

- None for roadmap review.

## Session Continuity

**Next Recommended Action:** Execute Phase 2 Plan 04 for trainer config loader wiring and a CPU-safe preflight CLI.

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

**Do Not Forget:** Commit approved planning artifacts only; leave unrelated worktree changes untouched.

---
*State initialized: 2026-05-04*
