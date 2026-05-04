# Project State: Diffusion Text Tuner

**Initialized:** 2026-05-04  
**Last updated:** 2026-05-04 after roadmap creation

## Project Reference

**Project:** Diffusion Text Tuner  
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.  
**Project Type:** Brownfield thesis ML research toolkit  
**Granularity:** standard  
**Parallelization:** true  
**Verification Preference:** adaptive; use explicit verification on high-risk phases and avoid over-gating obvious inventory/documentation work.

## Current Position

**Current Phase:** Phase 1 - Execution Surface and Pipeline Inventory  
**Current Plan:** Not planned yet  
**Status:** Roadmap ready for review  
**Progress:** [--------------------] 0%

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Execution Surface and Pipeline Inventory | Not started | Practical first step: inventory, install, command catalog, safe tests/smokes. |
| 2. Runtime Contracts and Run Provenance | Not started | Config validation, artifact contracts, manifests, runtime helpers. |
| 3. Data Curriculum and Dataset Quality | Not started | Prompt/synthetic curricula, validators, manifests, selected sample artifacts. |
| 4. CPU-Safe Characterization Tests | Not started | Behavior-locking tests before trainer/reward/pipeline refactors. |
| 5. Training Objective and Pipeline Comparability | Not started | Explicit training modes, run diffs, controlled comparisons, shared training utilities. |
| 6. Reward and Evaluation Validity | Not started | Canonical rewards, held-out eval, diagnostic/gold checks, thesis outputs. |
| 7. Moderate Structure and Extension Cleanup | Not started | Safe file structure cleanup, importable modules, extension seams. |

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| v1 requirement coverage | 58/58 mapped | 100% |
| Roadmap phases planned | 7 | 6-8 standard-granularity phases |
| Default test posture | Existing limited tests only | CPU-safe standard command |
| Reproducible environment | Missing manifest/lock | Committed dependency manifest and lock after Phase 1 |
| Run tracking | Ad hoc filesystem outputs | Local manifests after Phase 2 |

## Accumulated Context

### Decisions

- Treat the project as a brownfield thesis toolkit, not a hosted service or greenfield rewrite.
- Use moderate, behavior-preserving refactors rather than a big-bang package reorganization.
- Stabilize environment and command discovery before runtime contracts, tests, trainer/reward refactors, and evaluation claims.
- Use simple local run manifests first; defer MLflow, Weights & Biases, DVC, object storage, and plugin frameworks to v2 unless need becomes concrete.
- Keep generated images, tensors, checkpoints, logs, and large datasets out of git; only tiny fixtures and docs should be committed.

### Important Caveats

- `REQUIREMENTS.md` listed coverage as 57 v1 requirements, but the actual v1 ID list contains 58 requirements. Traceability now maps all 58.
- Exact CUDA/PyTorch/Diffusers/Transformers/Accelerate/PEFT/PaddleOCR/vLLM/SynthTIGER versions must be smoke-tested on real local/SLURM environments.
- Reward/evaluation semantics are research-critical and should receive explicit verification before thesis claims.

### Known Risks to Preserve in Planning

- Expensive diagnostics named like tests can accidentally trigger CUDA/model downloads.
- Trainer modules combine many responsibilities; refactor only after characterization tests exist.
- Reward logic is duplicated across scoring, training, evaluation, and experiments, creating drift risk.
- Artifact/path/tensor contracts are hidden and should be made explicit before GPU-heavy stages.
- Hardcoded personal paths and unpinned model revisions can break reproducibility.

### Open Todos

- Plan Phase 1 with `/gsd-plan-phase 1`.
- Confirm whether the user wants `uv` as the committed dependency/lock workflow or an alternate manifest strategy for the target cluster.
- Validate exact dependency pins and CUDA/module constraints during Phase 1 smoke checks.
- Keep ROADMAP.md and REQUIREMENTS.md traceability synchronized after phase revisions.

### Blockers

- None for roadmap review.

## Session Continuity

**Next Recommended Action:** Review the roadmap, then run `/gsd-plan-phase 1` to create an executable plan for execution surface discovery and stabilization.

**Files Created/Updated:**
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/REQUIREMENTS.md`

**Do Not Forget:** Commit approved planning artifacts only; leave unrelated worktree changes untouched.

---
*State initialized: 2026-05-04*
