# Project Research Summary

**Project:** Diffusion Text Tuner  
**Domain:** Brownfield thesis ML research toolkit for diffusion-based multilingual text rendering  
**Researched:** 2026-05-04  
**Confidence:** HIGH for roadmap direction; MEDIUM for exact CUDA/ML dependency pins until validated on target machines

## Executive Summary

Diffusion Text Tuner is not a production service or a greenfield ML platform; it is a brownfield thesis toolkit whose core research flows already exist: multilingual/Cyrillic prompt generation, FLUX image/latent/text-embedding generation, VLM/OCR reward scoring, SFT/DPO/masked-SFT LoRA training, SynthTIGER synthetic data, evaluation scripts, and local/SLURM launch patterns. Experts would stabilize this kind of repository by preserving runnable entry points while adding reproducible environments, typed configs, explicit artifact contracts, lightweight tests, and run provenance around the existing pipeline.

The recommended approach is conservative and dependency-driven: first make installation and commands repeatable with `pyproject.toml`, `uv.lock`, Python 3.11, Ruff, pytest, and optional dependency extras; then add strict config loading, canonical paths, local run manifests, and preflight validation before modularizing trainers or changing reward/generation logic. Keep the current PyTorch/Diffusers/Transformers/Accelerate/PEFT/FLUX/Qwen/PaddleOCR/SynthTIGER foundations; the immediate goal is reproducibility and comparability, not replacing core methods.

The major risks are environment drift, hidden filesystem/tensor contracts, expensive diagnostics being mistaken for automated tests, reward drift across training/scoring/evaluation, and trainer refactors that silently change research behavior. Mitigate them by pinning known-good versions after real GPU smoke tests, recording every run's resolved config/environment/model revisions/artifacts, validating paths and shapes before expensive GPU work, keeping default tests CPU-only, centralizing reward interfaces, and refactoring only after characterization tests exist.

## Key Findings

### Recommended Stack

Use a single `pyproject.toml` plus committed `uv.lock` as the source of truth. Retain Python 3.11 because it is the existing project baseline and narrows ML wheel ambiguity. Use uv for sync/locking, Ruff for lint/format, pytest for CPU-safe test automation, Pydantic v2 plus `pydantic-settings` for config validation and environment overlays, and local run manifests for experiment metadata. Keep Conda only as a thin optional cluster wrapper if SLURM/CUDA modules require it.

**Core technologies:**
- **Python 3.11** — current runtime baseline; avoid broader version support until thesis runs are reproducible.
- **uv + `pyproject.toml` + `uv.lock`** — reproducible dependency source with optional extras/groups for GPU, OCR, reward, synth, vLLM, MLX, test, lint, and analysis workflows.
- **PyTorch, Diffusers, Transformers, Accelerate, PEFT** — preserve existing training/generation foundation; pin exact versions only after target CUDA/SLURM smoke tests.
- **PaddleOCR/Qwen reward paths, bitsandbytes, SynthTIGER, vLLM, MLX** — keep as explicit optional capabilities so default installs/tests stay lightweight.
- **Pydantic v2 + pydantic-settings** — validate existing JSON configs and runtime environment overlays without a disruptive Hydra migration.
- **pytest + Ruff** — standardize CPU-only tests and fast lint/format through `pyproject.toml`.
- **Local `runs/<run_id>/manifest.json`** — first-line experiment tracking; defer MLflow/W&B until local manifests prove insufficient.

### Expected Features

The existing code has the ML capability; the missing features are mostly reproducibility, discoverability, validation, and safe extension points.

**Must have (table stakes):**
- Reproducible environment definition with known-good ML pins and optional dependency profiles.
- Standard command catalog for setup, tests, prompt generation, image generation, scoring, training, evaluation, synthesis, and SLURM.
- Config inventory, strict config loading, and naming conventions for SFT/DPO/masked-SFT/reward/eval variants.
- Canonical artifact path contract for prompts, generated images, latents, text embeddings, masks, scores, checkpoints, samples, and logs.
- Local run manifests capturing config snapshot, command, git state, environment, model IDs/revisions, seeds, inputs, outputs, and notes.
- Dataset/artifact validators and preflight checks that fail before model download or long GPU jobs.
- Lightweight automated tests and tiny fixtures for config parsing, datasets/collators, tensor shapes, prompt determinism, reward wrappers, and critical trainer math.
- Shared reward interface so scoring, training, and evaluation use the same Qwen/OCR semantics.

**Should have (differentiators):**
- Multilingual/script coverage reports for prompt and synthetic datasets.
- Reward calibration reports comparing OCR/VLM behavior.
- Experiment manifest diff tool for comparing runs by config, inputs, seeds, metrics, and artifacts.
- Pipeline resume/idempotency checks and sharded artifact indexes once scale requires them.
- Low-memory/debug presets and config templates for experiment families.
- Thesis figure/table generators and dataset/run cards.

**Defer (v2+ or later):**
- MLflow/W&B/DVC as primary tracking; local manifests come first.
- Hydra/OmegaConf migration until strict JSON configs and manifests are stable and sweeps become painful.
- Golden tiny E2E pipeline until mockable boundaries and fixtures exist.
- Text-rendering error taxonomy until reward/evaluation outputs are stable.
- Object-store/distributed data platform and major plugin architecture.
- Hosted web app/API or replacement of FLUX/Qwen/PaddleOCR/SynthTIGER without experimental reason.

### Architecture Approach

Evolve the repo as a file-backed ML research toolkit with thin CLI entry points, typed config schemas, explicit artifact contracts, centralized runtime helpers, and local manifests. Do not perform a big-bang package reorganization. Preserve current public commands while extracting reusable behavior behind them when touching a stage.

**Major components:**
1. **`configs/`** — committed experiment, stage, hardware, and synthesis settings; add organized `configs/experiments/<stage>/` variants.
2. **`src.runtime`** — cross-stage config I/O, path resolution, seed setup, run manifests, and preflight/artifact validation.
3. **`src.prompt_pipeline`** — prompt sampling and assembly; records generation distribution and seed context.
4. **`src.generation`** — reusable FLUX generation implementation called by `scripts.generate_images`.
5. **`src.scoring` + `src.rewards`** — batch scoring orchestration plus canonical Qwen/OCR reward implementations and result schemas.
6. **`src.training`** — trainers remain mode-specific orchestration layers; shared plumbing moves to config, dataset, flux utilities, losses, sampling, checkpointing, and schedulers.
7. **`scripts/` and `scripts/cluster`** — thin local/SLURM wrappers with no hidden experiment choices; cluster scripts pass config paths and launch topology.
8. **`tests/fixtures` and CPU-only tests** — tiny committed fixtures for contracts and math; expensive GPU/model diagnostics are opt-in `check_*`/`diagnose_*` scripts.

**Key patterns:** stable CLI shell with importable implementation; artifact contract validation before expensive work; centralized reproducibility layer; one reward interface used by all callers; trainers own objectives while shared modules own plumbing; promote one-off experiments to `src/` only after they become reusable.

### Critical Pitfalls

1. **Dependency prose instead of reproducible environment** — add `pyproject.toml`, `uv.lock`, optional extras/groups, setup docs, and import/GPU smoke checks before refactors.
2. **Seeds treated as full reproducibility** — record git, command, resolved config, package versions, CUDA/GPU, deterministic flags, DataLoader seeding, model revisions, and artifact checksums in manifests.
3. **Hidden artifact/path/tensor contracts** — define canonical layouts and schema versions; validate CSV/JSONL/PT keys, shapes, sample IDs, and paths before GPU-heavy runs.
4. **Expensive diagnostics masquerading as tests** — restrict pytest discovery to `tests/`, rename manual diagnostics, guard with `main()`, and keep default tests CPU/model-download-free.
5. **Trainer refactors before behavior is locked** — add characterization tests for config, dataset/collator, FLUX shapes, DPO math, schedulers, checkpointing, and sampling before moving code.
6. **Reward drift across scoring/training/evaluation** — centralize Qwen/OCR reward implementations, version reward prompts/settings, and require scorer metadata in score files/manifests.
7. **Unpinned external model/dataset revisions** — preflight Hugging Face access and record explicit model revisions/cache paths for comparison-grade runs.

## Implications for Roadmap

Based on the combined research, roadmap phases should stabilize the existing execution surface first, then add provenance and validation, then refactor under test coverage, and only then expand research features.

### Phase 1: Environment, Commands, and Discovery

**Rationale:** Every later change depends on a reproducible install and known commands; without this, failures are indistinguishable between code, environment, CUDA, and docs drift.  
**Delivers:** `pyproject.toml`, `.python-version`, committed `uv.lock`, Ruff/pytest config, optional dependency extras/groups, README/Makefile command catalog, CPU-safe test command, GPU/import smoke commands, documented Accelerate/SLURM launch templates, renamed manual diagnostics.  
**Addresses:** reproducible environment definition, standard command catalog, local/SLURM parity, lightweight test surface.  
**Avoids:** dependency prose, hidden personal launcher state, accidental GPU diagnostics during pytest.  
**Research flag:** Standard patterns; no extra research unless the target cluster has unusual CUDA/module restrictions.

### Phase 2: Runtime Manifests, Config Loading, and Artifact Contracts

**Rationale:** Run provenance and resolved artifact paths must exist before comparison-grade experiments or behavior-preserving refactors.  
**Delivers:** `src.runtime.config_io`, `src.runtime.manifests`, `src.runtime.paths`, `src.runtime.reproducibility`, strict JSON config validation, config snapshots, local `runs/<run_id>/`, canonical output layout, artifact schema/version definitions, path policy, basic preflight checks.  
**Uses:** Pydantic v2/settings, local JSON manifests, existing JSON configs.  
**Implements:** runtime layer shared by generation, scoring, training, synthesis, and evaluation.  
**Avoids:** seeds-only reproducibility, config drift, hidden artifact interfaces, relative/absolute path drift.  
**Research flag:** Standard patterns, but manifest schema details should be validated against actual thesis comparison needs.

### Phase 3: Lightweight Test Harness and Fixture Validators

**Rationale:** Tests should lock current behavior before trainer, reward, and generation code is moved.  
**Delivers:** tiny committed fixtures, config parsing tests, dataset/collator contract tests, FLUX latent/text/mask shape tests, loss/scheduler/DPO math tests, fake reward wrapper tests, prompt determinism tests, validator tests, pytest markers for `slow`, `gpu`, `model`, `ocr`, `integration`.  
**Addresses:** lightweight automated test suite, dataset/artifact validators, deterministic seed checks, small committed fixtures.  
**Avoids:** refactoring trainers blind, validating only loss math, overpromising GPU bitwise reproducibility.  
**Research flag:** Standard patterns for pytest; phase-specific research only if designing a golden mock E2E pipeline.

### Phase 4: Safe Modularization and Shared Interfaces

**Rationale:** Once commands, manifests, and tests exist, extract duplication without changing model behavior.  
**Delivers:** `src.training.sampling`, `src.training.checkpointing`, `src.training.schedulers`, trainer integration with runtime manifests/config, `src.rewards` canonical interfaces, `src.scoring.pipeline`, compatibility re-exports where needed, extension guide for new trainers/rewards/datasets.  
**Addresses:** shared reward interface, refactored shared trainer/pipeline utilities, extension guides.  
**Avoids:** reward drift, trainer semantic changes, overly abstract plugin framework, big-bang reorganization.  
**Research flag:** Needs targeted implementation research for Qwen/PaddleOCR reward semantics and FLUX/Accelerate edge cases if current code paths are ambiguous.

### Phase 5: Generation/Synthesis Pipeline Hardening and Evaluation Comparability

**Rationale:** GPU-heavy artifact-producing stages should be modularized after the runtime layer and validators are in place.  
**Delivers:** importable generation pipeline behind existing CLI, generation artifact index, synthetic dataset manifest/index validation, local/SLURM parity updates, evaluation suite outputs, score schema metadata, resume/idempotency checks where needed.  
**Addresses:** evaluation suite and comparison outputs, pipeline resume/idempotency, sharded indexes if scale demands, model/cache preflight.  
**Avoids:** artifact incompatibility, prompt/data distribution drift, unpinned model revisions, cluster/local drift.  
**Research flag:** Needs deeper research if introducing sharded dataset indexes, evaluation metrics, or human/OCR/VLM quality taxonomy.

### Phase 6: Research Differentiators and Optional Tooling

**Rationale:** Differentiators are valuable only after runs are reproducible and comparable.  
**Delivers:** multilingual/script coverage reports, reward calibration reports, manifest diff CLI, thesis figure/table generators, dataset/run cards, low-memory/debug presets, optional Hydra if JSON variants become unmanageable, optional MLflow/W&B/DVC if local manifests become insufficient.  
**Addresses:** differentiators and deferred experiment-scale tooling.  
**Avoids:** tracking/tooling becoming the project, major plugin architecture too early, unsupported v2 features creeping into foundational work.  
**Research flag:** Needs research for text-rendering error taxonomy, final thesis metrics, dataset licensing/provenance, and any external tracking/data-versioning choice.

### Phase Ordering Rationale

- Environment and commands come first because all later validation and refactoring require repeatable local/cluster execution.
- Manifests, strict config loading, and path contracts come before tests/refactors because they define what behavior and artifacts mean.
- Fixture tests come before modularization so code movement is reviewable and behavior-preserving.
- Reward/trainer/generation modularization is split across later phases because these are high-risk research-critical paths.
- Differentiators and external tooling are intentionally delayed until comparison-grade experiments have stable provenance and validators.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4:** Qwen/PaddleOCR reward centralization and FLUX/Accelerate trainer seams may require targeted code/path investigation.
- **Phase 5:** Evaluation comparability, sharded indexes, and resume semantics need project-specific design once artifact volume is known.
- **Phase 6:** Text-rendering error taxonomy, thesis metrics, dataset licensing, Hydra/MLflow/W&B/DVC decisions require separate validation.

Phases with standard patterns (skip research-phase unless environment surprises appear):
- **Phase 1:** uv/pyproject/Ruff/pytest/Accelerate command documentation is well documented.
- **Phase 2:** Pydantic config validation, local JSON manifests, path helpers, and seed capture are established patterns.
- **Phase 3:** pytest fixtures/markers and CPU-only contract tests are standard.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH/MEDIUM | HIGH for uv/pyproject/Ruff/pytest/Pydantic/local manifests; MEDIUM for exact PyTorch/CUDA/Paddle/vLLM/SynthTIGER pins until smoke-tested on target machines. |
| Features | HIGH | Strongly grounded in PROJECT.md, codebase context, and official PyTorch/Hugging Face expectations for reproducible research workflows. |
| Architecture | HIGH/MEDIUM | HIGH for conservative brownfield structure, runtime layer, manifests, contracts, and tests; MEDIUM for any future Hydra/config composition migration. |
| Pitfalls | HIGH | Supported by local audit plus official docs on PyTorch reproducibility, packaging, pytest discovery, Hugging Face Hub revisions, and Accelerate launch behavior. |

**Overall confidence:** HIGH for roadmap ordering and scope control; MEDIUM for hardware-specific dependency choices and final evaluation metric design.

### Gaps to Address

- **Exact CUDA/dependency matrix:** Validate torch/torchvision/diffusers/transformers/accelerate/peft/bitsandbytes/PaddleOCR/vLLM/SynthTIGER versions on the actual workstation/SLURM environment before pinning.
- **Cluster constraints:** Confirm CUDA modules, filesystem/cache locations, internet/model access, scheduler limits, and whether uv `.venv` is acceptable or needs a Conda/module wrapper.
- **Manifest schema final details:** Finalize required fields during Phase 2 with sample manifests and tests, especially model revisions, artifact checksums, SLURM metadata, and privacy notes.
- **Final thesis evaluation metric:** Reward/evaluation comparability is assumed central; validate whether OCR/VLM/human-evaluation safeguards or an error taxonomy are required.
- **Dataset/font/prompt licensing and provenance:** Not audited; add before publishing or sharing datasets/artifacts.
- **Artifact security/trust:** `.pt` artifacts should be treated as trusted local outputs; add full security review if consuming downloaded third-party manifests/artifacts.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` — project scope, validated/active requirements, constraints, and out-of-scope decisions.
- `.planning/research/STACK.md` — tooling, dependency, config, testing, and run metadata recommendations.
- `.planning/research/FEATURES.md` — table stakes, differentiators, anti-features, and feature dependency ordering.
- `.planning/research/ARCHITECTURE.md` — component boundaries, data/config/run flow, patterns, anti-patterns, and build order.
- `.planning/research/PITFALLS.md` — critical/moderate/minor pitfalls, phase warnings, and mitigations.
- PyTorch reproducibility docs — seeding, nondeterminism limits, deterministic algorithms, DataLoader worker seeding.
- Python Packaging User Guide and uv docs — `pyproject.toml`, optional dependencies, dependency groups, explicit PyTorch indexes, lock/sync workflows.
- pytest docs — pyproject configuration, custom markers, strict marker validation, discovery behavior.
- Pydantic and pydantic-settings docs — typed validation, JSON Schema, environment/dotenv/secrets/CLI settings.
- Hugging Face Accelerate/Diffusers/Hub docs — launch/config behavior, training script conventions, model revision/cache behavior.

### Secondary (MEDIUM confidence)
- Hydra docs — informs future config composition/output-directory tradeoffs; not recommended for first phase.
- DVC documentation — informs artifact/pipeline concepts; external data versioning is deferred.

---
*Research completed: 2026-05-04*  
*Ready for roadmap: yes*
