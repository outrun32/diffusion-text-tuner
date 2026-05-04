# Domain Pitfalls

**Domain:** Brownfield ML research toolkit for diffusion-based multilingual text rendering  
**Project:** Diffusion Text Tuner  
**Researched:** 2026-05-04  
**Research focus:** Common mistakes when making ML research repositories reproducible while continuing to add experiments/runs.  
**Overall confidence:** HIGH for environment/testing/reproducibility mechanics from official docs and local codebase audit; MEDIUM for roadmap phase mapping because exact future phase names are not yet fixed.

## Suggested Roadmap Phase Mapping

Use these phase labels when assigning pitfalls below. They are intentionally conservative for a brownfield thesis repo: stabilize the existing system before adding more experiment complexity.

| Phase | Working Name | Purpose |
|-------|--------------|---------|
| Phase 1 | Environment, Commands, and Preflight | Pin dependencies, document setup profiles, define canonical commands, add cheap preflight checks. |
| Phase 2 | Run Manifests and Artifact Contracts | Capture config snapshots, command lines, seeds, git commit, model revisions, output paths, artifact schemas, and trust boundaries. |
| Phase 3 | Test Harness and Lightweight Fixtures | Separate automated tests from expensive diagnostics; add unit/integration fixtures for config, datasets, rewards, prompt determinism, and tensor geometry. |
| Phase 4 | Modularization and Config Consolidation | Extract shared trainer/reward/config utilities without changing training semantics; reduce duplication and path drift. |
| Phase 5 | Experiment Expansion and Evaluation Comparability | Add new SFT/DPO/masked-SFT/reward/evaluation variants only after reproducibility, manifests, and tests make comparisons defensible. |

## Critical Pitfalls

Mistakes that can invalidate experiments, cause silent training drift, or force rewrites.

### Pitfall 1: Treating Dependency Prose as a Reproducible Environment

**What goes wrong:** A README lists approximate `pip install` commands, but there is no committed dependency manifest, lock/constraints file, or optional dependency grouping for CUDA, OCR, synthesis, tests, MLX, and vLLM workflows. New machines install different `torch`, `diffusers`, `transformers`, `accelerate`, `peft`, `paddleocr`, or auxiliary versions and produce failures or different training behavior.

**Why it happens:** Research repos often grow through interactive notebook/script installs. The environment only exists on the original workstation or cluster image, and transitive dependency versions are not captured.

**Warning signs:**
- Setup instructions are prose-only or spread across README snippets.
- `pip freeze` from the original machine is the only known dependency record.
- Optional tools like `paddleocr`, `synthtiger`, `vllm`, `mlx_lm`, and `pytest` are discovered only by running scripts until imports fail.
- A fresh environment fails before reaching the first real experiment.
- Cluster and local installs disagree on CUDA/PyTorch compatibility.

**Consequences:** Fresh reruns become archaeology; model loading and LoRA target matching can break without code changes; thesis results become hard to defend because the environment cannot be recreated.

**Prevention:**
- In Phase 1, add a `pyproject.toml` as the source of project metadata/tool config plus explicit optional extras or separate requirements profiles for `base`, `train`, `ocr`, `synth`, `cluster`, `mlx/vllm`, and `test`.
- Generate a constraints/lock artifact for known-good thesis environments. For pip-based workflows, pin package versions and use hash-checking or a wheelhouse for stricter repeatability when cluster internet access is unreliable.
- Document Python 3.11, CUDA/PyTorch compatibility, and which workflows are expected to work on CPU vs CUDA.
- Add a cheap `python -m ... doctor` or script-level preflight that reports Python, CUDA, GPU, torch, diffusers, transformers, accelerate, PEFT, PaddleOCR, SynthTIGER, and model access status before long jobs start.

**Detection:**
- Recreate the environment from scratch in a clean venv/container and run the canonical smoke commands.
- Run `pip check` and import smoke tests for all supported optional profiles.
- Compare `pip freeze`, CUDA version, GPU name, and key package versions in every run manifest.

**Roadmap phase:** Phase 1 must address this before reorganizing experiments; otherwise every later failure is ambiguous.

**Confidence:** HIGH. Local audit confirms no root manifest/lock/tool config. pip and PyPA docs recommend pinned repeatable installs and `pyproject.toml` for project/tool metadata.

### Pitfall 2: Assuming Seeds Alone Make Diffusion/Training Runs Reproducible

**What goes wrong:** A script records a seed but not the platform, PyTorch release, CUDA/cuDNN behavior, dataloader worker seeding, model revisions, generated artifact checksums, or whether deterministic algorithms were enabled. Two runs with the same seed can still differ.

**Why it happens:** ML teams often compress “reproducibility” to `torch.manual_seed(...)` while ignoring GPU nondeterminism, data loading randomness, and dependency/model version drift.

**Warning signs:**
- `seed` appears in configs but no run records `torch`, CUDA, GPU, cuDNN flags, or dataloader worker seeding policy.
- Prompt generation uses Python/NumPy/random coverage caches without snapshotting state.
- Training can be resumed or compared without checking that generated latents/text embeddings came from the same model revision/config.
- Small numeric differences are dismissed without separating expected GPU nondeterminism from real regressions.

**Consequences:** Experiment comparisons become noisy; regressions hide behind randomness; DPO/SFT/masked-SFT changes may appear better or worse because of uncontrolled data/order/model sources rather than the objective.

**Prevention:**
- In Phase 2, make run manifests capture: git commit, command, full resolved config, seed(s), Python/package versions, CUDA/GPU summary, deterministic flags, dataloader worker seed policy, input manifest checksums, output locations, and model repository revisions.
- In Phase 3, add deterministic unit tests for prompt generation under fixed seeds and for dataset/collator ordering with small fixtures.
- Use PyTorch reproducibility practices where needed: seed Python/NumPy/PyTorch, seed DataLoader workers with `worker_init_fn` and `torch.Generator`, record `torch.backends.cudnn.benchmark`, and optionally use `torch.use_deterministic_algorithms(True)` in debug/regression modes.
- Be explicit in docs: exact bitwise reproducibility across PyTorch releases/platforms is not guaranteed; thesis reproducibility should target “same environment + same inputs + same config produces comparable outputs.”

**Detection:**
- Run the same tiny prompt/dataset/loss fixture twice and diff outputs.
- Re-run a short smoke training/generation job twice on the same machine and compare manifest/config/artifact hashes plus tolerated metrics.
- Flag manifests missing seed, git commit, package versions, or model revisions as invalid for comparison.

**Roadmap phase:** Phase 2 for manifest metadata; Phase 3 for deterministic fixture coverage.

**Confidence:** HIGH. PyTorch official reproducibility docs explicitly warn that complete reproducibility is not guaranteed across releases/platforms and require seeding Python/NumPy/PyTorch plus DataLoader workers for reproducibility.

### Pitfall 3: Letting Generated Artifacts Define Hidden, Unversioned Interfaces

**What goes wrong:** Scripts exchange `.pt`, CSV, JSONL, generated images, masks, latents, prompt embeddings, and reward outputs through implicit paths and unstated schemas. A generator changes an output layout or tensor shape, and training fails later—or worse, silently trains on mismatched artifacts.

**Why it happens:** Research pipelines often evolve as filesystem handoffs: `outputs/generated/text_embeds`, `outputs/text_embeds`, `data/synth_cyrillic`, `runs/...`, and local absolute paths accumulate without a formal artifact contract.

**Warning signs:**
- Multiple scripts expect different paths for the same artifact type.
- Diagnostics assume old locations like `outputs/text_embeds/000000.pt` while current configs use `outputs/generated/text_embeds`.
- Dataset samplers fall back to scanning/loading many latent `.pt` files because shape metadata is missing.
- `.pt` files are loaded without provenance, schema version, model revision, or trust boundary notes.
- A “fix” involves copying files into whichever path a script expects.

**Consequences:** Reproducing old runs requires guessing directory layouts; large jobs fail after hours; artifact deserialization can become unsafe if `.pt` files are treated as portable/untrusted; training/evaluation comparisons mix incompatible generations.

**Prevention:**
- In Phase 2, define artifact contracts for prompts, generated images, latents, text embeddings, masks, reward CSV/JSONL, checkpoints, and evaluation summaries: required files, relative paths, key columns, tensor shapes/dtypes, producing command, consuming commands, and schema version.
- Add per-run `manifest.json` with config snapshot and artifact checksums for small metadata files; do not commit large artifacts.
- Use stable relative paths and require all scripts to resolve paths from config/run root rather than developer-specific absolute paths.
- Keep `torch.load(..., weights_only=True)` for tensor-only local artifacts, document that `.pt` artifacts are trusted local outputs only, and reject unexpected keys/shapes early.
- Make missing `shapes.csv` a validation error or a dedicated repair step rather than an implicit slow fallback during training.

**Detection:**
- Add a `validate-run`/`validate-dataset` command that checks expected files, CSV columns, tensor shapes, and config references before generation/training/scoring.
- Unit-test artifact readers with tiny temporary `.pt`/CSV/JSONL fixtures.
- Inspect new scripts for hardcoded `outputs/...`, `data/...`, or absolute paths.

**Roadmap phase:** Phase 2 owns contracts/manifests; Phase 3 adds fixture validation tests.

**Confidence:** HIGH. Local audit identifies path drift, local absolute paths, latent shape metadata fallback, many `.pt` loads, and ignored generated artifacts.

### Pitfall 4: Expensive Diagnostics Masquerading as Automated Tests

**What goes wrong:** Files named `test_*.py` outside `tests/` import CUDA models, download/load FLUX/Qwen/PaddleOCR, assume local artifacts, or run expensive work at import time. Broad pytest discovery accidentally launches multi-GB workloads or fails on CPU-only environments.

**Why it happens:** Researchers call exploratory scripts “tests” because they validate an idea manually, but pytest treats `test_*.py` as discoverable automated tests.

**Warning signs:**
- `scripts/test_gradient_flow.py` or `experiments/.../test_*.py` executes on import rather than under `if __name__ == "__main__"`.
- Test discovery is unconstrained because no pytest config exists.
- `python -m pytest .` tries to load GPU models or local data.
- Contributors avoid running tests because “tests are expensive and flaky.”

**Consequences:** CI adoption stalls; real unit tests are not trusted; refactors proceed without fast safety checks; accidental model loads waste cluster/GPU time.

**Prevention:**
- In Phase 3, reserve `tests/test_*.py` for fast automated tests and rename manual diagnostics to `check_*.py`, `diagnose_*.py`, or `bench_*.py`.
- Guard every diagnostic with `main()` and explicit CLI arguments.
- Add pytest config with `testpaths = ["tests"]`, strict markers/config, and markers such as `gpu`, `slow`, `manual`, `integration` if any heavier tests become formal.
- Keep default `python -m pytest tests` CPU-friendly and model-download-free.

**Detection:**
- Run pytest collection-only in a fresh CPU environment; it should not import large models or require local artifacts.
- Search newly added files for `test_*.py` outside `tests/` and for top-level execution.
- Treat any test requiring FLUX/Qwen/PaddleOCR as opt-in integration, never as default unit test.

**Roadmap phase:** Phase 3 should fix naming/config before expanding test coverage.

**Confidence:** HIGH. Local audit confirms expensive diagnostics named like tests. pytest docs define default discovery of `test_*.py`/`*_test.py` and recommend explicit layouts/config.

### Pitfall 5: Refactoring Trainers Before Locking Current Behavior

**What goes wrong:** Large trainer files are split or “cleaned up” while tests cover only one masked loss module. Subtle changes to optimizer/scheduler setup, DPO sign/scaling, reference model behavior, sampling, checkpointing, or Accelerate launch assumptions alter training results.

**Why it happens:** Brownfield research repos need cleanup, but trainer modules encode hard-won experimental behavior. Without characterization tests, refactors become unreviewable semantic changes.

**Warning signs:**
- A refactor PR changes file structure and training math in the same diff.
- SFT/DPO/masked-SFT trainers each copy similar logic but have mode-specific quirks no one can state confidently.
- No tests cover `compute_sigma`, DPO beta/log-ratio math, config override loading, checkpoint resume behavior, or Accelerate setup.
- “Looks cleaner” is the only validation criterion.

**Consequences:** Silent degradation of model quality; inability to compare pre/post-refactor results; time lost debugging whether a training change is a bug, environment issue, or expected algorithmic difference.

**Prevention:**
- In Phase 3, add characterization tests for pure math/config/dataset boundaries before moving trainer code.
- In Phase 4, extract utilities in narrow seams: `config_io`, `checkpointing`, `sampling`, `schedulers`, `accelerate_setup`, and shared reward interfaces. Keep trainer files as orchestration layers.
- Require each refactor to preserve old CLI/config behavior or explicitly document a migration.
- Use a tiny dry-run/fake-model integration test for trainer setup paths where feasible; avoid full FLUX jobs in unit tests.

**Detection:**
- Compare resolved configs and manifest fields before/after refactor.
- Add tests that fail if DPO signs/scales, mask downsampling shapes, collator padding, or checkpoint names change unexpectedly.
- During review, separate “move-only” commits from behavior-changing commits.

**Roadmap phase:** Phase 3 creates safety net; Phase 4 performs controlled modularization.

**Confidence:** HIGH. Local audit identifies large trainer modules, fragile DPO math, FLUX tensor contracts, and limited tests.

### Pitfall 6: Reward Drift Between Training, Scoring, and Evaluation

**What goes wrong:** Qwen/PaddleOCR reward logic exists in separate training, scoring, and evaluation paths. Prompt templates, yes/no token extraction, OCR settings, batching, preprocessing, or scoring aggregation diverge, so DPO pair selection, SFT filtering, and evaluation metrics no longer measure the same concept.

**Why it happens:** Reward code is often prototyped in experiment scripts, copied into trainers, then copied again into evaluation when a paper/thesis metric is needed quickly.

**Warning signs:**
- Multiple files instantiate Qwen/PaddleOCR separately with similar but not identical code.
- Evaluation scores cannot be reproduced from the scorer used for pair construction.
- Reward prompt changes are not versioned or recorded in run manifests.
- “Yes probability” experiments live under `experiments/` but production scoring imports a different implementation.

**Consequences:** The training signal and reported metric diverge; improvements may be artifacts of evaluator drift; thesis comparisons become hard to explain.

**Prevention:**
- In Phase 4, centralize reward interfaces and implementations under `src/rewards/` or one authoritative package imported by training, scoring scripts, and evaluation.
- Version reward prompts/settings and record them in run manifests.
- In Phase 3, add tests with fake OCR/VLM outputs for yes/no token extraction, OCR normalization, CSV resume/sharding, and score aggregation.
- Keep expensive Qwen/PaddleOCR checks as manual diagnostics; default tests should mock model-loading boundaries.

**Detection:**
- Compare outputs from training reward wrappers and evaluation scripts on the same tiny image/text fixture using mocked model responses.
- Search for duplicate reward prompt strings and PaddleOCR/Qwen instantiation outside the shared module.
- Require scorer name/version in every generated score CSV/JSONL.

**Roadmap phase:** Phase 3 for fake-wrapper tests; Phase 4 for centralization before Phase 5 comparisons.

**Confidence:** HIGH. Local audit confirms duplicated reward implementations and missing reward wrapper tests.

### Pitfall 7: Not Pinning External Model and Dataset Revisions

**What goes wrong:** Configs reference Hugging Face model IDs or external datasets by moving names/branches. Fresh runs download “latest” model files, tokenizer/chat templates, configs, or datasets that differ from the original experiment.

**Why it happens:** Model hubs make it easy to load by repo ID, but research reproducibility requires knowing the exact revision, local path, and access state used.

**Warning signs:**
- Configs contain `black-forest-labs/FLUX.2-klein-base-4B`, Qwen IDs, or dataset sources without commit hashes/revisions.
- A run cannot distinguish “same code, newer model snapshot” from “new experiment.”
- Gated model access failures happen after a long job starts.
- Cluster nodes use different Hugging Face caches.

**Consequences:** Reruns are not comparable; model-loading breaks when upstream repos change; results depend on cache state and credentials.

**Prevention:**
- In Phase 1, add preflight checks for required model access and local path alternatives.
- In Phase 2, require run manifests to capture model repo IDs, exact revisions/commit hashes, local cache paths, and whether files were loaded from local disk or downloaded.
- Prefer explicit `revision=` parameters or resolved snapshot paths when using Hugging Face Hub-backed downloads/loading.
- Document gated access setup without committing credentials.

**Detection:**
- Fail fast when a model config lacks a revision for comparison-grade runs.
- Record Hugging Face snapshot directory hashes/commit IDs in the manifest.
- Run offline/same-cache smoke tests for thesis-critical configurations.

**Roadmap phase:** Phase 1 for access/preflight; Phase 2 for manifest capture.

**Confidence:** HIGH. Hugging Face Hub docs state downloads default to the latest `main` branch and support `revision` including full commit hashes; local audit identifies gated/external model availability risk.

### Pitfall 8: Allowing Prompt/Data Generation Distribution to Drift Unnoticed

**What goes wrong:** Prompt generation, character coverage, scene/style pools, LLM fallback cleanup, deduplication, and synthetic rendering evolve without distribution checks. Models are trained on a different multilingual text distribution than intended, but no test fails.

**Why it happens:** Data generation code feels “upstream” of training and may be changed to unblock a dataset quickly. Coverage caches and randomness make drift hard to see by inspection.

**Warning signs:**
- Prompt pipeline has no deterministic tests under fixed seeds.
- Character/script coverage changes are only visible after training quality drops.
- Generated prompt datasets lack summary stats: language/script counts, length distribution, duplicate rate, special character coverage, renderability/failure rates.
- LLM fallback is optional but not recorded in manifests.

**Consequences:** Poor multilingual text rendering quality; invalid comparisons between runs trained on different prompt distributions; hidden regressions in Cyrillic/special-character coverage.

**Prevention:**
- In Phase 2, add dataset/prompt manifests with generation config, seed, coverage settings, LLM fallback mode/model, input pools, counts, duplicate rate, and script/character coverage summaries.
- In Phase 3, test deterministic prompt output for fixed seeds and cache refresh behavior.
- In Phase 5, require evaluation reports to include data distribution context, not only reward scores.

**Detection:**
- Generate a tiny fixed-seed prompt fixture and assert stable outputs or stable summary statistics.
- Add a `summarize-prompts` command to compare prompt datasets before training.
- Flag large shifts in script coverage, duplicate rate, or prompt length before accepting new experiments.

**Roadmap phase:** Phase 2 for manifests/summaries; Phase 3 for deterministic tests; Phase 5 for evaluation reporting.

**Confidence:** HIGH from local prompt-pipeline fragility audit; MEDIUM for exact metric thresholds because project-specific quality criteria need thesis input.

### Pitfall 9: Depending on Global/Personal Launcher State for Cluster Runs

**What goes wrong:** Local and SLURM launches depend on user-specific Accelerate default config in `~/.cache`, shell environment, or scheduler scripts not captured in the repo. Multi-GPU/multi-node behavior changes by machine or account.

**Why it happens:** `accelerate config` is convenient and stores defaults outside the project unless a `--config_file` is passed. Cluster workflows often start as personal shell scripts.

**Warning signs:**
- README says “run accelerate config” but does not specify committed config files per target.
- SLURM scripts set environment variables that are not reflected in run manifests.
- A command works for one user but silently changes GPU count/mixed precision for another.
- Multi-node runs require manual edits to machine rank/IP/ports with no template.

**Consequences:** Non-comparable training runs; expensive cluster failures; hidden differences in mixed precision, process count, gradient accumulation, or distributed setup.

**Prevention:**
- In Phase 1, commit named Accelerate config templates for supported local/cluster profiles and document canonical launch commands using `--config_file`.
- In Phase 2, record accelerate config path/content hash, SLURM job metadata, environment variables relevant to CUDA/distributed execution, GPU count, and mixed precision in manifests.
- Keep local CPU/debug launch commands for tests and config validation separate from GPU training commands.

**Detection:**
- Preflight prints resolved Accelerate config, process count, mixed precision, and visible GPUs.
- Short dry-run jobs write manifests before training begins.
- Review new SLURM scripts for hardcoded account/user/path assumptions.

**Roadmap phase:** Phase 1 for templates/commands; Phase 2 for launch metadata capture.

**Confidence:** HIGH. Accelerate docs state default configs are stored in cache locations and custom configs require `--config_file`; local project already has Accelerate/SLURM patterns.

## Moderate Pitfalls

### Pitfall 10: Over-Adopting Heavy Experiment Tracking Too Early

**What goes wrong:** The repo jumps directly to MLflow/W&B/DVC pipelines before basic manifests, stable commands, and artifact schemas exist. Tooling becomes the project instead of supporting thesis work.

**Prevention:** Follow the project decision to use simple local manifests first. Add DVC/MLflow/W&B only if Phase 2 manifests prove insufficient for comparing runs or sharing artifacts.

**Warning signs:**
- Tracking integration is proposed before there is a canonical setup/test/train command.
- Large generated artifacts are accidentally pulled into Git or remote tracking without clear retention rules.
- Students spend more time debugging tracking services than running experiments.

**Roadmap phase:** Defer until after Phase 2; revisit during Phase 5 only if comparison/sharing pain is concrete.

**Confidence:** MEDIUM. DVC docs provide robust pipeline/metadata features, but local project constraints explicitly defer external tracking.

### Pitfall 11: Committing or Sharing Sensitive/Heavy Generated Artifacts

**What goes wrong:** Generated images, prompt text, target text, logs, checkpoints, latents, and score files leak into commits or shared bundles. Private prompt data or large binaries bloat Git history.

**Prevention:** Keep generated outputs ignored; allow only tiny explicit fixtures under `tests/fixtures/` or `experiments/assets/`; sanitize logs before sharing; document prompt/artifact sensitivity.

**Warning signs:**
- Review diffs include `.pt`, checkpoints, generated images, CSVs with prompts, or logs.
- Private datasets are placed under committed `data/` paths.
- A fixture is large enough to be mistaken for real data.

**Roadmap phase:** Phase 1 for `.gitignore`/docs review; Phase 2 for artifact trust/privacy notes; Phase 3 for tiny fixtures.

**Confidence:** HIGH. Local audit identifies generated logs/artifacts may contain prompt text and are already ignored.

### Pitfall 12: Turning Configs Into Another Source of Drift

**What goes wrong:** Runtime JSON configs, Python dataclasses/defaults, README examples, SLURM scripts, and CLI flags all express overlapping defaults. Updating one leaves the others stale.

**Prevention:** In Phase 4, centralize config loading/validation and make scripts print/write resolved configs. In Phase 2, store resolved config snapshots per run. Keep README examples generated or minimal.

**Warning signs:**
- A default differs between `src/training/config.py`, `configs/*.json`, and README.
- Config parsing accepts missing or misspelled keys silently.
- New experiment variants copy an entire JSON and modify a few fields with no schema.

**Roadmap phase:** Phase 2 for snapshotting; Phase 3 for config parsing tests; Phase 4 for consolidation.

**Confidence:** HIGH from local audit of scattered config/default documentation.

### Pitfall 13: Using Relative Working Directories Without a Policy

**What goes wrong:** Scripts behave differently depending on where they are launched from, especially when run via SLURM, Accelerate, notebooks, or future config tools. Outputs and inputs land in unexpected directories.

**Prevention:** Resolve paths from explicit project/run roots, not ambient current working directory. Record working directory and command in manifests. If adopting Hydra later, understand that output directories and optional `chdir` behavior must be configured deliberately.

**Warning signs:**
- Running the same command from repo root vs script directory changes outputs.
- Scripts contain `Path("outputs/...")` without tying it to project root or config root.
- SLURM jobs fail because relative paths resolve from the submission directory.

**Roadmap phase:** Phase 2 for run-root policy; Phase 4 while consolidating config/path utilities.

**Confidence:** MEDIUM-HIGH. Local audit identifies path drift and absolute path bugs; Hydra docs confirm working directory behavior is a known config concern for experiment apps.

### Pitfall 14: Validating Only Loss Math While Ignoring Data Boundaries

**What goes wrong:** Current tests verify masked loss behavior, but dataset CSV parsing, collators, prompt embedding padding, preference pair loading, mask alignment, reward wrappers, and config loading remain untested. Most real failures occur at these boundaries.

**Prevention:** In Phase 3, add small fixtures for dataset/collator/config/reward boundaries before adding new experiment modes. Keep fixtures tiny, synthetic, and CPU-only.

**Warning signs:**
- Tests pass but training fails when loading a prepared dataset.
- Shape errors surface only after expensive generation.
- Reward CSV resume/sharding logic is changed without tests.

**Roadmap phase:** Phase 3.

**Confidence:** HIGH. Local testing audit lists these exact gaps.

## Minor Pitfalls

### Pitfall 15: Mistaking Notebook/Experiment Scripts for Supported Entry Points

**What goes wrong:** One-off scripts under `experiments/` become de facto workflows without CLI help, config validation, or manifest output.

**Prevention:** Label scripts as `manual`, `diagnostic`, or `supported`; move supported flows to `scripts/` or package entry points with documented arguments; require supported flows to write manifests.

**Roadmap phase:** Phase 1 for command inventory; Phase 4 for entry point cleanup.

**Confidence:** MEDIUM-HIGH from current experiment/diagnostic layout.

### Pitfall 16: Optimizing GPU Throughput Before Debuggability

**What goes wrong:** Everything is optimized for full FLUX/Qwen runs, leaving no cheap way to validate configs, shapes, or score aggregation.

**Prevention:** Maintain CPU/tiny fixture paths and fake model boundaries for tests; reserve full GPU diagnostics for opt-in checks.

**Roadmap phase:** Phase 3.

**Confidence:** HIGH from testing/performance audit.

### Pitfall 17: Failing to Document Negative Scope Decisions

**What goes wrong:** Future work reopens settled questions—hosted app, immediate MLflow/W&B, major rearchitecture, committing artifacts—because the repo does not explain why they are out of scope.

**Prevention:** Keep `.planning/PROJECT.md` and README aligned with thesis-toolkit scope; add short rationale when deferring large tooling or architecture changes.

**Roadmap phase:** Every phase transition; especially Phase 1 documentation and Phase 5 expansion planning.

**Confidence:** HIGH from project context.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Phase 1: Environment | Dependency prose remains the setup contract | Add `pyproject.toml`/requirements profiles, pinned constraints, setup docs, and import/preflight smoke tests. |
| Phase 1: Commands | Canonical commands differ between README, scripts, and cluster notes | Define one command table for setup, tests, prompt generation, image generation, scoring, training, evaluation, and diagnostics. |
| Phase 1: Accelerate/SLURM | User cache/global config controls training behavior | Commit named Accelerate templates and SLURM examples; require `--config_file` in documented commands. |
| Phase 2: Manifests | Manifest captures outputs but not inputs/provenance | Include git commit, command, config snapshot, seed, package versions, model revisions, input artifact checksums, and output schema version. |
| Phase 2: Artifacts | Large artifacts accidentally become versioned | Keep Git ignore strict; commit only tiny fixtures and metadata; document storage/cleanup policy. |
| Phase 2: Paths | Relative and absolute paths keep drifting | Introduce explicit project root/run root/path resolution helpers; fail on developer-specific absolute paths in supported configs. |
| Phase 3: Tests | Expensive diagnostics block CI | Rename diagnostics, restrict pytest discovery to `tests`, mark optional GPU/slow tests, and keep default tests CPU-only. |
| Phase 3: Fixtures | Fixtures become mini datasets | Use tiny synthetic tensors/images/CSVs; test schema/shape behavior, not model quality. |
| Phase 3: Determinism | Tests overpromise bitwise reproducibility | Assert deterministic behavior only for controlled CPU/tiny functions; document GPU tolerance separately. |
| Phase 4: Refactor | Cleanup changes training semantics | Add characterization tests first; split move-only and behavior-changing diffs; compare resolved configs/manifests. |
| Phase 4: Rewards | Training/evaluation reward implementations diverge | Centralize reward interface and version prompt/settings; make scripts import shared implementation. |
| Phase 5: New experiments | New variants multiply configs/artifacts without comparability | Require manifests, config schema, data summaries, and reward/evaluation versioning before accepting new variants. |

## Priority Recommendations

1. **Do first:** Add reproducible environment profiles and canonical command/preflight docs. Without this, every later test/refactor failure is ambiguous.
2. **Do second:** Add run manifests and artifact contracts before running new comparison-grade experiments.
3. **Do third:** Fix test discovery and add lightweight fixture coverage around dataset/config/reward/prompt boundaries.
4. **Do fourth:** Refactor shared trainer/reward/config code only after characterization tests exist.
5. **Do later:** Add heavier experiment tracking or new research variants only when the basic reproducibility loop is stable.

## What Might Be Missed / Gaps

- **Exact cluster environment:** The local planning docs mention SLURM/Accelerate but do not specify cluster CUDA modules, filesystem layout, scheduler constraints, or quota behavior. Validate Phase 1 recommendations against the actual cluster.
- **Exact thesis evaluation metric:** Pitfalls assume reward/evaluation comparability is central, but final thesis metrics may require additional OCR/VLM/human-evaluation safeguards.
- **Dataset licensing/permissions:** Research did not inspect actual datasets or generated prompt sources. If external images/fonts/text corpora are used, add a license/provenance audit.
- **Container strategy:** This document recommends manifests/constraints first. A later phase may decide whether Docker/Apptainer is necessary for the cluster.
- **Security depth:** Only artifact trust, URL download, credentials, and prompt leakage were considered. A full security review is needed if artifacts are shared publicly or downloaded from untrusted manifests.

## Sources

- Local project context: `/root/diffusion-text-tuner/.planning/PROJECT.md` (HIGH) — brownfield thesis toolkit goals, constraints, active requirements, out-of-scope decisions.
- Local codebase audit: `/root/diffusion-text-tuner/.planning/codebase/CONCERNS.md` (HIGH) — missing dependency manifest, duplicated rewards, expensive diagnostics named like tests, hardcoded paths, artifact/security/performance risks, fragile tensor/DPO/prompt areas.
- Local testing audit: `/root/diffusion-text-tuner/.planning/codebase/TESTING.md` (HIGH) — current pytest-compatible tests, discovery hazards, coverage gaps, fixture recommendations.
- PyTorch Reproducibility docs, last updated 2025-10-03: https://docs.pytorch.org/docs/2.11/notes/randomness.html (HIGH) — reproducibility limits, seeding Python/NumPy/PyTorch, cuDNN/deterministic algorithms, DataLoader worker seeding.
- pip Repeatable Installs docs: https://pip.pypa.io/en/stable/topics/repeatable-installs/ (HIGH) — pinned requirements, hash-checking, wheelhouse/install bundles.
- Python `venv` docs, Python 3.14.4: https://docs.python.org/3/library/venv.html (HIGH) — virtual environments are isolated, disposable, not checked into source control, and should be recreated from requirements.
- Python Packaging User Guide, `pyproject.toml`: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ (HIGH) — `pyproject.toml` for build metadata, dependencies, optional dependencies, and tool config.
- pytest Good Integration Practices: https://pytest.org/en/stable/explanation/goodpractices.html (HIGH) — default test discovery, test layout, venv/pip install, strict config, import modes.
- Hugging Face Hub download docs: https://huggingface.co/docs/huggingface_hub/en/guides/download (HIGH) — default latest `main`, explicit `revision`, snapshot downloads, cache behavior, dry-run.
- Hugging Face Accelerate launch docs: https://huggingface.co/docs/accelerate/en/basic_tutorials/launch (HIGH) — `accelerate launch`, `accelerate config`, custom `--config_file`, default cache locations, multi-node launch considerations.
- Hydra working directory docs, last updated 2025-10-27: https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/ (MEDIUM) — useful context for future config tooling and output directory/chdir policy; Hydra is not currently required by project scope.
- DVC `dvc.yaml` docs: https://dvc.org/doc/user-guide/project-structure/dvcyaml-files (MEDIUM) — pipeline stages, params, metrics, outputs, and reproducibility guidance; DVC is deferred by current project scope but informs artifact-contract pitfalls.
