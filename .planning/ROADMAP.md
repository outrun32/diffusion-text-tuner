# Roadmap: Diffusion Text Tuner

**Created:** 2026-05-04  
**Granularity:** standard  
**Project type:** brownfield thesis ML research toolkit  
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.

## Roadmap Principles

- Preserve existing runnable research flows while making them discoverable, reproducible, and safer to extend.
- Stabilize environment, command, config, artifact, and test contracts before heavy refactors.
- Keep default verification CPU-safe; use explicit smoke/diagnostic checks for CUDA, model, OCR, and SLURM paths.
- Avoid big-bang package reorganization; make moderate, behavior-preserving moves behind existing entry points.

## Phases

- [x] **Phase 1: Execution Surface and Pipeline Inventory** - Users can understand what already runs and reproduce the basic command/test surface before deeper changes.
- [x] **Phase 2: Runtime Contracts and Run Provenance** - Users can validate configs/artifacts and capture local run manifests before expensive pipeline work starts.
- [x] **Phase 3: Data Curriculum and Dataset Quality** - Users can define, generate, validate, and compare prompt/synthetic/reward-filtered training data. *(6/6 plans complete)*
- [x] **Phase 4: CPU-Safe Characterization Tests** - Users can rely on lightweight fixtures and deterministic tests for fragile config, data, reward, prompt, and objective behavior. *(6/6 plans complete)*
- [ ] **Phase 5: Training Objective and Pipeline Comparability** - Users can run and compare SFT, DPO, masked-SFT, and combined variants under explicit, controlled choices. *(6/6 plans complete; verification pending)*
- [ ] **Phase 6: Reward and Evaluation Validity** - Users can produce comparable held-out evaluations, reward diagnostics, and thesis-ready outputs from recorded runs. *(2/7 plans complete)*
- [ ] **Phase 7: Moderate Structure and Extension Cleanup** - Users can navigate clearer source/script/config homes and add future pipelines through documented extension points.

## Phase Details

### Phase 1: Execution Surface and Pipeline Inventory
**Goal**: Users can understand the current toolkit, install it reproducibly, and run safe baseline commands without triggering expensive GPU/model work by accident.  
**Depends on**: Nothing  
**Requirements**: INV-01, INV-02, INV-03, INV-04, ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, ENV-06, TEST-06, TEST-07  
**Success Criteria** (what must be TRUE):
  1. User can open a current pipeline inventory that separates supported entry points from outdated, duplicate, experimental, and manual diagnostic scripts.
  2. User can see what each prompt, generation, scoring, SFT, DPO, masked-SFT, synthetic, evaluation, plotting, and SLURM flow consumes, produces, optimizes, and supports in the thesis.
  3. User can install Python 3.11 dependencies from committed manifests with optional groups for GPU, OCR/reward, synthesis, vLLM/MLX, tests, linting, plotting, and analysis.
  4. User can run documented CPU-safe tests, lint/format commands, import/CUDA/model/OCR/cache smoke checks, and comparable local/SLURM command variants.
  5. User can distinguish default automated tests from slow, GPU, model, OCR, integration, and manual diagnostics.
**Plans**: 4 plans

Plans:

**Wave 1**
- [x] 01-01-PLAN.md — Inventory supported pipeline families, diagnostics, experiments, and historical tracks.
- [x] 01-02-PLAN.md — Add Python 3.11 uv/pyproject/lock tooling, pytest discovery, and Ruff configuration.

**Wave 2** *(blocked on Wave 1 tooling completion)*
- [x] 01-03-PLAN.md — Add tested, import-safe smoke checks for imports, CUDA, cache paths, model access, and OCR.

**Wave 3** *(blocked on Wave 1 inventory/tooling and Wave 2 smoke CLI)*
- [x] 01-04-PLAN.md — Publish command catalog, Makefile aliases, README links, and diagnostic separation.

Cross-cutting constraints:
- Default automated tests stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, or SynthTIGER.
- Generated images, tensors, checkpoints, logs, large datasets, `outputs/`, and `runs/` stay out of git unless intentionally tiny fixtures or documentation assets.

### Phase 2: Runtime Contracts and Run Provenance
**Goal**: Users can validate configs, paths, artifacts, and run metadata before long-running generation, scoring, training, or evaluation stages start.  
**Depends on**: Phase 1  
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04, ART-01, ART-02, ART-03, ART-04, RUN-01, RUN-03, RUN-04, STR-02  
**Success Criteria** (what must be TRUE):
  1. User can load existing and new configs through one validated path that reports missing fields, invalid values, model ID inconsistencies, and local/SLURM path problems before expensive work begins.
  2. User can create a local run directory whose manifest captures command, timestamp, git state, resolved config snapshot, environment summary, seeds, model IDs/revisions, inputs, outputs, metrics, and notes.
  3. User can rely on documented canonical paths and schema/version metadata for prompts, generated images, latents, embeddings, masks, scores, selected samples, preference pairs, checkpoints, logs, eval outputs, and manifests.
  4. User can preflight key artifacts and resume or inspect long-running stages using consistent manifest and output conventions without committing generated tensors, images, checkpoints, logs, or private outputs.
  5. User can use shared runtime helpers for config I/O, path resolution, seeds, manifests, and preflight validation across pipeline stages.
**Plans**: 5 plans

Plans:

**Wave 1**
- [x] 02-01-PLAN.md — Add shared validated config-loading contracts for SFT, DPO, and masked-SFT.
- [x] 02-02-PLAN.md — Define canonical runtime paths, artifact schemas, preflight validators, and generated-artifact safety docs.

**Wave 2** *(blocked on Wave 1 config/artifact contracts)*
- [x] 02-03-PLAN.md — Add local run manifests, config snapshots, reproducibility metadata, and manifest CLI commands.

**Wave 3** *(blocked on runtime helpers and manifests)*
- [x] 02-04-PLAN.md — Wire trainer config loaders to shared validation and expose a CPU-safe preflight CLI.

**Wave 4** *(blocked on manifest/preflight command implementations)*
- [x] 02-05-PLAN.md — Publish config-family organization, manifest/preflight command docs, Makefile aliases, and README links.

Cross-cutting constraints:
- Runtime helpers and preflight commands stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, or SynthTIGER.
- Generated images, tensors, checkpoints, logs, private outputs, and run directories stay ignored unless intentionally tiny fixtures or documentation assets.
- Existing root config files and documented local/SLURM commands remain runnable while new experiment variants gain organized family homes.

### Phase 3: Data Curriculum and Dataset Quality
**Goal**: Users can create and assess multilingual text-rendering datasets with explicit curriculum, provenance, quality checks, and versioned training selections.  
**Depends on**: Phase 2  
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07  
**Success Criteria** (what must be TRUE):
  1. User can choose explicit prompt/data curriculum stages for Cyrillic and multilingual rendering, from simple letters/words through harder punctuation, multiline, style, and scene cases.
  2. User can generate prompt datasets through explicit configs rather than monkey-patching and validate length, character coverage, duplicates, naturalness, malformed outputs, and content/style distributions.
  3. User can generate dataset manifests that record config, seeds, git commit, source hashes, model IDs, filtering stats, output counts, and relevant provenance for prompts and synthetic data.
  4. User can inspect synthetic dataset quality using OCR verification, mask/bbox/contrast filters, per-character and per-font coverage, resolution mix, and contact sheets.
  5. User can materialize selected SFT samples and DPO preference pairs as versioned artifacts and compare generated-image reward-filtered data against synthetic masked-SFT data.
**Plans**: 6 plans

Plans:

**Wave 1**
- [x] 03-01-PLAN.md — Add explicit prompt curriculum configs and config-driven prompt generation modes.
- [x] 03-02-PLAN.md — Add prompt dataset quality validation and dataset manifest tooling.
- [x] 03-03-PLAN.md — Add synthetic masked-SFT quality inspection, optional OCR summaries, and contact sheets.
- [x] 03-04-PLAN.md — Materialize selected SFT samples and DPO preference pairs as versioned artifacts.

**Wave 2** *(blocked on Wave 1 quality/selection artifacts)*
- [x] 03-05-PLAN.md — Compare generated-image reward-filtered data against synthetic masked-SFT data.

**Wave 3** *(blocked on Phase 3 implementation contracts)*
- [x] 03-06-PLAN.md — Publish Phase 3 runtime contracts, command aliases, README links, and docs tests.

Cross-cutting constraints:
- Default automated tests stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, or SynthTIGER.
- Generated prompt datasets, synthetic images/masks/tensors, quality reports under runtime roots, contact sheets, selected samples, preference pairs, checkpoints, logs, and run directories stay out of git unless intentionally tiny fixtures or documentation assets.
- Prompt, synthetic, selection, and comparison artifacts must record provenance through local manifests and source hashes before they are used as thesis evidence.

### Phase 4: CPU-Safe Characterization Tests
**Goal**: Users can verify fragile behavior with fast, deterministic tests before reward, trainer, prompt, dataset, or runtime code is moved.  
**Depends on**: Phase 3  
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TRN-01  
**Success Criteria** (what must be TRUE):
  1. User can run automated CPU-safe tests for config parsing/validation, dataset loading, collators, pair/sample selection, and artifact path/shape contracts using tiny fixtures.
  2. User can run deterministic tests for prompt generation under fixed seeds and mocked reward wrappers without loading Qwen, PaddleOCR, FLUX, or other large models.
  3. User can verify critical tensor math including masked losses, scheduler helpers, DPO objective sign, beta scaling, and winner/loser behavior before trusting training results.
  4. User can run the default test command without model downloads or CUDA while still having explicit markers/commands for optional slow, GPU, OCR, model, integration, and diagnostic checks.
**Plans**: 6 plans

Plans:

**Wave 1**
- [x] 04-01-PLAN.md — Add committed-config and tiny-artifact characterization tests.
- [x] 04-02-PLAN.md — Add dataset, collator, selection, and resolution bucket characterization tests.
- [x] 04-03-PLAN.md — Add objective math, scheduler, latent geometry, and DPO sign/beta tests.
- [x] 04-04-PLAN.md — Add fixed-seed prompt-generation determinism tests.
- [x] 04-05-PLAN.md — Add import-safe fake/mock reward wrapper tests.

**Wave 2** *(blocked on Wave 1 characterization surfaces)*
- [x] 04-06-PLAN.md — Publish characterization commands, Makefile aliases, README/runtime docs, and docs tests.

Cross-cutting constraints:
- Default automated tests stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, SynthTIGER, or external model weights.
- Characterization fixtures stay tiny and use `tmp_path` unless intentionally committed under `tests/fixtures/`.
- DPO objective sign, beta scaling, and winner/loser behavior are research-critical and must be verified before Phase 5 trainer comparability work.

### Phase 5: Training Objective and Pipeline Comparability
**Goal**: Users can configure, run, extend, and compare training approaches without hidden objective choices or accidental behavior changes.  
**Depends on**: Phase 4  
**Requirements**: TRN-02, TRN-03, TRN-04, TRN-05, TRN-06, TRN-07, RUN-02, STR-04  
**Success Criteria** (what must be TRUE):
  1. User can distinguish and configure SFT sample-selection modes, DPO pair-construction modes, and masked-SFT loss/LoRA/dataset/evaluation choices in configs and manifests.
  2. User can compare baseline, SFT, DPO, masked-SFT, and combined/curriculum runs under controlled prompts, seeds, inference settings, data sources, rewards, metrics, and artifact paths.
  3. User can compare two run manifests to see changed configs, data sources, rewards, seeds, inference settings, metrics, and outputs.
  4. User can identify training/inference mismatches such as step count, guidance, prompt embedding padding, model variants, and sampling configuration differences before using results as evidence.
  5. User can add or modify trainer variants through focused shared modules for sampling, checkpointing, schedulers, objective helpers, and runtime plumbing while preserving existing SFT, DPO, and masked-SFT behavior.
**Plans**: 6 plans

Plans:

**Wave 1**
- [x] 05-01-PLAN.md — Make SFT sample-selection and DPO pair-construction modes explicit in materialized artifacts, CLI flags, tests, and docs.
- [x] 05-02-PLAN.md — Add CPU-safe run-manifest diff module, CLI, tests, and runtime docs for RUN-02 comparisons.
- [x] 05-03-PLAN.md — Add CPU-safe training comparability mismatch detector, CLI, tests, and guide for controlled fields.

**Wave 2** *(blocked on Wave 1 selection and comparability contracts)*
- [x] 05-04-PLAN.md — Wire explicit SFT, DPO, and masked-SFT choices into config dataclasses, validation, snapshots, tests, and experiment config docs.
- [x] 05-05-PLAN.md — Add import-safe shared training modules for sampling, checkpointing, schedulers, runtime metadata, tests, and extension guidance.

**Wave 3** *(blocked on Wave 2 config/shared utility contracts plus Wave 1 manifest/comparability CLIs)*
- [x] 05-06-PLAN.md — Publish integrated training-run comparison CLI, Makefile alias, command docs, README links, and docs drift tests.

Cross-cutting constraints:
- Default automated tests stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, SynthTIGER, or external model weights.
- Selection artifacts, run manifests, comparison reports, checkpoints, logs, tensors, generated images, and private outputs remain ignored runtime artifacts unless intentionally tiny fixtures or documentation assets.
- DPO winner/loser semantics and current objective sign/beta behavior remain explicitly tested before any comparison-grade DPO changes are trusted.
- Comparability reports must surface missing or uncontrolled evidence explicitly rather than treating absent metrics/artifacts as comparable.

### Phase 6: Reward and Evaluation Validity
**Goal**: Users can trust reward scores, held-out evaluations, diagnostic reports, and thesis outputs as comparable evidence tied back to exact runs.  
**Depends on**: Phase 5  
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08, RUN-05, STR-03  
**Success Criteria** (what must be TRUE):
  1. User can use one canonical reward interface for Qwen/VLM, OCR/CER/entropy, and product reward paths across scoring, training, evaluation, and thesis reports.
  2. User can reproduce product score files with documented formula, scorer versions, component scores, thresholds, schema metadata, and manifest links.
  3. User can run held-out checkpoint-comparison evaluations with fixed prompts, seeds, inference settings, and automatic OCR/VLM/product/exact/character-level scoring.
  4. User can inspect evaluation by Russian text difficulty slices and reward disagreement reports, including scatter/correlation summaries, false-positive/false-negative contact sheets, and per-character confusion summaries.
  5. User can validate reward signals against a small gold diagnostic benchmark and generate thesis-ready tables, plots, and contact sheets from recorded run outputs that trace back to exact manifests/artifacts.
**Plans**: 7 plans

Plans:

**Wave 1**
- [x] 06-01-PLAN.md — Define the canonical reward result interface and reproducible product score formula.
- [x] 06-02-PLAN.md — Add a held-out checkpoint-comparison evaluation harness contract with fixed prompts, seeds, settings, and manifests.
- [ ] 06-03-PLAN.md — Add Russian text difficulty slicing and a small gold diagnostic benchmark contract.

**Wave 2** *(blocked on canonical reward and slice/gold contracts)*
- [ ] 06-04-PLAN.md — Wire scoring/evaluation outputs to canonical OCR/VLM/product/exact/character-level fields and sidecar validation.
- [ ] 06-05-PLAN.md — Generate reward disagreement diagnostics, false-positive/false-negative reports, confusion summaries, and contact-sheet manifests.

**Wave 3** *(blocked on scoring outputs and diagnostics)*
- [ ] 06-06-PLAN.md — Build thesis-ready output bundles from recorded manifests, score outputs, diagnostics, and artifact references.

**Wave 4** *(blocked on Phase 6 implementation contracts)*
- [ ] 06-07-PLAN.md — Publish Phase 6 command docs, README links, Makefile aliases, and docs drift tests.

Cross-cutting constraints:
- Default automated tests stay CPU-safe and do not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, SynthTIGER, or external model weights.
- Generated score files, held-out outputs, diagnostic reports, contact sheets, thesis tables/plots, checkpoints, logs, tensors, generated images, and private run outputs remain ignored runtime artifacts unless intentionally tiny fixtures.
- Missing reward/evaluation evidence must be surfaced explicitly rather than treated as comparable or thesis-ready.

### Phase 7: Moderate Structure and Extension Cleanup
**Goal**: Users can navigate a clearer brownfield toolkit and add future experiments through stable, documented seams instead of one-off scripts.  
**Depends on**: Phase 6  
**Requirements**: STR-01, STR-05, STR-06  
**Success Criteria** (what must be TRUE):
  1. User can find reusable source code, thin scripts, cluster launchers, configs, diagnostics, experiments, generated outputs, tests, and thesis artifacts in clear homes after moderate safe moves.
  2. User can invoke generation, scoring, synthesis, evaluation, plotting, and run comparison through thin CLI scripts backed by importable implementation modules.
  3. User can follow documented extension points to add future experiments, trainers, reward variants, datasets, or pipelines without creating hidden assumptions in unrelated scripts.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Execution Surface and Pipeline Inventory | 4/4 | Complete | 01-01, 01-02, 01-03, 01-04 |
| 2. Runtime Contracts and Run Provenance | 5/5 | Complete | 02-01, 02-02, 02-03, 02-04, 02-05 |
| 3. Data Curriculum and Dataset Quality | 6/6 | Complete | 03-01, 03-02, 03-03, 03-04, 03-05, 03-06 |
| 4. CPU-Safe Characterization Tests | 6/6 | Complete | 04-01, 04-02, 04-03, 04-04, 04-05, 04-06 |
| 5. Training Objective and Pipeline Comparability | 6/6 | Plans complete; verification pending | 05-01, 05-02, 05-03, 05-04, 05-05, and 05-06 complete |
| 6. Reward and Evaluation Validity | 2/7 | In progress | 06-01 and 06-02 |
| 7. Moderate Structure and Extension Cleanup | 0/TBD | Not started | - |

## Requirement Coverage

| Requirement | Phase |
|-------------|-------|
| INV-01 | Phase 1 |
| INV-02 | Phase 1 |
| INV-03 | Phase 1 |
| INV-04 | Phase 1 |
| ENV-01 | Phase 1 |
| ENV-02 | Phase 1 |
| ENV-03 | Phase 1 |
| ENV-04 | Phase 1 |
| ENV-05 | Phase 1 |
| ENV-06 | Phase 1 |
| TEST-06 | Phase 1 |
| TEST-07 | Phase 1 |
| CFG-01 | Phase 2 |
| CFG-02 | Phase 2 |
| CFG-03 | Phase 2 |
| CFG-04 | Phase 2 |
| ART-01 | Phase 2 |
| ART-02 | Phase 2 |
| ART-03 | Phase 2 |
| ART-04 | Phase 2 |
| RUN-01 | Phase 2 |
| RUN-03 | Phase 2 |
| RUN-04 | Phase 2 |
| STR-02 | Phase 2 |
| DATA-01 | Phase 3 |
| DATA-02 | Phase 3 |
| DATA-03 | Phase 3 |
| DATA-04 | Phase 3 |
| DATA-05 | Phase 3 |
| DATA-06 | Phase 3 |
| DATA-07 | Phase 3 |
| TEST-01 | Phase 4 |
| TEST-02 | Phase 4 |
| TEST-03 | Phase 4 |
| TEST-04 | Phase 4 |
| TEST-05 | Phase 4 |
| TRN-01 | Phase 4 |
| TRN-02 | Phase 5 |
| TRN-03 | Phase 5 |
| TRN-04 | Phase 5 |
| TRN-05 | Phase 5 |
| TRN-06 | Phase 5 |
| TRN-07 | Phase 5 |
| RUN-02 | Phase 5 |
| STR-04 | Phase 5 |
| EVAL-01 | Phase 6 |
| EVAL-02 | Phase 6 |
| EVAL-03 | Phase 6 |
| EVAL-04 | Phase 6 |
| EVAL-05 | Phase 6 |
| EVAL-06 | Phase 6 |
| EVAL-07 | Phase 6 |
| EVAL-08 | Phase 6 |
| RUN-05 | Phase 6 |
| STR-03 | Phase 6 |
| STR-01 | Phase 7 |
| STR-05 | Phase 7 |
| STR-06 | Phase 7 |

**Coverage:** 58/58 v1 requirements mapped exactly once.

## Verification Notes

- Use adaptive verification: explicit checks for dependency/environment changes, config/artifact contracts, objective math, reward semantics, trainer refactors, and evaluation outputs.
- Do not over-gate obvious documentation/inventory moves, but keep CPU-safe tests and smoke commands documented as early as Phase 1.
- Phase 4 is the main behavior-locking gate before high-risk modularization.

## Caveats

- `REQUIREMENTS.md` stated 57 v1 requirements, but the listed v1 IDs total 58. This roadmap maps all 58 listed requirements and updates traceability coverage accordingly.
- Exact CUDA, PaddleOCR, vLLM, SynthTIGER, and FLUX dependency pins should be validated on target local/SLURM machines during Phase 1 before treating them as thesis-grade reproducibility guarantees.
- Reward/evaluation details in Phase 6 may need targeted thesis-method validation before final claims are made.

---
*Roadmap created: 2026-05-04*
