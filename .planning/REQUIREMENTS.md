# Requirements: Diffusion Text Tuner

**Defined:** 2026-05-04
**Core Value:** Researchers can reproducibly create, run, compare, diagnose, and extend diffusion fine-tuning experiments for multilingual text rendering.

## v1 Requirements

Requirements for turning the current thesis research repo into a reproducible and scientifically useful multilingual text-rendering training toolkit.

### Project Understanding and Pipeline Inventory

- [x] **INV-01**: User can see a current inventory of all runnable pipelines: prompt generation, baseline image generation, reward scoring, SFT, DPO, masked-SFT, synthetic data, evaluation, plotting, and SLURM launchers.
- [x] **INV-02**: User can see what each pipeline optimizes, what data it consumes, what artifacts it produces, and which thesis question it supports.
- [x] **INV-03**: User can identify outdated, duplicate, diagnostic, or experimental scripts separately from supported toolkit entry points.
- [x] **INV-04**: User can trace the historical experiment tracks: reward-filtered generated-image SFT/DPO, synthetic masked-MSE training, OCR/VLM/product reward variants, and thesis plotting/report flows.

### Reproducible Environment and Commands

- [x] **ENV-01**: User can install the project from a committed dependency manifest for Python 3.11.
- [x] **ENV-02**: User can choose optional dependency groups for GPU training, OCR/reward scoring, synthesis, vLLM/MLX backends, tests, linting, plotting, and analysis.
- [x] **ENV-03**: User can run a standard CPU-safe test command without downloading large models or requiring CUDA.
- [x] **ENV-04**: User can run documented smoke commands that validate key imports, CUDA availability, model access, PaddleOCR availability, and expected cache/runtime paths before launching long jobs.
- [x] **ENV-05**: User can format and lint the repository with standard documented commands.
- [x] **ENV-06**: User can run local and SLURM variants of supported commands through documented, comparable entry points.

### Data Curriculum and Dataset Quality

- [ ] **DATA-01**: User can define explicit prompt/data curriculum stages for Cyrillic and multilingual text rendering, including single letters, short words, multi-word phrases, digits, punctuation, mixed case, multiline text, and harder scene/style cases.
- [ ] **DATA-02**: User can validate generated prompt datasets for target length, character set, rare-character coverage, duplicate rate, text naturalness, content-type distribution, style distribution, and illegal/malformed outputs.
- [ ] **DATA-03**: User can replace prompt-generation monkey-patching with explicit config objects or config files for simple, full, curriculum, and future prompt-generation modes.
- [ ] **DATA-04**: User can generate dataset manifests that record config, seed strategy, git commit, source word/scene/font/background hashes, model IDs, filtering stats, and output counts.
- [ ] **DATA-05**: User can inspect and report synthetic dataset quality, including OCR verification, mask area/bbox/contrast filters, per-character coverage, per-font coverage, resolution mix, and sample contact sheets.
- [ ] **DATA-06**: User can materialize selected SFT samples and DPO preference pairs as versioned artifacts instead of selecting them only inside dataset constructors.
- [ ] **DATA-07**: User can compare generated-image reward-filtered data against synthetic masked-SFT data and identify where each data source is expected to help or fail.

### Objective and Training Pipeline Validity

- [ ] **TRN-01**: User can verify DPO objective sign, beta scaling, and winner/loser behavior with deterministic unit tests before relying on DPO results.
- [ ] **TRN-02**: User can distinguish SFT modes such as all-above-threshold, top-1 winner per prompt, score-weighted SFT, and filtered hard-positive SFT.
- [ ] **TRN-03**: User can distinguish DPO pair-construction modes such as best-vs-worst, all separated pairs, margin-weighted pairs, and ambiguity-filtered pairs.
- [ ] **TRN-04**: User can run masked-SFT experiments with explicit masked/global loss weighting, LoRA target/rank choices, synthetic dataset variant, and evaluation suite references captured in config and manifest.
- [ ] **TRN-05**: User can compare baseline, SFT, DPO, masked-SFT, and combined/curriculum approaches under controlled prompts, seeds, inference settings, and evaluation metrics.
- [ ] **TRN-06**: User can identify training/inference mismatches such as step count, guidance, prompt embedding padding, model ID variants, and sampling configuration differences.
- [ ] **TRN-07**: User can add new trainer or pipeline variants without editing unrelated scripts or losing existing runnable behavior.

### Reward and Evaluation Validity

- [ ] **EVAL-01**: User can use one canonical reward interface for Qwen/VLM, OCR/CER/entropy, and product reward paths across scoring, training, evaluation, and thesis reports.
- [ ] **EVAL-02**: User can reproducibly generate product score files with documented formula, scorer versions, component scores, thresholds, and manifest metadata.
- [ ] **EVAL-03**: User can run a held-out checkpoint-comparison evaluation harness with fixed prompts, fixed seeds, fixed inference settings, and comparable outputs for baseline and trained LoRAs.
- [ ] **EVAL-04**: User can automatically score evaluation outputs with OCR CER, OCR detection rate, entropy/confidence, VLM score, product score, and exact/character-level text metrics where possible.
- [ ] **EVAL-05**: User can evaluate by Russian text difficulty slices: rare Cyrillic letters, word length, phrase length, digits, punctuation, mixed case, multiline layout, font/style, and scene/background type.
- [ ] **EVAL-06**: User can inspect reward disagreement through VLM-vs-OCR scatter/correlation, false-positive/false-negative contact sheets, and per-character confusion summaries.
- [ ] **EVAL-07**: User can validate reward signals against a small hand-labeled or gold diagnostic benchmark before using them as thesis evidence.
- [ ] **EVAL-08**: User can generate thesis-ready tables, plots, and contact sheets from recorded run outputs rather than static/manual numbers.

### Run Tracking and Experiment Comparison

- [ ] **RUN-01**: User can create a local run directory with a manifest containing command, timestamp, git state, resolved config, environment summary, seeds, model IDs/revisions, input paths, output paths, metrics, and notes.
- [ ] **RUN-02**: User can compare two local run manifests to see changed configs, data sources, rewards, seeds, inference settings, metrics, and artifacts.
- [ ] **RUN-03**: User can record long-running pipeline stage outputs in a consistent artifact layout without committing generated tensors, images, checkpoints, or logs.
- [ ] **RUN-04**: User can resume or inspect long-running generation, scoring, training, and evaluation stages using documented manifest and output conventions.
- [ ] **RUN-05**: User can map thesis plots/results back to the exact run manifests and artifacts that produced them.

### Configuration and Artifact Contracts

- [x] **CFG-01**: User can load existing JSON configs through a validated config-loading path that reports missing fields, invalid values, model ID inconsistencies, and path problems before expensive work starts.
- [ ] **CFG-02**: User can organize experiment configs by stage/family with consistent naming for SFT, DPO, masked-SFT, reward, synthesis, evaluation, and ablation variants.
- [ ] **CFG-03**: User can capture the resolved config used for a run as an immutable snapshot in that run's manifest directory.
- [x] **CFG-04**: User can define local and SLURM-compatible path settings without hardcoded personal absolute paths.
- [ ] **ART-01**: User can validate prompt JSONL, generated image directories, latent/text-embedding tensors, scores CSV files, masks, synthetic dataset indexes, selected sample manifests, DPO pair manifests, and checkpoint paths before GPU-heavy stages run.
- [ ] **ART-02**: User can rely on documented canonical paths for prompts, generated images, latents, text embeddings, masks, scores, selected samples, preference pairs, checkpoints, samples, logs, eval outputs, and run manifests.
- [ ] **ART-03**: User can see schema/version metadata for key generated artifacts and score files.
- [ ] **ART-04**: User can keep generated artifacts out of git while preserving small fixtures needed by tests and documentation.

### Testing and Diagnostics

- [ ] **TEST-01**: User can run automated tests for config parsing and validation.
- [ ] **TEST-02**: User can run automated tests for dataset loading, collators, pair/sample selection, and artifact path/shape contracts using tiny fixtures.
- [ ] **TEST-03**: User can run automated tests for critical tensor math including masked losses, scheduler helpers, DPO objective helpers, and beta/sign behavior.
- [ ] **TEST-04**: User can run automated tests for deterministic prompt generation behavior under fixed seeds.
- [ ] **TEST-05**: User can run automated tests for reward wrapper behavior using fakes/mocks rather than loading Qwen, PaddleOCR, or other large models.
- [x] **TEST-06**: User can distinguish default CPU tests from marked slow, GPU, model, OCR, integration, and manual diagnostic checks.
- [x] **TEST-07**: User can run optional diagnostic commands for gradient flow, generated sample quality, reward calibration, synthetic data validation, and eval contact sheets without confusing them with CI-safe tests.

### Structure and Refactoring

- [ ] **STR-01**: User can navigate a moderately cleaned file structure where reusable source code, thin scripts, cluster launchers, configs, diagnostics, experiments, generated outputs, tests, and thesis artifacts have clear homes.
- [x] **STR-02**: User can use shared runtime helpers for config I/O, path resolution, seeds, manifests, and preflight validation.
- [ ] **STR-03**: User can use shared scoring/reward/evaluation modules instead of duplicated Qwen/OCR logic across training and evaluation.
- [ ] **STR-04**: User can use focused shared training modules for sampling, checkpointing, schedulers, objective helpers, and config/runtime plumbing while preserving current SFT, DPO, and masked-SFT behavior.
- [ ] **STR-05**: User can use importable implementation modules behind CLI scripts for generation, scoring, synthesis, evaluation, plotting, and run comparison.
- [ ] **STR-06**: User can add future experiments/pipelines through documented extension points rather than new one-off scripts with hidden assumptions.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Tracking and Orchestration

- **TRK-01**: User can sync or mirror local manifests to MLflow, Weights & Biases, DVC, or another external tracking/data-versioning system.
- **TRK-02**: User can run multi-run sweep management through a higher-level orchestrator if JSON config variants become insufficient.
- **TRK-03**: User can publish reproducible dataset/run cards suitable for external release.

### Advanced Evaluation

- **ADV-01**: User can classify text-rendering failures with a formal error taxonomy.
- **ADV-02**: User can combine OCR, VLM, and human-evaluation signals into a calibrated thesis metric.
- **ADV-03**: User can maintain a broader curated Russian visual-text benchmark with licensing/provenance reviewed.

### Scaling and Productization

- **SCL-01**: User can publish or consume sharded datasets through object storage or Hugging Face Datasets.
- **SCL-02**: User can use a plugin-style architecture for new trainers, reward models, and dataset builders after stable extension seams are proven.
- **SCL-03**: User can expose parts of the toolkit through an API or UI if the thesis/toolkit scope later requires it.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Hosted web app or API service in v1 | The current project is a local/cluster thesis ML toolkit. |
| Committing generated model artifacts | Images, tensors, checkpoints, logs, and large datasets are environment-specific and should remain ignored. |
| Big-bang package reorganization | Moderate cleanup preserves existing runnable experiments and reduces refactor risk. |
| Immediate MLflow/W&B/DVC adoption | Simple local manifests solve the near-term reproducibility need with less operational overhead. |
| Replacing FLUX/Qwen/PaddleOCR/SynthTIGER stack as cleanup work | Core method changes need explicit experimental motivation, not refactor momentum. |
| Treating training loss or DPO accuracy as final thesis evidence | The thesis claim needs held-out text-rendering evaluation and reward validation. |
| Golden full E2E GPU test as a default CI gate | Too expensive and fragile; default tests should remain CPU-safe with optional GPU diagnostics. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INV-01 | Phase 1 | Complete |
| INV-02 | Phase 1 | Complete |
| INV-03 | Phase 1 | Complete |
| INV-04 | Phase 1 | Complete |
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| ENV-03 | Phase 1 | Complete |
| ENV-04 | Phase 1 | Complete |
| ENV-05 | Phase 1 | Complete |
| ENV-06 | Phase 1 | Complete |
| DATA-01 | Phase 3 | Pending |
| DATA-02 | Phase 3 | Pending |
| DATA-03 | Phase 3 | Pending |
| DATA-04 | Phase 3 | Pending |
| DATA-05 | Phase 3 | Pending |
| DATA-06 | Phase 3 | Pending |
| DATA-07 | Phase 3 | Pending |
| TRN-01 | Phase 4 | Pending |
| TRN-02 | Phase 5 | Pending |
| TRN-03 | Phase 5 | Pending |
| TRN-04 | Phase 5 | Pending |
| TRN-05 | Phase 5 | Pending |
| TRN-06 | Phase 5 | Pending |
| TRN-07 | Phase 5 | Pending |
| EVAL-01 | Phase 6 | Pending |
| EVAL-02 | Phase 6 | Pending |
| EVAL-03 | Phase 6 | Pending |
| EVAL-04 | Phase 6 | Pending |
| EVAL-05 | Phase 6 | Pending |
| EVAL-06 | Phase 6 | Pending |
| EVAL-07 | Phase 6 | Pending |
| EVAL-08 | Phase 6 | Pending |
| RUN-01 | Phase 2 | Pending |
| RUN-02 | Phase 5 | Pending |
| RUN-03 | Phase 2 | Pending |
| RUN-04 | Phase 2 | Pending |
| RUN-05 | Phase 6 | Pending |
| CFG-01 | Phase 2 | Complete |
| CFG-02 | Phase 2 | Pending |
| CFG-03 | Phase 2 | Pending |
| CFG-04 | Phase 2 | Complete |
| ART-01 | Phase 2 | Pending |
| ART-02 | Phase 2 | Pending |
| ART-03 | Phase 2 | Pending |
| ART-04 | Phase 2 | Pending |
| TEST-01 | Phase 4 | Pending |
| TEST-02 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |
| TEST-04 | Phase 4 | Pending |
| TEST-05 | Phase 4 | Pending |
| TEST-06 | Phase 1 | Complete |
| TEST-07 | Phase 1 | Complete |
| STR-01 | Phase 7 | Pending |
| STR-02 | Phase 2 | Complete |
| STR-03 | Phase 6 | Pending |
| STR-04 | Phase 5 | Pending |
| STR-05 | Phase 7 | Pending |
| STR-06 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 58 total
- Mapped to phases: 58
- Unmapped: 0
- Note: the previous count said 57, but the v1 section contains 58 listed requirement IDs; all listed IDs are mapped exactly once.

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 after Phase 2 Plan 01 execution*
