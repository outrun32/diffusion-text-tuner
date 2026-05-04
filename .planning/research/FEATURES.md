# Feature Landscape

**Domain:** diffusion-based thesis training toolkit for multilingual text rendering  
**Project:** Diffusion Text Tuner  
**Researched:** 2026-05-04  
**Scope:** Brownfield feature recommendations for making the existing toolkit reproducible, understandable, and extensible for new experiments and pipelines.

## Research Position

This repository should behave like a thesis-grade ML toolkit, not a production service and not a loose folder of experiment scripts. The expected feature set is therefore centered on repeatable experiment execution, inspectable artifacts, stable extension points, and lightweight validation around expensive GPU workflows.

The existing code already contains the core research capability: prompt generation, FLUX image/latent/text-embedding generation, reward scoring, SFT/DPO/masked-SFT LoRA training, synthetic Cyrillic data tooling, evaluation scripts, and SLURM launch patterns. The next features should make those capabilities reliable to rerun and safe to extend.

## Table Stakes

Features users expect in a reproducible thesis-oriented ML toolkit. Missing table-stakes features make the repository feel incomplete even if the models can train.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Reproducible environment definition | Fresh machines and cluster jobs need one discoverable dependency source for PyTorch, Diffusers, Transformers, Accelerate, PEFT, OCR/VLM, SynthTIGER, and test-only dependencies. | Medium | None; should be first | Use `pyproject.toml` or requirements files with extras such as `train`, `ocr`, `synth`, `test`, and `dev`. Pin known-good ML versions tightly enough to avoid silent API drift. |
| Standard command catalog | Researchers need copy-pasteable commands for setup, tests, prompt generation, image generation, scoring, training, evaluation, synthesis, and SLURM use. | Low | Environment definition | Prefer `README` plus optional `Makefile`/task runner. Include CPU-safe test commands and GPU-only commands clearly separated. |
| Config inventory and variant conventions | New SFT, DPO, masked-SFT, reward, and eval variants need consistent naming and discoverable fields. | Medium | Environment definition; command catalog | Keep JSON configs, but document schema and naming conventions. Add examples like `masked_sft_debug.json`, `dpo_ocr.json`, `sft_low_mem.json`. |
| Local run manifests | Reproducibility requires saving config snapshots, command, git commit, environment summary, seed, input/output paths, model IDs/local paths, and notes per run. | Medium | Command catalog; config inventory | This is the right current tracking feature because project scope explicitly defers MLflow/W&B. Store under ignored `runs/` or `outputs/<stage>/manifest.json`. |
| Deterministic seed and reproducibility controls | Prompt generation, DataLoader workers, PyTorch/CUDA operations, and synthetic data sampling must expose seeds and record determinism settings. | Medium | Config inventory; run manifests | PyTorch documents that perfect reproducibility is not guaranteed across releases/platforms, so record platform, versions, CUDA, seeds, and deterministic flags rather than promising bitwise reproduction everywhere. |
| Artifact path contract | Generated prompts, images, latents, text embeddings, masks, scores, checkpoints, samples, and logs need predictable locations. | Medium | Config inventory; run manifests | Existing concerns show drift between `outputs/text_embeds` and `outputs/generated/text_embeds`. Define canonical layout and migration aliases or clear validation errors. |
| Dataset and artifact validators | Fail fast when JSONL/CSV manifests, `.pt` tensors, masks, shapes, scores, or image files are missing, malformed, or mismatched. | Medium | Artifact path contract | Especially important for masked-SFT `shapes.csv`, latent/text embedding pairing, preference pairs, and score CSV resume/sharding. |
| Lightweight automated test suite | Safe refactoring requires tests that do not load FLUX/Qwen or require CUDA. | Medium | Environment definition | Add fixtures for config parsing, datasets/collators, prompt determinism, reward wrapper behavior via mocks, latent shape utilities, and trainer math. Keep expensive diagnostics outside pytest discovery. |
| Smoke/preflight checks | Long GPU jobs should fail before model download/training if model access, local paths, CUDA, dtype, required files, or output directories are invalid. | Medium | Environment definition; artifact path contract | Add `--dry-run` or `check_*` commands for generation, scoring, synthesis, and each trainer. |
| Shared reward interface | Scoring, training, and evaluation should call the same Qwen/PaddleOCR reward semantics to prevent drift. | Medium | Tests; config inventory | Existing reward duplication is a high-impact concern because reward changes can corrupt SFT filtering and DPO pair construction. |
| Clear experiment lifecycle documentation | A thesis toolkit needs an understandable narrative: generate prompts → generate images/latents/embeddings → score → train → evaluate → compare. | Low | Command catalog; run manifests | Include expected inputs/outputs for each stage and which artifacts are committed vs ignored. |
| Local/SLURM parity | Commands that run locally should map cleanly to cluster submissions, with matching config names and output layout. | Medium | Command catalog; artifact path contract | Preserve existing SLURM patterns but make them parameterized enough for new variants without editing job scripts each time. |
| Small committed fixtures | Automated tests and examples need tiny prompts, tensors, masks, scores, and maybe tiny images that are safe to commit. | Medium | Artifact path contract; tests | Do not commit generated datasets/checkpoints. Commit only intentionally tiny fixtures under `tests/fixtures/` or existing allowed asset paths. |
| Evaluation suite and comparison outputs | Thesis experiments need consistent metrics and artifacts for comparing baseline vs LoRA variants. | Medium | Run manifests; shared reward interface | Keep evaluation local-file based initially: CSV/JSONL summaries, sample grids, score distributions, and per-language/per-script breakdowns. |
| Extension guides for new pipelines | New experiments should have a documented path for adding a trainer, reward, dataset, prompt generator, or synthetic-data variant. | Low | Config inventory; docs; tests | Convert the current codebase map guidance into project docs with concrete templates and checklists. |

## Differentiators

Features that make the toolkit especially useful for multilingual text-rendering research. These are not all necessary for the first cleanup milestone, but they strengthen the thesis-toolkit narrative.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| Multilingual/script coverage reports | Shows whether prompt and synthetic datasets cover target scripts, characters, fonts, word lengths, and language buckets. | Medium | Prompt determinism tests; dataset validators | Particularly valuable for Cyrillic and future multilingual expansion. Output character histograms, rare-character coverage, and per-script counts. |
| Text-rendering error taxonomy | Makes evaluation explainable beyond one scalar reward. | High | Shared reward interface; evaluation suite | Categorize errors such as missing text, wrong characters, hallucinated glyphs, unreadable text, misplaced text, wrong language/script, and style/layout failure. Can begin manually or with OCR/VLM heuristics. |
| Reward calibration dashboard/report | Helps compare OCR and VLM reward behavior and detect drift between scoring methods. | Medium | Shared reward interface; evaluation suite | Could be static HTML/Markdown/CSV plots generated locally, not a hosted app. Include yes/no probability distributions and OCR confidence summaries. |
| Experiment manifest diff tool | Researchers can compare two runs by config, inputs, model IDs, seeds, metrics, and artifacts. | Medium | Run manifests; config inventory | Useful before adding external tracking. Can start as a CLI producing Markdown. |
| Pipeline resume and idempotency checks | Expensive generation/scoring/training should resume safely without corrupting outputs. | Medium | Artifact path contract; validators; manifests | Scoring already has sharding/resume patterns; standardize across stages. |
| Sharded dataset indexes | Larger synthetic/image-generation runs need explicit indexes instead of ad hoc directory scans. | High | Artifact validators; artifact path contract | Use CSV/JSONL manifests with stable sample IDs, shard IDs, paths, shape metadata, source config, and checksum/hash where practical. |
| Golden tiny end-to-end pipeline | A toy run validates the full logical flow without downloading full models or using a large GPU. | High | Fixtures; mockable interfaces; tests | Use mocks/fakes for FLUX/Qwen where possible. This is a powerful confidence feature but requires refactoring boundaries first. |
| Config templates for experiment families | Speeds creation of new SFT/DPO/masked-SFT/reward/evaluation variants while preserving comparability. | Medium | Config inventory | Could be JSON templates plus documented override patterns; avoid adding complex Hydra-style hierarchy unless needed. |
| Thesis figure/table generators | Converts run/evaluation outputs into repeatable thesis plots and tables. | Medium | Evaluation suite; run manifests | Existing `scripts/thesis_plots/` can become a documented output layer for results chapters. |
| Dataset card/run card generation | Produces human-readable summaries for synthetic datasets and training runs. | Medium | Manifests; coverage reports; validators | Aligns with Hugging Face dataset conventions: files plus a README/dataset card documenting splits, features, provenance, and integrity expectations. |
| Model access and cache preflight | Avoids late failures from gated Hugging Face models, missing credentials, wrong local model paths, or exhausted cache/disk. | Medium | Environment definition; preflight checks | Especially valuable for FLUX/Qwen dependencies and shared cluster environments. |
| Low-memory/debug presets | Enables contributors to test pipeline mechanics without full thesis-scale jobs. | Medium | Config templates; command catalog | Include tiny batch sizes, tiny sample counts, skipped sampling, and mock reward options. |
| Static architecture map generated from known entry points | Keeps docs from drifting as scripts and modules move. | Medium | Extension guides; structure cleanup | Could be a maintained Markdown map rather than fully automated initially. |

## Anti-Features

Features to explicitly avoid because they add operational burden or distract from thesis reproducibility.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Hosted web application or production API | The project target is a local/cluster research toolkit; web/API work consumes time without improving thesis experiments. | Provide CLIs, configs, run manifests, and static reports. |
| Immediate MLflow/W&B dependency as the primary tracker | External trackers add account/config/network complexity and are explicitly out of current scope. | Implement local JSON/YAML manifests first; optionally design the manifest so it can later be logged to Accelerate-supported trackers. |
| Major package rewrite before reproducibility | Large rearchitecture risks breaking existing experiments and delaying thesis work. | Do moderate, safe moves and extract shared utilities around tests. |
| Committing generated images, tensors, checkpoints, large datasets, or logs | Bloats git, may leak prompt text, and makes repository history hard to manage. | Keep artifacts ignored; commit tiny fixtures and metadata only. |
| Expensive GPU/model diagnostics named `test_*.py` | Pytest discovery can accidentally launch model downloads or CUDA jobs. | Rename to `check_*`/`diagnose_*`, guard with `main()`, and keep automated tests lightweight. |
| Replacing FLUX/Qwen/PaddleOCR/SynthTIGER without experimental reason | The current thesis work needs reproducibility around existing foundations before changing core methods. | Encapsulate model/reward interfaces so alternatives can be added later as variants. |
| Overly abstract plugin framework | Premature abstraction can obscure the simple experiment lifecycle and slow researchers down. | Use clear extension guides, shared interfaces, and focused modules only where duplication already exists. |
| Full deterministic reproduction guarantee | PyTorch explicitly notes reproducibility is not guaranteed across releases, commits, platforms, CPU/GPU even with identical seeds. | Offer best-effort determinism controls and record complete run metadata. |
| Object-store/distributed data platform as first milestone | Overkill for the current single-node filesystem/SLURM pipeline. | Standardize local artifact layouts and sharded manifests first; defer object-store/HF publishing until artifact scale demands it. |
| Training-feature additions without validation fixtures | New objectives or reward variants can silently change research conclusions. | Require small unit/integration fixtures for every new pipeline feature. |

## Feature Dependencies

```text
Reproducible environment definition
  → Standard command catalog
  → Lightweight automated test suite
  → Smoke/preflight checks

Config inventory and variant conventions
  → Standard command catalog
  → Local run manifests
  → Config templates for experiment families
  → Experiment manifest diff tool

Artifact path contract
  → Dataset and artifact validators
  → Small committed fixtures
  → Pipeline resume and idempotency checks
  → Sharded dataset indexes

Lightweight automated test suite
  → Safe shared reward interface refactor
  → Golden tiny end-to-end pipeline
  → Extension guides for new pipelines

Shared reward interface
  → Evaluation suite and comparison outputs
  → Reward calibration dashboard/report
  → Text-rendering error taxonomy

Prompt determinism controls
  → Multilingual/script coverage reports
  → Dataset card/run card generation

Local run manifests
  → Evaluation suite and comparison outputs
  → Experiment manifest diff tool
  → Dataset card/run card generation
  → Thesis figure/table generators

Local/SLURM parity
  → Pipeline resume and idempotency checks
  → Model access and cache preflight
```

## MVP Recommendation

Prioritize these features first because they unblock reproducible reruns and safer brownfield extension without destabilizing model behavior:

1. **Reproducible environment definition** — create the foundation for all local, GPU, OCR, synthesis, and test workflows.
2. **Standard command catalog** — make every existing pipeline stage executable from documented commands.
3. **Config inventory and variant conventions** — make future SFT/DPO/masked-SFT/reward/eval variants consistent.
4. **Artifact path contract** — eliminate path drift and clarify committed vs generated files.
5. **Local run manifests** — capture enough metadata to rerun and compare thesis experiments without external tracking.
6. **Lightweight automated tests and fixtures** — protect shape math, config parsing, datasets/collators, prompt determinism, and reward wrappers.
7. **Smoke/preflight checks** — prevent expensive cluster/GPU failures caused by missing files, model access, or invalid configs.
8. **Shared reward interface** — stop scoring/training/evaluation reward semantics from drifting.

Defer:

- **MLflow/W&B integration:** local manifests are sufficient for the next milestone; external tracking can be added once manifest fields stabilize.
- **Golden tiny E2E pipeline:** valuable, but it depends on fixtures, mocks, and clearer component boundaries.
- **Text-rendering error taxonomy:** thesis-valuable, but it should build on stable reward/evaluation outputs.
- **Sharded dataset indexes/object-store workflows:** only necessary once local manifests and validators expose real scale pain.
- **Major plugin architecture:** extension guides and focused refactors are safer for this brownfield phase.

## Complexity Notes

| Complexity | Features | Why |
|------------|----------|-----|
| Low | Command catalog, lifecycle docs, extension guides | Mostly documentation and standardization around existing scripts. |
| Medium | Environment definition, config inventory, run manifests, artifact contract, validators, tests, preflight checks, shared reward interface | Requires code and docs changes but can be introduced incrementally without changing training objectives. |
| High | Golden tiny E2E pipeline, text-rendering error taxonomy, sharded dataset indexes | Requires stronger abstraction, representative fixtures, and careful validation against expensive real workflows. |

## Roadmap Implications

Suggested feature ordering for the next brownfield milestone:

1. **Reproducibility foundation:** environment manifest, command catalog, config inventory, artifact path contract.
2. **Run observability:** local manifests, lifecycle docs, output summaries, preflight checks.
3. **Safety net:** lightweight tests, fixtures, diagnostic renaming, dataset/artifact validators.
4. **Extensibility cleanup:** shared reward interface, shared trainer utilities only where tests protect behavior, extension guides.
5. **Research differentiation:** coverage reports, reward calibration, run diffing, thesis plot/table generation.

This order keeps existing experiments runnable while making future experiments easier to add and compare.

## Sources

- Project context: `/root/diffusion-text-tuner/.planning/PROJECT.md` — HIGH confidence; defines brownfield scope, current capabilities, active requirements, and explicit out-of-scope items.
- Codebase structure: `/root/diffusion-text-tuner/.planning/codebase/STRUCTURE.md` — HIGH confidence; identifies existing entry points, configs, artifact roots, and extension locations.
- Codebase concerns: `/root/diffusion-text-tuner/.planning/codebase/CONCERNS.md` — HIGH confidence; identifies missing dependency manifest, duplicated reward logic, path drift, fragile FLUX tensor contracts, and test gaps.
- PyTorch reproducibility documentation: `https://docs.pytorch.org/docs/2.11/notes/randomness.html` — HIGH confidence; last updated 2025-10-03; supports best-effort reproducibility controls and the warning that complete reproducibility is not guaranteed across releases/platforms/devices.
- Hugging Face Diffusers training overview: `https://huggingface.co/docs/diffusers/en/training/overview` — HIGH confidence; supports self-contained, easy-to-tweak, beginner-friendly, single-purpose training scripts as a useful design target for research training code.
- Hugging Face Accelerate experiment tracking documentation: `https://huggingface.co/docs/accelerate/en/usage_guides/tracking` — HIGH confidence; confirms Accelerate supports multiple trackers but also motivates local manifest-first tracking because external tracker integration can add complexity in multiprocessing environments.
- Hugging Face Datasets build/load documentation: `https://huggingface.co/docs/datasets/en/about_dataset_load` — HIGH confidence; supports dataset documentation, typed features, split organization, and integrity checks as expectations for dataset-like artifacts.

## Notable Gaps

- No direct empirical competitor survey was performed; recommendations are based on current project context plus official PyTorch/Hugging Face documentation.
- Context7 CLI fallback was attempted for Diffusers and Accelerate documentation but failed due an environment-local npm `ENOENT` issue (`/root/.cursor-server/bin/linux-x64/lib` missing). Official docs were used instead.
- Exact current package versions are unknown because the repository lacks a committed dependency manifest; environment recommendations should be validated against the working thesis machine/cluster before pinning.
- Feature priorities assume the immediate milestone is cleanup/reproducibility rather than adding a new ML method.
