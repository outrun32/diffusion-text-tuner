# Architecture Patterns

**Domain:** Reproducible diffusion-based training toolkit for multilingual text rendering  
**Project:** `diffusion-text-tuner`  
**Researched:** 2026-05-04  
**Overall confidence:** HIGH for brownfield structure and reproducibility patterns; MEDIUM for optional future config-framework migration.

## Executive Recommendation

Evolve this repository as a **file-backed ML research toolkit with thin CLI entry points, typed config schemas, explicit artifact contracts, and local run manifests**. Do not start with a major package reorganization. The current pipeline already has the right high-level stages — prompt generation, image generation, reward scoring, SFT/DPO/masked-SFT training, synthetic data, evaluation, and SLURM launchers — but the boundaries need to become more explicit so future experiments do not require rediscovering path conventions or copying trainer code.

The safest brownfield direction is:

1. Keep existing public commands runnable (`python -m scripts.generate_images`, `python -m scripts.score_images`, `accelerate launch -m src.training.*_trainer`).
2. Add a small shared runtime layer for **config loading, path resolution, seeding, run manifests, and artifact validation**.
3. Refactor large trainer and pipeline files incrementally behind the same CLIs.
4. Treat generated tensors/images/checkpoints/logs as ignored artifacts, but record their provenance in run manifests.
5. Add tests around artifact contracts and tensor-shape contracts before changing FLUX latent utilities or trainer math.

This matches the current thesis goal: make experiments reproducible and comparable without destabilizing already-working research flows.

## Recommended Architecture

```text
diffusion-text-tuner/
├── configs/                         # committed, human-readable experiment inputs
│   ├── sft.json                     # keep stable compatibility configs initially
│   ├── dpo.json
│   ├── masked_sft.json
│   ├── eval_suite.json
│   ├── accelerate/                  # execution topology only
│   ├── synth/                       # synthetic rendering configs
│   └── experiments/                 # add: named variants by stage/mode
│       ├── sft/
│       ├── dpo/
│       ├── masked_sft/
│       ├── generation/
│       └── scoring/
├── src/
│   ├── runtime/                     # add: cross-stage reproducibility layer
│   │   ├── config_io.py             # load dataclass + JSON override; write resolved config
│   │   ├── manifests.py             # create/update run manifest files
│   │   ├── paths.py                 # canonical artifact path helpers
│   │   ├── reproducibility.py       # seed Python/NumPy/PyTorch/DataLoader workers
│   │   └── validation.py            # preflight checks for files/models/artifact schemas
│   ├── prompt_pipeline/             # keep: prompt sampling/assembly logic
│   ├── generation/                  # add gradually: reusable FLUX generation implementation
│   │   └── pipeline.py              # called by scripts/generate_images.py
│   ├── scoring/                     # add gradually: reusable scoring orchestration
│   │   └── pipeline.py              # called by scripts/score_images.py
│   ├── rewards/                     # add gradually: canonical reward interfaces/classes
│   │   ├── base.py
│   │   ├── qwen.py
│   │   └── ocr.py
│   ├── training/
│   │   ├── config.py                # keep schemas, split later only if needed
│   │   ├── dataset.py               # file-backed datasets and collators
│   │   ├── flux2_utils.py           # FLUX latent/text geometry contract
│   │   ├── losses.py                # objective math
│   │   ├── sampling.py              # add: sample-image generation shared by trainers
│   │   ├── checkpointing.py         # add: save/resume helpers
│   │   ├── schedulers.py            # add: flow/sigma/LR helpers
│   │   ├── sft_trainer.py           # keep as mode orchestration
│   │   ├── dpo_trainer.py           # keep as mode orchestration
│   │   └── masked_sft_trainer.py    # keep as mode orchestration
│   └── evaluation/                  # evaluation CLIs import rewards/generation helpers
├── scripts/                         # thin CLI wrappers + cluster/synthesis utilities
│   ├── generate_images.py           # eventually delegates to src.generation.pipeline
│   ├── score_images.py              # eventually delegates to src.scoring.pipeline
│   ├── synth/                       # keep synthetic builder; extract shared pieces only as needed
│   └── cluster/                     # SLURM wrappers; no unique business logic
├── tests/                           # lightweight CPU/fixture tests only
│   ├── fixtures/
│   ├── test_config_io.py
│   ├── test_dataset_contracts.py
│   ├── test_flux2_shapes.py
│   ├── test_losses.py
│   └── test_run_manifest.py
├── outputs/                         # ignored heavy artifacts
├── runs/                            # ignored local run manifests/logs
└── docs/                            # thesis-facing docs; sanitized run summaries only
```

### Why this shape

- **Thin scripts preserve current commands.** The existing README and SLURM jobs call `scripts/*.py` and `src.training.*_trainer`; changing these too early would break known workflows.
- **`src/runtime/` avoids training-package coupling.** Run manifests, seeding, path helpers, and config snapshots are needed by generation, scoring, synthetic data, training, and evaluation. Keeping them in `src/training/` would make non-training stages depend on training internals.
- **`src/rewards/` prevents reward drift.** Current reward logic is duplicated across `src/training/rewards.py`, `scripts/score_images.py`, `src/evaluation/evaluate_rewards.py`, and `experiments/ocr_reward_tests/`. Canonical reward classes should live in one importable package; old modules can re-export for compatibility during migration.
- **Trainer files become orchestration layers, not utility dumpsters.** Diffusers' own training examples are intentionally self-contained, easy to tweak, and single-purpose. This repo should keep one trainer per method, but extract common sampling/checkpoint/scheduler/config behaviors so SFT, DPO, and masked-SFT do not diverge accidentally.
- **Artifacts remain filesystem-native.** A database or external experiment tracker is not necessary for the current thesis scope. Local manifests provide provenance without adding MLflow/W&B operational complexity.

## Component Boundaries

| Component | Responsibility | Owns | Must Not Own | Communicates With |
|-----------|----------------|------|--------------|-------------------|
| `configs/` | Declarative experiment, stage, hardware, and synthesis settings | Committed JSON/YAML config files and named variants | Generated run state or environment secrets | CLI loaders, `src.runtime.config_io` |
| `src.runtime` | Cross-stage reproducibility primitives | Config snapshots, run IDs, manifest schema, path helpers, seeding, preflight checks | Model-specific math, reward semantics, training loops | All scripts and trainers |
| `src.prompt_pipeline` | Prompt sampling and assembly | Text/style/scene sampling, LLM client wrapper, prompt JSONL schema | FLUX image generation, scoring, trainer data loading | Prompt-generation CLI, generation stage |
| `src.generation` | Reusable FLUX generation implementation | Loading FLUX pipeline, generating image variants, saving image/latent/embed artifacts | CLI parsing, scoring, trainer updates | `scripts.generate_images`, runtime manifests, training datasets |
| `src.scoring` | Batch scoring orchestration | Iterating images, resume/shard logic, writing score tables | Reward model internals duplicated per caller | `scripts.score_images`, `src.rewards`, runtime manifests |
| `src.rewards` | Canonical reward interfaces and implementations | Qwen yes-probability reward, OCR/CER reward, reward result schema | Batch CLI sharding, trainer objective math | scoring, evaluation, training if reward-in-loop returns |
| `src.training.config` | Typed training schemas | `SFTConfig`, `DPOConfig`, `MaskedSFTConfig`, LoRA config dataclasses | File reading policy once `config_io` exists | trainers, tests, config loader |
| `src.training.dataset` | File-backed dataset contracts | CSV/PT loading, filtering, collators, resolution buckets | Generating artifacts, scoring, model loading | trainers, tests, runtime validators |
| `src.training.flux2_utils` | FLUX latent/text tensor contract | Encode/decode helpers, packing/unpacking, position IDs, embedding helpers | CLI parsing or experiment branching | generation, synthesis, trainers, tests |
| `src.training.losses` | Objective math | Masked flow-matching loss, mask downsampling, future objective helpers | Data I/O, accelerator setup | trainers, tests |
| Trainer modules | Mode-specific orchestration | Model loading, dataset creation, accelerator preparation, train loop, mode-specific checkpoints/sampling hooks | Shared reward logic, generic config I/O, duplicated sampling utilities | runtime, datasets, losses, flux utils, Accelerate |
| `scripts/cluster` | Cluster launch topology | SLURM arrays, resource requests, environment activation, command wrappers | Different behavior from local CLIs | existing CLI entry points |
| `experiments/` | Disposable/manual research probes | Explicitly manual diagnostics and committed tiny assets | Formal tests, reusable pipeline logic, import-time model loading | docs, future promoted modules |

## Data, Config, and Run Flow

### End-to-end artifact flow

```text
Committed source/configs
    │
    ├─ prompt generation
    │     configs/experiments/generation/*.json
    │     └─> data/*.jsonl                         # prompt records; large files ignored
    │
    ├─ FLUX image generation
    │     scripts.generate_images + runtime manifest
    │     └─> outputs/generated/
    │           ├── images/{prompt_id}/v{n}.png
    │           ├── latents/{prompt_id}/v{n}.pt
    │           ├── text_embeds/{prompt_id}.pt
    │           └── generation_manifest.jsonl       # per-sample artifact index
    │
    ├─ reward scoring
    │     scripts.score_images + runtime manifest
    │     └─> outputs/generated/scores.csv
    │
    ├─ SFT/DPO training
    │     accelerate launch -m src.training.*_trainer --config configs/...json
    │     └─> outputs/{sft,dpo}/
    │           ├── checkpoints/
    │           ├── samples/
    │           └── metrics.jsonl
    │
    ├─ synthetic masked-SFT dataset
    │     scripts.synth.build_dataset + configs/synth/*.yaml
    │     └─> data/synth_cyrillic/masked_sft/
    │           ├── latents/{sample_id}.pt
    │           ├── text_embeds/{sample_id}.pt
    │           ├── shapes.csv
    │           └── dataset_manifest.jsonl
    │
    └─ masked-SFT training/evaluation
          └─> outputs/masked_sft/{checkpoints,samples,metrics.jsonl}
```

### Config flow

Use the existing dataclass + JSON pattern first. Do **not** introduce Hydra in the first cleanup phase. Hydra is a good future option for hierarchical config composition and multiruns, but it changes config semantics and default output directory behavior; in this brownfield repo that should come after stable manifests, tests, and command wrappers.

Recommended immediate flow:

```text
configs/*.json or configs/experiments/<stage>/<name>.json
    │
    ▼
src.runtime.config_io.load_config(ConfigClass, path)
    │
    ├─ instantiate dataclass defaults
    ├─ apply JSON overrides recursively, including nested LoRA configs
    ├─ validate unknown keys fail fast by default
    ├─ resolve paths relative to repo root unless explicitly absolute
    └─ return immutable/resolved config object
    │
    ▼
src.runtime.manifests.create_run(...)
    │
    ├─ runs/<YYYYMMDD-HHMMSS>_<stage>_<experiment_name>_<short_id>/
    │   ├── manifest.json
    │   ├── config.resolved.json
    │   ├── command.txt
    │   ├── git.json
    │   ├── environment.txt
    │   ├── artifacts.json
    │   ├── metrics.jsonl
    │   └── notes.md
    └─ output_dir remains under outputs/<stage>/<experiment_name or run_id>/
```

Minimum manifest fields:

| Field | Purpose |
|-------|---------|
| `run_id` | Stable local identifier used in output paths and logs |
| `stage` | `prompt_generation`, `generation`, `scoring`, `sft`, `dpo`, `masked_sft`, `evaluation`, `synth_build` |
| `experiment_name` | Human-readable config name from dataclass/config |
| `command` | Exact shell command used, including `accelerate launch` flags |
| `config_path` | Source config file |
| `resolved_config_path` | Snapshot of effective config after defaults/overrides |
| `accelerate_config_path` | Hardware/process topology for trainer runs |
| `git_commit`, `git_dirty` | Source provenance; warn on dirty tree but do not block research runs by default |
| `python`, `torch`, `cuda`, `diffusers`, `transformers`, `accelerate`, `peft` | Dependency/runtime provenance |
| `seed` | Primary run seed |
| `artifact_inputs` | Prompt JSONL, latents dir, text embeds dir, scores CSV, dataset root, checkpoint inputs |
| `artifact_outputs` | Images, latents, embeddings, scores, checkpoints, samples, metrics paths |
| `status` | `created`, `running`, `completed`, `failed`, `interrupted` |
| `started_at`, `finished_at` | Run timing |
| `notes` | Freeform thesis notes, caveats, and links |

### Local vs cluster parity

Every stage should have one canonical Python entry point and two launch surfaces:

```text
Local:   python -m scripts.generate_images --config configs/experiments/generation/baseline.json
Cluster: sbatch scripts/cluster/generate_images.sbatch CONFIG=configs/experiments/generation/baseline.json

Local:   accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/experiments/sft/baseline.json
Cluster: sbatch scripts/cluster/sft.sbatch CONFIG=configs/experiments/sft/baseline.json ACCELERATE_CONFIG=configs/accelerate/multi_gpu.yaml
```

Cluster scripts must not encode hidden experiment choices. They should set resources and pass through config paths, shard IDs, and output roots.

## Patterns to Follow

### Pattern 1: Stable CLI Shell, Importable Implementation

**What:** Keep the current commands but move reusable behavior into importable modules when touching a stage.

**When:** Any time `scripts/generate_images.py`, `scripts/score_images.py`, or a trainer grows new reusable logic.

**Example target shape:**

```python
# scripts/generate_images.py
from src.generation.pipeline import GenerationConfig, run_generation
from src.runtime.config_io import load_config
from src.runtime.manifests import run_context


def main() -> None:
    args = parse_args()
    cfg = load_config(GenerationConfig, args.config, cli_overrides=args.overrides)
    with run_context(stage="generation", cfg=cfg, command_args=args) as run:
        run_generation(cfg, run=run)


if __name__ == "__main__":
    main()
```

**Why:** Scripts remain runnable; logic becomes testable without importing CUDA-heavy models at module import time.

### Pattern 2: Artifact Contract Validation Before Expensive Work

**What:** Validate expected files, CSV columns, `.pt` tensor keys, tensor shapes, and matching sample IDs before launching GPU-heavy loops.

**When:** At the start of generation, scoring, dataset build, and every trainer.

**Example contracts:**

| Artifact | Required contract |
|----------|-------------------|
| Prompt JSONL | One JSON object per line; stable `id`; prompt text; target text if scoring/training needs it |
| SFT latent | `latents/{prompt_id}/v{version}.pt` with tensor key `latent` |
| Text embedding | `text_embeds/{prompt_id}.pt` with tensor key `prompt_embeds` |
| Scores CSV | `id`, `version`, `score`, and optionally `target_text`, `reward_method`, `model_id` |
| Masked-SFT latent | `latents/{sample_id}.pt` with `latent` and `mask_lat` |
| Shapes CSV | Complete `id,H,W` rows for every masked-SFT sample; no silent fallback in large runs |
| Checkpoint | LoRA adapter path plus config snapshot identifying base model and target modules |

**Why:** Most failures in this repo are currently path/schema/shape mismatches that surface after model loading. Preflight checks make failures fast and cheap.

### Pattern 3: Reproducibility Layer Is Centralized

**What:** Put seed and deterministic-mode handling in `src.runtime.reproducibility`, then call it from every CLI and trainer.

**Example:**

```python
def configure_reproducibility(seed: int, deterministic: bool = False) -> torch.Generator:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**Notes:** PyTorch explicitly documents that complete reproducibility is not guaranteed across releases, commits, platforms, or CPU/GPU execution. This repo should therefore record exact environment metadata and treat deterministic mode as an opt-in debugging/regression setting, not a free performance default.

### Pattern 4: One Reward Interface, Many Callers

**What:** Define a canonical reward result and reward model interface in `src.rewards`.

**Example:**

```python
@dataclass(frozen=True)
class RewardResult:
    score: float
    method: str
    target_text: str | None
    raw: dict[str, object]


class RewardModel(Protocol):
    method: str

    def score_image(self, image_path: Path, prompt: str | None, target_text: str | None) -> RewardResult:
        ...
```

**Why:** Generation scoring, evaluation, DPO pair creation, and future reward-in-the-loop methods must agree on what a reward score means. Duplicated prompts/token logic will corrupt comparability.

### Pattern 5: Trainers Own Objectives; Shared Modules Own Plumbing

**What:** Keep SFT, DPO, and masked-SFT as separate entry points because their objectives and data semantics differ. Extract only shared plumbing.

**Extract first:**

| New module | Move from | Why |
|------------|-----------|-----|
| `src.training.sampling` | SFT/DPO/masked-SFT sampling helpers | Keep eval sample rendering consistent |
| `src.training.checkpointing` | trainer save/resume snippets | Prevent incompatible checkpoint layouts |
| `src.training.schedulers` | `compute_sigma`, LR schedule helpers, flow timestep helpers | Test math once; avoid DPO/SFT drift |
| `src.runtime.config_io` | trainer `main()` config load blocks | Validate nested config overrides consistently |
| `src.runtime.manifests` | ad hoc logging/output directory setup | Ensure every run records provenance |

**Do not extract yet:** mode-specific objective code and special cases until tests exist. DPO sign/scaling and masked latent geometry are fragile and should remain close to their tests during refactor.

### Pattern 6: Promote Experiments Through Stages

**What:** One-off research code starts under `experiments/`, then moves to `src/` only after it becomes reusable.

```text
experiments/ocr_reward_tests/check_qwen_yes_prob.py   # manual probe
    ↓ once reused by scoring/evaluation/training
src/rewards/qwen.py                                   # canonical implementation
    ↓ caller-specific orchestration
src/scoring/pipeline.py + src/evaluation/evaluate_rewards.py
```

**Rule:** Files named `test_*.py` should be lightweight automated tests under `tests/`, not CUDA/model-loading diagnostics.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Big-Bang Package Reorganization

**What:** Moving all scripts/trainers/configs into a new architecture before tests and manifests exist.  
**Why bad:** Existing local/SLURM commands can break, and behavior changes become indistinguishable from file moves.  
**Instead:** Add runtime helpers first, preserve CLIs, and refactor one stage at a time behind compatibility wrappers.

### Anti-Pattern 2: Hidden Path Conventions in Configs and Scripts

**What:** A config says `outputs/latents` while the README uses `outputs/generated/latents`, or diagnostics assume `outputs/text_embeds/000000.pt`.  
**Why bad:** Runs fail late and are hard to reproduce.  
**Instead:** Use `src.runtime.paths` and preflight validators; every run manifest should list resolved input/output paths.

### Anti-Pattern 3: Recomputing Text Embeddings Inside Trainers

**What:** Trainer loops load FLUX text encoders or recompute prompt embeddings per training sample.  
**Why bad:** Wastes VRAM/time and undermines the current precomputed-embedding design.  
**Instead:** Keep generation/synthesis responsible for writing `text_embeds/*.pt`; trainers load them through datasets.

### Anti-Pattern 4: Config Framework Migration Before Stabilizing Contracts

**What:** Introducing Hydra/OmegaConf immediately to solve config sprawl.  
**Why bad:** It adds new output directory, override, and composition semantics while the repo still lacks dependency pins, config tests, and manifest contracts.  
**Instead:** First implement strict dataclass JSON loading and run manifests. Revisit Hydra after config variants and sweeps become the bottleneck.

### Anti-Pattern 5: Reward Logic Forks

**What:** Changing Qwen/PaddleOCR reward logic separately in scoring, evaluation, and training code.  
**Why bad:** SFT filtering and DPO pair selection become incomparable across experiments.  
**Instead:** Centralize reward implementations and record `reward_method`, `reward_model_id`, and scorer config in every `scores.csv`/manifest.

## Suggested Build Order

### Phase 1: Stabilize Execution Surface and Environment

**Goal:** Make the repo installable and commands discoverable without changing ML behavior.

Build:
- `pyproject.toml` or equivalent dependency manifest with extras for `train`, `ocr`, `synth`, `vllm`, `mlx`, and `test`.
- Standard commands in README/Makefile/task runner for setup, tests, generation, scoring, SFT, DPO, masked-SFT, synth build, and evaluation.
- Basic pytest config that only discovers `tests/test_*.py`.
- Rename manual CUDA diagnostics from `test_*.py` to `check_*.py`/`diagnose_*.py` and add `main()` guards.

Why first:
- Every later refactor depends on being able to run the same commands repeatedly.
- Prevents accidental execution of expensive diagnostics during test discovery.

### Phase 2: Add Runtime Manifest and Strict Config Loading

**Goal:** Every run records what happened and where artifacts went.

Build:
- `src/runtime/config_io.py` with strict dataclass JSON override handling.
- `src/runtime/manifests.py` with `run_context(...)` for lifecycle updates.
- `src/runtime/paths.py` for canonical artifact path conventions.
- `src/runtime/reproducibility.py` for seed setup and DataLoader worker seeding.
- Tests for config overrides, unknown keys, manifest creation, and path resolution.

Why second:
- The project requirement explicitly calls for local run manifests.
- Manifests also make later behavior-preserving refactors auditable.

### Phase 3: Validate Artifact Contracts With Tiny Fixtures

**Goal:** Catch path/schema/shape mistakes before GPU jobs.

Build:
- `tests/fixtures/` containing tiny synthetic `.pt`, `.csv`, `.jsonl` artifacts.
- Dataset/collator tests for SFT, DPO, and masked-SFT.
- FLUX latent pack/unpack/text-ID shape tests that avoid loading full models.
- Preflight validators for scores CSV, latent dirs, text embedding dirs, `shapes.csv`, and checkpoint inputs.

Why third:
- The known fragile areas are latent geometry, DPO math, prompt determinism, reward wrappers, and dataset loading.
- Tests should guard the contracts before code is moved.

### Phase 4: Extract Shared Trainer Plumbing

**Goal:** Reduce copy/paste without changing objectives.

Build:
- `src.training.sampling` for sample/eval image generation helpers.
- `src.training.checkpointing` for LoRA/checkpoint save/resume helpers.
- `src.training.schedulers` for `compute_sigma`, timestep helpers, and LR schedule helpers.
- Trainer updates to use runtime config/manifests and shared helpers.
- Unit tests for DPO beta/log-ratio math and scheduler helpers before moving them.

Why fourth:
- Trainer modules are large and cross-import helpers from each other, but moving objective code too early is risky.
- Shared plumbing is easier to test and less likely to alter training semantics.

### Phase 5: Centralize Rewards and Scoring

**Goal:** Ensure scoring, evaluation, and training agree on reward semantics.

Build:
- `src.rewards` package with Qwen and OCR reward implementations.
- `src.scoring.pipeline` used by `scripts.score_images`.
- Evaluation imports shared rewards instead of duplicating logic.
- Score CSV schema versioning: include method/model/config metadata where practical.
- Lightweight tests around reward result parsing using mocks/fakes, not real VLM/OCR loads.

Why fifth:
- Reward drift can silently corrupt SFT/DPO data selection.
- Real Qwen/PaddleOCR tests are expensive, so first build mockable interfaces.

### Phase 6: Split Generation and Synthetic Pipelines Behind Existing CLIs

**Goal:** Make data-generation stages importable and testable.

Build:
- `src.generation.pipeline` called by `scripts.generate_images`.
- Generation artifact index/manifest writing per prompt/version.
- Synthetic dataset manifest/index validation in `scripts.synth.build_dataset`.
- Local/SLURM parity updates so cluster scripts pass config paths instead of hardcoding choices.

Why sixth:
- Generation is GPU-heavy and artifact-rich; by this point manifests, path helpers, and validators already exist.

### Phase 7: Optional Config Composition/Sweeps

**Goal:** Support many named variants and sweeps if JSON variants become painful.

Decision point:
- If experiments remain mostly single-run named JSON configs, keep strict dataclass JSON.
- If the thesis work needs many cross-product sweeps across model/path/hardware/objective settings, introduce Hydra/OmegaConf deliberately.

Hydra migration prerequisites:
- Config tests exist.
- Run manifests already snapshot resolved config.
- Existing JSON configs have documented equivalents.
- Output directory behavior is explicitly configured so it does not conflict with `outputs/` and `runs/`.

## Scalability Considerations

| Concern | At current thesis scale | At larger local/SLURM scale | At publish/share scale |
|---------|-------------------------|-----------------------------|------------------------|
| Artifact indexing | CSV/JSONL plus directory paths are enough | Require complete per-stage manifest/index files; avoid listing huge directories repeatedly | Publish curated datasets/checkpoints to Hugging Face or object storage with checksums |
| Run tracking | Local `runs/<run_id>/manifest.json` | Add metrics JSONL and command/environment snapshots for every shard | Export sanitized run summaries to `docs/` or thesis appendix; defer W&B/MLflow unless collaboration requires it |
| Config variants | Root `configs/*.json` plus `configs/experiments/*` | Strict schema validation and naming conventions | Consider Hydra after tests/manifests if multirun sweeps dominate |
| GPU memory | Precomputed embeddings, bf16, LoRA, gradient checkpointing | Separate low-memory configs; preflight model/component loading; avoid reward model in training process unless necessary | Document exact hardware/dependency matrix and model access requirements |
| Distributed execution | Existing Accelerate configs and SLURM scripts | Use Accelerate process-aware APIs for save/log/gather and keep one process writing manifests where needed | Package reproducible launch recipes per environment |
| Determinism | Seed Python/NumPy/PyTorch and log versions | Optional deterministic mode for regression/debug runs; expect slower performance | Publish seeds, resolved configs, exact dependency versions, and note PyTorch nondeterminism limits |
| Tests | CPU-only fixture tests for contracts and math | Add smoke tests that validate configs without model downloads | Separate expensive GPU smoke/manual checks from CI tests |

## Brownfield Migration Map

| Current issue | Target architecture response | First safe change |
|---------------|------------------------------|-------------------|
| No dependency manifest | Phase 1 environment definition | Add `pyproject.toml`/lock guidance and command docs |
| Large trainer modules | Trainers as orchestration; shared modules for plumbing | Extract config loading and manifest wiring before moving math |
| Duplicated reward logic | `src.rewards` canonical package | Define interface and re-export from `src.training.rewards` for compatibility |
| Expensive diagnostics named tests | `tests/` only for lightweight automated tests | Rename diagnostics and add `pytest` discovery config |
| Inconsistent output paths | `src.runtime.paths` and manifest artifacts | Define canonical `outputs/generated/{images,latents,text_embeds}` and update config examples |
| Sparse run provenance | `runs/<run_id>/manifest.json` | Add manifest context manager and snapshot resolved config/command/env |
| Fragile tensor geometry | Contract tests before utility changes | Add tests for latent/mask/text embedding shapes with tiny tensors |
| SLURM/local drift | One Python entry point per stage; cluster wrappers pass through configs | Update sbatch templates to accept `CONFIG`/`ACCELERATE_CONFIG` variables |

## Source Notes and Confidence

| Finding | Confidence | Source |
|---------|------------|--------|
| PyTorch cannot guarantee full reproducibility across releases/platforms; seed Python/NumPy/PyTorch and optionally deterministic algorithms/DataLoader worker seeds | HIGH | PyTorch reproducibility docs, last updated 2025-10-03: https://docs.pytorch.org/docs/2.11/notes/randomness.html |
| Accelerate is appropriate for distributed/mixed precision training and provides process-aware prepare/save/gather/checkpoint APIs | HIGH | Hugging Face Accelerate `Accelerator` docs v1.13.0: https://huggingface.co/docs/accelerate/package_reference/accelerator |
| Diffusers training examples favor self-contained, easy-to-tweak, single-purpose scripts; this supports separate trainer entry points with shared utilities extracted only where helpful | HIGH | Hugging Face Diffusers training overview: https://huggingface.co/docs/diffusers/training/overview |
| Hydra supports hierarchical config composition, command-line overrides, and multirun sweeps; useful later but not first in this brownfield repo | MEDIUM | Hydra docs, last updated 2025-10-27: https://hydra.cc/docs/intro/ |
| Current repo's primary risks are dependency/config gaps, large trainers, reward duplication, expensive diagnostics named tests, latent geometry fragility, and path drift | HIGH | `.planning/PROJECT.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`, `.planning/codebase/CONCERNS.md`, `README.md`, `src/training/config.py`, `src/training/dataset.py` |

## Notable Gaps / Follow-up Research

- **Exact dependency pins are unknown.** Architecture assumes a future dependency manifest will capture known-good versions for Torch/CUDA, Diffusers, Transformers, Accelerate, PEFT, PaddleOCR, vLLM, MLX, and SynthTIGER.
- **Current trainer import-time behavior was not exhaustively audited.** The target architecture assumes expensive model loading remains inside command/training functions; verify while refactoring.
- **No recommendation yet on Hydra adoption timing beyond “not first.”** Reassess after strict JSON configs and manifests are implemented.
- **Manifest schema should be finalized during implementation.** The fields above are sufficient for roadmap planning, but implementation should include tests and sample manifests.
- **Artifact schema versioning needs a concrete version field.** Add this when writing validators so old generated artifacts can be detected rather than silently accepted.
