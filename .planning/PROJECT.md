# Diffusion Text Tuner

## What This Is

Diffusion Text Tuner is a thesis-driven research codebase for developing a diffusion-based training toolkit for multilingual text rendering. It focuses on prompt generation, synthetic Cyrillic/multilingual data preparation, FLUX.2 Klein image generation, reward scoring, and LoRA-based fine-tuning pipelines for improving rendered text quality in diffusion outputs.

The immediate goal is to turn the existing research repository into a more understandable, reproducible, and extensible project so new experiments, training pipelines, configs, and runs can be added without rediscovering how the system fits together.

## Core Value

Researchers can reproducibly create, run, compare, and extend diffusion fine-tuning experiments for multilingual text rendering.

## Requirements

### Validated

- ✓ Prompt generation pipeline exists for multilingual/Cyrillic text-rendering prompts — existing
- ✓ FLUX image generation pipeline saves generated images, latents, and prompt text embeddings — existing
- ✓ Reward scoring exists for generated images using VLM/OCR-oriented paths — existing
- ✓ SFT, DPO, and masked-SFT LoRA training entry points exist — existing
- ✓ Synthetic Cyrillic masked-SFT dataset tooling exists with SynthTIGER integration — existing
- ✓ Accelerate and SLURM launch patterns exist for local and cluster execution — existing
- ✓ Lightweight tensor-loss tests exist for masked flow-matching loss behavior — existing
- ✓ Codebase map documents current architecture, stack, structure, conventions, tests, and concerns — existing

### Active

- [ ] Add a reproducible dependency and environment definition for local, GPU, OCR, synthesis, and test workflows.
- [ ] Define standard commands for setup, tests, data preparation, generation, scoring, training, and evaluation.
- [ ] Clean up file structure with moderate, safe moves that make scripts, diagnostics, configs, and experiments easier to find.
- [ ] Introduce simple local run manifests that capture config snapshots, commands, metadata, outputs, and run notes.
- [ ] Organize experiment configs so new SFT, DPO, masked-SFT, reward, and evaluation variants can be added consistently.
- [ ] Add lightweight tests and fixtures for config parsing, dataset/collator behavior, prompt determinism, reward wrappers, and critical training math.
- [ ] Refactor obvious shared trainer/pipeline utilities without destabilizing existing training behavior.
- [ ] Improve documentation so the repo reads as a thesis toolkit rather than a collection of one-off scripts.

### Out of Scope

- Building a hosted web application or production service — the project target is a local/cluster ML research toolkit.
- Committing generated images, tensors, checkpoints, large datasets, or logs — these remain environment-specific artifacts.
- Major package rearchitecture as the first step — moderate cleanup is preferred so existing experiments remain runnable.
- Immediate MLflow or Weights & Biases integration — simple local manifests are the current tracking target.
- Replacing the core FLUX/Qwen/PaddleOCR technology choices without a concrete experimental reason — current work should make the existing stack reproducible first.

## Context

This is a brownfield ML research repository with existing prompt generation, image generation, scoring, training, synthetic data, evaluation, and cluster-launch code. The thesis topic is "Developing a Diffusion-based Training Toolkit for Multilingual Text Rendering," so project decisions should prioritize reproducibility, experiment comparability, and clear extension points for new multilingual text-rendering pipelines.

The codebase map in `.planning/codebase/` identifies the main current risks: no dependency manifest or lockfile, no standard tool configuration, limited formal tests, duplicated reward logic, expensive diagnostics named like tests, large trainer modules with many responsibilities, and inconsistent experiment/run organization. Existing generated artifacts live under ignored roots such as `outputs/`, `runs/`, and parts of `data/`, and should stay out of version control.

Expected future work includes refactoring and adding experiments/pipelines, launching new runs with new configs, and keeping enough project memory that the repository remains understandable between thesis iterations.

## Constraints

- **Thesis scope**: Changes should support the thesis toolkit narrative and make experiments explainable and reproducible.
- **Runtime**: CUDA-capable GPU execution is assumed for core generation/training paths; SLURM cluster execution should remain supported.
- **Stack**: Python 3.11, PyTorch, Diffusers, Transformers, Accelerate, PEFT, PaddleOCR/Qwen reward paths, and SynthTIGER are current project foundations.
- **Artifacts**: Generated images, tensors, checkpoints, logs, and large datasets must remain ignored unless intentionally added as tiny fixtures.
- **Refactor safety**: Prefer moderate cleanup and stable behavior over a major package reorganization that breaks existing runs.
- **Run tracking**: Use simple local manifests first; defer external tracking systems unless the need becomes concrete.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as a brownfield thesis toolkit | Existing code already implements core research flows; initialization should preserve and clarify them rather than restart from scratch. | — Pending |
| Prioritize reproducibility, structure, tests, configs, and pipeline readiness together | The next work needs the repo to be understandable before adding more experiments and runs. | — Pending |
| Use moderate cleanup rather than major reorganization | Existing experiments should remain runnable while structure improves. | — Pending |
| Track runs with simple local manifests first | Captures reproducibility metadata without adding external service complexity. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-04 after initialization*
