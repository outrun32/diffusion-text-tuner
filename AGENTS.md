# Diffusion Text Tuner Agent Guide

## Project Context

Diffusion Text Tuner is a brownfield thesis ML research toolkit for developing diffusion-based training methods for multilingual text rendering. The thesis topic is "Developing a Diffusion-based Training Toolkit for Multilingual Text Rendering."

The core value is reproducible creation, execution, comparison, diagnosis, and extension of diffusion fine-tuning experiments for multilingual text rendering.

## Planning Source of Truth

Read these before substantial work:

- `.planning/PROJECT.md` - project framing, validated capabilities, active scope, constraints, and decisions.
- `.planning/ROADMAP.md` - current 7-phase roadmap and requirement mapping.
- `.planning/STATE.md` - current phase, continuity notes, risks, and next action.
- `.planning/REQUIREMENTS.md` - v1 requirements and traceability.
- `.planning/codebase/` - architecture, structure, stack, testing, conventions, and concerns map.
- `.planning/research/SUMMARY.md` - research synthesis for reproducibility, data, training, reward, and evaluation priorities.

## Current Roadmap Position

Current phase: Phase 1 - Execution Surface and Pipeline Inventory.

Next command: `/gsd-plan-phase 1`.

Phase 1 should stabilize understanding and basic execution before deeper refactors:

- Inventory supported pipelines and separate them from experiments/diagnostics.
- Add reproducible install/tooling surfaces.
- Document safe CPU tests, smoke checks, and local/SLURM command variants.
- Keep expensive GPU/model/OCR diagnostics outside default test discovery.

## Engineering Priorities

- Preserve existing runnable research flows while improving structure.
- Prefer moderate, behavior-preserving refactors over big-bang package reorganization.
- Add tests and characterization before moving high-risk trainer, reward, dataset, or generation code.
- Keep generated images, tensors, checkpoints, logs, and large datasets out of git.
- Treat training loss and DPO accuracy as internal signals, not final evidence of Russian text rendering quality.
- Tie thesis plots/results back to exact runs, configs, rewards, seeds, and artifacts.

## Known High-Risk Areas

- DPO objective sign/beta scaling and winner/loser semantics need deterministic tests.
- Reward logic is duplicated and may drift across scoring, training, evaluation, and experiments.
- Product-score generation is underspecified and must be reproducible before relying on it.
- Synthetic masked-SFT improves reconstruction loss but does not by itself prove better prompt-following text rendering.
- Prompt/data quality, rare Cyrillic coverage, and reward calibration are central research risks.
- Hardcoded paths, unpinned model revisions, and hidden artifact contracts undermine reproducibility.

## Verification Style

Use adaptive verification:

- Always verify dependency/tooling, config/artifact contracts, objective math, reward semantics, trainer refactors, and evaluation outputs.
- Do not over-gate simple documentation or inventory-only changes.
- Keep default automated tests CPU-safe.
- Use explicit smoke or diagnostic commands for CUDA, model, OCR, integration, and SLURM checks.

## Git Discipline

- Commit planning artifacts and approved changes atomically.
- Do not revert or modify unrelated worktree changes.
- Do not commit generated artifacts unless they are intentionally tiny fixtures or documentation assets.
- Keep `.planning/ROADMAP.md`, `.planning/STATE.md`, and `.planning/REQUIREMENTS.md` synchronized after phase changes.
