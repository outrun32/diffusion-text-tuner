# Phase 3 Context: Data Curriculum and Dataset Quality

**Phase:** 03 - Data Curriculum and Dataset Quality  
**Prepared:** 2026-05-04  
**Mode:** Standard plan-phase, no optional flags

## Workflow Gate Results

1. **ROADMAP validation:** Phase 3 exists in `.planning/ROADMAP.md` and is not planned: `**Plans:** TBD`, progress `0/TBD`, no `03-*` phase directory existed before this planning run.
2. **Prior phases:** Phase 1 and Phase 2 verification reports both pass. Phase 3 can build on `docs/pipeline_inventory.md`, `docs/runtime_contracts.md`, and `src.runtime` helpers.
3. **SDK fallback:** `gsd-sdk` is unavailable in this workspace; plan validation is performed manually against required frontmatter, task XML fields, requirement coverage, source audit, and wave/file-conflict checks.
4. **Unrelated worktree:** The workspace contains pre-existing code/config/data changes outside `.planning/ROADMAP.md`, `.planning/STATE.md`, and this Phase 3 directory. Phase 3 planning must not stage or modify those unrelated changes.

## Locked User Scope

No separate Phase 3 `CONTEXT.md` from `/gsd-discuss-phase` exists, but the user supplied the following non-negotiable scope for this planning run:

- Implement DATA-01 through DATA-07.
- Preserve Phase 3 goal: users can create and assess multilingual text-rendering datasets with explicit curriculum, provenance, quality checks, and versioned training selections.
- Include prompt/data curriculum stages for Cyrillic and multilingual text rendering.
- Include prompt dataset validation for length, character set, rare-character coverage, duplicate rate, malformed outputs, content/style distribution, and naturalness/malformed checks.
- Replace prompt-generation monkey-patching with explicit config objects or config files for simple, full, curriculum, and extension modes.
- Add dataset manifests/provenance for prompts and synthetic data.
- Add synthetic dataset quality inspection: OCR verification, masks/bbox/contrast filters, character/font/resolution reports, and contact sheets.
- Add materialized selected SFT samples and DPO pair artifacts.
- Add comparison of generated-image reward-filtered data vs synthetic masked-SFT data.

## Existing Constraints To Preserve

- Default automated tests stay CPU-safe and do not load FLUX, Qwen, PaddleOCR, CUDA, vLLM, MLX, or SynthTIGER.
- Generated images, tensors, checkpoints, logs, large prompt datasets, and `data/synth_cyrillic/` outputs stay out of git.
- Use local file-backed manifests and runtime contracts before introducing external data-versioning or tracking services.
- Prefer focused modules and thin CLIs over adding more responsibilities to large trainer files.
- Continue using `src.runtime` path/artifact/manifest helpers and documented command surfaces from Phase 2.

## Phase Dependency Inputs

- `src.prompt_pipeline.generate` currently creates prompt JSONL records from hardcoded constants and CLI args.
- `src.prompt_pipeline.config` already defines Cyrillic characters, rare characters, content types, tier weights, case weights, and style pools.
- `scripts.synth.build_dataset` already renders/collates synthetic masked-SFT data and writes `index.csv`, `prompts.jsonl`, `shapes.csv`, `latents/`, and `text_embeds/`.
- `scripts.synth.filter_masked_dataset` already filters masked-SFT data by text length, mask fraction, bbox height, and missing artifacts, but it is not manifest-backed and does not produce full quality reports.
- `src.training.dataset.SFTDataset` and `DPODataset` currently select SFT samples and DPO pairs inside constructors rather than materializing versioned selection artifacts first.
- `src.runtime.paths` and `src.runtime.artifacts` reserve selected sample and preference pair paths but do not yet own Phase 3 quality/manifests/comparison contracts.

## Planning Decision Summary

- Create six executable plans across three waves.
- Wave 1 creates independent implementation surfaces for prompt curriculum, prompt/manifests validation, synthetic quality inspection, and materialized training selections.
- Wave 2 compares generated reward-filtered data with synthetic masked-SFT data using outputs from Wave 1.
- Wave 3 wires docs, Makefile aliases, README links, and runtime contract updates after the implementation contracts exist.
