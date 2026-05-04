# Phase 3 Research: Data Curriculum and Dataset Quality

**Phase:** 03 - Data Curriculum and Dataset Quality  
**Researched:** 2026-05-04  
**Discovery level:** Level 2 standard research, because Phase 3 designs new data-quality contracts across prompt generation, synthetic data, and training-selection artifacts.

## Sources Reviewed

- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/research/SUMMARY.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/CONCERNS.md`
- Phase 1 and Phase 2 `VERIFICATION.md`
- `docs/pipeline_inventory.md`
- `docs/runtime_contracts.md`
- `src/runtime/*`
- `src/prompt_pipeline/generate.py`, `config.py`, `text_generator.py`, `assembler.py`, `style_generator.py`
- `scripts/synth/build_dataset.py`, `filter_masked_dataset.py`, `word_sampler.py`, `make_fixture.py`
- `src/training/dataset.py`
- Runtime test patterns in `tests/test_runtime_artifacts.py`, `tests/test_runtime_docs.py`, and `tests/test_runtime_preflight.py`

## Findings

### Prompt curriculum

The prompt pipeline already contains the raw pieces needed for DATA-01: Cyrillic alphabets and rare characters, weighted content types, text tiers, case weights, style pools, and scene/style assembly. The missing contract is an explicit curriculum object/config that says which stages are generated, how many samples each stage contributes, which scripts/characters/content types each stage targets, and which prompt mode was used.

The current CLI accepts primitive flags (`--n`, `--seed`, `--no-llm`, `--model`, `--backend`, `--batch-size`, `--expand-scenes`) and initializes `TextGenerator`, `StyleGenerator`, `ScenePool`, and `Assembler` directly. Phase 3 should keep this runnable entry point while adding explicit config loading so simple/full/curriculum modes are data-driven rather than monkey-patched.

### Prompt dataset quality

Existing prompt JSONL records include `prompt`, `target_text`, `tier`, `content_type`, `scene_id`, `style`, `lang`, and `char_coverage`. These support CPU-safe validation for length, character set, rare-character coverage, duplicates, malformed JSON/records, content distribution, style distribution, and rough naturalness checks. No model/OCR work is needed for prompt validation.

Recommended output: JSON report plus optional schema metadata sidecar with counts, warnings/errors, duplicate examples, coverage by character/stage/content/style, and pass/fail thresholds from config.

### Dataset manifests/provenance

Phase 2 `src.runtime.manifests` records run provenance but does not yet define dataset manifests for prompts or synthetic datasets. Phase 3 needs file-backed dataset manifest helpers that record schema version, dataset kind, config path/hash/snapshot, seed strategy, git commit, source hashes, model IDs/revisions where relevant, filtering stats, output counts, and artifact paths. These manifests should be emitted next to generated prompt/synthetic/selection outputs under ignored runtime roots, and small test fixtures should stay under `tmp_path`.

### Synthetic quality inspection

`scripts.synth.build_dataset` and `filter_masked_dataset` provide raw material for DATA-05: masks, bboxes from meta annotations, contrast via raw images/masks, character coverage via text/index rows, resolution via metadata/shapes, and filter summaries. The current filter script handles length, mask fraction, bbox height, missing artifacts, and mask dilation, but it does not produce a comprehensive quality report, contact sheet, per-font/per-resolution distribution, or OCR-verification integration.

OCR should be explicit and opt-in. Default checks must stay CPU-safe by inspecting image/mask/meta/index files and accepting optional OCR result CSV/JSONL produced by a separate diagnostic or reward/scoring run.

### Materialized SFT/DPO selections

`SFTDataset` filters rows above a score threshold inside its constructor, and `DPODataset` constructs best-vs-worst pairs inside its constructor. That makes selections hard to version, inspect, compare, or tie back to manifests. Phase 3 should add materialization helpers and a CLI that write `selected_samples.jsonl` and `preference_pairs.jsonl` with schema metadata, selection mode, thresholds, score columns, source file hashes, counts, and per-record provenance.

### Source comparison

Phase 3 needs a comparison report for generated-image reward-filtered data vs synthetic masked-SFT data. This can be CPU-safe by comparing existing selected-sample/pair artifacts, prompt-quality reports, synthetic-quality reports, and manifests. The report should identify coverage overlaps/gaps, selection distributions, expected strengths, expected failure modes, and which source is intended to support SFT, DPO, or masked-SFT experiments.

## Recommended Architecture

- Add `src/data_quality/` for CPU-safe curriculum, validation, manifest, synthetic-quality, and comparison helpers.
- Keep `src.prompt_pipeline.generate` as the prompt generation CLI, but make it accept explicit config files and pass a resolved config into generation functions.
- Add thin scripts under `scripts/` for validation, inspection, selection materialization, and source comparison.
- Extend runtime path/artifact docs and validators only after the Phase 3 implementation contracts exist.
- Add tests under `tests/` using text/CSV/JSON fixtures and small generated PIL images in `tmp_path`; do not commit generated data.

## Risks and Mitigations

| Risk | Mitigation in plans |
|------|---------------------|
| Prompt distribution changes silently | Add deterministic config/curriculum tests and validation report tests before wiring generation. |
| Expensive OCR/model checks enter default tests | Keep OCR input as optional report input; default tests inspect local text/image metadata only. |
| Selection artifacts drift from dataset constructors | Materialize selections first and optionally teach dataset loaders to consume materialized artifacts while preserving existing threshold behavior. |
| Synthetic data quality reports load unsafe tensors or generated images from git | Use CPU-safe metadata/image inspection, `torch.load(..., weights_only=True)` for trusted local tensors only, and maintain git-safety docs. |
| Generated artifacts are staged accidentally | Keep outputs under `outputs/`, `runs/`, or generated data roots; tests use `tmp_path`; docs state non-committable boundaries. |

## Research Conclusion

Existing code provides the necessary data producers and runtime contracts. Phase 3 should not introduce new external services or data platforms. The implementation should add explicit config/curriculum contracts, CPU-safe quality reports, local dataset manifests, materialized selection artifacts, and comparison tooling while preserving current runnable flows.
