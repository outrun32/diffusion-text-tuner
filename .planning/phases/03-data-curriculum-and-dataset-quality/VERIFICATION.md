---
phase: 03-data-curriculum-and-dataset-quality
verified: 2026-05-04T16:15:28Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "Synthetic dataset manifests now accept and record build config, seed strategy, model metadata, template/runner metadata, and word/font/scene/background source hashes."
  gaps_remaining: []
  regressions: []
---

# Phase 3: Data Curriculum and Dataset Quality Verification Report

**Phase Goal:** Users can create and assess multilingual text-rendering datasets with explicit curriculum, provenance, quality checks, and versioned training selections.
**Verified:** 2026-05-04T16:15:28Z
**Status:** passed / PASS
**Re-verification:** Yes — after synthetic provenance gap closure

## Goal Achievement

Phase 3 now satisfies the roadmap goal and DATA-01 through DATA-07. Re-verification focused on the previously blocked DATA-04 synthetic provenance gap, then performed quick regression checks on the previously verified Phase 3 truths. The latest implementation provides user-facing synthetic manifest flags, passes those values into `create_dataset_manifest`, tests the manifest contents, and documents the provenance workflow.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can choose explicit prompt/data curriculum stages for Cyrillic and multilingual rendering, from simple letters/words through harder punctuation, multiline, style, and scene cases. | ✓ VERIFIED | `configs/prompts/curriculum.json` and `src/data_quality/curriculum.py` expose the required families (`single_letters`, short words/phrases, digits, punctuation, mixed case, `multiline`, `style`, `scene`); grep confirmed config loader and stage-family support. |
| 2 | User can generate prompt datasets through explicit configs rather than monkey-patching and validate length, character coverage, duplicates, naturalness, malformed outputs, and content/style distributions. | ✓ VERIFIED | `src/prompt_pipeline/generate.py` wires `--config` through `load_prompt_generation_config` and tags records with `prompt_mode`, `curriculum_stage`, and `curriculum_family`; `src/data_quality/prompt_validation.py` implements malformed JSONL, required-field, length/script, rare-character, duplicate, naturalness, content-type, and style checks. |
| 3 | User can generate dataset manifests that record config, seeds, git commit, source hashes, model IDs, filtering stats, output counts, and relevant provenance for prompts and synthetic data. | ✓ VERIFIED | Previous blocker closed. `scripts/inspect_synthetic_dataset.py` now accepts `--config`, `--seed`, `--template`, `--runner`, `--model-id`, `--model-revision`, `--word-source`, `--font-source`, `--scene-source`, and `--background-source`; manifest creation passes `config_path`, config snapshot, seed strategy, source paths, filtering stats, output counts, model metadata, and template/runner metadata into `create_dataset_manifest`. `tests/test_synthetic_quality.py` asserts config path, seed, model metadata, template/runner metadata, and hashes for word/font/scene/background sources. |
| 4 | User can inspect synthetic dataset quality using OCR verification, mask/bbox/contrast filters, per-character and per-font coverage, resolution mix, and contact sheets. | ✓ VERIFIED | `src/data_quality/synthetic_quality.py` computes sample/missing-file counts, mask area, bbox height/area, contrast, character/font coverage, resolution distribution, threshold rejections, optional OCR summary, and PIL contact sheets; `scripts/inspect_synthetic_dataset.py` exposes report, manifest, OCR, threshold, and contact-sheet options without importing OCR/model stacks. |
| 5 | User can materialize selected SFT samples and DPO preference pairs as versioned artifacts and compare generated-image reward-filtered data against synthetic masked-SFT data. | ✓ VERIFIED | `src/training/selection.py` writes schema-versioned selected samples and DPO preference pairs with source score hashes, thresholds, counts, and strict winner-over-loser semantics; `src/data_quality/source_comparison.py` compares generated and synthetic evidence with counts, rare-character coverage, score/margin summaries, synthetic health, provenance evidence, and expected help/failure sections. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/data_quality/curriculum.py` | Prompt curriculum config models, validation, and stage expansion | ✓ VERIFIED | Contains `PromptGenerationConfig`, `CurriculumStage`, `load_prompt_generation_config`, deterministic allocation, and required curriculum families. |
| `configs/prompts/{simple,full,curriculum}.json` | Explicit prompt-generation configs | ✓ VERIFIED | Present and covered by full test suite; curriculum config exposes DATA-01 stages. |
| `src/prompt_pipeline/generate.py` | Config-driven prompt generation CLI | ✓ VERIFIED | `--config` loads explicit configs and records curriculum provenance while preserving legacy flags. |
| `src/data_quality/prompt_validation.py` | Prompt JSONL quality report | ✓ VERIFIED | Substantive aggregate validator for DATA-02 dimensions. |
| `src/data_quality/manifests.py` | Dataset manifest creation/loading and source hashing | ✓ VERIFIED | Supports `dataset-manifest/v1`, config snapshots/hashes, seed strategy, git state, safe source hashing, filtering stats, output counts, model metadata, and manifest loading validation. |
| `scripts/validate_prompt_dataset.py` | Prompt validation/report/manifest CLI | ✓ VERIFIED | Prompt-side manifest path supplies config and validation stats to manifest helper. |
| `src/data_quality/synthetic_quality.py` | Synthetic quality inspection and contact sheets | ✓ VERIFIED | CPU-safe PIL/CSV/JSON implementation with OCR handoff only through precomputed files. |
| `scripts/inspect_synthetic_dataset.py` | Synthetic report/manifest/contact sheet CLI | ✓ VERIFIED | Previous partial artifact is now complete: provenance flags are present and wired to manifest creation. |
| `tests/test_synthetic_quality.py` | Regression tests for synthetic quality/provenance | ✓ VERIFIED | Test asserts report/manifest/contact sheet outputs and synthetic provenance fields/source hashes. |
| `docs/synthetic_quality.md` | Synthetic quality and provenance docs | ✓ VERIFIED | Documents required layout, report fields, optional OCR, contact sheets, `dataset-manifest/v1`, provenance flags, and artifact safety. |
| `src/training/selection.py` / `scripts/materialize_training_data.py` | Versioned SFT/DPO selection artifacts | ✓ VERIFIED | Selection and CLI paths remain substantive and covered by full test suite. |
| `src/data_quality/source_comparison.py` / `scripts/compare_data_sources.py` | Generated-vs-synthetic comparison reports | ✓ VERIFIED | Evidence-aware comparison implementation and CLI remain wired. |
| `docs/commands.md`, `README.md`, `Makefile`, `docs/runtime_contracts.md` | Phase 3 command/runtime discovery | ✓ VERIFIED | Phase 3 command, runtime contract, and Makefile surfaces exist; detailed synthetic provenance flags are documented in `docs/synthetic_quality.md`. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `src/prompt_pipeline/generate.py` | `src/data_quality/curriculum.py` | `load_prompt_generation_config` | ✓ WIRED | Import and CLI call verified by grep and tests. |
| `scripts/validate_prompt_dataset.py` | `src/data_quality/prompt_validation.py` | `validate_prompt_dataset` | ✓ WIRED | Prompt validation CLI remains connected to validator. |
| `scripts/validate_prompt_dataset.py` | `src/data_quality/manifests.py` | `create_dataset_manifest` | ✓ WIRED | Prompt manifest generation supplies config/seed/source/filter/count provenance. |
| `scripts/inspect_synthetic_dataset.py` | `src/data_quality/synthetic_quality.py` | `inspect_synthetic_dataset`, `create_synthetic_contact_sheet` | ✓ WIRED | CLI creates report/contact sheet from synthetic quality report. |
| `scripts/inspect_synthetic_dataset.py` | `src/data_quality/manifests.py` | `create_dataset_manifest` | ✓ WIRED | Previous partial link fixed: call now passes `config_path`, config snapshot, seed strategy, source paths including word/font/scene/background sources, filtering stats, output counts, model metadata, and template/runner metadata. |
| `src/data_quality/manifests.py` | `src/runtime/reproducibility.py` | `collect_git_state`, `collect_model_revisions` | ✓ WIRED | Manifest helper records git state and derives model revisions from config snapshots, then merges explicit synthetic model metadata. |
| `scripts/materialize_training_data.py` | `src/training/selection.py` | `materialize_sft_samples`, `materialize_dpo_pairs` | ✓ WIRED | CLI and implementation remain connected. |
| `scripts/compare_data_sources.py` | `src/data_quality/source_comparison.py` | `compare_data_sources` | ✓ WIRED | CLI and implementation remain connected. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `generate.py` | prompt records | `PromptGenerationConfig` + generator constants/backends | Yes; config drives count/output/seed/backend/stages and record provenance | ✓ FLOWING |
| `validate_prompt_dataset.py` | prompt quality report/manifest | JSONL input + thresholds + config | Yes; line-by-line parsing feeds aggregate report and prompt manifest | ✓ FLOWING |
| `inspect_synthetic_dataset.py` | synthetic quality report/contact sheet | `index.csv`, `prompts.jsonl`, `shapes.csv`, images/masks/meta, optional OCR | Yes; report metrics flow into JSON, contact sheet, filtering stats, and output counts | ✓ FLOWING |
| `inspect_synthetic_dataset.py` | synthetic manifest provenance | CLI `--config/--seed/--model-*` and source path flags | Yes; values flow through `_config_snapshot`, `_seed_strategy`, `_model_metadata`, `_source_paths`, and `create_dataset_manifest` into persisted manifest JSON | ✓ FLOWING |
| `src/data_quality/manifests.py` | source hashes/config hash/git/models | Safe text source files, config file, runtime git/model helpers | Yes; safe source files hash deterministically; binary/generated artifacts are referenced rather than read | ✓ FLOWING |
| `materialize_training_data.py` | selected samples / preference pairs | `scores.csv` | Yes; parsed scores are filtered into JSONL artifacts and summary manifests | ✓ FLOWING |
| `compare_data_sources.py` | comparison report | Phase 3 JSON/JSONL reports/manifests/selections | Yes; present evidence is aggregated and missing optional evidence remains explicit | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Synthetic provenance regression tests pass | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_synthetic_quality.py -q` | `5 passed in 1.51s` | ✓ PASS |
| Full CPU-safe test suite passes | `PATH="/root/.local/bin:$PATH" uv run pytest -q` | `120 passed in 4.04s` | ✓ PASS |
| Synthetic inspector exposes provenance flags | `PATH="/root/.local/bin:$PATH" uv run python scripts/inspect_synthetic_dataset.py --help` | Help output includes `--config`, `--seed`, `--template`, `--runner`, `--model-id`, `--model-revision`, `--word-source`, `--font-source`, `--scene-source`, and `--background-source` | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|---|---|---|---|---|
| DATA-01 | 03-01, 03-06 | Explicit prompt/data curriculum stages | ✓ SATISFIED | Curriculum contracts and configs cover Cyrillic/multilingual stage families from letters/words through multiline/style/scene. |
| DATA-02 | 03-02, 03-06 | Prompt dataset validation dimensions | ✓ SATISFIED | Prompt validation covers length, script/character/rare coverage, duplicates, malformed/naturalness checks, and content/style distributions. |
| DATA-03 | 03-01, 03-06 | Replace monkey-patching with explicit config objects/files | ✓ SATISFIED | Prompt generation supports `--config` and explicit `PromptGenerationConfig` while preserving legacy invocation. |
| DATA-04 | 03-02, 03-03, 03-04, 03-06 | Dataset manifests with config, seed strategy, git commit, source word/scene/font/background hashes, model IDs, filtering stats, and output counts | ✓ SATISFIED | Prompt manifests were already complete; synthetic manifests now expose and persist config path/snapshot, seed, model metadata, template/runner metadata, source hashes for word/font/scene/background files, filtering stats, and output counts. |
| DATA-05 | 03-03, 03-06 | Synthetic quality inspection and reports | ✓ SATISFIED | Synthetic quality module and CLI implement OCR handoff, masks, bbox, contrast, character/font coverage, resolution mix, threshold filters, and contact sheets. |
| DATA-06 | 03-04, 03-06 | Materialized SFT samples and DPO preference pairs | ✓ SATISFIED | Selection implementation and CLI create versioned JSONL artifacts with provenance and winner/loser semantics. |
| DATA-07 | 03-05, 03-06 | Compare generated reward-filtered data vs synthetic masked-SFT data | ✓ SATISFIED | Source comparison implementation and CLI produce generated-vs-synthetic reports with evidence availability, coverage gaps, expected help, and expected failure sections. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| Relevant Phase 3 synthetic provenance files | n/a | TODO/FIXME/placeholder/not-implemented | ✓ None found | No blocker stub markers found in `scripts/inspect_synthetic_dataset.py`, `src/data_quality/*.py`, or `docs/synthetic_quality.md`. |
| `src/data_quality/source_comparison.py`, `src/data_quality/synthetic_quality.py`, `src/data_quality/prompt_validation.py` | various | `return {}` / `return []` / `return None` | ℹ️ Info | Reviewed as normal optional-evidence/missing-file/default-return behavior, not user-visible stubs. |

### Human Verification Required

None for the Phase 3 gate. The implementation produces machine-verifiable report and contact-sheet artifacts; manual visual review of contact sheets remains recommended before using generated images as thesis evidence, but no unresolved must-have requires a human decision to mark Phase 3 complete.

### Gaps Summary

No remaining gaps. The previous synthetic provenance blocker is closed, no regressions were found in previously verified Phase 3 truths, and DATA-01 through DATA-07 are satisfied.

---

_Verified: 2026-05-04T16:15:28Z_
_Verifier: the agent (gsd-verifier)_
