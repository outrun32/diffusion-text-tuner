---
phase: 05-training-objective-and-pipeline-comparability
verified: 2026-05-05T19:38:02Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 5: Training Objective and Pipeline Comparability Verification Report

**Phase Goal:** Users can configure, run, extend, and compare training approaches without hidden objective choices or accidental behavior changes.
**Verified:** 2026-05-05T19:38:02Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can distinguish and configure SFT sample-selection modes, DPO pair-construction modes, and masked-SFT loss/LoRA/dataset/evaluation choices in configs and manifests. | ✓ VERIFIED | `src/training/selection.py` defines `SFT_SELECTION_MODES` with `threshold`, `top_k_per_prompt`, `score_weighted`, `hard_positive` and `DPO_PAIR_MODES` with `best_vs_worst`, `all_separated_pairs`, `margin_weighted`, `ambiguity_filtered` (lines 17-23). `src/training/config.py` exposes `selection_mode`, `selected_samples_path`, `sample_weighting`, `pair_construction_mode`, `preference_pairs_path`, `pair_weighting`, and masked-SFT `masked_lambda`/LoRA/eval fields (lines 30-34, 143-161, 190-194). `src/runtime/config_io.py` validates exact literals and emits sorted snapshots via `resolve_config_snapshot` (lines 88-100, 181-239, 270-310). |
| 2 | User can compare baseline, SFT, DPO, masked-SFT, and combined/curriculum runs under controlled prompts, seeds, inference settings, data sources, rewards, metrics, and artifact paths. | ✓ VERIFIED | `src/training/comparability.py` compares controlled groups for training, inference, prompt, model, data source, reward, metrics, and artifacts (lines 13-24) and blocks model/prompt/seed/inference/data/reward mismatches (line 24). `scripts/compare_training_runs.py` composes manifest diffs and comparability reports (lines 67-72). `docs/training_comparability.md` and `docs/commands.md` explicitly cover baseline, SFT, DPO, masked-SFT, combined, and curriculum comparison posture. Spot-check: integrated CLI returned `{"blocking": 2, "exit": 1, "schema": "training-run-comparison/v1"}` for seed/inference mismatches. |
| 3 | User can compare two run manifests to see changed configs, data sources, rewards, seeds, inference settings, metrics, and outputs. | ✓ VERIFIED | `src/runtime/manifest_diff.py` creates `config_changes`, `data_source_changes`, `reward_changes`, `seed_changes`, `inference_changes`, `metric_changes`, and `artifact_changes` sections (lines 14-22, 55-63). It redacts environment/cache metadata to presence booleans only (lines 152-166). `scripts/compare_run_manifests.py` exposes `--left`, `--right`, `--markdown`, and `--output` (lines 34-45). |
| 4 | User can identify training/inference mismatches such as step count, guidance, prompt embedding padding, model variants, and sampling configuration differences before using results as evidence. | ✓ VERIFIED | `CONTROLLED_FIELD_GROUPS` includes `num_training_steps`, `num_inference_steps`, `guidance_scale`, `prompt_embedding_padding`, `seed`, prompt/target, `model_id`, data-source fields, rewards, metrics, and `samples_dir` (src/training/comparability.py lines 13-22). Missing-vs-present fields are reported explicitly as `missing_left`/`missing_right` (lines 144-170). `scripts/check_training_comparability.py` exits 1 for blocking mismatches unless `--allow-blocking` is used (lines 19-34). |
| 5 | User can add or modify trainer variants through focused shared modules for sampling, checkpointing, schedulers, objective helpers, and runtime plumbing while preserving existing SFT, DPO, and masked-SFT behavior. | ✓ VERIFIED | Focused modules exist and are import-safe: `src/training/sampling.py`, `checkpointing.py`, `schedulers.py`, and `runtime.py`. `src/training/schedulers.py` re-exports existing objective scheduler helpers from `src.training.dpo_objective` (line 5), preserving the Phase 4 objective source of truth. `docs/training_comparability.md` documents `Shared trainer seams` and `Adding a trainer variant` (lines 106-139). Existing trainer loops are not refactored in this phase, which preserves behavior; future wiring is a residual risk, not a Phase 5 blocker. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/training/selection.py` | CPU-safe SFT/DPO selection modes and artifact summaries | ✓ VERIFIED | Substantive implementation with mode validation, deterministic sorting, source hashes, JSONL writers, summaries, strict DPO winner/loser handling, and weighted modes. |
| `scripts/materialize_training_data.py` | CLI flags for explicit SFT/DPO mode materialization | ✓ VERIFIED | Imports selection helpers and forwards `--mode`, `--score-column`, thresholds, margin, ambiguity, and `--hard-negative-threshold`. |
| `src/runtime/manifest_diff.py` | Manifest diff model and pure comparison function | ✓ VERIFIED | Pure local JSON/manifest comparison; categorized sections and Markdown formatter. |
| `scripts/compare_run_manifests.py` | CLI for JSON/Markdown manifest diffs | ✓ VERIFIED | Delegates to `compare_run_manifests`; handles malformed manifests with exit 2. |
| `src/training/comparability.py` | Pure comparability checks and report formatting | ✓ VERIFIED | Standard-library comparison logic plus manifest loader inside manifest-specific function only. |
| `scripts/check_training_comparability.py` | CLI for config/manifest mismatch reports | ✓ VERIFIED | Supports config mode, manifest mode, Markdown/output, and blocking-aware exit codes. |
| `src/training/config.py` | Dataclass fields for explicit Phase 5 training choices | ✓ VERIFIED | SFT/DPO choice fields and masked-SFT fields present with backward-compatible defaults. Ruff caveat noted below. |
| `src/runtime/config_io.py` | Pydantic validation and snapshots for explicit choices | ✓ VERIFIED | Exact literal validation, path-policy validation, forbidden extras, and snapshot generation. |
| `src/training/sampling.py` | Sampling interval and eval-suite helpers | ✓ VERIFIED | Pure helper functions with immutable normalization semantics. |
| `src/training/checkpointing.py` | Checkpoint interval/path helpers | ✓ VERIFIED | Pure interval helper and standard checkpoint directory formatter. |
| `src/training/schedulers.py` | Scheduler helper re-exports/wrappers | ✓ VERIFIED | Re-exports `compute_sigma` and `time_dependent_beta` from `dpo_objective`. |
| `src/training/runtime.py` | Training runtime metadata helpers | ✓ VERIFIED | Extracts sorted manifest input/output metadata from config snapshots. |
| `scripts/compare_training_runs.py` | Integrated manifest diff + comparability CLI | ✓ VERIFIED | Produces `training-run-comparison/v1`, combines `manifest_diff` and `comparability`, and exits 1 on blocking mismatches. |
| `docs/training_comparability.md`, `docs/commands.md`, `README.md`, `Makefile` | Discoverable Phase 5 command/docs surface | ✓ VERIFIED | Exact commands, Make alias, README link, shared seams, generated-artifact safety, and approach posture are present and guarded by tests. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/materialize_training_data.py` | `src/training/selection.py` | Direct import | ✓ WIRED | `from src.training.selection import materialize_dpo_pairs, materialize_sft_samples` at line 15; `main()` calls both helpers based on `--kind`. |
| `scripts/compare_run_manifests.py` | `src/runtime/manifest_diff.py` | Direct import/delegation | ✓ WIRED | Imports `compare_run_manifests` and `format_manifest_diff_markdown` at line 10; calls comparison at line 18. |
| `scripts/check_training_comparability.py` | `src/training/comparability.py` | Direct import/delegation | ✓ WIRED | Imports `compare_training_configs`, `compare_training_manifests`, and formatter at lines 12-16; delegates in config and manifest modes. |
| `src/runtime/config_io.py` | `src/training/config.py` | Validated models convert to dataclasses | ✓ WIRED | Imports SFT/DPO/masked dataclasses at lines 12-18; `_SFTModel`, `_DPOModel`, `_MaskedSFTModel` return dataclass instances. |
| `scripts/compare_training_runs.py` | `src/runtime/manifest_diff.py` + `src/training/comparability.py` | Integrated CLI composition | ✓ WIRED | Imports both comparison modules at lines 10-15 and combines their outputs at lines 67-72. |
| `Makefile` | `scripts.compare_training_runs` | `compare-training-runs` target | ✓ WIRED | Target invokes `uv run python -m scripts.compare_training_runs --left-manifest $(LEFT_MANIFEST) --right-manifest $(RIGHT_MANIFEST) ...`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `materialize_sft_samples` | `rows`, `selected_rows`, `summary` | Local score CSV via `_read_score_rows`, SHA-256 via `_sha256_file` | Yes — writes JSONL rows and optional manifest from parsed CSV, not hardcoded data | ✓ FLOWING |
| `materialize_dpo_pairs` | `pairs`, `filtering_stats`, `summary` | Local score CSV grouped by prompt; strict score comparisons | Yes — emits only score-derived strict winner/loser pairs and optional weights | ✓ FLOWING |
| `compare_run_manifests` | Diff sections | `load_run_manifest` for two local manifest JSON files | Yes — computes differences from config snapshots, metrics, outputs, and presence-only environment metadata | ✓ FLOWING |
| `compare_training_manifests` | `report` | `load_run_manifest` then manifest config/input/output/metric metadata | Yes — reports blocking/warning mismatches from manifest-controlled fields | ✓ FLOWING |
| `compare_training_runs` | `manifest_diff`, `comparability` | Composes the two comparison modules from the same manifest paths | Yes — integrated report is generated from local manifests | ✓ FLOWING |
| `resolve_config_snapshot` | `snapshot` | Validated SFT/DPO/masked-SFT dataclass | Yes — sorted, JSON-serializable dataclass snapshot includes explicit choices | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full CPU-safe test suite | `PATH="/root/.local/bin:$PATH" uv run pytest -q` | Orchestrator reported `202 passed` | ✓ PASS |
| SFT/DPO explicit selection materialization | Temporary Python script calling `materialize_sft_samples(... mode="score_weighted")` and `materialize_dpo_pairs(... mode="margin_weighted")` | `{"dpo_count": 2, "dpo_mode": "margin_weighted", "sft_count": 3, "sft_mode": "score_weighted"}` | ✓ PASS |
| Integrated run comparison blocks uncontrolled differences | Temporary Python script invoking `scripts.compare_training_runs.main([...])` on two tiny manifests | `{"blocking": 2, "exit": 1, "schema": "training-run-comparison/v1"}` | ✓ PASS |
| Makefile alias is discoverable | Tests run `make -n compare-training-runs LEFT_MANIFEST=... RIGHT_MANIFEST=...` | Covered by `tests/test_training_comparison_docs.py`; dry-run asserts exact integrated CLI | ✓ PASS |
| Ruff caveat | `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/config.py` | Fails with 7 E501 line-length errors in long prompt literals at lines 59, 152, 219, 277, 281, 285, 289 | ⚠️ WARNING — pre-existing lint debt, not a Phase 5 goal blocker |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TRN-02 | 05-01, 05-04 | Distinguish SFT modes such as threshold, top-1, score-weighted, hard-positive | ✓ SATISFIED | Selection modes implemented in `src/training/selection.py`; config fields and validation in `src/training/config.py` and `src/runtime/config_io.py`; tests cover all modes and CLI forwarding. |
| TRN-03 | 05-01, 05-04 | Distinguish DPO pair modes such as best-vs-worst, all pairs, margin-weighted, ambiguity-filtered | ✓ SATISFIED | DPO pair modes implemented with strict winner/loser semantics and config validation; tests cover pair generation, equal-score rejection, weighting, and ambiguity filtering. |
| TRN-04 | 05-04 | Masked-SFT explicit masked/global loss weighting, LoRA choices, synthetic dataset variant, eval suite in config/manifest | ✓ SATISFIED | `MaskedSFTConfig` and `_MaskedSFTModel` expose `masked_lambda`, multi-rank LoRA, `data_dir`, `eval_suite_path`, `validation_interval`, and `eval_suite_n_per_step`; tests assert snapshots/docs. |
| TRN-05 | 05-03, 05-06 | Compare baseline, SFT, DPO, masked-SFT, combined/curriculum under controlled fields | ✓ SATISFIED | Comparability report and integrated CLI compare prompts, seeds, inference settings, data sources, rewards, metrics, and artifacts; docs cover all approach names and safe command posture. |
| TRN-06 | 05-03 | Identify training/inference mismatches before using results as evidence | ✓ SATISFIED | Blocking/warning mismatch detector reports step count, guidance, prompt embedding padding, model ID, sampling/inference/data/reward differences and exits nonzero on blockers. |
| TRN-07 | 05-05, 05-06 | Add new trainer/pipeline variants without editing unrelated scripts or losing behavior | ✓ SATISFIED | Shared import-safe helper modules plus documented `Adding a trainer variant` flow provide focused seams; existing trainer loops are preserved. |
| RUN-02 | 05-02, 05-06 | Compare two local run manifests for configs, data, rewards, seeds, inference, metrics, artifacts | ✓ SATISFIED | `src/runtime/manifest_diff.py`, `scripts/compare_run_manifests.py`, and integrated CLI provide categorized JSON/Markdown diffs. |
| STR-04 | 05-05, 05-06 | Focused shared training modules for sampling, checkpointing, schedulers, objective helpers, config/runtime plumbing | ✓ SATISFIED | `sampling`, `checkpointing`, `schedulers`, `runtime`, existing `dpo_objective`, config validation, docs, and tests provide the shared seams. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/training/selection.py` | 303, 484 | `return {}` | ℹ️ Info | Empty dict is valid behavior when no weights are needed; not a stub and not user-visible placeholder data. |
| `src/training/comparability.py` | 214 | `return []` | ℹ️ Info | Empty list is valid formatter behavior when no controlled fields are supplied; not a stub. |
| `src/training/config.py` | 59, 152, 219, 277, 281, 285, 289 | Ruff E501 long prompt literals | ⚠️ Warning | Confirmed documented Plan 05-04 caveat. This blocks broad Ruff on `src/training/config.py` in the current dirty/pre-existing worktree, but it does not block Phase 5 behavior, tests, config validation, or comparability goal achievement. |

### Human Verification Required

None. Phase 5 deliverables are CPU-safe CLI/config/manifest/test/documentation surfaces. GPU training runs, OCR/model diagnostics, and visual evaluation remain explicit research operations outside this phase's default verification contract.

### Gaps Summary

No blocking gaps found. The Phase 5 goal is achieved by actual code evidence: explicit training objective/data choices exist, are validated into snapshots/manifests, can be materialized as artifacts, can be compared through manifest and mismatch reports, and are discoverable through the command/docs surface.

Residual risks to carry forward:

- The shared training utility modules are intentionally not yet wired into the large SFT/DPO/masked-SFT trainer loops. This preserves current behavior and satisfies Phase 5's extension-seam goal, but future trainer refactors should migrate loop logic through these helpers with compatibility tests.
- The integrated comparability command reads local manifest/config metadata only. It does not prove GPU training or visual text-rendering quality; Phase 6 must validate reward/evaluation evidence.
- The documented `src/training/config.py` Ruff E501 caveat is real and pre-existing lint debt. It should be cleaned separately but is unrelated to Phase 5 goal achievement.

---

_Verified: 2026-05-05T19:38:02Z_
_Verifier: the agent (gsd-verifier)_
