---
phase: 07-moderate-structure-and-extension-cleanup
verified: 2026-05-06T16:19:50Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
---

# Phase 7: Moderate Structure and Extension Cleanup Verification Report

**Phase Goal:** Users can navigate a clearer brownfield toolkit and add future experiments through stable, documented seams instead of one-off scripts.
**Verified:** 2026-05-06T16:19:50Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can find reusable source code, thin scripts, cluster launchers, configs, diagnostics, experiments, generated outputs, tests, and thesis artifacts in clear homes after moderate safe moves. | ✓ VERIFIED | `docs/structure_and_extension.md` lines 9-25 define canonical homes for `src/`, `scripts/`, `scripts/cluster/`, `scripts/synth/`, `scripts/thesis_plots/`, `configs/`, `configs/experiments/`, `tests/`, `experiments/`, ignored runtime outputs, and thesis evidence. `scripts/README.md` lines 5-18 separates supported wrappers, diagnostics, cluster jobs, synthesis helpers, plotting helpers, historical scripts, and generated-output boundaries. README links the Phase 7 guide at lines 54-58. |
| 2 | User can invoke generation, scoring, synthesis, evaluation, plotting, and run comparison through thin CLI scripts backed by importable implementation modules. | ✓ VERIFIED | Thin wrappers delegate to importable seams: `scripts/generate_images.py` imports `GenerationConfig`/`run_generation` from `src.generation.pipeline` and calls `run_generation(...)`; `scripts/score_images.py` imports `ScoringConfig`/`run_scoring` from `src.scoring.pipeline` and calls `run_scoring(config)`; `scripts/synth/build_dataset.py` imports `SynthesisBuildConfig`/`build_dataset` from `src.synthesis.dataset_builder`; `scripts/plot_metrics.py` imports and delegates to `src.plotting.training_metrics.plot_training_metrics`; existing evaluation/run-comparison wrappers delegate to `src.evaluation.heldout.write_evaluation_plan`, `src.runtime.manifest_diff.compare_run_manifests`, and training comparability helpers. CLI help spot-checks for generation, scoring, synthesis, and plotting succeeded. |
| 3 | User can follow documented extension points to add future experiments, trainers, reward variants, datasets, or pipelines without creating hidden assumptions in unrelated scripts. | ✓ VERIFIED | `src/toolkit/extension_points.py` defines a standard-library-only frozen `ExtensionPoint` registry with 10 entries: prompt generation, image generation, scoring, synthesis, training, evaluation, plotting, run comparison, diagnostics, thesis outputs. `docs/structure_and_extension.md` lines 55-82 mirror the registry and provide an extension checklist covering config, artifact/manifest contracts, importable modules, thin CLI wrappers, CPU-safe tests, command docs, and generated-artifact safety. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `docs/structure_and_extension.md` | Canonical structure map, extension rules, registry mirror, and generated-artifact boundaries | ✓ VERIFIED | Substantive guide with canonical homes, extension rules, registry table, and checklist; linked from README and commands docs. |
| `scripts/README.md` | Script-family navigation and wrapper/diagnostic boundaries | ✓ VERIFIED | Classifies thin wrappers, diagnostics, cluster jobs, synthesis helpers, plotting helpers, historical scripts, and runtime output safety. |
| `src/generation/pipeline.py` | Importable generation seam | ✓ VERIFIED | Exports `GenerationConfig`, `GenerationPaths`, prompt loading, path resolution, seed planning, and `run_generation`; heavy FLUX/torchvision imports are inside `run_generation`. |
| `scripts/generate_images.py` | Thin generation CLI wrapper | ✓ VERIFIED | Parses existing arguments, builds `GenerationConfig`, delegates to `run_generation`, supports `python -m scripts.generate_images`. |
| `src/scoring/pipeline.py` | Importable scoring seam | ✓ VERIFIED | Owns canonical score columns, `ScoringConfig`, task discovery, canonical row conversion, schema sidecar writing, sharding/resume behavior, scorer selection, and `run_scoring`. |
| `scripts/score_images.py` | Thin scoring CLI wrapper | ✓ VERIFIED | Parses scoring arguments, builds `ScoringConfig`, delegates to `run_scoring`, and re-exports compatibility helpers. |
| `src/synthesis/dataset_builder.py` | Importable synthetic dataset builder seam | ✓ VERIFIED | Exposes `SynthesisBuildConfig`, render/collate/fan-out/schema writer phases, gated latent/text phases, and `build_dataset`. Heavy imports are inside explicit phase functions. |
| `scripts/synth/build_dataset.py` | Thin synthesis CLI wrapper | ✓ VERIFIED | Parses existing flags, builds `SynthesisBuildConfig`, delegates to `build_dataset`, and preserves compatibility re-exports. |
| `src/plotting/training_metrics.py` | Importable plotting seam | ✓ VERIFIED | Provides typed `TrainingMetrics`, `load_metrics`, `smooth`, `summarize_metrics`, and `plot_training_metrics`; Matplotlib imports are lazy inside plotting. |
| `scripts/plot_metrics.py` | Thin plotting CLI wrapper | ✓ VERIFIED | Delegates to `plot_training_metrics`, preserves direct script and module invocation and compatibility aliases. |
| `src/toolkit/extension_points.py` | Importable extension-point registry | ✓ VERIFIED | Descriptive-only registry; no dynamic plugin loading, command execution, or runtime artifact validation at import time. |
| `tests/test_*phase7*` and seam contract tests | CPU-safe drift/contract tests | ✓ VERIFIED | Focused Phase 7 suite passed: 41 tests. Orchestrator full suite passed: 294 tests. |
| `docs/commands.md`, `README.md`, `Makefile` | Discoverable Phase 7 command/docs surface | ✓ VERIFIED | `phase7-structure-tests` documented in commands/README and implemented in Makefile. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `README.md` | `docs/structure_and_extension.md` | Phase 7 front-door link | ✓ WIRED | README lines 54-58 link the guide and verification alias. |
| `scripts/README.md` | `docs/structure_and_extension.md` | Script navigation guidance | ✓ WIRED | `scripts/README.md` line 3 links the structure guide and line 22 instructs reusable modules under `src/`. |
| `scripts/generate_images.py` | `src/generation/pipeline.py` | CLI parser builds `GenerationConfig`, calls `run_generation` | ✓ WIRED | Lines 24 and 77-94. |
| `scripts/score_images.py` | `src/scoring/pipeline.py` | CLI parser builds `ScoringConfig`, calls `run_scoring` | ✓ WIRED | Lines 32-38 and 126-140. |
| `src/scoring/pipeline.py` | `src/evaluation/reward_interface.py` | Product score/metadata helpers | ✓ WIRED | Lines 23-27 import `ProductScoreFormula`, `build_score_metadata`, and `compute_product_score`; `build_canonical_score_row` and sidecar writing use them. |
| `scripts/synth/build_dataset.py` | `src/synthesis/dataset_builder.py` | CLI parser builds `SynthesisBuildConfig`, calls `build_dataset` | ✓ WIRED | Lines 26-36 and 107-125. |
| `scripts/plot_metrics.py` | `src/plotting/training_metrics.py` | CLI delegates to `plot_training_metrics` | ✓ WIRED | Lines 13 and 31-35. |
| `docs/structure_and_extension.md` | `src/toolkit/extension_points.py` | Registry mirror and `list_extension_points` reference | ✓ WIRED | Guide lines 55-70 mirror registry entries and line 57 names `list_extension_points()`. |
| `docs/commands.md` | `Makefile` | `phase7-structure-tests` command | ✓ WIRED | Commands lines 201-216 document the alias; Makefile lines 140-141 implement it. |

### Data-Flow Trace (Level 4)

Phase 7 does not add UI components or live dashboards that render dynamic data. For code seams, the relevant flow is command/config-to-module delegation:

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `scripts/generate_images.py` | `GenerationConfig` | argparse CLI args | Yes — passed into `run_generation`, which loads prompt JSONL and writes configured outputs | ✓ FLOWING |
| `scripts/score_images.py` | `ScoringConfig` | argparse CLI args | Yes — passed into `run_scoring`, which discovers image/text-embedding tasks and writes score CSV/sidecar | ✓ FLOWING |
| `scripts/synth/build_dataset.py` | `SynthesisBuildConfig` | argparse CLI args | Yes — passed into `build_dataset`, which runs render/collate/fan-out/index and optional gated model phases | ✓ FLOWING |
| `scripts/plot_metrics.py` | `metrics_csv`, `output_dir` | argparse CLI args | Yes — passed into `plot_training_metrics`, which loads CSV metrics and writes `training_curves.png` when invoked | ✓ FLOWING |
| `src/toolkit/extension_points.py` | `_EXTENSION_POINTS` | Static registry tuple | Yes — `list_extension_points()` and `get_extension_point()` expose complete registry metadata | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Extension registry imports and returns all stages | `uv run python - <<'PY' ... list_extension_points() ... PY` | Returned 10 entries: prompt generation, image generation, scoring, synthesis, training, evaluation, plotting, run comparison, diagnostics, thesis outputs; `get_extension_point('scoring')` returned `src.scoring.pipeline`. | ✓ PASS |
| Phase 7 Makefile alias is dry-run discoverable | `make -n phase7-structure-tests` | Printed `uv run pytest tests/test_structure_extension_docs.py tests/test_generation_pipeline_contracts.py tests/test_scoring_pipeline_contracts.py tests/test_synthesis_pipeline_contracts.py tests/test_plotting_pipeline_contracts.py tests/test_extension_points_docs.py -q`. | ✓ PASS |
| Thin CLI wrappers import and expose help without executing heavy work | `uv run python -m scripts.generate_images --help`, `uv run python -m scripts.score_images --help`, `uv run python -m scripts.synth.build_dataset --help`, `uv run python -m scripts.plot_metrics --help` | All commands exited successfully. | ✓ PASS |
| Focused Phase 7 seam/docs tests pass | `uv run pytest tests/test_structure_extension_docs.py tests/test_generation_pipeline_contracts.py tests/test_scoring_pipeline_contracts.py tests/test_synthesis_pipeline_contracts.py tests/test_plotting_pipeline_contracts.py tests/test_extension_points_docs.py -q` | `41 passed in 1.45s`. | ✓ PASS |
| Full CPU-safe suite passes | Orchestrator-provided `PATH="/root/.local/bin:$PATH" uv run pytest -q` | `294 passed`. | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| STR-01 | 07-01, 07-06 | User can navigate a moderately cleaned file structure where reusable source code, thin scripts, cluster launchers, configs, diagnostics, experiments, generated outputs, tests, and thesis artifacts have clear homes. | ✓ SATISFIED | `docs/structure_and_extension.md` canonical homes table; `scripts/README.md` script-family guide; README/commands/Makefile links. |
| STR-05 | 07-02, 07-03, 07-04, 07-05, 07-06 | User can use importable implementation modules behind CLI scripts for generation, scoring, synthesis, evaluation, plotting, and run comparison. | ✓ SATISFIED | Verified generation/scoring/synthesis/plotting wrappers delegate to importable modules; evaluation/run comparison wrappers already delegate to `src.evaluation.heldout`, `src.runtime.manifest_diff`, and `src.training.comparability`; extension registry indexes all seams. |
| STR-06 | 07-01 through 07-06 | User can add future experiments/pipelines through documented extension points rather than new one-off scripts with hidden assumptions. | ✓ SATISFIED | `src.toolkit.extension_points` registry plus docs extension-point table/checklist and Phase 7 focused tests. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| — | — | No TODO/FIXME/placeholder/not-implemented markers found in Phase 7-owned source seams (`src/generation`, `src/scoring`, `src/synthesis`, `src/plotting`, `src/toolkit`). | — | No blocker. |

### Human Verification Required

None for the Phase 7 structural goal. The verification did not run expensive FLUX/Qwen/PaddleOCR/SynthTIGER/CUDA jobs; that is intentional for this CPU-safe structure cleanup phase and remains covered by explicit runtime smoke/diagnostic commands when those environments are available.

### Gaps Summary

No blocking gaps found. Phase 7 goal is achieved: the repository has clear documented homes, thin wrapper/importable-module seams for the targeted pipeline families, a registry/checklist for extension points, and CPU-safe verification surfaces. Residual risk is limited to environment-specific heavy runtime execution (GPU/model/OCR/synthesis), which was intentionally out of default verification scope and should be validated with explicit smoke/diagnostic commands on target local/SLURM machines.

---

_Verified: 2026-05-06T16:19:50Z_  
_Verifier: the agent (gsd-verifier)_
