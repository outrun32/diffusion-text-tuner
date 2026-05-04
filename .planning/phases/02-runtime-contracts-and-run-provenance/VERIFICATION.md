---
phase: 02-runtime-contracts-and-run-provenance
verified: 2026-05-04T14:54:25Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "Users can capture local run manifests before expensive generation, scoring, training, or evaluation stages start."
  gaps_remaining: []
  regressions: []
---

# Phase 2: Runtime Contracts and Run Provenance Verification Report

**Phase Goal:** Users can validate configs, paths, artifacts, and run metadata before long-running generation, scoring, training, or evaluation stages start.
**Verified:** 2026-05-04T14:54:25Z
**Status:** passed
**Re-verification:** Yes — after manifest gap closure

## Goal Achievement

Phase 2 now satisfies the roadmap contract. The previous blocker was trainer-only run manifest creation. Re-verification confirmed that `src.runtime.manifests` and `scripts.run_manifest` now support `generate`, `score`, `sft`, `dpo`, `masked_sft`, `synthetic`, and `evaluation`; non-training stages can be initialized before a stage-specific config exists, raw JSON config snapshots are preserved when provided, and training stages still require validated trainer configs.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can load existing and new configs through one validated path that reports missing fields, invalid values, model ID inconsistencies, and local/SLURM path problems before expensive work begins. | ✓ VERIFIED | Quick regression check: `src/runtime/config_io.py` still exposes strict Pydantic-backed `load_stage_config`, `resolve_config_snapshot`, model/path validators, and secret-safe `RuntimeConfigError` context. `uv run pytest tests/test_runtime_config_io.py ...` passed as part of the 48 runtime-contract test run. |
| 2 | User can create a local run directory whose manifest captures command, timestamp, git state, resolved config snapshot, environment summary, seeds, model IDs/revisions, inputs, outputs, metrics, and notes. | ✓ VERIFIED | Full re-check of previous gap: `MANIFEST_STAGES` includes `generate`, `score`, `sft`, `dpo`, `masked_sft`, `synthetic`, `evaluation`; `create_run_manifest` writes `manifest.json` and `config_snapshot.json`; `collect_git_state`, `collect_environment_summary`, `collect_seeds`, and `collect_model_revisions` populate provenance. CLI spot-check created and inspected a configless generate manifest, created score/synthetic/evaluation manifests, and created a raw-config generate manifest whose payload contained command, timestamp, git state, env summary, raw config snapshot, seed 77, model ID, and model revision `rev1`. |
| 3 | User can rely on documented canonical paths and schema/version metadata for prompts, generated images, latents, embeddings, masks, scores, selected samples, preference pairs, checkpoints, logs, eval outputs, and manifests. | ✓ VERIFIED | Quick regression check: `src/runtime/paths.py` still defines canonical roots and stage mappings; `src/runtime/artifacts.py` still declares `runtime-artifacts/v1` and validators for required artifact families; `docs/runtime_contracts.md` documents the artifact matrix and schema/version conventions. Runtime artifact/docs tests passed. |
| 4 | User can preflight key artifacts and resume or inspect long-running stages using consistent manifest and output conventions without committing generated tensors, images, checkpoints, logs, or private outputs. | ✓ VERIFIED | Quick regression check: `scripts/preflight_runtime.py` supports `generate`, `score`, `sft`, `dpo`, `masked-sft`, `synthetic`, and `evaluation`; it imports runtime config/artifact/path/manifest helpers and emits JSON/human reports without launching jobs. Manifest inspect, note, and metrics behavior remains tested. `.gitignore` and `assert_artifact_git_safety` keep generated roots and binaries out of git by default. |
| 5 | User can use shared runtime helpers for config I/O, path resolution, seeds, manifests, and preflight validation across pipeline stages. | ✓ VERIFIED | Quick regression check: shared helpers are exported from `src/runtime/__init__.py`; trainer `load_config` functions delegate to `config_io.load_stage_config`; `scripts/preflight_runtime.py` uses `resolve_stage_paths`, `validate_artifacts`, `load_stage_config`, and `load_run_manifest`; `scripts/run_manifest.py` uses manifest helpers. Full CPU-safe suite passed (`68 passed`). |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/runtime/config_io.py` | Shared validated config loading and snapshots | ✓ VERIFIED | Substantive implementation with strict models, path policy, errors, dataclass conversion, and immutable snapshot metadata. Training-stage configs remain validated before manifest snapshots. |
| `src/runtime/paths.py` | Canonical runtime path helpers and git-safety classification | ✓ VERIFIED | Covers generation/scoring/SFT/DPO/masked-SFT/synthetic/evaluation/plotting/manifests and non-committable classifications. |
| `src/runtime/artifacts.py` | CPU-safe artifact schema validators | ✓ VERIFIED | Validates JSONL, CSV, generated tensor/image layouts, masked-SFT tensors, training inputs, manifests, selected samples, preference pairs, logs/checkpoints/eval outputs. |
| `src/runtime/manifests.py` | Run directory creation, manifest schema, config snapshots, update/inspect helpers | ✓ VERIFIED | Previous partial is closed. `MANIFEST_STAGES` covers non-training and training stages; `_resolve_manifest_config_snapshot` requires validated configs for training stages and returns either minimal stage snapshots or raw JSON snapshots for non-training stages. |
| `src/runtime/reproducibility.py` | Git, environment, seed, model metadata collectors | ✓ VERIFIED | Secret-safe env presence, cache presence, package versions, git state, seed/model extraction. Raw-config snapshots are supported by `_metadata_source`. |
| `scripts/run_manifest.py` | Manifest CLI | ✓ VERIFIED | `init`, `inspect`, `note`, and `metrics` are wired to runtime helpers. `--stage` choices are now derived from full `MANIFEST_STAGES`; configless `generate`, `score`, `synthetic`, and `evaluation` init spot-checks succeeded. |
| `scripts/preflight_runtime.py` | Runtime preflight CLI | ✓ VERIFIED | Supports Phase 2 stages and returns JSON/human reports without launching jobs. |
| `docs/runtime_contracts.md` | Canonical paths/artifact schemas/git-safety docs | ✓ VERIFIED | Covers required artifact families and local/SLURM path guidance. |
| `docs/commands.md`, `README.md`, `Makefile`, `configs/experiments/README.md` | Discoverable command/config surfaces | ✓ VERIFIED | `docs/commands.md` includes generate/SFT/evaluation manifest examples and explains non-training raw-config behavior; README and Makefile expose runtime contract entry points. |
| `tests/test_runtime_*.py` | CPU-safe runtime tests | ✓ VERIFIED | Runtime tests passed: manifest tests `9 passed`; combined runtime contract tests `48 passed`; full CPU-safe suite `68 passed`. New manifest tests cover configless generate, raw-config generate, and CLI generate init/inspect. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/training/sft_trainer.py` | `src/runtime/config_io.py` | `load_config` delegates to `config_io.load_stage_config("sft", path)` | ✓ WIRED | Verified by code grep and runtime config tests. |
| `src/training/dpo_trainer.py` | `src/runtime/config_io.py` | `load_config` delegates to `config_io.load_stage_config("dpo", path)` | ✓ WIRED | Verified by code grep and runtime config tests. |
| `src/training/masked_sft_trainer.py` | `src/runtime/config_io.py` | `load_config` delegates to `config_io.load_stage_config("masked_sft", path)` | ✓ WIRED | Verified by code grep and runtime config tests. |
| `scripts/preflight_runtime.py` | runtime helpers | imports/uses config, paths, artifacts, manifests | ✓ WIRED | Uses `load_stage_config`, `resolve_stage_paths`, `validate_artifacts`, and `load_run_manifest`. |
| `scripts/run_manifest.py` | `src/runtime/manifests.py` | CLI calls `create_run_manifest`, `load_run_manifest`, `update_run_manifest`, `print_manifest_summary` | ✓ WIRED | Previous partial is closed. CLI accepts all manifest stages from `MANIFEST_STAGES`; non-training init and inspect spot-checks passed. |
| Docs/Makefile | Runtime CLIs | documented commands and aliases | ✓ WIRED | `docs/commands.md`, `README.md`, and `Makefile` expose preflight and manifest commands. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `scripts/run_manifest.py` | `manifest` | `create_run_manifest(...)` | Yes | ✓ FLOWING — CLI prints created run dir; inspect loads actual `manifest.json`. |
| `src/runtime/manifests.py` | `config_snapshot` | Validated `load_stage_config` for training stages; minimal/raw JSON snapshot for non-training stages | Yes | ✓ FLOWING — raw-config smoke showed `raw_config`, seed, model ID, and revision flow into manifest `config_snapshot`, `seeds`, `models`, and `inputs`. |
| `src/runtime/reproducibility.py` | `models`, `seeds`, `environment`, `git` | Snapshot flattening, package metadata, torch CUDA presence, git commands | Yes | ✓ FLOWING — generated smoke manifest includes actual git commit/dirty state, package versions, env presence booleans, model revision, and seed. |
| `scripts/preflight_runtime.py` | `report` | Runtime helper calls | Yes | ✓ FLOWING — preflight tests cover JSON reports and blocking error aggregation. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Runtime manifest tests cover gap fixes | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifests.py` | `9 passed` | ✓ PASS |
| Runtime contract tests still pass | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py tests/test_runtime_artifacts.py tests/test_runtime_preflight.py tests/test_runtime_docs.py` | `48 passed` | ✓ PASS |
| Full CPU-safe suite still passes | `PATH="/root/.local/bin:$PATH" uv run pytest` | `68 passed` | ✓ PASS |
| Manifest lint target passes | `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/runtime/manifests.py src/runtime/reproducibility.py scripts/run_manifest.py tests/test_runtime_manifests.py` | `All checks passed!` | ✓ PASS |
| Configless generate manifest can be initialized and inspected | `uv run python -m scripts.run_manifest init --stage generate --command "python -m scripts.generate_images" --run-root /tmp/opencode/runs/phase2-generate-reverify`; then `inspect .../manifest.json` | Created `/tmp/opencode/runs/phase2-generate-reverify/20260504T145317Z-generate-generate`; inspect printed Stage `generate`, command, config snapshot, outputs, metrics, notes. | ✓ PASS |
| Other non-training manifest stages can be initialized before launch | Programmatic CLI loop for `score`, `synthetic`, `evaluation` with configless `init` under `/tmp/opencode/runs/phase2-stage-reverify` | Each returned exit code `0` and printed a run directory. | ✓ PASS |
| Raw JSON config snapshots flow into non-training manifests | `uv run python` wrote `/tmp/opencode/generate-config-reverify.json` and ran `scripts.run_manifest init --stage generate --config ...` | Manifest contains `raw_config`, `schema_version: runtime-config/v1`, `stage: generate`, `seeds: {seed: 77}`, and `models: {model_id: ..., model_revision: rev1}`. | ✓ PASS |
| Training manifests still require validated config snapshots | `uv run python -m scripts.run_manifest init --stage sft --command "python -m src.training.sft_trainer" --run-root /tmp/opencode/runs/phase2-sft-no-config-reverify` | Reported `sft: --config is required for training-stage manifests`. | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CFG-01 | 02-01, 02-04 | Validated JSON configs with field/path/model errors before expensive work | ✓ SATISFIED | `config_io.py`, trainer loader delegation, runtime config tests. |
| CFG-02 | 02-05 | Organize experiment configs by stage/family with consistent naming | ✓ SATISFIED | `configs/experiments/README.md` and docs tests. |
| CFG-03 | 02-03 | Capture resolved config as immutable run snapshot | ✓ SATISFIED | `create_run_manifest` writes `config_snapshot.json`; training snapshots are validated; non-training raw JSON snapshots are preserved when supplied. |
| CFG-04 | 02-01 | Local/SLURM-compatible path settings without hardcoded personal absolutes | ✓ SATISFIED | Path policy rejects traversal/home/off-repo absolute paths; runtime docs describe relative local/SLURM roots. |
| ART-01 | 02-02, 02-04 | Validate key artifacts before GPU-heavy stages | ✓ SATISFIED | Artifact validators and preflight CLI cover prompt/generated/score/masked/training/eval/manifest families. |
| ART-02 | 02-02, 02-05 | Document canonical paths | ✓ SATISFIED | `paths.py` and `docs/runtime_contracts.md` cover required families. |
| ART-03 | 02-02 | Schema/version metadata for artifacts and score files | ✓ SATISFIED | `ARTIFACT_SCHEMA_VERSION`, manifest schema, optional score schema sidecar, docs. |
| ART-04 | 02-02, 02-05 | Keep generated artifacts out of git with fixture exceptions | ✓ SATISFIED | `.gitignore`, `assert_artifact_git_safety`, docs. |
| RUN-01 | 02-03, 02-05 | Local run directory with complete manifest metadata | ✓ SATISFIED | Gap closed: manifest creation now covers training and non-training long-running stages; raw-config smoke confirms config, command, git, environment, seeds, models, inputs, outputs, metrics, notes. |
| RUN-03 | 02-02, 02-03, 02-05 | Consistent artifact layout without committing generated tensors/images/checkpoints/logs | ✓ SATISFIED | Canonical paths, docs, git-safety checks, `.gitignore`. |
| RUN-04 | 02-03, 02-04, 02-05 | Resume/inspect long-running generation, scoring, training, evaluation via manifest/output conventions | ✓ SATISFIED | CLI `init` and `inspect` work for `generate`; configless `score`, `synthetic`, and `evaluation` init works; SFT/DPO/masked-SFT remain supported with config validation. |
| STR-02 | 02-01 through 02-04 | Shared helpers for config I/O, paths, seeds, manifests, preflight | ✓ SATISFIED | `src/runtime/*`, trainer wiring, preflight CLI, manifest CLI, runtime tests. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `Makefile` | 50 | Comment says `# Placeholder form for docs/tests: python -m scripts.run_manifest inspect runs/<run_id>/manifest.json` | ℹ️ Info | Documentation-style placeholder only; not executable code and not a stub. |
| Worktree | n/a | Dirty/untracked unrelated files present during verification | ℹ️ Info | Manifest smoke correctly records dirty git state. Not a Phase 2 gap; avoid staging unrelated changes. |

### Human Verification Required

None. Phase 2 is a code/docs/runtime-contract surface; the blocking behavior and regression checks are verifiable through static review plus CPU-safe commands.

### Gaps Summary

No remaining blocking gaps. The prior run-manifest stage coverage gap is closed: generation, scoring, synthesis, and evaluation manifests can now be initialized before launch, while SFT/DPO/masked-SFT manifests still require validated config snapshots.

---

_Verified: 2026-05-04T14:54:25Z_
_Verifier: the agent (gsd-verifier)_
