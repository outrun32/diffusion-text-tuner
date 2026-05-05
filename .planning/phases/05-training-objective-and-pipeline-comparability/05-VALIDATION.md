---
phase: 05
slug: training-objective-and-pipeline-comparability
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-05
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during Phase 5 execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest via uv |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py tests/test_runtime_manifest_diff.py tests/test_training_comparability.py -q` |
| **Full suite command** | `PATH="/root/.local/bin:$PATH" uv run pytest -q` |
| **Estimated runtime** | < 60 seconds for focused CPU-safe commands; full suite depends on local machine but remains CPU/model-download-free |

## Sampling Rate

- **After every task commit:** Run the task's PLAN.md `<automated>` command.
- **After every plan wave:** Run `PATH="/root/.local/bin:$PATH" uv run pytest -q`.
- **Before `/gsd-verify-work`:** Full default pytest must be green.
- **Max feedback latency:** One task; no plan may defer test creation beyond its first task.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | TRN-02 | T-05-01-01 | Score CSV parsing is validated before selection artifacts are trusted. | unit | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -x` | ✅ | ⬜ pending |
| 05-01-02 | 01 | 1 | TRN-02/TRN-03 | T-05-01-01/T-05-01-02 | Explicit SFT/DPO modes preserve strict winner/loser and provenance semantics. | unit | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` | ✅ | ⬜ pending |
| 05-01-03 | 01 | 1 | TRN-02/TRN-03 | T-05-01-03 | CLI/docs keep generated artifact safety visible. | unit+lint | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/selection.py scripts/materialize_training_data.py tests/test_training_selection_artifacts.py` | ✅ | ⬜ pending |
| 05-02-* | 02 | 1 | RUN-02 | T-05-02-01/T-05-02-02 | Manifest diffs are CPU-safe and secret-presence-only. | unit+lint | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_manifest_diff.py -q` | ❌ until Plan 02 Task 1 | ⬜ pending |
| 05-03-* | 03 | 1 | TRN-05/TRN-06 | T-05-03-01/T-05-03-02 | Mismatch reports block uncontrolled comparisons. | unit+lint | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparability.py -q` | ❌ until Plan 03 Task 1 | ⬜ pending |
| 05-04-* | 04 | 2 | TRN-02/TRN-03/TRN-04 | T-05-04-01/T-05-04-03 | Explicit config choices validate and snapshot. | unit+lint | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py tests/test_characterization_config_artifacts.py -q` | ✅ | ⬜ pending |
| 05-05-* | 05 | 2 | TRN-07/STR-04 | T-05-05-01/T-05-05-02 | Shared modules stay import-safe and deterministic. | unit+lint | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_shared_utilities.py -q` | ❌ until Plan 05 Task 1 | ⬜ pending |
| 05-06-* | 06 | 3 | TRN-05/RUN-02/TRN-07/STR-04 | T-05-06-01/T-05-06-03 | Integrated command/docs remain CPU-safe and synchronized. | unit+docs+lint | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_comparison_docs.py tests/test_training_comparability.py tests/test_runtime_manifest_diff.py -q` | ❌ until Plan 06 Task 1 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements

Existing pytest/uv infrastructure covers Phase 5. Each plan creates its own missing test file as Task 1 before production implementation.

## Manual-Only Verifications

All Phase 5 planned behaviors have automated CPU-safe verification. GPU training runs remain explicit user-launched research commands, not default verification.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references through Task 1 test scaffolds.
- [x] No watch-mode flags.
- [x] Default validation remains CPU-safe and model-download-free.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending execution
