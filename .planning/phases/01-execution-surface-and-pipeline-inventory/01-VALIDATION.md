---
phase: 1
slug: execution-surface-and-pipeline-inventory
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest after Plan 02; script fallback before Plan 02 |
| **Config file** | `pyproject.toml` created in Plan 02 |
| **Quick run command** | `python tests/test_losses.py` before Plan 02; `uv run pytest` after Plan 02 |
| **Full suite command** | `uv run pytest && uv run ruff check . && uv run ruff format --check .` after Plan 02 |
| **Estimated runtime** | < 60 seconds for CPU-safe tests/lint on normal development hardware |

---

## Sampling Rate

- **After every task commit:** Run the task's `<automated>` command.
- **After every plan wave:** Run `uv run pytest` once Plan 02 is complete; before then run `python tests/test_losses.py`.
- **Before `/gsd-verify-work`:** `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and documentation string checks must be green.
- **Max feedback latency:** 60 seconds for default CPU-safe checks. GPU/model/OCR smokes are explicit diagnostics, not default gates.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | INV-01, INV-02 | T-01-01 | No generated artifacts committed | doc assertion | `python -c "from pathlib import Path; text=Path('docs/pipeline_inventory.md').read_text(); assert all(s in text for s in ['Prompt generation','Image generation','Reward scoring','SFT training','DPO training','Masked-SFT training','Synthetic data','Evaluation','Plotting','SLURM launchers'])"` | ✅ | ⬜ pending |
| 1-01-02 | 01 | 1 | INV-03, INV-04 | T-01-02 | Historical/manual paths labeled before use | doc assertion | `python -c "from pathlib import Path; text=Path('docs/pipeline_inventory.md').read_text(); assert all(s in text for s in ['Supported toolkit entry points','Manual diagnostics','Experimental scripts','Historical experiment tracks'])"` | ✅ | ⬜ pending |
| 1-02-01 | 02 | 1 | ENV-01, ENV-02 | T-01-03 | Optional dependency groups prevent accidental heavy installs | tooling | `uv lock --check` | ✅ | ⬜ pending |
| 1-02-02 | 02 | 1 | ENV-03, ENV-05, TEST-06 | T-01-04 | Pytest restricted to `tests/` | tooling/test | `uv run pytest --collect-only` | ✅ | ⬜ pending |
| 1-03-01 | 03 | 2 | ENV-04, TEST-07 | T-01-05 | Smoke CLI does not reveal secrets or load models by default | unit RED gate | `python -c "from pathlib import Path; text=Path('tests/test_smoke_environment.py').read_text(); assert all(s in text for s in ['test_list_outputs_all_checks','test_main_list_prints_checks','test_unknown_check_returns_nonzero','test_import_has_no_heavy_side_effects'])" && (uv run pytest tests/test_smoke_environment.py && exit 1 || exit 0)` | ✅ | ⬜ pending |
| 1-03-02 | 03 | 2 | ENV-04, TEST-07 | T-01-05 | Explicit checks only for GPU/model/OCR | smoke | `uv run python -m scripts.smoke_environment --list` | ✅ | ⬜ pending |
| 1-04-01 | 04 | 3 | TEST-06, TEST-07 | T-01-06 | Expensive diagnostics are opt-in and guarded | collect-only | `uv run pytest --collect-only` | ✅ | ⬜ pending |
| 1-04-02 | 04 | 3 | ENV-03, ENV-04, ENV-05, ENV-06 | T-01-07 | Command catalog separates local/SLURM/default/diagnostic paths | doc/tooling | `make -n test lint format smoke-imports smoke-cuda smoke-model-access smoke-ocr` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers Phase 1 planning needs. Plan 02 installs pytest/Ruff tooling; Plan 03 adds the first new test scaffold before the smoke CLI implementation.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CUDA, model access, PaddleOCR, and SLURM runtime viability on target machines | ENV-04, ENV-06, TEST-07 | Requires target GPU/cluster credentials, cache state, and scheduler access | Run documented smoke commands on local GPU and SLURM login/compute environments after Phase 1 execution. Treat failures as environment findings, not default CPU test failures. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers missing test references through Plan 03 TDD task.
- [x] No watch-mode flags.
- [x] Feedback latency < 60 seconds for default CPU-safe checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
