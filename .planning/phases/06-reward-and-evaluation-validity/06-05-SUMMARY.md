---
phase: 06-reward-and-evaluation-validity
plan: 05
subsystem: evaluation-diagnostics
tags: [reward-diagnostics, evaluation, vlm-ocr-disagreement, contact-sheets, cpu-safe-tests]

requires:
  - phase: 06-reward-and-evaluation-validity
    provides: Canonical reward score fields and product-score outputs from Plans 06-01 and 06-04
  - phase: 06-reward-and-evaluation-validity
    provides: Russian difficulty slices and gold diagnostic benchmark contracts from Plan 06-03
provides:
  - CPU-safe reward disagreement reports for recorded VLM/OCR/product score rows
  - False-positive and false-negative diagnostics linked to exact-match or gold expectations
  - Per-character confusion and per-slice disagreement summaries
  - Optional bounded PIL contact-sheet generation and CLI report writing
affects: [phase-6-thesis-outputs, reward-validity, evaluation-command-docs]

tech-stack:
  added: []
  patterns: [recorded-score-diagnostics, bounded-runtime-contact-sheets, deterministic-json-markdown-cli, docs-drift-tests]

key-files:
  created:
    - src/evaluation/diagnostics.py
    - scripts/analyze_reward_diagnostics.py
    - tests/test_reward_diagnostics.py
  modified:
    - docs/evaluation_diagnostics.md

key-decisions:
  - "Keep reward diagnostics model-free by analyzing only recorded score rows, optional gold metadata, and user-requested local images for bounded contact sheets."
  - "Classify false positives/false negatives from product-score thresholds against gold human labels when present, falling back to recorded exact-match fields."
  - "Treat malformed inputs as CLI errors while treating discovered disagreements, missing evidence, and confusions as normal diagnostic findings."

patterns-established:
  - "Diagnostics reports expose schema version, thresholds, record counts, missing-evidence counters, VLM/OCR correlation, false rows, character confusions, per-slice counts, and contact-sheet entries."
  - "CLI scripts run from the repository root or direct script path without importing FLUX, Qwen, PaddleOCR, CUDA, model weights, or OCR backends."
  - "Generated reports and contact sheets are runtime artifacts and remain outside git unless intentionally added later as tiny fixtures."

requirements-completed: [EVAL-05, EVAL-06, EVAL-07]

metrics:
  duration: 4min 11s
  completed: 2026-05-06T14:58:52Z
  tasks: 3
  files: 5
---

# Phase 6 Plan 05: Reward Disagreement Diagnostics Summary

**CPU-safe VLM/OCR/product-score disagreement diagnostics with false-row reports, per-character confusions, slice summaries, and bounded contact sheets from recorded outputs.**

## Performance

- **Duration:** 4min 11s
- **Started:** 2026-05-06T14:54:41Z
- **Completed:** 2026-05-06T14:58:52Z
- **Tasks:** 3
- **Files modified:** 5 including this summary

## Accomplishments

- Added `src.evaluation.diagnostics` with `analyze_reward_disagreement`, `format_diagnostics_markdown`, CSV/JSON/JSONL score loading, deterministic JSON/Markdown writers, missing-evidence accounting, Pearson VLM/OCR correlation, scatter summaries, false-positive/false-negative rows, per-character confusion counts, per-slice disagreement counts, and optional bounded PIL contact sheets.
- Added `scripts/analyze_reward_diagnostics.py` as a thin CPU-safe CLI supporting `--scores`, `--gold`, `--output-report`, `--markdown-summary`, `--contact-sheet`, threshold flags, and bounded contact-sheet limits.
- Added `tests/test_reward_diagnostics.py` with CPU-safe fixtures and temp-path image/report generation; tests use fakes/fixtures only and do not load FLUX, Qwen, PaddleOCR, CUDA, model weights, OCR engines, tensors, checkpoints, or logs.
- Extended `docs/evaluation_diagnostics.md` with reward disagreement concepts, CLI examples, gold benchmark linkage, contact-sheet fields, per-character/per-slice summaries, and generated-artifact safety guidance.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -x` failed as intended with `ModuleNotFoundError: No module named 'src.evaluation.diagnostics'` after writing reward diagnostic tests.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -q` passed with 5 tests after implementing the diagnostics module and CLI.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -q` failed as intended because `docs/evaluation_diagnostics.md` did not yet contain the required reward diagnostics CLI terms.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/diagnostics.py scripts/analyze_reward_diagnostics.py tests/test_reward_diagnostics.py` passed with 6 tests and Ruff clean.

## Task Commits

1. **Task 1 RED: Specify reward disagreement reports** - `8fe002f` (`test`)
2. **Task 2 GREEN: Implement diagnostics module and CLI** - `3d94011` (`feat`)
3. **Task 3 RED: Add diagnostics documentation drift assertions** - `d16cdca` (`test`)
4. **Task 3 GREEN: Extend diagnostics documentation** - `7e75a3c` (`docs`)

**Plan metadata:** recorded in final docs commit.

## Files Created/Modified

- `src/evaluation/diagnostics.py` - CPU-safe reward disagreement implementation, score-file parsing, false-row classification, correlation/scatter summaries, character confusion summaries, per-slice counts, contact-sheet generation, and report formatting.
- `scripts/analyze_reward_diagnostics.py` - Thin CLI for recorded score inputs, optional gold benchmarks, deterministic JSON/Markdown output, threshold flags, and bounded contact sheets.
- `tests/test_reward_diagnostics.py` - TDD coverage for report schema, missing evidence, VLM/OCR correlation, false positives/negatives, character confusions, slice counts, contact sheets, CLI behavior, and docs drift.
- `docs/evaluation_diagnostics.md` - Extended guide for reward disagreement diagnostics, CLI examples, gold linkage, contact-sheet fields, and artifact safety.
- `.planning/phases/06-reward-and-evaluation-validity/06-05-SUMMARY.md` - This execution summary.

## Decisions Made

- Used recorded `product_score` thresholds for diagnostic polarity and compared them against gold `human_label` values (`pass`/`fail`) when available, with recorded exact-match fields as the fallback.
- Kept the CLI nonzero behavior limited to malformed inputs; reward disagreements, missing evidence, and confusions are successful diagnostic outputs rather than command failures.
- Imported PIL only inside the user-requested contact-sheet path, keeping default report generation model-free and avoiding unnecessary image work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed direct script import path for the CLI**
- **Found during:** Task 2 (Implement diagnostics module and CLI)
- **Issue:** Running `python scripts/analyze_reward_diagnostics.py ...` from the test invoked the script with `scripts/` on `sys.path`, so `import src.evaluation.diagnostics` failed.
- **Fix:** Added a small repository-root insertion at script startup before importing the implementation module.
- **Files modified:** `scripts/analyze_reward_diagnostics.py`
- **Verification:** Targeted pytest passed afterward.
- **Committed in:** `3d94011`

**2. [Rule 3 - Blocking] Fixed plan-owned Ruff violations**
- **Found during:** Task 3 (Extend diagnostics documentation)
- **Issue:** Ruff flagged direct-script import ordering and several line-length violations in plan-owned files after implementation and docs tests were added.
- **Fix:** Added a targeted `# noqa: E402` for the intentional root-path bootstrap and wrapped long lines without changing behavior.
- **Files modified:** `src/evaluation/diagnostics.py`, `scripts/analyze_reward_diagnostics.py`
- **Verification:** Task 3 verification command passed with 6 tests and Ruff clean.
- **Committed in:** `7e75a3c`

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 3 blocking verification fix)
**Impact on plan:** Both fixes were limited to plan-owned files and required for the documented CLI and verification command to work. No scope expansion, model/OCR/CUDA loading, or generated runtime artifacts were introduced.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no `TODO`, `FIXME`, placeholder, coming-soon, or not-available markers in files created or modified by this plan. Empty list/dict initializers are runtime accumulators, not user-visible stubs.

## Threat Flags

None. The plan threat model covered score/gold input parsing and optional local-image contact sheets. Mitigations implemented here include required-field validation, CLI malformed-input errors, explicit missing-evidence counts, thresholds/source paths in reports, bounded contact-sheet limits, and runtime-output documentation.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -x` — RED failed as intended on missing `src.evaluation.diagnostics` before implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -q` — passed with 5 tests after diagnostics implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -q` — RED failed as intended on missing reward diagnostics docs terms before documentation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_diagnostics.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/evaluation/diagnostics.py scripts/analyze_reward_diagnostics.py tests/test_reward_diagnostics.py` — passed with 6 tests and Ruff clean.
- Final verification repeated the Task 3 command successfully; tests generate JSON/Markdown reports and contact sheets only under pytest `tmp_path`.

## TDD Gate Compliance

- RED gate commits exist: `8fe002f` and `d16cdca`.
- GREEN gate commits exist after their RED commits: `3d94011` and `7e75a3c`.
- No refactor-only commit was needed; lint cleanup was included in Task 3's GREEN commit after verification failures.

## Deferred Issues

- The worktree still contains unrelated pre-existing dirty and untracked files in training, configs, data, docs, scripts, and tests. They were left untouched and excluded from all plan commits.
- The GSD SDK CLI was unavailable in this environment (`gsd-sdk: command not found` and no local `node_modules/@gsd-build/sdk`), so state/roadmap/requirements updates were applied manually instead of through SDK query handlers.
- No GPU/model/OCR diagnostics were run, per plan constraints.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 6 thesis-output work can consume `reward-diagnostics/v1` JSON reports, Markdown summaries, contact-sheet entries, and per-slice/character findings from recorded score files.
- Phase 6 command-doc work can publish the `scripts/analyze_reward_diagnostics.py` CLI and generated-artifact safety policy without adding model/OCR/CUDA requirements.

## Self-Check: PASSED

- Found all created/modified task files: `src/evaluation/diagnostics.py`, `scripts/analyze_reward_diagnostics.py`, `tests/test_reward_diagnostics.py`, `docs/evaluation_diagnostics.md`, and this summary.
- Found task commits `8fe002f`, `3d94011`, `d16cdca`, and `7e75a3c` in git history.
- Required targeted pytest and Ruff verification commands passed; generated reports/contact sheets were confined to pytest `tmp_path`.

---
*Phase: 06-reward-and-evaluation-validity*  
*Completed: 2026-05-06T14:58:52Z*
