---
phase: 05-training-objective-and-pipeline-comparability
plan: 01
subsystem: training-data-selection
tags: [selection-artifacts, sft, dpo, jsonl, provenance, cpu-safe-tests, tdd]

requires:
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Versioned selected-sample and preference-pair artifacts plus CLI/docs baseline.
  - phase: 04-cpu-safe-characterization-tests
    provides: CPU-safe dataset/selection and DPO winner/loser characterization before trainer comparability.
provides:
  - Explicit SFT selection modes `threshold`, `top_k_per_prompt`, `score_weighted`, and `hard_positive` in `src.training.selection`.
  - Explicit DPO pair modes `best_vs_worst`, `all_separated_pairs`, `margin_weighted`, and `ambiguity_filtered` with strict winner-over-loser semantics.
  - CLI support for exact mode names and hard-positive `--hard-negative-threshold` materialization.
  - User documentation and CPU-safe tests for comparison-grade selection artifacts.
affects: [phase-5-config-choice-snapshots, phase-5-run-comparability, phase-5-command-docs]

tech-stack:
  added: []
  patterns: [pure-csv-jsonl-selection, explicit-mode-artifacts, source-hash-provenance, thin-cli, tdd]

key-files:
  created: []
  modified:
    - src/training/selection.py
    - scripts/materialize_training_data.py
    - tests/test_training_selection_artifacts.py
    - docs/data_selection.md

key-decisions:
  - "Preserve current default semantics by keeping `threshold` SFT and `best_vs_worst` DPO as the default modes while naming them explicitly in summaries and rows."
  - "Keep all selection materialization CPU-safe and metadata-only; generated JSONL/manifests remain runtime artifacts unless intentionally promoted as tiny fixtures."
  - "Reject equal-score DPO candidates in every pair mode so winner/loser labels remain strictly score-ordered."

patterns-established:
  - "Weighted modes add normalized `sample_weight`/`pair_weight` fields rounded to 12 decimals for deterministic artifact diffs."
  - "Hard-positive SFT records `hard_negative_threshold` in summaries so prompt-level filtering is reproducible."
  - "Ambiguity-filtered DPO checks best-vs-second-best margin separately from best-vs-worst pair margin."

requirements-completed: [TRN-02, TRN-03]

duration: 5min 16s
completed: 2026-05-05T19:02:19Z
---

# Phase 5 Plan 01: Explicit Selection Mode Summary

**Named SFT sample-selection and DPO pair-construction artifacts with deterministic weights, strict winner/loser semantics, CLI flags, tests, and docs.**

## Performance

- **Duration:** 5 min 16 sec
- **Started:** 2026-05-05T18:57:03Z
- **Completed:** 2026-05-05T19:02:19Z
- **Tasks:** 3
- **Files modified:** 4 task files plus this summary

## Accomplishments

- Added explicit SFT modes: `threshold`, `top_k_per_prompt`, `score_weighted`, and `hard_positive`, with summaries recording mode, thresholds, source score hash, counts, and filtering stats.
- Added explicit DPO modes: `best_vs_worst`, `all_separated_pairs`, `margin_weighted`, and `ambiguity_filtered`, preserving strict winner-over-loser behavior and deterministic `(prompt_id, winner_version, loser_version)` ordering.
- Exposed the SFT hard-positive `--hard-negative-threshold` CLI flag and documented exact command examples for every mode.
- Expanded CPU-safe tests from 9 to 16 targeted tests without introducing CUDA, model, OCR, image, or tensor dependencies.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -x` failed on `score_weighted` with `ValueError: mode must be one of: threshold, top_k_per_prompt` after adding SFT mode tests.
- **Task 2 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` failed on five missing SFT/DPO explicit modes after adding DPO mode tests.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` passed with 14 tests after implementing explicit SFT/DPO modes.
- **Task 3 RED:** The targeted pytest command failed because the CLI did not recognize `--hard-negative-threshold` and docs lacked the new mode sections.
- **Task 3 GREEN:** The plan verification command passed with 16 tests and Ruff clean after adding the CLI flag and docs.

## Task Commits

1. **Task 1 RED: Specify explicit SFT selection mode behavior** - `73a7b67` (`test`)
2. **Task 2 RED: Specify explicit DPO pair construction behavior** - `1864652` (`test`)
3. **Task 2 GREEN: Implement explicit SFT and DPO mode materialization** - `b48fe2c` (`feat`)
4. **Task 3 RED: Expose mode flags through CLI and docs** - `9be8b7d` (`test`)
5. **Task 3 GREEN: CLI flags and mode documentation** - `5d921ab` (`feat`)

**Plan metadata:** recorded in the final docs commit after this summary is created.

## Files Created/Modified

- `src/training/selection.py` - Adds explicit SFT/DPO mode validation, deterministic selection/pair construction, normalized mode weights, strict equal-score rejection, and mode-specific summaries.
- `scripts/materialize_training_data.py` - Adds `--hard-negative-threshold` and forwards exact `--mode` values to selection helpers.
- `tests/test_training_selection_artifacts.py` - Adds tiny CSV fixture tests for SFT modes, DPO modes, CLI mode forwarding, docs drift, provenance fields, and weighted rows.
- `docs/data_selection.md` - Documents exact SFT/DPO mode names, commands, generated-artifact safety, summaries, and filtering counters.

## Decisions Made

- Kept default `threshold` SFT and `best_vs_worst` DPO behavior as the backwards-compatible default while making those names explicit in artifacts.
- Used metadata-only CSV/JSONL operations and pytest `tmp_path` fixtures to keep the plan CPU-safe and model-download-free.
- Added optional weighted fields only for weighted modes so existing unweighted artifact rows remain compatible.
- Counted DPO ambiguity for `ambiguity_filtered` using best-vs-second-best margin, while preserving best-vs-worst margin gating for default pair construction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected an impossible DPO all-separated fixture expectation**
- **Found during:** Task 2 GREEN verification.
- **Issue:** The new test fixture expected one below-margin candidate even though every strict non-equal candidate met `margin >= 0.3`; the only rejected candidate was equal-score.
- **Fix:** Updated the expected `pairs_below_margin` counter to `0` so the test matched strict winner-over-loser semantics.
- **Files modified:** `tests/test_training_selection_artifacts.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` passed with 14 tests.
- **Committed in:** `b48fe2c`

**2. [Rule 3 - Blocking] Fixed Ruff line-length failure**
- **Found during:** Task 3 verification.
- **Issue:** Ruff rejected a 101-character sorted-pair line in `src/training/selection.py`.
- **Fix:** Wrapped the deterministic pair sorting call across multiple lines.
- **Files modified:** `src/training/selection.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/selection.py scripts/materialize_training_data.py tests/test_training_selection_artifacts.py` passed.
- **Committed in:** `5d921ab`

---

**Total deviations:** 2 auto-fixed issues (1 bug, 1 blocking lint issue).
**Impact on plan:** Both fixes were necessary for correct tests and required verification; no scope expansion beyond explicit mode materialization.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan.

## Threat Flags

None. The new CSV parsing, CLI filesystem writes, summaries, source hashes, strict numeric score/version parsing, and generated-artifact safety guidance are within the plan threat model and covered by tests/docs.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -x` — failed as expected for Task 1 RED, then passed after implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` — passed, 16 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/selection.py scripts/materialize_training_data.py tests/test_training_selection_artifacts.py` — passed.

## Issues Encountered

- The worktree contained unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all commits.
- Local `gsd-sdk` initialization/state queries returned no output in this checkout, so state/roadmap/requirements synchronization was prepared by direct file edits rather than SDK handlers.
- `uv` is available with `/root/.local/bin` added to `PATH`, so verification commands used the plan's explicit PATH prefix.

## User Setup Required

None - no external service configuration required.

## Next Plan Readiness

- Plan 05-02 can compare run manifests knowing selection artifacts now carry explicit mode names, weights/margins, thresholds, counts, source score hashes, and filtering stats.
- Plan 05-04 can wire these exact SFT/DPO mode names into training config dataclasses and snapshots without changing default trainer CSV semantics.
- Plan 05-06 can publish integrated command docs using the exact CLI examples added here.

## Self-Check: PASSED

- Found all created/modified task files: `src/training/selection.py`, `scripts/materialize_training_data.py`, `tests/test_training_selection_artifacts.py`, `docs/data_selection.md`, and this summary.
- Found task commits `73a7b67`, `1864652`, `b48fe2c`, `9be8b7d`, and `5d921ab` in git history.
- Required verification commands passed, and generated test artifacts stayed under pytest temporary directories rather than git.

---
*Phase: 05-training-objective-and-pipeline-comparability*
*Completed: 2026-05-05T19:02:19Z*
