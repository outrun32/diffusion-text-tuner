---
phase: 03-data-curriculum-and-dataset-quality
plan: 04
subsystem: training-data-selection
tags: [selection-artifacts, sft, dpo, jsonl, provenance, cpu-safe-tests]

requires:
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Runtime artifact contracts, generated-artifact safety guidance, and CPU-safe pytest tooling.
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Dataset manifest/source-hashing conventions and synthetic/prompt data-quality context from plans 03-01 through 03-03.
provides:
  - Deterministic selected SFT sample materialization in `src.training.selection`
  - Deterministic DPO best-vs-worst preference pair materialization with explicit winner/loser semantics
  - `scripts/materialize_training_data.py` CLI for `selected_samples.jsonl` and `preference_pairs.jsonl`
  - User documentation for selection schemas, manifests, default equivalence, and artifact safety
affects: [phase-3-source-comparison, phase-3-runtime-docs, phase-4-dataset-tests, phase-5-training-comparability]

tech-stack:
  added: []
  patterns: [pure-csv-jsonl-selection, stable-artifact-ids, source-hash-provenance, thin-cli, tdd]

key-files:
  created:
    - src/training/selection.py
    - scripts/materialize_training_data.py
    - tests/test_training_selection_artifacts.py
    - docs/data_selection.md
  modified: []

key-decisions:
  - "Preserve current trainer behavior by adding optional materialized-manifest paths without changing SFTDataset or DPODataset loaders in this plan."
  - "Use deterministic JSONL metadata artifacts with stable IDs and score CSV hashes rather than committing generated images, tensors, or training outputs."
  - "Reject equal-score or ambiguity-margin DPO pairs so winner/loser labels remain strictly score-ordered."

patterns-established:
  - "Selection artifacts record schema version, score column, mode, source path/hash, thresholds, counts, and per-row stable IDs."
  - "The CLI prints the same deterministic summary JSON that it can write as a manifest sidecar."
  - "CPU-safe tests use tiny temporary CSV fixtures and never load CUDA, FLUX, Qwen, PaddleOCR, tensors, or generated images."

requirements-completed: [DATA-04, DATA-06]

duration: 5min
completed: 2026-05-04T15:46:54Z
---

# Phase 3 Plan 04: Training Selection Artifact Summary

**Versioned SFT selected-sample and DPO preference-pair JSONL artifacts with score provenance, stable IDs, and a CPU-safe materialization CLI.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-04T15:41:30Z
- **Completed:** 2026-05-04T15:46:54Z
- **Tasks:** 3
- **Files modified:** 4 task files plus this summary and planning metadata

## Accomplishments

- Added `src.training.selection` with `materialize_sft_samples(...)` and `materialize_dpo_pairs(...)` for deterministic JSONL artifacts from score CSV files.
- Preserved default equivalence with current in-constructor selection behavior: SFT threshold filtering on `score`, and DPO best-vs-worst pairs gated by winner threshold and score margin.
- Added stable per-row IDs, schema versions, source score CSV SHA-256 hashes, score columns, threshold/margin settings, counts, and filtering stats.
- Added `scripts/materialize_training_data.py` so users can materialize `selected_samples.jsonl` or `preference_pairs.jsonl` and optional summary manifests before training.
- Documented selection artifact schemas, command examples, runtime-contract links, and generated-artifact safety in `docs/data_selection.md`.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` failed with `ModuleNotFoundError: No module named 'src.training.selection'` after adding SFT materialization tests only.
- **Task 1 GREEN:** The same test command passed with 3 tests after implementing `materialize_sft_samples(...)`.
- **Task 2 RED:** The same test command failed with `ImportError: cannot import name 'materialize_dpo_pairs'` after adding DPO pair tests only.
- **Task 2 GREEN:** The same test command passed with 6 tests after implementing DPO pair materialization and margin rounding.
- **Task 3 RED:** The same test command failed because `scripts.materialize_training_data` and `docs/data_selection.md` were missing after adding CLI/docs tests.
- **Task 3 GREEN:** The same test command passed with 9 tests after adding the CLI and documentation.

## Task Commits

1. **Task 1 RED: SFT selection artifact tests** - `b43451a` (`test`)
2. **Task 1 GREEN: SFT selected-sample materialization** - `7ddf054` (`feat`)
3. **Task 2 RED: DPO preference pair tests** - `6b8c9be` (`test`)
4. **Task 2 GREEN: DPO preference pair materialization** - `0d70025` (`feat`)
5. **Task 3 RED: CLI and docs tests** - `ac4825f` (`test`)
6. **Task 3 GREEN: CLI and docs implementation** - `0e34764` (`feat`)

**Plan metadata:** recorded in the final docs commit after this summary is created.

## Files Created/Modified

- `src/training/selection.py` - Pure CSV/JSONL helper module for SFT selected samples and DPO preference pairs with validation, source hashing, stable IDs, and summaries.
- `scripts/materialize_training_data.py` - Thin CLI for `--kind sft|dpo`, score column, threshold, margin, output path/dir, manifest, and JSON stdout.
- `tests/test_training_selection_artifacts.py` - CPU-safe TDD coverage for selection semantics, provenance fields, validation, CLI outputs, manifests, docs, and deterministic ordering.
- `docs/data_selection.md` - User-facing schema, command, runtime-contract, config-connection, and generated-artifact safety guide.

## Decisions Made

- Kept SFT and DPO trainer/dataset loaders unchanged because the plan explicitly creates artifacts and docs only; future plans can optionally consume materialized JSONL paths.
- Used source score CSV hashes directly in every JSONL row and summary manifest so artifacts can be traced to the exact scoring output.
- Treated DPO equal-score pairs as ambiguous even when `--margin 0.0`, satisfying the threat-model requirement that winners are strictly higher scoring than losers.
- Wrote CLI manifests as deterministic summary JSON sidecars rather than full run manifests, because this plan owns selection artifacts while run manifests remain the broader Phase 2 provenance layer.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Rounded DPO pair margins for stable artifact semantics**
- **Found during:** Task 2 GREEN verification.
- **Issue:** Python float subtraction serialized `0.8 - 0.2` as `0.6000000000000001`, which made pair margins less inspectable and caused the deterministic artifact test to fail.
- **Fix:** Rounded materialized DPO margins to 12 decimal places while preserving numeric threshold comparisons.
- **Files modified:** `src/training/selection.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` passed.
- **Committed in:** `0d70025`

---

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** The fix improves deterministic artifact readability without changing selection scope or trainer behavior.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scans found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan. Internal empty lists and `None` CLI defaults do not feed UI rendering and are not stubs.

## Threat Flags

None. The new score-CSV parsing, selection summaries, and DPO winner/loser semantics were covered by the plan threat model and mitigated through required-column validation, numeric version/score parsing, source SHA-256 hashes, count summaries, and strict winner-over-loser tests.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_training_selection_artifacts.py -q` — passed, 9 tests.
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/selection.py scripts/materialize_training_data.py tests/test_training_selection_artifacts.py` — passed.
- Temporary CLI smoke under `/tmp/opencode/selection-smoke` — passed; generated one `preference-pairs/v1` row with winner version 2, loser version 1, and manifest `pair_count: 1`.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirements updates were prepared manually.
- `uv` is available only with `/root/.local/bin` added to `PATH` in this shell, so verification commands used `PATH="/root/.local/bin:$PATH"`.
- The worktree had unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all commits.

## User Setup Required

None - no external service configuration required.

## Next Plan Readiness

- Plan 03-05 can compare generated reward-filtered selections against synthetic masked-SFT reports using `selected_samples.jsonl`, `preference_pairs.jsonl`, and their summary manifests.
- Plan 03-06 can publish command aliases and runtime contract updates for the new selection CLI and schema names.
- Phase 4 dataset/collator characterization can reuse these tiny CSV/JSONL fixtures to lock selection semantics before loader refactors.

## Self-Check: PASSED

- Found all created/modified task files: `src/training/selection.py`, `scripts/materialize_training_data.py`, `tests/test_training_selection_artifacts.py`, `docs/data_selection.md`, and this summary.
- Found task commits `b43451a`, `7ddf054`, `6b8c9be`, `0d70025`, `ac4825f`, and `0e34764` in git history.
- Required verification commands passed, and generated smoke outputs stayed under `/tmp/opencode` rather than git.

---
*Phase: 03-data-curriculum-and-dataset-quality*
*Completed: 2026-05-04T15:46:54Z*
