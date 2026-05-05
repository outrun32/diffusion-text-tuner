---
phase: 05-training-objective-and-pipeline-comparability
plan: 04
subsystem: training-config-validation
tags: [training, config-validation, sft, dpo, masked-sft, snapshots, docs, cpu-safe, tdd]

requires:
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Explicit SFT selection and DPO pair-construction mode names from Plan 05-01.
  - phase: 05-training-objective-and-pipeline-comparability
    provides: Training comparability snapshot expectations from Plan 05-03.
provides:
  - Explicit SFT selection choice fields in dataclasses, validated runtime configs, and snapshots.
  - Explicit DPO pair-construction choice fields in dataclasses, validated runtime configs, and snapshots.
  - Masked-SFT loss, LoRA, dataset, and eval-suite snapshot/doc coverage.
  - Experiment README choice-field tables guarded by CPU-safe docs assertions.
affects: [phase-5-shared-training-utilities, phase-5-integrated-comparison-docs, phase-6-evaluation-validity]

tech-stack:
  added: []
  patterns: [pydantic-literal-validation, dataclass-snapshot-provenance, secret-safe-errors, docs-drift-tests]

key-files:
  created:
    - .planning/phases/05-training-objective-and-pipeline-comparability/deferred-items.md
  modified:
    - src/training/config.py
    - src/runtime/config_io.py
    - tests/test_runtime_config_io.py
    - tests/test_characterization_config_artifacts.py
    - configs/experiments/sft/README.md
    - configs/experiments/dpo/README.md
    - configs/experiments/masked_sft/README.md

key-decisions:
  - "Keep backwards-compatible SFT and DPO defaults as `threshold` and `best_vs_worst` while allowing comparison-grade explicit modes from Plan 05-01."
  - "Validate materialized selected-sample and preference-pair paths through the existing CPU-safe path policy before trainer launch."
  - "Track pre-existing dirty Ruff failures as deferred instead of touching unrelated user edits in `src/training/config.py`."

patterns-established:
  - "Config snapshots expose explicit choice fields by relying on dataclass fields and immutable `asdict` snapshot generation."
  - "Invalid mode values use Pydantic `Literal` validation so error messages include field context without echoing raw secret-like inputs."
  - "Experiment-family README tables are guarded by focused docs assertions in runtime config tests."

requirements-completed: [TRN-02, TRN-03, TRN-04]

duration: 3min 32s
completed: 2026-05-05T19:21:46Z
---

# Phase 5 Plan 04: Explicit Training Config Choice Summary

**SFT, DPO, and masked-SFT configs now validate comparison-grade choice fields and expose them in manifest-ready snapshots.**

## Performance

- **Duration:** 3 min 32 sec
- **Started:** 2026-05-05T19:18:14Z
- **Completed:** 2026-05-05T19:21:46Z
- **Tasks:** 3
- **Files modified:** 7 plan files plus this summary and deferred-items note

## Accomplishments

- Added failing then passing CPU-safe tests for SFT selection fields, DPO pair-construction fields, masked-SFT snapshot fields, and secret-safe invalid mode errors.
- Added `selection_mode`, `selected_samples_path`, `score_column`, `hard_negative_threshold`, and `sample_weighting` to `SFTConfig`, with runtime validation and snapshot inclusion.
- Added `pair_construction_mode`, `preference_pairs_path`, `score_column`, `ambiguity_margin`, and `pair_weighting` to `DPOConfig`, with runtime validation and snapshot inclusion.
- Extended runtime path validation to `selected_samples_path` and `preference_pairs_path` while keeping unknown fields forbidden.
- Published SFT, DPO, and masked-SFT experiment README field tables and guarded the required field strings with a docs assertion.

## Task Commits

Each task was committed atomically where possible:

1. **Task 1 RED: Specify explicit config fields and validation** - `55eb657` (test)
2. **Task 2 GREEN: Implement config dataclass and validation fields** - `796477a` (feat)
3. **Task 3 RED: Guard experiment config choice docs** - `d569bfa` (test)
4. **Task 3 GREEN: Document experiment config choice fields** - `677d70f` (docs)

**Plan metadata:** pending final documentation commit.

## Files Created/Modified

- `src/training/config.py` - Adds explicit SFT and DPO comparison-choice dataclass fields with backwards-compatible defaults; committed only the 05-04 field hunks and left pre-existing dirty edits unstaged.
- `src/runtime/config_io.py` - Adds Literal validation for explicit SFT/DPO modes and weighting labels, validates materialized selection/pair paths, and returns updated dataclasses for snapshots.
- `tests/test_runtime_config_io.py` - Adds CPU-safe snapshot, invalid-mode, and docs drift assertions for explicit training choices.
- `tests/test_characterization_config_artifacts.py` - Adds characterization coverage proving explicit SFT, DPO, and masked-SFT choices appear in snapshots.
- `configs/experiments/sft/README.md` - Documents SFT selection choice fields and allowed modes.
- `configs/experiments/dpo/README.md` - Documents DPO pair-construction/objective choice fields and allowed modes.
- `configs/experiments/masked_sft/README.md` - Documents masked-SFT loss, LoRA, dataset, and evaluation-suite fields.
- `.planning/phases/05-training-objective-and-pipeline-comparability/deferred-items.md` - Records the out-of-scope pre-existing Ruff blocker in the dirty worktree.

## Decisions Made

- Preserved the existing default SFT and DPO behavior by defaulting `selection_mode` to `threshold` and `pair_construction_mode` to `best_vs_worst`.
- Used existing `validate_path_policy` for materialized selected-sample and preference-pair paths rather than adding artifact existence checks, keeping validation CPU-safe.
- Did not touch or commit unrelated pre-existing dirty edits in `src/training/config.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, or untracked experiment artifacts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected masked-SFT LoRA expectation in the new characterization test**
- **Found during:** Task 2 (Implement config dataclass and validation fields)
- **Issue:** The new test hardcoded `joint_attn_r == 16`, but the committed `configs/masked_sft.json` fixture uses `joint_attn_r == 32`.
- **Fix:** Asserted that the snapshot matches the fixture payload's `lora.joint_attn_r` value instead of a stale default.
- **Files modified:** `tests/test_characterization_config_artifacts.py`
- **Verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py tests/test_characterization_config_artifacts.py -q` passed with 35 tests before docs assertions and 36 tests after docs assertions.
- **Committed in:** `796477a`

---

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** The fix kept the test aligned with committed config fixtures and did not expand scope.

## Issues Encountered

- The worktree had unrelated pre-existing dirty and untracked files before execution. Task commits staged only plan-related hunks/files; unrelated files were left untouched.
- The plan's Ruff command fails on the current dirty worktree because pre-existing uncommitted prompt strings in `src/training/config.py` exceed line length. Per scope-boundary rules, this was documented in `deferred-items.md` instead of modifying unrelated user edits.
- Local GSD SDK state queries returned no output in this checkout, so state, roadmap, and requirements updates were prepared by direct file edits.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern matches were optional path defaults such as `selected_samples_path: str | None = None`, `preference_pairs_path: str | None = None`, and existing optional resume/eval fields; these do not render placeholder UI or replace required data sources.

## Threat Flags

None. This plan added CPU-safe local config validation and docs only; no new network endpoint, auth path, file-content reader beyond config JSON loading, or schema migration was introduced outside the plan threat model.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py tests/test_characterization_config_artifacts.py -x` — **failed as expected for RED** after Task 1 because `_SFTModel` rejected the new explicit SFT fields as forbidden extras.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_runtime_config_io.py tests/test_characterization_config_artifacts.py -q` — **passed** (`36 passed`).
- `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/config.py src/runtime/config_io.py tests/test_runtime_config_io.py tests/test_characterization_config_artifacts.py` — **blocked by pre-existing dirty worktree edits** in `src/training/config.py` with seven E501 line-length errors outside the committed 05-04 hunks.

## TDD Gate Compliance

- RED gate: `55eb657` added failing explicit config choice tests; targeted pytest failed on forbidden extra fields before implementation.
- GREEN gate: `796477a` added dataclass/runtime validation fields and targeted pytest passed.
- Task 3 RED gate: `d569bfa` added docs drift assertions; targeted pytest failed because experiment READMEs lacked the exact fields.
- Task 3 GREEN gate: `677d70f` added the README tables and targeted pytest passed.

## Deferred Issues

- Pre-existing dirty `src/training/config.py` prompt string line-length violations should be handled by the owner of those uncommitted changes or in a dedicated cleanup plan. They currently block the exact Ruff command when run against the dirty worktree.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 05-05 can build shared training utilities knowing explicit SFT/DPO/masked-SFT choice fields load through the runtime config path and snapshot into manifests.
- Plan 05-06 can publish integrated training-run comparison docs using the exact field names now guarded by tests.

## Self-Check: PASSED

- Found modified task files in the worktree/history: `src/training/config.py`, `src/runtime/config_io.py`, `tests/test_runtime_config_io.py`, `tests/test_characterization_config_artifacts.py`, and the three experiment README files.
- Found task commits `55eb657`, `796477a`, `d569bfa`, and `677d70f` in git history.
- Confirmed targeted pytest passes with 36 tests; Ruff blocker is documented as an out-of-scope pre-existing dirty-worktree issue.

---
*Phase: 05-training-objective-and-pipeline-comparability*
*Completed: 2026-05-05T19:21:46Z*
