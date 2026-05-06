---
phase: 06-reward-and-evaluation-validity
plan: 04
subsystem: scoring-output-contracts
tags: [rewards, evaluation, scoring, product-score, metadata, cpu-safe-validation, tdd]

requires:
  - phase: 06-reward-and-evaluation-validity
    provides: Canonical RewardResult, ProductScoreFormula, compute_product_score, and build_score_metadata from Plan 06-01.
  - phase: 06-reward-and-evaluation-validity
    provides: Held-out evaluation manifest links and evaluation score output paths from Plan 06-02.
provides:
  - Canonical Phase 6 score CSV rows with VLM, OCR, CER, entropy, exact-match, character-level, product, missing-evidence, and manifest fields.
  - Canonical score `.schema.json` sidecars with formula metadata, scorer versions, thresholds, source manifests, and schema versions.
  - CPU-safe artifact validation for Phase 6 score CSV/JSONL files and sidecars.
  - Documentation and docs drift coverage for canonical score-file fields, commands, sidecars, and validation.
affects: [phase-6-reward-evaluation-validity, scoring-outputs, heldout-evaluation-scores, thesis-evidence-validation]

tech-stack:
  added: []
  patterns: [canonical-boundary-conversion, metadata-sidecar-validation, missing-evidence-accounting, cpu-safe-fixture-tests]

key-files:
  created:
    - tests/test_evaluation_scoring_outputs.py
  modified:
    - scripts/score_images.py
    - src/evaluation/evaluate_rewards.py
    - src/runtime/artifacts.py
    - docs/reward_evaluation.md

key-decisions:
  - "Preserve legacy score CSV fields while adding canonical Phase 6 fields so existing SFT/DPO consumers remain compatible."
  - "Use `compute_product_score` and `build_score_metadata` at score/evaluation boundaries instead of duplicating product formula logic."
  - "Validate score CSV/JSONL files and sidecars shallowly without opening generated images, tensors, CUDA devices, OCR engines, Qwen, or model weights."

patterns-established:
  - "Canonical score outputs include `product_score`, `missing_components`, `formula_complete`, `manifest_path`, exact-match, detection status, and character metrics."
  - "Score sidecars declare `reward-score-metadata/v1` plus `phase6-score-file/v1` or `phase6-score-jsonl/v1` file schema versions."
  - "Docs drift tests guard score field names, manifest-link flags, sidecar fields, and validation snippets."

requirements-completed: [EVAL-04]

metrics:
  duration: 4min 28s
  completed: 2026-05-06T14:52:50Z
  tasks: 3
  files: 6
---

# Phase 6 Plan 04: Canonical Scoring Outputs Summary

**Canonical scoring and evaluation outputs now carry reproducible OCR, VLM, product, exact-match, character-level, missing-evidence, manifest, and sidecar metadata contracts that can be validated CPU-safely before diagnostics or thesis reporting.**

## Performance

- **Duration:** 4 min 28 sec
- **Started:** 2026-05-06T14:48:22Z
- **Completed:** 2026-05-06T14:52:50Z
- **Tasks:** 3
- **Files modified:** 5 task files plus this summary and planning metadata

## Accomplishments

- Added `tests/test_evaluation_scoring_outputs.py` with CPU-safe fixtures covering canonical score rows, score sidecars, evaluation JSONL conversion, artifact validation acceptance, artifact validation rejection, and documentation drift.
- Updated `scripts/score_images.py` to build canonical Phase 6 score CSV rows through the Plan 06-01 product formula, including legacy-compatible `id/version/score/target_text` fields plus `sample_id`, `product_score`, VLM/OCR/CER/entropy evidence, detection status, exact match, character metrics, `missing_components`, `formula_complete`, `manifest_path`, JSON metrics, and threshold flags.
- Added score `.schema.json` sidecar creation for CSV scoring with formula name, weights, thresholds, scorer versions, source manifests, required fields, and `phase6-score-file/v1` schema metadata.
- Updated `src/evaluation/evaluate_rewards.py` to convert Qwen/PaddleOCR reward outputs into canonical JSONL fields and write matching metadata sidecars with `phase6-score-jsonl/v1`.
- Extended `src/runtime/artifacts.py` with `validate_artifacts("evaluation_scores", ...)` support for shallow, CPU-safe CSV/JSONL row and sidecar validation.
- Expanded `docs/reward_evaluation.md` with canonical score CSV/JSONL fields, sidecar schema fields, product formula metadata, manifest-link command examples, validation snippets, missing-evidence behavior, and runtime artifact safety guidance.

## RED/GREEN Evidence

- **Task 1 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py -x` failed as expected with `ImportError: cannot import name 'build_canonical_score_row'` after the scoring-output contract tests were written.
- **Task 2 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py tests/test_reward_wrapper_contracts.py -q` passed with 13 tests after wiring canonical score builders, evaluation conversion, sidecars, and validators.
- **Task 3 RED:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py -x` failed as expected on the docs drift assertion for missing score-file documentation.
- **Task 3 GREEN:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check scripts/score_images.py src/evaluation/evaluate_rewards.py src/runtime/artifacts.py tests/test_evaluation_scoring_outputs.py` passed with 6 tests and Ruff clean.
- **Final verification:** `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py tests/test_reward_wrapper_contracts.py -q` passed with 14 tests.

## Task Commits

1. **Task 1: Specify canonical scoring output contracts** - `5f8da5f` (`test`)
2. **Task 2: Wire score scripts and artifact validators to canonical interface** - `8adf527` (`feat`)
3. **Task 3 RED: Add scoring output docs assertion** - `57f75b8` (`test`)
4. **Task 3 GREEN: Document canonical score files** - `69bd279` (`docs`)

**Plan metadata:** committed separately after summary creation.

## Files Created/Modified

- `tests/test_evaluation_scoring_outputs.py` - CPU-safe score row, sidecar, JSONL conversion, validator, and docs drift tests using fakes and temporary files only.
- `scripts/score_images.py` - Canonical CSV row conversion, sidecar writer, manifest-link flags, and product-score use through `compute_product_score`.
- `src/evaluation/evaluate_rewards.py` - Canonical evaluation JSONL conversion, JSONL sidecar writer, and manifest-link flags.
- `src/runtime/artifacts.py` - `evaluation_scores` validation stage for Phase 6 score CSV/JSONL and `.schema.json` sidecars.
- `docs/reward_evaluation.md` - Score-file field, sidecar, command, validation, missing-evidence, and artifact-safety documentation.
- `.planning/phases/06-reward-and-evaluation-validity/06-04-SUMMARY.md` - This execution summary.

## Decisions Made

- Kept `id`, `version`, `score`, and `target_text` in score CSVs to avoid breaking existing training selection flows while adding canonical `sample_id` and Phase 6 fields for evaluation validity.
- Used `ProductScoreFormula`, `compute_product_score`, and `build_score_metadata` for scoring/evaluation boundary conversion so product formula and sidecar metadata remain shared with Plan 06-01.
- Treated missing VLM/OCR/CER/entropy/exact evidence as first-class score row data through `missing_components` and `formula_complete` rather than silently substituting zeros or hiding absent components.
- Kept artifact validation shallow: it validates CSV/JSONL field shape, finite numeric fields, JSON-encoded fields, sidecar schema metadata, formula metadata, and manifest links without inspecting generated artifacts or loading optional stacks.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed plan-owned Ruff issues**
- **Found during:** Task 3 verification
- **Issue:** Ruff flagged line length and several legacy lint issues in plan-owned `src/evaluation/evaluate_rewards.py` after the targeted Ruff command was introduced for the plan files.
- **Fix:** Wrapped long sidecar calls, removed an unused OCR confidence assignment, removed unnecessary read modes, removed f-string prefixes without placeholders, and added explicit `zip(..., strict=False)` in summary printing without changing runtime behavior.
- **Files modified:** `scripts/score_images.py`, `src/evaluation/evaluate_rewards.py`
- **Verification:** Task 3 pytest and Ruff command passed afterward.
- **Committed in:** `69bd279`

---

**Total deviations:** 1 auto-fixed blocking verification issue.
**Impact on plan:** The fix stayed within plan-owned files and did not add GPU/model/OCR execution, generated artifacts, or unrelated changes.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no `TODO`, `FIXME`, placeholder, coming-soon text, not-available placeholders, hardcoded empty UI data, or unwired mock behavior in plan-created/modified files.

## Threat Flags

None. The new score row conversion, sidecar metadata, and artifact validation surfaces match the plan threat model. Validation remains CPU-safe and shallow; no network endpoints, auth paths, schema migrations, generated-image inspection, tensor loading, CUDA calls, OCR/model initialization, or unplanned trust boundaries were introduced.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py -x` — RED failed as expected on missing `build_canonical_score_row` before implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py tests/test_reward_wrapper_contracts.py -q` — passed with 13 tests after Task 2 implementation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py -x` — RED failed as expected on missing docs phrases before documentation.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py -q && PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check scripts/score_images.py src/evaluation/evaluate_rewards.py src/runtime/artifacts.py tests/test_evaluation_scoring_outputs.py` — passed with 6 tests and Ruff clean.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_evaluation_scoring_outputs.py tests/test_reward_wrapper_contracts.py -q` — passed with 14 tests for final plan verification.

## Deferred Issues

- Unrelated pre-existing dirty and untracked files remain in training modules, configs, data roots, thesis docs, scripts, loss helpers, and loss tests. They were present before execution and were left untouched/excluded from all Plan 06-04 commits.
- The GSD SDK CLI was unavailable in this checkout (`gsd-sdk: command not found` and no local `node_modules/@gsd-build/sdk`), so planning state files were updated directly instead of through SDK query handlers.
- No FLUX/Qwen/PaddleOCR/CUDA/model-weight diagnostics were run, per plan and user constraints.

## TDD Gate Compliance

- RED gate commits exist: `5f8da5f` and `57f75b8`.
- GREEN gate commits exist after RED commits: `8adf527` for implementation and `69bd279` for docs/lint cleanup.
- No separate refactor-only commit was needed.

## Next Phase Readiness

- Phase 6 Plan 05 can consume complete/missing-aware score files through `validate_artifacts("evaluation_scores", ...)` before building reward disagreement diagnostics.
- Thesis-output plans can rely on sidecar metadata for formula/scorer/threshold provenance and manifest traceability.

## Self-Check: PASSED

- Found created/modified files: `tests/test_evaluation_scoring_outputs.py`, `scripts/score_images.py`, `src/evaluation/evaluate_rewards.py`, `src/runtime/artifacts.py`, `docs/reward_evaluation.md`, and this summary.
- Found task commits in git history: `5f8da5f`, `8adf527`, `57f75b8`, and `69bd279`.
- Required targeted pytest and Ruff verification commands passed.

---
*Phase: 06-reward-and-evaluation-validity*  
*Completed: 2026-05-06T14:52:50Z*
