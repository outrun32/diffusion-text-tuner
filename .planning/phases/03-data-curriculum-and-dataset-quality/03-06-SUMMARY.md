---
phase: 03-data-curriculum-and-dataset-quality
plan: 06
subsystem: runtime-docs-command-surface
tags: [runtime-contracts, data-quality, command-surface, makefile, docs-tests, cpu-safe-tests]

requires:
  - phase: 03-data-curriculum-and-dataset-quality
    provides: Prompt curriculum, prompt validation/manifests, synthetic quality, selection artifacts, and source comparison from plans 03-01 through 03-05.
  - phase: 02-runtime-contracts-and-run-provenance
    provides: Runtime path resolution, artifact validators, manifest/preflight helpers, command catalog, and generated-artifact safety conventions.
provides:
  - Phase 3 runtime path keys and artifact validators for quality reports, dataset manifests, selection JSONL, preference pairs, contact sheets, and source comparison reports.
  - Command catalog, README links, and Makefile aliases for Phase 3 data curriculum and quality workflows.
  - Runtime contract documentation with Phase 3 schema names, producers, consumers, preflight hooks, and git-safety classifications.
affects: [phase-4-characterization-tests, phase-5-training-comparability, phase-6-evaluation-validity, thesis-provenance]

tech-stack:
  added: []
  patterns: [docs-drift-tests, shallow-json-schema-validation, cpu-safe-command-aliases, generated-artifact-safety]

key-files:
  created: [tests/test_data_quality_docs.py]
  modified: [src/runtime/paths.py, src/runtime/artifacts.py, docs/runtime_contracts.md, docs/commands.md, README.md, Makefile]

key-decisions:
  - "Keep Phase 3 validators shallow and CPU-safe by checking only JSON/JSONL presence, schema_version, and required fields rather than generated images, tensors, OCR, or model outputs."
  - "Expose Phase 3 workflows through docs/commands.md and Makefile aliases while preserving Phase 1 and Phase 2 command surfaces."
  - "Classify generated data-quality reports, contact sheets, selections, preference pairs, and comparison outputs as non-committable runtime artifacts by default."

patterns-established:
  - "Phase 3 docs drift tests assert command, README, Makefile, and runtime-contract strings so workflow discovery does not regress silently."
  - "Runtime artifact validation for Phase 3 accepts required-ready gates but remains model-free and suitable for preflight automation."

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07]

duration: 7min
completed: 2026-05-04T16:05:01Z
---

# Phase 3 Plan 06: Runtime Contract and Command Surface Summary

**Phase 3 data-quality workflows are path-resolved, shallow-validated, documented, and discoverable through CPU-safe command surfaces.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-04T15:58:26Z
- **Completed:** 2026-05-04T16:05:01Z
- **Tasks:** 3
- **Files modified:** 7 task files plus this summary and planning metadata

## Accomplishments

- Extended `resolve_stage_paths` with Phase 3 data-quality path families for prompt reports/manifests, synthetic reports/manifests/contact sheets, selected samples, preference pairs, and generated-vs-synthetic comparisons.
- Extended `validate_artifacts` with CPU-safe JSON/JSONL checks for `dataset-manifest/v1`, `prompt-quality/v1`, `synthetic-quality/v1`, `selected-samples/v1`, `preference-pairs/v1`, and `data-source-comparison/v1`.
- Added docs-drift tests that guard Phase 3 runtime contracts, command docs, Makefile aliases, README links, schema names, and generated-artifact safety wording.
- Published Phase 3 command examples and Makefile aliases for prompt config generation, prompt validation, synthetic inspection, SFT/DPO selection materialization, and source comparison without implying GPU/OCR/model work in CPU-safe defaults.
- Updated runtime contract docs with Phase 3 artifact producers, consumers, required fields, schema names, preflight hooks, and git-safety classifications.

## RED/GREEN Evidence

- **Task 1 RED:** `uv run pytest tests/test_data_quality_docs.py tests/test_runtime_artifacts.py -q` failed with missing `data_selection` path resolution and unsupported Phase 3 artifact stages after adding tests only.
- **Task 1 GREEN:** The same test target passed with 13 tests after extending `src/runtime/paths.py` and `src/runtime/artifacts.py`; Ruff also passed for the touched runtime/test files.
- **Task 2 RED:** `uv run pytest tests/test_data_quality_docs.py -q` failed because the Phase 3 command docs, README links, and Makefile aliases were absent.
- **Task 2 GREEN:** The same docs test target passed after updating `docs/commands.md`, `README.md`, and `Makefile`; Makefile dry-run printed the expected Phase 3 commands.
- **Task 3 RED:** `uv run pytest tests/test_data_quality_docs.py tests/test_runtime_docs.py tests/test_runtime_artifacts.py -q` failed because runtime contract docs lacked Phase 3 schema/producers/field coverage.
- **Task 3 GREEN:** The same test target passed with 21 tests after updating `docs/runtime_contracts.md`.

## Task Commits

Each TDD gate was committed atomically:

1. **Task 1 RED: Phase 3 runtime artifact contract tests** - `436a6d5` (`test`)
2. **Task 1 GREEN: Phase 3 runtime paths and artifact validators** - `5113037` (`feat`)
3. **Task 2 RED: Phase 3 command surface docs tests** - `5c5756f` (`test`)
4. **Task 2 GREEN: Phase 3 command docs, README links, and Makefile aliases** - `786eb4a` (`docs`)
5. **Task 3 RED: Phase 3 runtime docs coverage** - `abc3ac3` (`test`)
6. **Task 3 GREEN: Phase 3 runtime contract documentation** - `8a1524c` (`docs`)

**Plan metadata:** recorded in the final docs commit after this summary is created.

## Files Created/Modified

- `tests/test_data_quality_docs.py` - CPU-safe docs/runtime regression tests for Phase 3 paths, artifact schemas, command docs, Makefile aliases, README links, and runtime contract docs.
- `src/runtime/paths.py` - Adds Phase 3 path keys for prompt quality, synthetic quality, data selection, and source comparison artifacts.
- `src/runtime/artifacts.py` - Adds shallow Phase 3 JSON/JSONL validators and required-ready behavior for reports, manifests, selected samples, preference pairs, and source comparisons.
- `docs/commands.md` - Publishes Phase 3 command examples and generated-artifact safety guidance.
- `README.md` - Links Phase 3 data-quality docs and reiterates generated artifact safety.
- `Makefile` - Adds CPU-safe Phase 3 aliases while preserving existing Phase 1/2 aliases.
- `docs/runtime_contracts.md` - Documents Phase 3 artifact families, schemas, producers, consumers, required fields, preflight hooks, and git-safety classification.

## Decisions Made

- Used shallow schema validation for Phase 3 runtime validators to keep default preflight/model-free behavior and avoid inspecting generated images/tensors or invoking OCR/model stacks.
- Kept Phase 3 Makefile aliases as reviewable command surfaces that write to ignored runtime roots; users can override variables for local or SLURM workspaces.
- Documented selection and comparison outputs as generated runtime artifacts even though they are metadata, because they can include private prompt text, reward scores, and run-specific provenance.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed Ruff line-length issues in new runtime docs tests and validators**
- **Found during:** Task 1 (Extend runtime path and artifact contracts for Phase 3)
- **Issue:** Ruff flagged overlong lines in `src/runtime/artifacts.py` and `tests/test_data_quality_docs.py` after the GREEN implementation.
- **Fix:** Wrapped long validator calls and path assertions without changing behavior.
- **Files modified:** `src/runtime/artifacts.py`, `tests/test_data_quality_docs.py`
- **Verification:** `uv run --extra lint ruff check src/runtime/paths.py src/runtime/artifacts.py tests/test_data_quality_docs.py` passed.
- **Committed in:** `5113037`

**2. [Rule 3 - Blocking] Matched exact artifact-safety wording required by docs tests**
- **Found during:** Task 2 (Publish Phase 3 command and README surfaces)
- **Issue:** The command docs and README communicated generated-artifact safety, but the new drift test required the exact phrase `generated reports, images, tensors, contact sheets, selections, and comparisons`.
- **Fix:** Added the exact phrase while preserving the safety guidance.
- **Files modified:** `docs/commands.md`, `README.md`
- **Verification:** `uv run pytest tests/test_data_quality_docs.py -q` passed.
- **Committed in:** `786eb4a`

---

**Total deviations:** 2 auto-fixed blocking issues.
**Impact on plan:** Both were formatting/docs-test alignment fixes required for the planned verification gates; no scope expansion or generated data artifacts were introduced.

## Auth Gates

None.

## Known Stubs

None. Stub-pattern scan found no TODO/FIXME/placeholder/coming-soon/not-available markers in files created or modified by this plan. Empty lists/dicts in tests and validators are normal accumulators or expected report structures, not placeholder UI/data stubs.

## Threat Flags

None. The plan threat model covered command-doc execution surfaces and artifact validators. Changes remained local filesystem/docs/test surfaces, used shallow JSON/JSONL validation, and preserved generated-artifact non-committable guidance.

## Verification Results

- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_data_quality_docs.py tests/test_runtime_artifacts.py -q` — passed during Task 1 GREEN, 13 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_data_quality_docs.py -q` — passed during Task 2 GREEN, 7 tests.
- `make -n phase3-generate-prompts phase3-validate-prompts phase3-inspect-synthetic phase3-materialize-sft phase3-materialize-dpo phase3-compare-sources` — passed and printed CPU-safe command lines.
- `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_data_quality_docs.py tests/test_runtime_docs.py tests/test_runtime_artifacts.py -q` — passed, 21 tests.
- `PATH="/root/.local/bin:$PATH" uv run pytest` — passed, 120 CPU-safe tests.

## Issues Encountered

- `gsd-sdk` was unavailable under local `node_modules` and on `PATH`; state, roadmap, and requirement updates were prepared manually.
- `uv` is available only with `/root/.local/bin` added to `PATH` in this shell, so verification commands used `PATH="/root/.local/bin:$PATH"`.
- The worktree had unrelated pre-existing dirty and untracked files before execution; they were left untouched and excluded from all task commits.

## User Setup Required

None - no external service credentials, CUDA device, OCR package, model cache, or generated artifact is required for the docs/runtime tests.

## Next Phase Readiness

- Phase 3 now has complete DATA-01 through DATA-07 workflow implementation plus runtime contract, command, README, and Makefile discovery.
- Phase 4 CPU-safe characterization tests can rely on documented Phase 3 artifact schemas and command surfaces without adding GPU/OCR/model defaults.
- Phase 5/6 can consume materialized selection and comparison reports while tracing thesis claims back to exact manifests, configs, and hashes.

## Self-Check: PASSED

- Found created/modified task files: `src/runtime/paths.py`, `src/runtime/artifacts.py`, `docs/runtime_contracts.md`, `docs/commands.md`, `README.md`, `Makefile`, `tests/test_data_quality_docs.py`, and this summary.
- Found task commits `436a6d5`, `5113037`, `5c5756f`, `786eb4a`, `abc3ac3`, and `8a1524c` in git history.
- Required verification commands passed, including the final targeted Phase 3 docs/runtime suite and full CPU-safe pytest suite.

---
*Phase: 03-data-curriculum-and-dataset-quality*
*Completed: 2026-05-04T16:05:01Z*
