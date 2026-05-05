# Deferred Items

## Phase 5 Plan 05-04

- Pre-existing dirty edits in `src/training/config.py` include long prompt string literals that make the plan's Ruff command fail with E501 line-length errors. These lines were present before this plan execution and are not part of the committed 05-04 changes, so they were left untouched per worktree-safety constraints.
