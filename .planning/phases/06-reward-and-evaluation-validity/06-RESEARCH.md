# Phase 6 Research: Reward and Evaluation Validity

**Phase:** 06-reward-and-evaluation-validity  
**Goal:** Users can trust reward scores, held-out evaluations, diagnostic reports, and thesis outputs as comparable evidence tied back to exact runs.  
**Discovery level:** Level 1 / project-specific verification. No new external dependency is required; the work uses existing Python 3.11, pytest, Ruff, PIL, torch where already used, and optional runtime Qwen/PaddleOCR stacks behind explicit scoring commands.

## Inputs Reviewed

- `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/REQUIREMENTS.md`
- `.planning/research/SUMMARY.md`
- `.planning/codebase/ARCHITECTURE.md`, `CONVENTIONS.md`, `TESTING.md`, `STACK.md`, `INTEGRATIONS.md`
- Phase 3 summaries for dataset manifests and synthetic quality reports
- Phase 4 summaries for fake reward wrapper tests and characterization command docs
- Phase 5 summaries for run manifest diffing, training comparability, explicit config choices, shared trainer seams, and integrated comparison command docs
- Current source interfaces in `src/training/rewards.py`, `scripts/score_images.py`, `src/evaluation/evaluate_rewards.py`, `src/runtime/manifests.py`, and `src/runtime/artifacts.py`

## Findings

1. Reward logic is still split between `src/training/rewards.py`, `scripts/score_images.py`, and `src/evaluation/evaluate_rewards.py`. Phase 4 made the training reward wrappers import-safe and fake-testable, so Phase 6 can add a canonical pure reward result interface and then wire scoring/evaluation scripts to it without loading Qwen/PaddleOCR during default tests.
2. Product score output needs an explicit formula, component metadata, thresholds, scorer versions, schema metadata, and manifest links. The existing score CSV only guarantees `id`, `version`, `score`, and `target_text`, with optional OCR/VLM columns. Phase 6 should add schema sidecars/report helpers instead of changing generated image/tensor artifacts.
3. Held-out evaluation should be a local file-backed harness: fixed prompt JSONL, fixed seeds, fixed inference settings, baseline/trained LoRA labels, generated output paths, score output paths, and run manifest links. Default tests should validate the plan/config/report contracts only; actual FLUX/Qwen/PaddleOCR execution remains explicit runtime work.
4. Russian text difficulty slicing can reuse prompt metadata fields and target text analysis. Useful CPU-safe slices include rare Cyrillic, word length, phrase length, digits, punctuation, mixed case, multiline, font/style, and scene/background where metadata is present.
5. Reward disagreement reports should be generated from recorded score files and metadata: VLM-vs-OCR scatter tables, correlation summaries, false-positive/false-negative rows, contact sheet manifests, and per-character confusion summaries. Contact sheet generation can be optional and PIL-only.
6. Gold diagnostic validation can be a small JSONL schema for hand-labeled/gold examples with expected text, labels, and metric expectations. Tests should use tiny committed fixtures and never require real model/OCR calls.
7. Thesis-ready outputs should be built from recorded manifests/reports only. Tables/plots/contact-sheet manifests must include source manifest paths, artifact paths, schema versions, and git/config provenance so RUN-05 is satisfied.

## Architecture Pattern

- Add pure, import-safe modules under `src/evaluation/` for contracts, held-out config/report planning, slices, diagnostics, gold checks, and thesis output manifests.
- Keep Qwen/PaddleOCR/FLUX imports inside runtime scorer/generation branches only.
- Extend `scripts/score_images.py` and `src/evaluation/evaluate_rewards.py` through canonical result conversion rather than duplicating formula logic.
- Extend `src/runtime/artifacts.py` with CPU-safe validation for Phase 6 JSON/JSONL/CSV outputs.
- Publish commands through `docs/commands.md`, `README.md`, and `Makefile`, guarded by docs drift tests.

## Constraints

- Default pytest remains CPU-safe and model-download-free.
- Generated score files, held-out outputs, evaluation reports, contact sheets, thesis tables/plots, checkpoints, images, tensors, logs, and run directories remain ignored runtime artifacts unless intentionally tiny fixtures.
- Missing/uncontrolled evidence must be surfaced explicitly, not treated as comparable.
- Existing runnable score/training commands stay compatible while gaining canonical metadata.
