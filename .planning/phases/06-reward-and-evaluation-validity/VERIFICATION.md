---
phase: 06-reward-and-evaluation-validity
verified: 2026-05-06T15:29:43Z
status: passed
score: 10/10 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 8/10
  gaps_closed:
    - "EVAL-01 training-side canonical reward interface gap closed by importing RewardResult/ProductScoreFormula/compute_product_score in src.training.rewards and adding build_training_reward_result."
    - "STR-03 evaluator-local Qwen/OCR duplication gap closed by moving file-path evaluation adapters to src.training.rewards and importing them from src.evaluation.evaluate_rewards."
  gaps_remaining: []
  regressions: []
---

# Phase 6: Reward and Evaluation Validity Verification Report

**Phase Goal:** Users can trust reward scores, held-out evaluations, diagnostic reports, and thesis outputs as comparable evidence tied back to exact runs.  
**Verified:** 2026-05-06T15:29:43Z  
**Status:** passed  
**Re-verification:** Yes — after reward-interface gap closure

## Goal Achievement

Phase 6 now satisfies the roadmap and requirement contract. The previous blockers were specifically re-checked against code, not SUMMARY claims: `src.training.rewards` now imports the canonical reward interface, exposes `build_training_reward_result(...) -> RewardResult`, owns the file-path Qwen/OCR evaluation adapters, and `src.evaluation.evaluate_rewards` reuses those training adapters instead of defining evaluator-local scorer classes. Previously verified scoring sidecars, held-out plans, slice/gold diagnostics, reward disagreement reports, thesis bundles, command docs, and CPU-safe tests remain intact.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | EVAL-01: one canonical reward interface covers Qwen/VLM, OCR/CER/entropy, product paths across scoring, training, evaluation, and thesis reports | ✓ VERIFIED | `src/evaluation/reward_interface.py` defines `RewardResult`, `ProductScoreFormula`, and `compute_product_score`. `scripts/score_images.py` and `src/evaluation/evaluate_rewards.py` use canonical product/metadata helpers. `src/training/rewards.py:17-21` imports canonical symbols and `src/training/rewards.py:350-382` converts training/offline reward dictionaries into canonical `RewardResult` rows. Docs at `docs/reward_evaluation.md:48-54` document the shared contract. |
| 2 | EVAL-02: product score files are reproducible with formula, scorer versions, component scores, thresholds, schema metadata, and manifest links | ✓ VERIFIED | `ProductScoreFormula` and `build_score_metadata` record formula, weights, thresholds, scorer versions, and source manifests; `scripts/score_images.py:174-201` and `src/evaluation/evaluate_rewards.py:138-177` write `.schema.json` sidecars. |
| 3 | EVAL-03: held-out checkpoint-comparison evaluation harness fixes prompts, seeds, settings, baseline/trained targets, outputs, and manifests | ✓ VERIFIED | `src/evaluation/heldout.py` exports `HeldoutEvaluationConfig`, `EvaluationTarget`, `build_evaluation_plan`, and `write_evaluation_plan`; `scripts/run_heldout_evaluation.py` imports the harness; tests passed in full suite. |
| 4 | EVAL-04: evaluation outputs can be scored with OCR CER/detection, entropy, VLM, product, exact, and character-level fields | ✓ VERIFIED | `scripts/score_images.py:119-171` builds canonical CSV rows with `score_vlm`, `score_ocr`, `cer`, `entropy`, `detection_status`, `exact_text_match`, `char_accuracy`, `product_score`, and `missing_components`; `src/evaluation/evaluate_rewards.py:83-135` emits equivalent JSONL records. |
| 5 | EVAL-05: evaluation can be inspected by Russian text difficulty slices | ✓ VERIFIED | `src/evaluation/slices.py` exports `classify_text_slices` and `summarize_slices`; `src/evaluation/gold_benchmark.py` and `src/evaluation/diagnostics.py` consume slice labels; CPU-safe slice/gold tests passed. |
| 6 | EVAL-06: reward disagreement diagnostics include VLM-vs-OCR scatter/correlation, false positives/negatives, contact sheets, and character confusions | ✓ VERIFIED | `src/evaluation/diagnostics.py` exports `analyze_reward_disagreement` and `format_diagnostics_markdown`, with correlation/scatter, false-row, confusion, per-slice, and optional contact-sheet paths verified by tests. |
| 7 | EVAL-07: reward signals can be validated against a small gold diagnostic benchmark | ✓ VERIFIED | `src/evaluation/gold_benchmark.py` exports `load_gold_benchmark` and `evaluate_gold_predictions`; committed fixture `tests/fixtures/evaluation/gold_diagnostic.jsonl` is used by tests and docs. |
| 8 | EVAL-08: thesis-ready tables, plots, and contact sheets are generated from recorded outputs rather than manual/static numbers | ✓ VERIFIED | `src/evaluation/thesis_outputs.py` reads source manifests, score reports, diagnostic reports, and table specs; writes bundle/Markdown/tables/SVG/contact-sheet references; missing provenance becomes readiness errors. |
| 9 | RUN-05: thesis plots/results map back to exact run manifests and artifacts | ✓ VERIFIED | Thesis bundle code loads manifests through `load_run_manifest`, records manifest paths and evidence report paths, and blocks missing manifest/report provenance. Held-out plans also link targets to manifests. |
| 10 | STR-03: shared scoring/reward/evaluation modules replace duplicated Qwen/OCR logic across training and evaluation | ✓ VERIFIED | `src.evaluation.evaluate_rewards` imports `EvaluationQwenYesProbReward as QwenYesProbReward` and `PaddleOCRAccuracyReward as PaddleOCRReward` from `src.training.rewards` (`evaluate_rewards.py:37-42`). Grep found no evaluator-local `class QwenYesProbReward` or `class PaddleOCRReward`; tests assert identity with shared training classes. |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/evaluation/reward_interface.py` | Canonical reward result dataclasses, product formula, metadata helpers | ✓ VERIFIED | 320 lines; pure Python; import spot-check loaded no heavy modules and computed a complete product score. |
| `src/training/rewards.py` | Training reward wrappers plus canonical training/evaluation adapters | ✓ VERIFIED | Imports canonical `RewardResult`, `ProductScoreFormula`, and `compute_product_score`; `build_training_reward_result` returns `RewardResult`; owns shared file-path `EvaluationQwenYesProbReward` and `PaddleOCRAccuracyReward`. |
| `scripts/score_images.py` | Canonical CSV rows and product score sidecars | ✓ VERIFIED | Uses `ProductScoreFormula`, `compute_product_score`, `build_score_metadata`; scoring branches lazily import training reward wrappers. |
| `src/evaluation/evaluate_rewards.py` | Canonical evaluation JSONL records and metadata sidecars using shared adapters | ✓ VERIFIED | Imports evaluation adapters from `src.training.rewards`; canonical record builder and sidecar writer remain wired. |
| `src/runtime/artifacts.py` | CPU-safe Phase 6 score validation | ✓ VERIFIED | Tests validate CSV/JSONL Phase 6 fields and `.schema.json` sidecar contracts. |
| `src/evaluation/heldout.py` / `scripts/run_heldout_evaluation.py` | Held-out evaluation plan/report materialization | ✓ VERIFIED | Exports/imports present; Makefile dry-run invokes CLI without launching expensive jobs. |
| `src/evaluation/slices.py` / `src/evaluation/gold_benchmark.py` | Russian slice and gold diagnostic contracts | ✓ VERIFIED | Exports present; fixture and tests validate CPU-safe metadata paths. |
| `src/evaluation/diagnostics.py` / `scripts/analyze_reward_diagnostics.py` | Reward disagreement diagnostics | ✓ VERIFIED | Exports present; CLI alias dry-runs expected command. |
| `src/evaluation/thesis_outputs.py` / `scripts/build_thesis_outputs.py` | Thesis output bundle builder and provenance validator | ✓ VERIFIED | Loads manifests/reports and records artifact paths; CLI alias dry-runs expected command. |
| `docs/reward_evaluation.md`, `docs/evaluation_harness.md`, `docs/evaluation_diagnostics.md`, `docs/thesis_outputs.md`, `docs/commands.md`, `README.md`, `Makefile` | Discoverable Phase 6 contracts and command surface | ✓ VERIFIED | Docs contain shared training/evaluation adapter contract and Makefile aliases; docs drift tests included in full suite. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `src/training/rewards.py` | `src/evaluation/reward_interface.py` | imports `RewardResult`, `ProductScoreFormula`, `compute_product_score` | ✓ WIRED | Lines 17-21 import canonical symbols; `build_training_reward_result` constructs `RewardResult` from Qwen/OCR/CER/entropy/exact evidence. |
| `src/evaluation/evaluate_rewards.py` | `src/training/rewards.py` | imports shared file-path evaluation adapters | ✓ WIRED | Lines 37-42 alias `EvaluationQwenYesProbReward` and `PaddleOCRAccuracyReward`; tests assert object identity and absence of local scorer classes. |
| `scripts/score_images.py` | `src/evaluation/reward_interface.py` | canonical row/metadata helpers | ✓ WIRED | `build_canonical_score_row` calls `compute_product_score`; sidecar writer calls `build_score_metadata`; `main` calls both. |
| `src/evaluation/evaluate_rewards.py` | `src/evaluation/reward_interface.py` | canonical JSONL/metadata helpers | ✓ WIRED | `build_canonical_evaluation_record` calls product formula; `write_evaluation_score_metadata` calls metadata helper. |
| `src/runtime/artifacts.py` | Phase 6 score CSV/JSONL sidecars | `evaluation_scores` validation | ✓ WIRED | Focused scoring-output tests passed, including accept/reject cases. |
| `scripts/run_heldout_evaluation.py` | `src/evaluation/heldout.py` | imports harness functions | ✓ WIRED | Makefile dry-run invokes `python -m scripts.run_heldout_evaluation ...`. |
| `src/evaluation/gold_benchmark.py` | `src/evaluation/slices.py` | slice labels in gold reports | ✓ WIRED | Grep confirms `classify_text_slices`; tests passed. |
| `src/evaluation/diagnostics.py` | `src/evaluation/slices.py` and gold benchmark | per-slice/gold diagnostic evidence | ✓ WIRED | Diagnostics exports present and covered by tests. |
| `scripts/build_thesis_outputs.py` | `src/evaluation/thesis_outputs.py` | thesis bundle builder | ✓ WIRED | Makefile dry-run invokes expected CLI. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `src/training/rewards.py` | canonical `RewardResult` fields | Training/offline reward output dictionaries from Qwen/OCR wrappers | Yes — `build_training_reward_result` maps `reward_qwen_yes_prob`, `reward_ocr`/`reward_paddleocr`, `cer`, `entropy`, and `exact_text_match` into canonical evidence and product score | ✓ FLOWING |
| `src/evaluation/evaluate_rewards.py` | Qwen/OCR reward outputs and canonical JSONL fields | Shared `src.training.rewards` file-path adapters + canonical product helpers | Yes — scorer outputs update `reward_outputs`, which are converted into canonical records and written to JSONL | ✓ FLOWING |
| `scripts/score_images.py` | canonical CSV score row fields | Runtime VLM/OCR scorer outputs plus text embedding `target_text` | Yes — scorer evidence feeds `build_canonical_score_row`; missing evidence is explicit | ✓ FLOWING |
| `src/evaluation/heldout.py` | plan `targets`, `manifest_links`, planned commands | JSON config + prompt JSONL + run manifests | Yes — validated local config/manifests materialize planned commands and reports | ✓ FLOWING |
| `src/evaluation/diagnostics.py` | disagreement/correlation/false rows | Recorded CSV/JSONL/JSON score rows and optional gold JSONL | Yes — recorded inputs are parsed and summarized without model calls | ✓ FLOWING |
| `src/evaluation/thesis_outputs.py` | bundle manifests, reports, tables, plots, sheets | Recorded run manifests, score reports, diagnostic reports, table specs | Yes — missing provenance is recorded as readiness errors | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Canonical reward interface import stays lightweight and product scoring works | `PATH="/root/.local/bin:$PATH" uv run python -c "... import src.evaluation.reward_interface ... compute_product_score(...) ..."` | Printed `True set()` — complete formula and no newly loaded heavy modules among torch/transformers/paddleocr/diffusers/vLLM/MLX | ✓ PASS |
| Evaluation reuses training adapters and training adapter returns canonical `RewardResult` | `PATH="/root/.local/bin:$PATH" uv run python -c "... evaluate_rewards aliases ... build_training_reward_result(...) ..."` | Printed `True True True True` | ✓ PASS |
| Focused reward-interface/scoring tests | `PATH="/root/.local/bin:$PATH" uv run pytest tests/test_reward_wrapper_contracts.py tests/test_evaluation_reward_interface.py tests/test_evaluation_scoring_outputs.py -q` | `22 passed in 1.38s` | ✓ PASS |
| Full CPU-safe test suite | `PATH="/root/.local/bin:$PATH" uv run pytest -q` | `253 passed in 7.81s` | ✓ PASS |
| Ruff on changed reward/evaluation files | `PATH="/root/.local/bin:$PATH" uv run --extra lint ruff check src/training/rewards.py src/evaluation/evaluate_rewards.py tests/test_reward_wrapper_contracts.py tests/test_evaluation_reward_interface.py` | `All checks passed!` | ✓ PASS |
| Phase 6 CLI docs/aliases are discoverable without running expensive jobs | `make -n phase6-heldout-plan phase6-reward-diagnostics phase6-thesis-outputs` | Printed expected `run_heldout_evaluation`, `analyze_reward_diagnostics`, and `build_thesis_outputs` commands | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| EVAL-01 | 06-01, 06-04, 06-07 | Canonical reward interface across scoring, training, evaluation, thesis reports | ✓ SATISFIED | Canonical interface is used by scoring/evaluation; training module imports it and emits `RewardResult` through `build_training_reward_result`; docs define shared adapter contract. |
| EVAL-02 | 06-01, 06-04, 06-07 | Reproducible product score files with formula/scorer/component/threshold/manifest metadata | ✓ SATISFIED | Product formula, sidecars, scorer versions, thresholds, and manifest links are implemented and tested. |
| EVAL-03 | 06-02, 06-07 | Held-out checkpoint-comparison harness with fixed prompts/seeds/settings/comparable outputs | ✓ SATISFIED | Held-out harness exports and CLI wiring present; tests passed. |
| EVAL-04 | 06-04, 06-07 | Automatic OCR/VLM/product/exact/character scoring fields | ✓ SATISFIED | Canonical CSV/JSONL builders include required fields and validation tests. |
| EVAL-05 | 06-03, 06-05, 06-07 | Russian difficulty slices | ✓ SATISFIED | Slice classifier/summarizer and diagnostics/gold integration present. |
| EVAL-06 | 06-05, 06-07 | Reward disagreement diagnostics | ✓ SATISFIED | Diagnostics module/CLI/tests cover correlation, false rows, confusions, slices, and contact sheets. |
| EVAL-07 | 06-03, 06-05, 06-07 | Gold diagnostic benchmark validation | ✓ SATISFIED | Gold benchmark module and fixture exist; diagnostics can consume gold labels. |
| EVAL-08 | 06-06, 06-07 | Thesis-ready outputs from recorded outputs | ✓ SATISFIED | Thesis bundle builder consumes manifests/reports and emits traceable artifacts with readiness errors. |
| RUN-05 | 06-02, 06-06, 06-07 | Map thesis plots/results to exact manifests/artifacts | ✓ SATISFIED | Held-out and thesis outputs carry source manifest/report/artifact references; missing provenance blocks readiness. |
| STR-03 | 06-01, 06-04, 06-07 | Shared scoring/reward/evaluation modules instead of duplicated Qwen/OCR logic | ✓ SATISFIED | Evaluation imports Qwen/OCR adapters from `src.training.rewards`; no evaluator-local scorer classes remain. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `src/training/rewards.py` | 98 | Literal string `"placeholder"` inside Qwen chat message image slot | ℹ️ Info | Not a stub: the processor receives actual PIL/image tensor separately; this does not block reward-interface sharing. |

No blocker anti-patterns remain. Empty-return matches found in evaluation helpers are normal parser/error-path sentinels, not user-visible stubs or disconnected data flows.

### Human Verification Required

None for Phase 6 completion. The phase contracts are CPU-safe code, metadata, provenance, docs, and command-surface capabilities, all programmatically verified. Separate thesis-quality claims still require explicit target-environment Qwen/PaddleOCR/FLUX/CUDA/SLURM runs and research review before final thesis use, but that is outside this phase completion gate.

### Gaps Summary

No remaining blocking gaps. The prior reward-interface blockers are closed: training can emit canonical `RewardResult` rows, and evaluation reuses shared training-owned Qwen/OCR file-path adapters rather than duplicating scorer classes.

---

_Verified: 2026-05-06T15:29:43Z_  
_Verifier: the agent (gsd-verifier)_
