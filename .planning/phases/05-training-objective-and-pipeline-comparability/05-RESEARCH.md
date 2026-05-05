# Phase 5 Research: Training Objective and Pipeline Comparability

**Phase:** 05-training-objective-and-pipeline-comparability  
**Researched:** 2026-05-05  
**Mode:** codebase-grounded planning research; no new external dependencies selected

## Planning-Relevant Findings

- Existing Phase 3 selection tooling already materializes `selected-samples/v1` and `preference-pairs/v1` JSONL from score CSVs in `src/training/selection.py` and exposes `scripts/materialize_training_data.py`.
- Existing `SFTDataset` and `DPODataset` still select rows directly from scores CSVs inside constructors. Phase 5 must make explicit mode choices consumable by training configs/loaders without removing current CSV behavior.
- Phase 4 extracted `src.training.dpo_objective` and verified current negative-beta behavior. Phase 5 may add selectable data/pair construction modes, but objective math must remain delegated through that helper and CPU-safe tests must preserve sign/beta expectations.
- Existing `MaskedSFTConfig` already exposes `masked_lambda`, multi-rank LoRA fields, `data_dir`, `eval_suite_path`, validation intervals, and sampling fields. Phase 5 must make these choices visible in configs/manifests/comparison reports rather than treating them as implicit trainer defaults.
- Local run manifests (`src.runtime.manifests`) capture command, config snapshot, seeds, models, inputs, outputs, and metrics, but there is no manifest diff tool yet. RUN-02 should be implemented as a CPU-safe manifest comparison module and CLI.
- Existing runtime preflight can validate individual stage readiness. Phase 5 needs a comparability layer that compares runs/configs for controlled prompts, seeds, inference settings, reward/data sources, metrics, and mismatches before thesis evidence is trusted.
- The codebase architecture warns against adding new pipeline logic directly to large trainer modules. Shared training utilities should be import-safe modules with focused APIs, wired conservatively through tests and compatibility wrappers.

## Validation Architecture

Default validation remains CPU-safe and must not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, SynthTIGER, or external model weights.

Required validation surfaces:

1. **Selection/data-mode tests:** tiny CSV/JSONL fixtures covering SFT modes (`threshold`, `top_k_per_prompt`, `score_weighted`, `hard_positive`) and DPO modes (`best_vs_worst`, `all_separated_pairs`, `margin_weighted`, `ambiguity_filtered`).
2. **Config/manifest tests:** committed JSON config fixtures that prove explicit SFT/DPO/masked-SFT choices load through `src.runtime.config_io` and snapshot into run manifests.
3. **Manifest diff tests:** tiny manifest fixtures comparing config snapshots, data sources, rewards, seeds, inference settings, metrics, and artifact paths.
4. **Mismatch detector tests:** pure dictionary/config fixtures reporting step-count, guidance, prompt-embedding padding, model variant, seed, and sampling setting mismatches.
5. **Shared utility tests:** pure helper tests for sampling/checkpoint/scheduler/runtime plumbing APIs without trainer model loading.
6. **Docs drift tests:** concrete command strings in `docs/commands.md`, `README.md`, `Makefile`, and Phase 5 docs must stay synchronized.

## Threats and Constraints

- Generated images, tensors, checkpoints, logs, and private run outputs remain ignored runtime artifacts.
- Config and manifest parsing must reject malformed JSON and traversal/home paths using existing runtime policies.
- Manifest diffs must not expose secret environment values; existing manifests record presence flags only and diff output should preserve that property.
- Comparison reports must surface missing evidence explicitly instead of fabricating metrics.
- Heavy GPU/model/OCR diagnostics are opt-in commands, not default pytest checks.
