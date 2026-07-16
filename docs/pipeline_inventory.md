# Pipeline Inventory

This inventory records the current command surfaces and separates supported toolkit commands from
manual diagnostics and historical experiments. Platform support is part of the contract: Apple
Silicon handles CPU-safe analysis and MLX prompt generation, while FLUX execution remains CUDA-only.

## Supported toolkit entry points

| Pipeline family | Current entry point | Consumes | Produces | Optimizes / measures | Thesis support | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Prompt generation | `uv run python -m src.prompt_pipeline.generate` | Prompt-pipeline resources in `data/`, scene/style/text generators, optional local LLM backend configuration | Prompt JSONL records under `data/` | Prompt diversity, Cyrillic/multilingual text coverage, scene/style coverage | Supplies reproducible prompt inputs for multilingual text-rendering generation and training experiments | Supported toolkit entry point |
| Dataset download | `uv run python -m scripts.download_dataset` | Hugging Face dataset `Outrun32/cyrillic-prompts-15k` and local output path arguments | Local prompt JSONL dataset files under `data/` | Dataset availability rather than model objective quality | Provides a baseline prompt corpus for SFT/DPO image generation pipelines | Supported toolkit entry point |
| Image generation | `uv run python -m scripts.generate_images` | Prompt JSONL files, FLUX.2 Klein model access, generation arguments, CUDA runtime | Generated PNGs, latents, and prompt text embeddings, commonly under `outputs/generated/` | Image sample generation for each prompt and seed/version setting | Creates candidate rendered-text images and reusable latent/text-embedding artifacts for scoring and training | Supported toolkit entry point |
| Reward scoring | `uv run python -m scripts.score_images` | Generated images, text embeddings, target prompt/text metadata, Qwen VLM and/or OCR reward dependencies | `scores.csv` files with reward scores, often under `outputs/generated/` | VLM/OCR-oriented text-rendering reward scores used for filtering and preference construction | Converts generated-image quality signals into SFT sample selection and DPO winner/loser evidence | Supported toolkit entry point |
| SFT training | `uv run accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/sft.json` | Generated latents, text embeddings, score CSVs, `configs/sft.json`, FLUX/LoRA dependencies | SFT LoRA checkpoints, logs, and sample outputs under the configured `output_dir` such as `outputs/sft` | Flow-matching MSE on high-reward generated samples | Tests whether reward-filtered supervised fine-tuning improves Cyrillic/multilingual text rendering | Supported toolkit entry point |
| DPO training | `uv run accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.dpo_trainer --config configs/dpo.json` | Winner/loser pairs derived from generated latents, text embeddings, scores, `configs/dpo.json`, policy/reference LoRA setup | DPO LoRA checkpoints, logs, and sample outputs under the configured `output_dir` such as `outputs/dpo` | Diffusion DPO preference objective over winner/loser image variants | Tests whether preference optimization improves text-rendering quality beyond SFT or baseline behavior | Supported toolkit entry point |
| Masked-SFT training | `uv run python -m src.training.masked_sft_trainer --config configs/masked_sft.json` | Synthetic masked-SFT latents, text embeddings, masks, metadata, shapes, and `configs/masked_sft.json` | Masked-SFT LoRA checkpoints, validation samples, metrics, and logs under the configured output directory | Region-weighted masked flow-matching loss over text regions plus global reconstruction terms | Tests whether synthetic text-region supervision improves reconstruction and text rendering | Supported toolkit entry point |
| Synthetic data | `uv run python -m scripts.synth.build_dataset` | Explicit external SynthTIGER template/config/runner plus fonts/backgrounds and Cyrillic resources; optional CUDA baking | Raw synthetic samples, masks, metadata, AnyWord layouts, latents, text embeddings, and `shapes.csv` under `data/synth_cyrillic/` | Synthetic dataset coverage, mask/text-region quality, and latent/text-embedding preparation | Integration surface for controlled masked-SFT data; rendering is not turnkey without external SynthTIGER inputs | Supported adapter with explicit external inputs |
| Evaluation | `uv run python -m scripts.run_heldout_evaluation`, followed by the emitted generation/scoring commands and `uv run python -m scripts.aggregate_heldout_scores` | Fixed target-disjoint prompts, fixed seeds, source run manifests, baseline and LoRA paths | Materialized execution specification, per-seed scores, and aggregate CSV | Baseline/LoRA generation quality and reward behavior with checked prompt coverage | Supports controlled comparisons when checkpoints and manifests exist | Supported materialize-and-execute contract |
| Plotting | `uv run python -m scripts.plot_metrics`; `uv run python -m scripts.build_thesis_outputs` | Recorded training/evaluation metrics, manifests, and evidence reports | Metric plots, tables, SVGs, and bounded contact sheets | Training trends and report evidence; losses remain internal signals | Produces report artifacts tied to recorded inputs | Supported toolkit entry point |
| SLURM launchers | `sbatch scripts/cluster/*.sbatch` | Cluster environment, configured datasets/artifacts, Accelerate configs, job array parameters, and stage-specific `.sbatch` files | Cluster logs, sharded generated/scored artifacts, training checkpoints, and merged scores | Local-vs-cluster execution parity, sharding throughput, and long-running job orchestration | Keeps generation, scoring, and training runnable on the target SLURM research environment | Supported toolkit entry point |

## Non-default diagnostics and experiments

The paths below remain useful for investigation and thesis history, but they are intentionally not part of the default supported command surface above. Do not include these in broad automated test discovery unless they are later renamed, guarded, and made CPU-safe.

### Manual diagnostics

| Path | Purpose | Why non-default | Expected prerequisites |
| --- | --- | --- | --- |
| `scripts/diagnose_gradient_flow.py` | Checks whether gradients flow through a generated/training sample path. | It is a manual CUDA/model diagnostic with a guarded entry point and can load heavyweight model or tensor artifacts when run explicitly. | CUDA-capable environment plus locally generated embeddings/artifacts matching the script assumptions. |
| `scripts/diagnose_grad_magnitude.py` | Inspects gradient magnitudes for generated/training artifacts. | It is a manual CUDA/model diagnostic, not a formal isolated unit test. | CUDA-capable environment plus local generated tensors. |
| `experiments/ocr_reward_tests/test_qwen_yes_prob.py` | Probes Qwen yes-probability reward behavior on OCR/VLM experiment assets. | It depends on local model/backend availability and has historically included workstation-specific paths. | Qwen/VLM runtime, local assets, and any backend-specific setup. |
| `experiments/ocr_reward_tests/test_paddleocr.py` | Probes PaddleOCR behavior on experiment assets. | It is an OCR experiment, not a CPU-safe default test. | PaddleOCR installation, local OCR model/cache availability, and experiment assets. |

### Experimental scripts

| Path | Purpose | Relationship to supported flows | Status |
| --- | --- | --- | --- |
| `experiments/ocr_reward_tests/test_qwen_yes_prob.py` | Tests VLM yes/no scoring behavior for reward calibration. | Informs `uv run python -m scripts.score_images` and reward evaluation semantics, but does not replace the supported scorer. | Experimental script |
| `experiments/ocr_reward_tests/test_paddleocr.py` | Tests PaddleOCR recognition and OCR reward feasibility. | Informs OCR reward variants and `src.evaluation.evaluate_rewards`, but remains a one-off experiment. | Experimental script |

### Legacy or superseded configs

| Config | Family | Current interpretation | Status |
| --- | --- | --- | --- |
| `configs/experiments/sft/sft_ocr_final.json` | SFT reward variant | Historical final OCR-filtered SFT settings. | Historical final config; raw run manifest missing |
| `configs/experiments/sft/sft_vlm_final.json` | SFT reward variant | Historical final VLM-filtered SFT settings. | Historical final config; raw run manifest missing |
| `configs/experiments/sft/sft_product_final.json` | SFT reward variant | Historical final thesis Product SFT settings. | Historical final config; raw run manifest missing |
| `configs/experiments/dpo/dpo_ocr_final.json` | DPO reward variant | Historical final OCR-preference settings. | Historical final config; raw run manifest missing |
| `configs/experiments/dpo/dpo_vlm_final.json` | DPO reward variant | Historical final VLM-preference settings. | Historical final config; raw run manifest missing |
| `configs/experiments/dpo/dpo_product_final.json` | DPO reward variant | Historical final Product DPO settings. | Historical final config; raw run manifest missing |

### Supported toolkit entry points

Supported toolkit entry points are the commands in the first table: prompt generation, dataset
download, `uv run python -m scripts.generate_images`, `uv run python -m scripts.score_images`, SFT,
DPO, masked-SFT, synthetic data, evaluation, plotting, and `sbatch scripts/cluster/*.sbatch`.
Diagnostics, experiments, and historical configs remain opt-in.

## Historical experiment tracks

### reward-filtered generated-image SFT/DPO

The primary historical track generates multiple FLUX image variants from prompt JSONL records,
scores those variants with VLM/OCR reward paths, and uses the resulting `scores.csv` files for
high-reward SFT sample selection and DPO winner/loser pair construction. It connects
`uv run python -m scripts.generate_images`, `uv run python -m scripts.score_images`,
`src.training.sft_trainer`, and `src.training.dpo_trainer` to the thesis question of whether
reward-filtered generated examples and preference optimization improve text rendering.

### synthetic masked-MSE training

The synthetic masked-MSE track builds controlled Cyrillic/text-region datasets through
`uv run python -m scripts.synth.build_dataset`, saves images, masks, metadata, latents, text
embeddings, and shape indexes under `data/synth_cyrillic/`, then trains
`src.training.masked_sft_trainer` with region-weighted masked flow-matching losses.

### OCR/VLM/product reward variants

The OCR/VLM/product reward variants compare how different quality signals select SFT samples or
construct DPO pairs. Their surviving settings live under `configs/experiments/{sft,dpo}/`; they are
historical config records, not proof of a run without a matching manifest and checkpoint hash.

### thesis plotting/report flows

The plotting/report flow consumes recorded metrics through `uv run python -m scripts.plot_metrics`
and `uv run python -m scripts.build_thesis_outputs`. Loss and DPO accuracy remain internal signals; final claims
need exact run, config, seed, reward, and generated-output provenance.

## Artifact safety notes

Generated ML artifacts stay out of git unless they are intentionally tiny fixtures or documentation assets. This includes `outputs/`, `runs/`, generated images, tensors, checkpoints, large generated datasets, and logs. These artifacts may contain prompt text, model outputs, scores, or environment-specific paths, so keep them under ignored runtime roots and commit only source code, configuration, tests, small approved fixtures, and documentation.
