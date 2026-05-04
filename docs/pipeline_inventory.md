# Pipeline Inventory

This inventory documents the current runnable surfaces in the brownfield diffusion text-tuning toolkit before Phase 1 changes introduce new installation, smoke-check, or command-alias layers. It preserves the existing entry points from `README.md` and the codebase map while separating supported toolkit commands from diagnostics and historical experiment tracks in later sections.

## Supported toolkit entry points

| Pipeline family | Current entry point | Consumes | Produces | Optimizes / measures | Thesis support | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Prompt generation | `python -m src.prompt_pipeline.generate` | Prompt-pipeline resources in `data/`, scene/style/text generators, optional local LLM backend configuration | Prompt JSONL records under `data/` | Prompt diversity, Cyrillic/multilingual text coverage, scene/style coverage | Supplies reproducible prompt inputs for multilingual text-rendering generation and training experiments | Supported toolkit entry point |
| Dataset download | `python scripts/download_dataset.py` | Hugging Face dataset `Outrun32/cyrillic-prompts-15k` and local output path arguments | Local prompt JSONL dataset files under `data/` | Dataset availability rather than model objective quality | Provides a baseline prompt corpus for SFT/DPO image generation pipelines | Supported toolkit entry point |
| Image generation | `python -m scripts.generate_images` | Prompt JSONL files, FLUX.2 Klein model access, generation arguments, CUDA runtime | Generated PNGs, latents, and prompt text embeddings, commonly under `outputs/generated/` | Image sample generation for each prompt and seed/version setting | Creates candidate rendered-text images and reusable latent/text-embedding artifacts for scoring and training | Supported toolkit entry point |
| Reward scoring | `python -m scripts.score_images` | Generated images, text embeddings, target prompt/text metadata, Qwen VLM and/or OCR reward dependencies | `scores.csv` files with reward scores, often under `outputs/generated/` | VLM/OCR-oriented text-rendering reward scores used for filtering and preference construction | Converts generated-image quality signals into SFT sample selection and DPO winner/loser evidence | Supported toolkit entry point |
| SFT training | `accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.sft_trainer --config configs/sft.json` | Generated latents, text embeddings, score CSVs, `configs/sft.json`, FLUX/LoRA dependencies | SFT LoRA checkpoints, logs, and sample outputs under the configured `output_dir` such as `outputs/sft` | Flow-matching MSE on high-reward generated samples | Tests whether reward-filtered supervised fine-tuning improves Cyrillic/multilingual text rendering | Supported toolkit entry point |
| DPO training | `accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.training.dpo_trainer --config configs/dpo.json` | Winner/loser pairs derived from generated latents, text embeddings, scores, `configs/dpo.json`, policy/reference LoRA setup | DPO LoRA checkpoints, logs, and sample outputs under the configured `output_dir` such as `outputs/dpo` | Diffusion DPO preference objective over winner/loser image variants | Tests whether preference optimization improves text-rendering quality beyond SFT or baseline behavior | Supported toolkit entry point |
| Masked-SFT training | `python -m src.training.masked_sft_trainer --config configs/masked_sft.json` | Synthetic masked-SFT latents, text embeddings, masks, metadata, shapes, and `configs/masked_sft.json` | Masked-SFT LoRA checkpoints, validation samples, metrics, and logs under the configured output directory | Region-weighted masked flow-matching loss over text regions plus global reconstruction terms | Tests whether synthetic text-region supervision improves reconstruction and text rendering | Supported toolkit entry point |
| Synthetic data | `python -m scripts.synth.build_dataset` | SynthTIGER configuration, fonts/backgrounds, Cyrillic word/text resources, FLUX text/latent baking dependencies | Raw synthetic samples, masks, metadata, AnyWord-style layouts, latents, text embeddings, and `shapes.csv` under `data/synth_cyrillic/` | Synthetic dataset coverage, mask/text-region quality, and latent/text-embedding preparation | Provides controlled multilingual/Cyrillic masked-SFT training data independent of generated-image rewards | Supported toolkit entry point |
| Evaluation | `python -m src.evaluation.generate_baseline`; `python -m src.evaluation.evaluate_rewards` | Held-out prompts/images, baseline generation settings, reward model dependencies, evaluation metadata | Baseline image outputs and reward/evaluation summaries | Baseline generation quality and reward-model behavior on evaluation samples | Supports comparisons between baseline, trained LoRA outputs, and reward variants for thesis evidence | Supported toolkit entry point |
| Plotting | `python scripts/thesis_plots/plot_sft_losses.py`; `python scripts/thesis_plots/plot_dpo_metrics.py` | Training/evaluation logs, metric exports, and thesis result artifacts | Thesis figures/plots for SFT losses and DPO metrics | Trends in training loss and DPO metrics, treated as internal evidence rather than final text-rendering proof | Produces report-ready plots tied to exact runs and metrics when inputs are recorded | Supported toolkit entry point |
| SLURM launchers | `sbatch scripts/cluster/*.sbatch` | Cluster environment, configured datasets/artifacts, Accelerate configs, job array parameters, and stage-specific `.sbatch` files | Cluster logs, sharded generated/scored artifacts, training checkpoints, and merged scores | Local-vs-cluster execution parity, sharding throughput, and long-running job orchestration | Keeps generation, scoring, and training runnable on the target SLURM research environment | Supported toolkit entry point |

## Non-default diagnostics and experiments

The paths below remain useful for investigation and thesis history, but they are intentionally not part of the default supported command surface above. Do not include these in broad automated test discovery unless they are later renamed, guarded, and made CPU-safe.

### Manual diagnostics

| Path | Purpose | Why non-default | Expected prerequisites |
| --- | --- | --- | --- |
| `scripts/test_gradient_flow.py` | Checks whether gradients flow through a generated/training sample path. | It is a manual CUDA/model diagnostic named like a test and can load heavyweight model or tensor artifacts at import/runtime. | CUDA-capable environment plus locally generated embeddings/artifacts matching the script assumptions. |
| `scripts/test_grad_magnitude.py` | Inspects gradient magnitudes for generated/training artifacts. | It is a manual CUDA/model diagnostic, not a formal isolated unit test. | CUDA-capable environment plus local generated tensors. |
| `experiments/ocr_reward_tests/test_qwen_yes_prob.py` | Probes Qwen yes-probability reward behavior on OCR/VLM experiment assets. | It depends on local model/backend availability and has historically included workstation-specific paths. | Qwen/VLM runtime, local assets, and any backend-specific setup. |
| `experiments/ocr_reward_tests/test_paddleocr.py` | Probes PaddleOCR behavior on experiment assets. | It is an OCR experiment, not a CPU-safe default test. | PaddleOCR installation, local OCR model/cache availability, and experiment assets. |

### Experimental scripts

| Path | Purpose | Relationship to supported flows | Status |
| --- | --- | --- | --- |
| `experiments/ocr_reward_tests/test_qwen_yes_prob.py` | Tests VLM yes/no scoring behavior for reward calibration. | Informs `python -m scripts.score_images` and reward evaluation semantics, but does not replace the supported scorer. | Experimental script |
| `experiments/ocr_reward_tests/test_paddleocr.py` | Tests PaddleOCR recognition and OCR reward feasibility. | Informs OCR reward variants and `src.evaluation.evaluate_rewards`, but remains a one-off experiment. | Experimental script |

### Legacy or superseded configs

| Config | Family | Current interpretation | Status |
| --- | --- | --- | --- |
| `configs/train_base.json` | Earlier training configuration | Historical baseline-style config predating the current explicit SFT/DPO/masked-SFT command inventory. | Legacy or superseded config |
| `configs/train_v4_highLR.json` | Earlier training configuration | High-learning-rate historical variant useful for comparison notes, not a default supported launch target. | Legacy or superseded config |
| `configs/sft_ocr.json` | SFT reward variant | OCR-filtered SFT variant that should be understood relative to the canonical `configs/sft.json` SFT command. | Variant / non-default config |
| `configs/sft_vlm.json` | SFT reward variant | VLM-filtered SFT variant used to compare reward families. | Variant / non-default config |
| `configs/sft_product.json` | SFT reward variant | Product-score SFT variant for combined reward experiments. | Variant / non-default config |
| `configs/dpo_ocr.json` | DPO reward variant | OCR-preference DPO variant that should be compared against the canonical `configs/dpo.json` command. | Variant / non-default config |
| `configs/dpo_vlm.json` | DPO reward variant | VLM-preference DPO variant for reward-family comparison. | Variant / non-default config |
| `configs/dpo_product.json` | DPO reward variant | Product-score DPO variant for combined reward experiments. | Variant / non-default config |

### Supported toolkit entry points

Supported toolkit entry points are the commands in the first table: prompt generation, dataset download, `python -m scripts.generate_images`, `python -m scripts.score_images`, SFT, DPO, masked-SFT, synthetic data, evaluation, plotting, and `sbatch scripts/cluster/*.sbatch`. These are the commands users should start from when they want reproducible local or SLURM runs; diagnostics, experiments, and legacy configs should be opt-in and documented as such.

## Historical experiment tracks

### reward-filtered generated-image SFT/DPO

The primary historical track generates multiple FLUX image variants from prompt JSONL records, scores those variants with VLM/OCR reward paths, and uses the resulting `scores.csv` files for high-reward SFT sample selection and DPO winner/loser pair construction. It connects `python -m scripts.generate_images`, `python -m scripts.score_images`, `src.training.sft_trainer`, and `src.training.dpo_trainer` into the thesis question of whether reward-filtered generated examples and preference optimization improve multilingual text rendering.

### synthetic masked-MSE training

The synthetic masked-MSE track builds controlled Cyrillic/text-region datasets through `python -m scripts.synth.build_dataset`, saves images, masks, metadata, latents, text embeddings, and shape indexes under `data/synth_cyrillic/`, then trains `src.training.masked_sft_trainer` with region-weighted masked flow-matching losses. It supports the thesis comparison between generated-image reward filtering and controlled synthetic text-region supervision.

### OCR/VLM/product reward variants

The OCR/VLM/product reward variants compare how different quality signals select SFT samples or construct DPO pairs. Configs such as `configs/sft_ocr.json`, `configs/sft_vlm.json`, `configs/sft_product.json`, `configs/dpo_ocr.json`, `configs/dpo_vlm.json`, and `configs/dpo_product.json` should be interpreted as reward-family variants of the supported SFT/DPO flows rather than separate default pipelines.

### thesis plotting/report flows

The thesis plotting/report flows consume recorded training and evaluation metrics to produce figures such as SFT loss curves and DPO metric plots through `python scripts/thesis_plots/plot_sft_losses.py` and `python scripts/thesis_plots/plot_dpo_metrics.py`. These plots help communicate internal training dynamics, but thesis claims should trace them back to exact runs, configs, seeds, rewards, and generated artifacts rather than treating loss or DPO accuracy alone as final evidence of rendered-text quality.

## Artifact safety notes

Generated ML artifacts stay out of git unless they are intentionally tiny fixtures or documentation assets. This includes `outputs/`, `runs/`, generated images, tensors, checkpoints, large generated datasets, and logs. These artifacts may contain prompt text, model outputs, scores, or environment-specific paths, so keep them under ignored runtime roots and commit only source code, configuration, tests, small approved fixtures, and documentation.
