# External Integrations

**Analysis Date:** 2026-05-04

## APIs & External Services

**Model Hubs:**
- Hugging Face Hub - Downloads FLUX.2 Klein, Qwen models, and prompt datasets.
  - SDK/Client: `diffusers`, `transformers`, `datasets`
  - Auth: standard Hugging Face environment/config authentication if private or gated models require it; no explicit token environment variable is read in the codebase.
  - Usage: `Flux2KleinPipeline.from_pretrained(...)` in `scripts/generate_images.py`, `src/training/flux2_utils.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, and `src/training/masked_sft_trainer.py`; `load_dataset(...)` in `scripts/download_dataset.py`.

**Vision-language Reward Models:**
- Qwen VLM - Computes yes/no probability reward for rendered text correctness.
  - SDK/Client: `transformers.AutoModelForImageTextToText`, `transformers.AutoProcessor`, `transformers.BitsAndBytesConfig`
  - Auth: standard Hugging Face model access; no project-specific env var detected.
  - Usage: `src/training/rewards.py`, `src/evaluation/evaluate_rewards.py`, and experiments under `experiments/ocr_reward_tests/`.

**Local LLM Backends:**
- Transformers text generation - Default backend for prompt phrase/scene generation.
  - SDK/Client: `transformers.AutoModelForCausalLM`, `transformers.AutoTokenizer`
  - Auth: standard Hugging Face model access.
  - Usage: `src/prompt_pipeline/llm_client.py`.
- vLLM - Optional high-throughput CUDA backend for prompt generation.
  - SDK/Client: `vllm.LLM`, `vllm.SamplingParams`
  - Auth: standard local model/Hugging Face access.
  - Usage: `src/prompt_pipeline/llm_client.py`.
- MLX - Optional Apple Silicon backend for prompt generation.
  - SDK/Client: `mlx_lm.load`, `mlx_lm.generate`
  - Auth: standard local model/Hugging Face access.
  - Usage: `src/prompt_pipeline/llm_client.py`.

**OCR Services:**
- PaddleOCR - Local OCR reward/scoring for Cyrillic text.
  - SDK/Client: `paddleocr.PaddleOCR`
  - Auth: none detected.
  - Usage: `src/training/rewards.py`, `src/evaluation/evaluate_rewards.py`, and `experiments/ocr_reward_tests/`.

**Image/Data Sources:**
- Unsplash backgrounds - Background-fetch tooling exists for synthetic datasets.
  - SDK/Client: script-local HTTP/download logic in `scripts/synth/fetch_unsplash.py`.
  - Auth: not detected in mapped files; metadata and docs live under `data/backgrounds/unsplash_meta/`.
- SynthTIGER - Local text-image rendering engine for synthetic masked-SFT datasets.
  - SDK/Client: CLI invocation through `scripts/synth/run_synthtiger.py` from `scripts/synth/build_dataset.py`.
  - Auth: none.

## Data Storage

**Databases:**
- Not detected. There is no relational, document, or vector database client in `src/` or `scripts/`.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
  - Prompt JSONL files live under `data/` and are read by `src/prompt_pipeline/generate.py`, `scripts/generate_images.py`, and `scripts/download_dataset.py`.
  - Generated images, latents, text embeddings, scores, checkpoints, samples, and TensorBoard logs live under `outputs/` and `runs/`.
  - Synthetic data layouts are written under `data/synth_cyrillic/` by `scripts/synth/build_dataset.py`.
  - Model checkpoints are saved under trainer `output_dir` values such as `outputs/sft`, `outputs/dpo`, and `outputs/masked_sft`.

**Caching:**
- No Redis/Memcached/service cache detected.
- File-level caches/checkpoints are used: `scripts/generate_images.py` skips existing latents/images, `scripts/score_images.py` supports `--resume`, and trainers write checkpoint directories under `outputs/*/checkpoints`.

## Authentication & Identity

**Auth Provider:**
- No application authentication system is present.
  - Implementation: All external model/dataset access relies on the user's local Hugging Face configuration or environment outside this codebase.
  - No API keys or auth-specific env var reads were detected in `src/` or `scripts/`.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, OpenTelemetry, Prometheus, or hosted error tracking integration is detected.

**Logs:**
- Python `logging` is used in pipeline/training modules including `scripts/generate_images.py`, `scripts/score_images.py`, `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/masked_sft_trainer.py`, and `src/prompt_pipeline/generate.py`.
- TensorBoard logging is configured through Hugging Face Accelerate in `src/training/sft_trainer.py` and `src/training/dpo_trainer.py`.
- Some utility modules use `print` for model-load and summary output in `src/training/rewards.py`, `src/training/flux2_utils.py`, and `src/evaluation/evaluate_rewards.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable. The repository is a research/training pipeline, not a deployed web service.

**CI Pipeline:**
- None detected. No `.github/workflows/`, `gitlab-ci.yml`, or similar CI configuration was found during fast mapping.
- Cluster execution is handled manually via SLURM scripts in `scripts/cluster/`.

## Environment Configuration

**Required env vars:**
- None are explicitly required by project code.
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is set internally in `src/training/rewards.py`.
- Optional Hugging Face credentials may be required in the user environment for gated/private model access, but no specific env var is read by repository code.

**Secrets location:**
- No secrets storage is committed.
- `.env`, `.envrc`, virtual environments, logs, outputs, runs, tensors, model weights, and generated images are ignored by `.gitignore`.
- No `.env*` file was detected during mapping and no forbidden secret file contents were read.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server, webhook route, or callback endpoint is present.

**Outgoing:**
- None. Model/dataset downloads are pull-based through Hugging Face libraries in `diffusers`, `transformers`, and `datasets`.

---

*Integration audit: 2026-05-04*
