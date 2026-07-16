# Diffusion Text Tuner agent guide

## Project boundary

This is the final public codebase for an individual bachelor-thesis study of reward-filtered FLUX.2
Klein alignment for Russian/Cyrillic text rendering. Preserve the scientific distinction between:

- historical defense results, whose raw rows and checkpoints are not in the repository;
- code paths and CPU-safe contracts that can be verified from the checkout;
- future reruns that require a Linux/CUDA host.

The old seven-phase `.planning/` workspace was completed and deliberately removed before the public
release. Do not require or recreate it. Current sources of truth are:

- `README.md` for project claims and supported workflows;
- `reports/final/` for public evidence and evidence-status labels;
- `docs/runtime_contracts.md` for manifests and artifact paths;
- `docs/commands.md` for command ownership;
- `docs/structure_and_extension.md` for module boundaries;
- `configs/experiments/README.md` for config placement.

## Platform rules

The local development machine is Apple Silicon. MLX supports prompt-language-model inference only.
Do not describe MLX or MPS as a FLUX training backend.

Safe local work includes tests, lint, prompt generation, data validation, manifests, selection from
recorded scores, benchmark planning, diagnostics, and report generation. FLUX image generation,
latent baking, PyTorch VLM scoring, SFT, DPO, masked-SFT, and ReFL remain Linux/CUDA stages. Keep
their imports behind explicit execution functions and make preflight fail before model loading on an
unsupported host.

## Scientific invariants

- `thesis_vlm_ocr_product_v1` means exactly `score_vlm * score_ocr`.
- The five-component geometric formula is a separate diagnostic metric.
- VLM-only score files use VLM as `score`; OCR-only files use OCR; combined files use Product.
- Training loss and DPO accuracy are internal signals, not evidence of rendered-text quality.
- Benchmark targets must be unique and disjoint from the pinned prompt-training pool.
- Never convert historical aggregate-only metrics into verified claims without the raw rows.
- Tie new plots and tables to run manifests, immutable config snapshots, model revisions, seeds, and
  artifact hashes.

## Engineering rules

- Keep default pytest CPU-safe; model, CUDA, OCR, integration, and manual probes stay opt-in.
- Run `make check` before handing off a change. On the Mac, run `make mac-check` when MLX setup or
  platform code changes.
- Use `uv.lock` with Python 3.11. Do not replace frozen installs with ad hoc `pip install` commands in
  committed launchers.
- Keep reusable logic under `src/` and CLI parsing under `scripts/`.
- Reject config options that the execution path does not implement; never accept a field and then
  ignore it silently.
- Preserve unrelated worktree changes. The OCR research document and its local assets may be under
  active development.

## Artifact and security rules

Generated images, tensors, checkpoints, logs, score files, private manifests, chat exports, editor
databases, and handoff bundles stay outside Git under ignored runtime roots. Small reviewed fixtures,
project-page figures, and `reports/final/` evidence are the only deliberate exceptions.

Never print, commit, or copy credential values. A deleted file remains reachable in Git history, so
removing it in a later commit is not a security fix. Secret-history cleanup requires credential
rotation, a full-history scan, coordinated history rewrite, and force-push approval.
