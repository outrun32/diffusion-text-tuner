# Diffusion Text Tuner

[![Quality](https://github.com/outrun32/diffusion-text-tuner/actions/workflows/quality.yml/badge.svg)](https://github.com/outrun32/diffusion-text-tuner/actions/workflows/quality.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Reward-filtered LoRA alignment of FLUX.2 Klein Base 4B for Russian and Cyrillic text rendering.

[Project page](https://outrun32.github.io/diffusion-text-tuner/project-page/)
[Prompt dataset](https://huggingface.co/datasets/Outrun32/cyrillic-prompts-15k)
[Evidence bundle](reports/final/README.md)

![Comparison of Base, Product SFT, and Product DPO samples](docs/project-page/assets/teaser_success.webp)

*Base is the unadapted model; Product SFT and Product DPO are LoRA checkpoints trained with the
reward-filtered pipeline described below.*

## Why this project exists

The work began with a client request: make an open-source image model place supplied names and short
phrases inside generated images. English was the first target. Russian exposed the harder problem:
models dropped letters, substituted Cyrillic glyphs, or mixed them with Latin homoglyphs even when
the surrounding image looked correct.

Qwen Image 2.0, FLUX.2 Klein 9B, and FLUX.2 Klein 4B were tested as starting points. Klein 4B gave
the best balance of model size and capability. Like the 9B model, it could already render some
Cyrillic letters and short Russian words correctly, which made it a practical base for alignment
rather than training the writing capability from scratch.

Ordinary image-reconstruction training was a poor match for that failure. Clean Russian text-image
data was scarce, and the error of interest was discrete: did the model write the requested text?
This project treats the task as alignment instead. It generates several candidates, scores the text
inside each image, then trains on the candidates that pass the reward.

## Results

On the recorded benchmark, Product SFT reduced normalized character error rate from `0.859`
to `0.126`. Product DPO reached the highest normalized exact-match rate, but its normalized CER was
worse than Product SFT.

| Checkpoint | Normalized CER ↓ | Normalized exact match ↑ |
| --- | ---: | ---: |
| Base | 0.859 | 41.7% |
| Product SFT | **0.126** | 50.0% |
| Product DPO | 0.168 | **52.5%** |

The table records the aggregate results of the experiment. Product SFT had the lowest CER among the
three runs, while Product DPO produced the highest full-string exact-match rate. The full table is in
[reports/final](reports/final/README.md).

## Method

1. Generate several images for each prompt with the base model.
2. Score whether the requested text appears using a VLM and OCR.
3. Keep high-scoring candidates for LoRA self-training; turn the best and worst candidates into
   preference pairs.
4. Compare Base, Product SFT, and Product DPO on prompts excluded from training.

![Candidate scoring and routing into SFT and DPO](docs/project-page/assets/method_candidates.webp)

OCR and VLM fail differently. OCR is sensitive to glyph recognition and character-level errors;
the VLM judges whether the requested text is present in the generated scene, but can accept a
visually plausible string with a wrong character. The two signals can disagree on these edge cases.
The selection score is therefore:

```text
score_product = score_vlm × score_ocr
```

A candidate receives a high `score_product` only when both checks agree. This keeps images with
legible text that also matches the requested string.

The SFT path fits a LoRA adapter to selected generated samples with flow-matching MSE. The DPO path
uses policy-versus-reference flow-matching errors for winner/loser pairs; it is a diffusion
surrogate, not language-model DPO.

A separate five-component geometric score exists for diagnostics. It combines VLM, OCR, CER,
entropy, and exact match; it is not the `score_product` used for candidate selection in the reported
experiment.

## Quick start

With Python 3.11, [uv](https://docs.astral.sh/uv/), and ShellCheck installed:

```bash
uv python install 3.11
uv sync --frozen --group dev --extra lint
make check
```

This runs lint, formatting checks, the CPU-safe test suite, and evidence verification. Model loading,
OCR probes, integration jobs, and GPU work stay outside the default test run.

FLUX image generation, PyTorch VLM scoring, latent baking, SFT, DPO, masked-SFT, and ReFL require a
Linux/CUDA host. The runnable command sequence and artifact contracts are documented in
[docs/commands.md](docs/commands.md) and [docs/runtime_contracts.md](docs/runtime_contracts.md).

## Prompt generation and evaluation

The repository includes a config-driven prompt dataset generator. Its curriculum starts with single
letters and short words, then adds phrases, digits, punctuation, mixed case, multiline text, styles,
and scenes. Sampling weights increase coverage of Cyrillic glyphs that the base model renders poorly;
dataset validation tracks the rare letters `щ`, `ъ`, `ё`, `э`, `ю`, `ф`, and `ц` explicitly.

```bash
uv run python -m src.prompt_pipeline.generate --config configs/prompts/curriculum.json
```

The benchmark contains 120 unique targets across six difficulty slices, with no exact target
overlap against the pinned training pool. It is committed as
[benchmark_prompts_v2.jsonl](reports/final/benchmark_prompts_v2.jsonl) and provides one fixed
evaluation surface for Base, SFT, and DPO runs.

What the checkout can verify:

- dataset hashes, target disjointness, and evidence manifests;
- reward and objective math, config validation, and selection contracts;
- CPU-safe tests, lint, security checks, and report generation;
- command paths that fail before model loading on unsupported hosts.

Generated images, score files, tensors, checkpoints, private manifests, and logs remain outside Git.
Small reviewed fixtures, the project-page figures, and `reports/final/` are the deliberate
exceptions.

## Where to look

- [`src/`](src/) contains reusable generation, scoring, training, runtime, and evaluation code.
- [`scripts/`](scripts/) contains command-line entry points and manual diagnostics.
- [`configs/`](configs/) contains experiment, prompt, accelerator, and evaluation configs.
- [`reports/final/`](reports/final/) separates checkout-verifiable artifacts from historical
  aggregates.
- [`docs/`](docs/) explains command ownership, evaluation, provenance, and extension rules.

Start with [the command index](docs/commands.md), [runtime contracts](docs/runtime_contracts.md),
[reward and evaluation validity](docs/reward_evaluation.md), or
[repository boundaries](docs/structure_and_extension.md).

## Scope

- The reported experiment covers Russian and Cyrillic text rather than multilingual rendering in
  general.
- Filtering by `score_product` favored shorter samples: median target length changed from 15 to 8
  characters.
- OCR and Qwen participated in selection and evaluation, so they are not independent judges.
- DPO pairs came from Base while the policy started from SFT, so the preference data is off-policy.

## Next direction

Reward alignment improves many near-correct generations, but it is weaker when the base model does
not know a glyph at all. The next stage adds direct glyph supervision for those characters, using
targeted synthetic examples and text-region masks alongside the alignment pipeline. The same data
curriculum and evaluation scheme can then be extended beyond Cyrillic to other writing systems.

## License

Code and documentation are available under the [Apache License 2.0](LICENSE). Model weights and
third-party datasets keep their upstream licenses and access terms.
