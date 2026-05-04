# Dataset Quality and Prompt Provenance

Phase 3 adds CPU-safe prompt dataset quality checks and local dataset manifests before prompt JSONL files feed generation, scoring, or training.

## Prompt validation CLI

Use `scripts/validate_prompt_dataset.py` from the repository root. The command imports only pure-Python validation and runtime provenance helpers; it does not load FLUX, Qwen, PaddleOCR, CUDA, OCR, or SynthTIGER.

### Simple prompt datasets

Use this for simple prompt datasets generated from `configs/prompts/simple.json`.

```bash
uv run python scripts/validate_prompt_dataset.py \
  --input data/prompts_simple.jsonl \
  --report runs/prompt-quality/simple-report.json \
  --manifest runs/prompt-quality/dataset-manifest.json \
  --config configs/prompts/simple.json \
  --required-rare-characters ё,ж,ц,щ,ъ \
  --min-rare-character-coverage 0.5 \
  --max-duplicate-rate 0.05
```

### Full prompt datasets

Use this for full prompt datasets generated from `configs/prompts/full.json`.

```bash
uv run python scripts/validate_prompt_dataset.py \
  --input data/prompts_full.jsonl \
  --report runs/prompt-quality/full-report.json \
  --manifest runs/prompt-quality/full-manifest.json \
  --config configs/prompts/full.json \
  --allowed-scripts cyrillic,latin,digits,punctuation
```

### Curriculum prompt datasets

Use this for curriculum prompt datasets generated from `configs/prompts/curriculum.json`.

```bash
uv run python scripts/validate_prompt_dataset.py \
  --input data/prompts_curriculum.jsonl \
  --report runs/prompt-quality/curriculum-report.json \
  --manifest runs/prompt-quality/curriculum-manifest.json \
  --config configs/prompts/curriculum.json \
  --strict-warnings
```

When `--report` is omitted, the command writes the JSON quality report to stdout. It returns `2` when blocking validation errors exist, `1` when only warnings exist and `--strict-warnings` is used, and `0` when errors are absent.

## Quality dimensions

Prompt reports include:

- malformed JSONL lines and missing required fields with line numbers;
- target-text length buckets and optional min/max length warnings;
- Cyrillic, Latin, digit, and punctuation coverage counts;
- required rare-character coverage and missing rare-character lists;
- duplicate target-text rate and small duplicate examples;
- content-type and style distributions for drift checks;
- deterministic naturalness/malformed heuristics for empty text, instruction-like outputs, excessive repeated tokens, unmatched quotes, illegal characters, and disallowed scripts.

Reports intentionally avoid echoing full prompt text. Duplicate examples are limited to small `target_text` examples so private prompt datasets are not copied into aggregate reports.

## Manifest fields

Dataset manifests use `dataset-manifest/v1` and record:

- dataset kind and prompt JSONL paths;
- config path, config hash, and config snapshot;
- seed strategy extracted from the config when available;
- git commit/dirty state through Phase 2 reproducibility helpers;
- source hashes for safe text/CSV/JSON/JSONL inputs;
- referenced paths for generated binary tensors/images unless explicitly marked safe to hash;
- model IDs/revisions discovered from config metadata;
- filtering stats and output counts from the prompt quality report.

## Generated artifact safety

Generated reports and manifests are runtime artifacts. Keep them under ignored roots such as `runs/` or `outputs/` unless a tiny fixture is intentionally reviewed for documentation or tests. Generated prompt datasets, tensors, images, checkpoints, contact sheets, and private reports should not be committed.
