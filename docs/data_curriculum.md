# Data Curriculum and Prompt Generation Configs

Phase 3 introduces explicit prompt-generation config files so prompt modes can be
selected without editing Python constants or monkey-patching module globals.

## Config files

| Config | Purpose | Default output |
| --- | --- | --- |
| `configs/prompts/simple.json` | CPU-safe no-LLM letters and short-word prompts for quick checks. | `data/prompts/simple.jsonl` |
| `configs/prompts/full.json` | Broad historical distribution with explicit model/backend/output settings. | `data/prompts/full.jsonl` |
| `configs/prompts/curriculum.json` | Named staged curriculum from easy Cyrillic cases through harder style and scene cases. | `data/prompts/curriculum.jsonl` |

Each config records `schema_version`, `mode`, `seed`, `output_path`,
`generation`, `curriculum_stages`, and `validation_thresholds`. Output paths are
repository-relative and must stay under generated-artifact roots such as `data/`,
`outputs/`, or `runs/`.

## Local commands

Generate a quick CPU-safe prompt sample without constructing an LLM client:

```bash
python -m src.prompt_pipeline.generate --config configs/prompts/simple.json --no-llm
```

Generate the full distribution from the committed full config:

```bash
python -m src.prompt_pipeline.generate --config configs/prompts/full.json
```

Generate the staged curriculum:

```bash
python -m src.prompt_pipeline.generate --config configs/prompts/curriculum.json
```

Legacy flag-only invocation remains supported for compatibility:

```bash
python -m src.prompt_pipeline.generate --n 100 --no-llm --output data/prompts_test.jsonl
```

## Stage provenance

When a config is supplied, generated JSONL records include:

- `prompt_mode` — config mode such as `simple`, `full`, or `curriculum`.
- `curriculum_stage` — named stage such as `single_letters` or `style_heavy`.
- `curriculum_family` — normalized DATA-01 family such as `digits`,
  `punctuation`, `multiline`, `style`, or `scene`.

These fields allow later validation, manifests, selection artifacts, and thesis
plots to trace prompts back to explicit curriculum choices.

## Artifact safety

The config files are source contracts and are committed. Generated prompt JSONL
outputs can contain large prompt datasets and should remain under ignored runtime
roots (`data/`, `outputs/`, or `runs/`) unless a future plan intentionally creates
tiny fixtures. Do not commit generated prompt datasets from these commands by
default.
