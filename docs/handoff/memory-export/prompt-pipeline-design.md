# Prompt Dataset Generation Pipeline — IMPLEMENTED

## Location: src/prompt_pipeline/
## Run: python -m src.prompt_pipeline.generate --n 30000 --no-llm (38s no-LLM, ~1-2h with LLM on 5090)

## Modules
- config.py — all constants, content types, tiers, styles, RU translations, LLM prompts
- text_generator.py — freq dict loading, coverage-weighted sampling, tier 1-2 algorithmic, fallback for 3-5
- style_generator.py — combinatoric (fonts x colors x effects x sizes) with per-content constraints
- scene_pool.py — load/sample/expand scene descriptions from JSON pool
- llm_client.py — transformers + mlx backends, phrase generation (T3-T5), scene expansion
- assembler.py — 5 RU + 4 EN prompt templates, style translations, placement by content type
- generate.py — CLI with argparse, content-type quotas, JSONL output

## Data files: data/
- ru_freq_50k.txt — FrequencyWords (OpenSubtitles 2018), 48696 valid Cyrillic words after filter
- scenes_seed.json — 67 seed scenes across 9 content types (RU + EN descriptions)
- thematic.json — cities (50), names (60), brands (30), slogans (25), number templates

## Content types (no documents/signs — focused on creative/visual):
poster 25%, photo_text 20%, typography 15%, product 10%, social_media 10%,
clothing 8%, book_cover 7%, street_art 3%, niche 2%

## Coverage (30k samples, no-LLM): all rare chars >=0.9% (Щ:9814, Ъ:9393, Ё:10882)
## Key optimization: weight caching with refresh every 200 updates (90x speedup)
