# Synthesis Configs

Place synthetic prompt, font, background, mask, and rendered dataset variants here.

## Naming

Use `{stage}_{reward_or_data}_{purpose}.json` with `stage=synthetic`.

Examples:

- `synthetic_cyrillic_base.json`
- `synthetic_cyrillic_font_ablation.json`
- `synthetic_multilingual_layout_stress.json`

## Required Contract

Synthesis configs should identify:

- `schema_version`, `stage: synthetic`, and `experiment_name`.
- Prompt/text source, language/script coverage, font/background assets, render parameters, masks, and split strategy.
- Model IDs/revisions only when a generator or encoder is used; pure local rendering configs should still record tool versions and seeds.
- Input roots, output roots, manifest paths, and generated artifact schemas for indexes, shapes, masks, rendered images, latents, and text embeddings.
- Seeds for prompt selection, font/background sampling, rendering, and any latent/text embedding bake steps.

Generated datasets under `data/synth_cyrillic/` and related tensor/image roots remain non-committable runtime artifacts unless a tiny reviewed fixture is intentionally added.
