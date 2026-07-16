#!/usr/bin/env python3
"""Generate a SIMPLIFIED dataset for curriculum-style training.

Focus: short text (1-3 words), clean fonts only, no cursive/handwritten.
Goal: teach the model to render simple Cyrillic correctly before scaling up.

Usage:
  python -m scripts.generate_simple_dataset --n 15000 --model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.prompt_pipeline import config
from src.prompt_pipeline.assembler import Assembler
from src.prompt_pipeline.scene_pool import ScenePool
from src.prompt_pipeline.text_generator import TextGenerator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── Simplified config overrides ──────────────────────────────────────────

# Only clean, geometric fonts that diffusion models render well
SIMPLE_FONTS = [
    "bold sans-serif",
    "thin sans-serif",
    "serif",
    "monospace",
    "stencil",
    "minimalist",
    "retro",
    "futuristic",
    "gothic",
]

# Only clean, high-contrast effects
SIMPLE_EFFECTS = [
    "clean",
    "embossed",
    "shadow",
    "outlined",
    "gradient",
    "3D",
    "neon glow",
]

# Larger text only — small/tiny is too hard for the model
SIMPLE_SIZES = ["large", "medium"]

# Tier distribution: heavily weighted toward short text
# Tier 1: single word (2-8 chars)
# Tier 2: 2-3 words (up to ~25 chars)
# Tier 3: short phrase 3-5 words (LLM) — small amount for variety
SIMPLE_TIER_WEIGHTS = {
    1: 0.35,  # single word
    2: 0.45,  # 2-3 words
    3: 0.20,  # short phrase (3-5 words, LLM)
    4: 0.00,  # disabled — title+subtitle too complex
    5: 0.00,  # disabled — sentences too complex
}

# Content types — keep variety but remove hardest ones
SIMPLE_CONTENT_TYPES = {
    "poster": 0.30,
    "photo_text": 0.20,
    "typography": 0.15,
    "product": 0.10,
    "social_media": 0.10,
    "clothing": 0.08,
    "book_cover": 0.07,
}

# 100% Russian
LANG_RU_RATIO = 1.0

# Case: more uppercase (model renders CAPS better)
SIMPLE_CASE_WEIGHTS = {
    "upper": 0.60,
    "title": 0.30,
    "lower": 0.10,
    "mixed": 0.00,
}

# Shorter LLM prompt for tier 3: cap at 5 words
LLM_SIMPLE_PROMPT = {
    3: (
        "Придумай короткую фразу из 3-5 слов для {content_type_ru}. "
        "Обязательно используй слова: {words}. "
        "Фраза должна быть короткой, максимум 5 слов. "
        "Ответь одной строкой."
    ),
}


def patch_config():
    """Override module-level config constants for simplified dataset.

    IMPORTANT: dicts/lists are mutated IN-PLACE so that modules which did
    ``from .config import TIER_WEIGHTS`` see the updated values.
    """
    # Mutable containers: clear + update in-place
    config.FONTS.clear()
    config.FONTS.extend(SIMPLE_FONTS)

    config.EFFECTS.clear()
    config.EFFECTS.extend(SIMPLE_EFFECTS)

    config.SIZES.clear()
    config.SIZES.extend(SIMPLE_SIZES)

    config.TIER_WEIGHTS.clear()
    config.TIER_WEIGHTS.update(SIMPLE_TIER_WEIGHTS)

    config.TIER_OVERRIDES.clear()

    config.CONTENT_TYPES.clear()
    config.CONTENT_TYPES.update(SIMPLE_CONTENT_TYPES)

    config.CASE_WEIGHTS.clear()
    config.CASE_WEIGHTS.update(SIMPLE_CASE_WEIGHTS)

    config.LLM_PHRASE_PROMPTS.clear()
    config.LLM_PHRASE_PROMPTS.update(LLM_SIMPLE_PROMPT)

    # Scalars: simple attribute assignment is fine
    config.LANG_RU_RATIO = LANG_RU_RATIO

    # Remove cursive/handwritten from all style constraints
    config.STYLE_CONSTRAINTS.clear()
    config.STYLE_CONSTRAINTS.update(
        {
            "typography": {
                "fonts": ["gothic", "futuristic", "bold sans-serif", "stencil"],
                "effects": ["clean", "gradient", "3D", "outlined", "neon glow"],
                "sizes": ["large", "medium"],
            },
            "book_cover": {
                "fonts": ["serif", "gothic", "minimalist"],
            },
            "clothing": {
                "fonts": ["bold sans-serif", "minimalist", "stencil", "retro", "futuristic"],
                "sizes": ["large", "medium"],
            },
        }
    )


def generate_dataset(
    n: int,
    output_path: str,
    llm=None,
    seed: int = 42,
    batch_size: int = 1,
):
    patch_config()

    rng = random.Random(seed)

    text_gen = TextGenerator(
        freq_dict_path=str(DATA_DIR / "ru_freq_50k.txt"),
        thematic_path=str(DATA_DIR / "thematic.json"),
        seed=seed,
    )
    # Override the tier weights in the text generator too
    # These are already patched by patch_config()

    from src.prompt_pipeline.style_generator import StyleGenerator

    style_gen = StyleGenerator(seed=seed)

    expanded = DATA_DIR / "scenes_expanded.json"
    scene_pool = ScenePool(
        seed_path=str(DATA_DIR / "scenes_seed.json"),
        expanded_path=str(expanded) if expanded.exists() else None,
        seed=seed,
    )
    assembler = Assembler(seed=seed)

    logger.info("Scene pool: %d scenes", len(scene_pool))
    logger.info("Frequency dict: %d words", len(text_gen.words))
    logger.info("LLM: %s", "enabled" if llm else "disabled (fallback mode)")
    logger.info("Fonts: %s", SIMPLE_FONTS)
    logger.info("Tiers: %s", SIMPLE_TIER_WEIGHTS)

    ct_names = list(SIMPLE_CONTENT_TYPES.keys())
    ct_weights = list(SIMPLE_CONTENT_TYPES.values())

    # Pre-sample metadata
    meta: list[tuple[str, int, str, str]] = []
    for _ in range(n):
        ct = rng.choices(ct_names, ct_weights, k=1)[0]
        tier = text_gen.sample_tier(ct)
        case = text_gen.sample_case()
        lang = "ru"  # Always Russian
        meta.append((ct, tier, case, lang))

    records: list[dict] = []
    seen_hashes: set[str] = set()
    llm_calls = 0
    skipped_long = 0

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **_kw):
            return it

    pbar = tqdm(total=n, desc="Generating", unit="prompt")
    pos = 0

    while pos < n:
        chunk_end = min(pos + batch_size, n)
        chunk = meta[pos:chunk_end]

        texts: dict[int, str] = {}
        llm_jobs: list[tuple[int, int, list[str], str]] = []

        for j, (ct, tier, case, _lang) in enumerate(chunk):
            if tier <= 2:
                texts[j] = text_gen.generate_text(tier, case)
            elif llm is not None:
                must_include = text_gen.get_must_include_words(tier)
                llm_jobs.append((j, tier, must_include, ct))
            else:
                texts[j] = text_gen.generate_text_fallback(tier, case)

        if llm_jobs:
            batch_results = llm.generate_phrases_batch(
                [(tier, mi, ct) for _, tier, mi, ct in llm_jobs]
            )
            for (j, _tier, _mi, _ct), txt in zip(llm_jobs, batch_results, strict=True):
                texts[j] = txt
            llm_calls += len(llm_jobs)

        for j, (content_type, tier, _case, lang) in enumerate(chunk):
            i = pos + j
            target_text = texts[j]

            # 15% number append for posters/social/product
            if content_type in ("poster", "social_media", "product") and rng.random() < 0.15:
                target_text = f"{target_text} {text_gen.generate_number_text()}"

            # Enforce max length: skip if too long (regeneration would be complex)
            if len(target_text) > 30:
                # Truncate to last complete word within limit
                words = target_text.split()
                truncated = ""
                for w in words:
                    test = f"{truncated} {w}".strip() if truncated else w
                    if len(test) <= 30:
                        truncated = test
                    else:
                        break
                if truncated:
                    target_text = truncated
                    skipped_long += 1
                else:
                    target_text = words[0][:30]  # Take first word truncated
                    skipped_long += 1

            # Deduplicate
            h = f"{target_text}|{content_type}"
            if h in seen_hashes:
                target_text += f" {text_gen.generate_number_text()}"
                h = f"{target_text}|{content_type}"
            seen_hashes.add(h)

            # Scene & style
            scene = scene_pool.sample(content_type, lang)
            style = style_gen.sample(content_type)

            # Assemble prompt
            prompt = assembler.assemble(target_text, scene, style, content_type, lang)

            # Char coverage
            text_gen.update_coverage(target_text)
            char_cov = {}
            for ch in target_text.lower():
                if ch in config.CYRILLIC_LOWER:
                    char_cov[ch] = char_cov.get(ch, 0) + 1

            records.append(
                {
                    "id": f"p_{i:05d}",
                    "prompt": prompt,
                    "target_text": target_text,
                    "tier": tier,
                    "content_type": content_type,
                    "scene_id": scene["id"],
                    "style": style,
                    "lang": lang,
                    "char_coverage": char_cov,
                }
            )

        pbar.update(chunk_end - pos)
        pos = chunk_end

    pbar.close()

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to %s", len(records), out)
    logger.info("LLM calls: %d", llm_calls)
    logger.info("Truncated to <=30 chars: %d", skipped_long)

    # Distribution stats
    tier_counts = Counter(r["tier"] for r in records)
    ct_counts = Counter(r["content_type"] for r in records)
    text_lens = [len(r["target_text"]) for r in records]

    logger.info("\n=== DISTRIBUTION ===")
    logger.info("Tier distribution:")
    for t in sorted(tier_counts):
        logger.info(
            "  Tier %d: %d (%.1f%%)", t, tier_counts[t], 100 * tier_counts[t] / len(records)
        )
    logger.info("Content type distribution:")
    for ct in sorted(ct_counts, key=ct_counts.get, reverse=True):
        logger.info("  %s: %d (%.1f%%)", ct, ct_counts[ct], 100 * ct_counts[ct] / len(records))
    logger.info(
        "Text length: min=%d, max=%d, mean=%.1f, median=%.1f",
        min(text_lens),
        max(text_lens),
        sum(text_lens) / len(text_lens),
        sorted(text_lens)[len(text_lens) // 2],
    )

    # Character coverage
    cov = text_gen.coverage_report()
    total_chars = sum(cov.values())
    rare = "щъёэюфц"
    logger.info("Character coverage (rare):")
    for ch in rare:
        cnt = cov.get(ch, 0)
        logger.info("  '%s': %d  (%.1f%%)", ch, cnt, 100 * cnt / total_chars if total_chars else 0)


def main():
    parser = argparse.ArgumentParser(description="Generate simplified text-rendering dataset")
    parser.add_argument("--n", type=int, default=15_000)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "prompts_simple.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--backend", type=str, default="transformers", choices=["transformers", "mlx", "vllm"]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Patch config BEFORE anything imports it
    patch_config()

    llm = None
    if not args.no_llm:
        logger.info("Loading LLM: %s (backend=%s)", args.model, args.backend)
        from src.prompt_pipeline.llm_client import LLMClient

        llm = LLMClient(
            model_id=args.model,
            backend=args.backend,
            temperature=args.temperature,
            seed=args.seed,
        )
        logger.info("LLM loaded")

    generate_dataset(
        n=args.n,
        output_path=args.output,
        llm=llm,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
