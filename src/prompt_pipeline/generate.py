#!/usr/bin/env python3
"""Generate a JSONL dataset of text-rendering prompts.

Usage
-----
# Quick test without LLM (algorithmic fallback for all tiers):
  python -m src.prompt_pipeline.generate --n 100 --no-llm --output data/prompts_test.jsonl

# Full generation with local LLM on GPU:
  python -m src.prompt_pipeline.generate --n 30000 --model Qwen/Qwen3-4B --output data/prompts.jsonl

# With MLX backend (Apple Silicon):
  python -m src.prompt_pipeline.generate --n 30000 --model mlx-community/Qwen3-4B-MLX-4bit --backend mlx

# Expand scene pool before generating:
  python -m src.prompt_pipeline.generate --expand-scenes 50 --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

# Resolve project root so `python -m src.prompt_pipeline.generate` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

from .assembler import Assembler
from .config import CONTENT_TYPES, CYRILLIC_LOWER, LANG_RU_RATIO
from .scene_pool import ScenePool
from .style_generator import StyleGenerator
from .text_generator import TextGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene pool expansion
# ---------------------------------------------------------------------------

def expand_scene_pool(llm, scene_pool: ScenePool, per_type: int = 50):
    """Use the LLM to generate additional scenes for each content type."""
    for ct in CONTENT_TYPES:
        logger.info("Generating %d scenes for '%s' …", per_type, ct)
        descriptions = llm.generate_scenes(ct, n=per_type)
        existing_count = len(scene_pool.scenes)
        new_scenes = []
        for i, desc in enumerate(descriptions):
            sid = f"{ct}_gen_{existing_count + i:04d}"
            new_scenes.append({
                "id": sid,
                "content_type": ct,
                "ru": desc,
                "en": "",  # can be translated later
            })
        scene_pool.add_scenes(new_scenes)
        logger.info("  → added %d scenes (total pool: %d)", len(new_scenes), len(scene_pool))

    out_path = DATA_DIR / "scenes_expanded.json"
    scene_pool.save(str(out_path))
    logger.info("Expanded scene pool saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int,
    output_path: str,
    llm=None,
    seed: int = 42,
):
    rng = random.Random(seed)

    # --- Initialise components ---
    text_gen = TextGenerator(
        freq_dict_path=str(DATA_DIR / "ru_freq_50k.txt"),
        thematic_path=str(DATA_DIR / "thematic.json"),
        seed=seed,
    )
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

    # --- Content-type quota ---
    ct_names = list(CONTENT_TYPES.keys())
    ct_weights = list(CONTENT_TYPES.values())

    records: list[dict] = []
    seen_hashes: set[str] = set()
    llm_calls = 0

    for i in range(n):
        content_type = rng.choices(ct_names, ct_weights, k=1)[0]
        tier = text_gen.sample_tier(content_type)
        case = text_gen.sample_case()
        lang = "ru" if rng.random() < LANG_RU_RATIO else "en"

        # --- Generate target text ---
        if tier <= 2:
            target_text = text_gen.generate_text(tier, case)
        elif llm is not None:
            must_include = text_gen.get_must_include_words(tier)
            target_text = llm.generate_phrase(tier, must_include, content_type)
            llm_calls += 1
        else:
            target_text = text_gen.generate_text_fallback(tier, case)

        # Occasionally append a number (15% chance for posters/social)
        if content_type in ("poster", "social_media", "product") and rng.random() < 0.15:
            target_text = f"{target_text} {text_gen.generate_number_text()}"

        # --- Deduplicate ---
        h = f"{target_text}|{content_type}"
        if h in seen_hashes:
            # Retry with different text (simple approach)
            target_text += f" {text_gen.generate_number_text()}"
            h = f"{target_text}|{content_type}"
        seen_hashes.add(h)

        # --- Scene & style ---
        scene = scene_pool.sample(content_type, lang)
        style = style_gen.sample(content_type)

        # --- Assemble prompt ---
        prompt = assembler.assemble(target_text, scene, style, content_type, lang)

        # --- Char coverage ---
        text_gen.update_coverage(target_text)
        char_cov = {}
        for ch in target_text.lower():
            if ch in CYRILLIC_LOWER:
                char_cov[ch] = char_cov.get(ch, 0) + 1

        record = {
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
        records.append(record)

        if (i + 1) % 1000 == 0:
            logger.info(
                "  [%d/%d] llm_calls=%d  unique_chars=%d",
                i + 1, n, llm_calls, len(text_gen.char_counts),
            )

    # --- Write output ---
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to %s", len(records), out)
    logger.info("LLM calls: %d", llm_calls)

    # --- Coverage summary ---
    cov = text_gen.coverage_report()
    total = sum(cov.values())
    logger.info("Total characters placed: %d", total)
    rare = "щъёэюфц"
    for ch in rare:
        cnt = cov.get(ch, 0)
        logger.info("  '%s': %d  (%.1f%%)", ch, cnt, 100 * cnt / total if total else 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text-rendering prompt dataset")
    parser.add_argument("--n", type=int, default=30_000, help="Number of prompts to generate")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "prompts.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM, use algorithmic fallback for all tiers")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace model ID for phrase generation")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "mlx"],
                        help="LLM backend")
    parser.add_argument("--expand-scenes", type=int, default=0,
                        help="Generate N additional scenes per content type before main run")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    llm = None
    if not args.no_llm:
        from .llm_client import LLMClient
        llm = LLMClient(
            model_id=args.model,
            backend=args.backend,
            temperature=args.temperature,
        )

    # Optional: expand scene pool
    if args.expand_scenes > 0:
        if llm is None:
            logger.error("--expand-scenes requires LLM (remove --no-llm)")
            sys.exit(1)
        scene_pool = ScenePool(str(DATA_DIR / "scenes_seed.json"), seed=args.seed)
        expand_scene_pool(llm, scene_pool, per_type=args.expand_scenes)

    generate_dataset(
        n=args.n,
        output_path=args.output,
        llm=llm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
