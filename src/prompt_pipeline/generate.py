#!/usr/bin/env python3
"""Generate a JSONL dataset of text-rendering prompts.

Usage
-----
# Quick test without LLM (algorithmic fallback for all tiers):
  python -m src.prompt_pipeline.generate --n 100 --no-llm --output data/prompts_test.jsonl

# Full generation with local LLM on GPU:
  python -m src.prompt_pipeline.generate --n 30000 --model Qwen/Qwen3.5-4B \
      --output data/prompts.jsonl

# With MLX backend (Apple Silicon):
  python -m src.prompt_pipeline.generate --n 30000 \
      --model mlx-community/Qwen3.5-4B-MLX-4bit --backend mlx

# Expand scene pool before generating:
  python -m src.prompt_pipeline.generate --expand-scenes 50 --model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:          # graceful fallback
    def tqdm(it, **_kw):     # type: ignore[misc]
        return it

from src.data_quality.curriculum import (
    CurriculumConfigError,
    CurriculumStage,
    PromptGenerationConfig,
    load_prompt_generation_config,
)

from .assembler import Assembler
from .config import CONTENT_TYPES, CYRILLIC_LOWER, LANG_RU_RATIO
from .scene_pool import ScenePool
from .style_generator import StyleGenerator
from .text_generator import TextGenerator

# Resolve project root so `python -m src.prompt_pipeline.generate` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

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
    batch_size: int = 1,
    prompt_config: PromptGenerationConfig | None = None,
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
    if batch_size > 1:
        logger.info("Batch size: %d", batch_size)

    meta = _build_generation_plan(
        n=n,
        rng=rng,
        text_gen=text_gen,
        prompt_config=prompt_config,
    )

    records: list[dict] = []
    seen_hashes: set[str] = set()
    llm_calls = 0

    pbar = tqdm(total=n, desc="Generating", unit="prompt")
    pos = 0

    while pos < n:
        chunk_end = min(pos + batch_size, n)
        chunk = meta[pos:chunk_end]

        # --- Phase 1: generate target texts (batch LLM calls) ---
        texts: dict[int, str] = {}      # local index -> text
        llm_jobs: list[tuple[int, int, list[str], str]] = []  # (j, tier, words, ct)

        for j, item in enumerate(chunk):
            ct = item["content_type"]
            tier = item["tier"]
            case = item["case"]
            lang = item["lang"]
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

        # --- Phase 2: assemble records ---
        for j, item in enumerate(chunk):
            content_type = item["content_type"]
            tier = item["tier"]
            case = item["case"]
            lang = item["lang"]
            i = pos + j
            target_text = _apply_stage_text_policy(texts[j], item, text_gen, rng)

            # Occasionally append a number (15% chance for posters/social)
            if content_type in ("poster", "social_media", "product") and rng.random() < 0.15:
                target_text = f"{target_text} {text_gen.generate_number_text()}"

            # --- Deduplicate ---
            h = f"{target_text}|{content_type}"
            if h in seen_hashes:
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
            if prompt_config is not None:
                record.update(
                    {
                        "prompt_mode": prompt_config.mode,
                        "curriculum_stage": item["stage_name"],
                        "curriculum_family": item["stage_family"],
                    }
                )
            records.append(record)

        pbar.update(chunk_end - pos)
        pos = chunk_end

    pbar.close()

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


def _build_generation_plan(
    *,
    n: int,
    rng: random.Random,
    text_gen: TextGenerator,
    prompt_config: PromptGenerationConfig | None,
) -> list[dict[str, object]]:
    if prompt_config is None:
        return [_sample_legacy_metadata(rng, text_gen) for _ in range(n)]

    plan: list[dict[str, object]] = []
    for stage, count in prompt_config.expanded_stages():
        for _ in range(count):
            plan.append(_sample_stage_metadata(stage, rng, text_gen))
    if len(plan) > n:
        return plan[:n]
    while len(plan) < n:
        plan.append(_sample_legacy_metadata(rng, text_gen))
        plan[-1]["stage_name"] = "unallocated"
        plan[-1]["stage_family"] = "fallback"
    return plan


def _sample_legacy_metadata(rng: random.Random, text_gen: TextGenerator) -> dict[str, object]:
    ct_names = list(CONTENT_TYPES.keys())
    ct_weights = list(CONTENT_TYPES.values())
    content_type = rng.choices(ct_names, ct_weights, k=1)[0]
    return {
        "content_type": content_type,
        "tier": text_gen.sample_tier(content_type),
        "case": text_gen.sample_case(),
        "lang": "ru" if rng.random() < LANG_RU_RATIO else "en",
        "stage_name": None,
        "stage_family": None,
    }


def _sample_stage_metadata(
    stage: CurriculumStage, rng: random.Random, text_gen: TextGenerator
) -> dict[str, object]:
    content_types = list(stage.content_types) or list(CONTENT_TYPES.keys())
    content_type = rng.choice(content_types)
    tiers = list(stage.tiers) or _default_tiers_for_family(stage.family)
    cases = list(stage.cases) or _default_cases_for_family(stage.family)
    languages = list(stage.languages) or ["ru"]
    return {
        "content_type": content_type,
        "tier": rng.choice(tiers) if tiers else text_gen.sample_tier(content_type),
        "case": rng.choice(cases),
        "lang": rng.choice(languages),
        "stage_name": stage.name,
        "stage_family": stage.family,
    }


def _default_tiers_for_family(family: str) -> list[int]:
    if family == "single_letters":
        return [1]
    if family in {"short_words", "digits"}:
        return [1, 2]
    if family == "multiline":
        return [4, 5]
    return [2, 3, 4]


def _default_cases_for_family(family: str) -> list[str]:
    if family == "mixed_case":
        return ["mixed", "title"]
    return ["upper", "title", "lower"]


def _apply_stage_text_policy(
    text: str,
    item: dict[str, object],
    text_gen: TextGenerator,
    rng: random.Random,
) -> str:
    family = item.get("stage_family")
    if family == "single_letters":
        char = rng.choice(CYRILLIC_LOWER)
        case = item.get("case")
        return char.upper() if case == "upper" else char
    if family == "digits":
        return f"{text} {text_gen.generate_number_text()}"
    if family == "punctuation":
        return f"{text}{rng.choice(['!', '?', '.', ':'])}"
    if family == "mixed_case":
        return _to_mixed_case(text)
    if family == "multiline" and "\n" not in text:
        parts = text.split()
        if len(parts) >= 2:
            midpoint = len(parts) // 2
            return f"{' '.join(parts[:midpoint])}\n{' '.join(parts[midpoint:])}"
    return text


def _to_mixed_case(text: str) -> str:
    chars = [ch.upper() if index % 2 == 0 else ch.lower() for index, ch in enumerate(text)]
    return "".join(chars)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text-rendering prompt dataset")
    parser.add_argument("--config", type=str, default=None,
                        help="Prompt generation config JSON (simple/full/curriculum)")
    parser.add_argument("--n", type=int, default=30_000, help="Number of prompts to generate")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "prompts.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM, use algorithmic fallback for all tiers")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model ID for phrase generation")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "mlx", "vllm"],
                        help="LLM backend (vllm enables FP8 + batch inference)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="LLM batch size (>1 useful with vllm backend)")
    parser.add_argument("--expand-scenes", type=int, default=0,
                        help="Generate N additional scenes per content type before main run")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    prompt_config = None
    generation_n = args.n
    output_path = args.output
    seed = args.seed
    no_llm = args.no_llm
    model = args.model
    backend = args.backend
    batch_size = args.batch_size
    temperature = args.temperature
    expand_scenes = args.expand_scenes

    if args.config:
        try:
            prompt_config = load_prompt_generation_config(args.config)
        except CurriculumConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        generation_n = prompt_config.generation.n
        output_path = str(prompt_config.output_path)
        seed = prompt_config.seed
        no_llm = args.no_llm or prompt_config.generation.no_llm
        model = prompt_config.generation.model
        backend = prompt_config.generation.backend
        batch_size = prompt_config.generation.batch_size
        temperature = prompt_config.generation.temperature
        expand_scenes = prompt_config.generation.expand_scenes

    llm = None
    if not no_llm:
        from .llm_client import LLMClient
        llm = LLMClient(
            model_id=model,
            backend=backend,
            temperature=temperature,
        )

    # Optional: expand scene pool
    if expand_scenes > 0:
        if llm is None:
            logger.error("--expand-scenes requires LLM (remove --no-llm)")
            return 1
        scene_pool = ScenePool(str(DATA_DIR / "scenes_seed.json"), seed=seed)
        expand_scene_pool(llm, scene_pool, per_type=expand_scenes)

    generate_dataset(
        n=generation_n,
        output_path=output_path,
        llm=llm,
        seed=seed,
        batch_size=batch_size,
        prompt_config=prompt_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
