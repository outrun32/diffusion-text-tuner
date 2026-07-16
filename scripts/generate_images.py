"""
Generate multiple image versions per prompt, saving latents and PNGs.

This is the first step of the SFT+DPO data pipeline:
  1. generate_images.py  — this file
  2. score_images.py     — score each image with VLM
  3. SFT/DPO training   — train on curated data

Usage:
  python -m scripts.generate_images \
    --prompts data/prompts_simple.jsonl \
    --output_dir outputs/generated \
    --model_id black-forest-labs/FLUX.2-klein-base-4B \
    --versions_per_prompt 5 \
    --batch_size 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.generation.pipeline import GenerationConfig, run_generation

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images for SFT/DPO pipeline")
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL file with prompts")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Optional immutable Hugging Face commit hash.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional LoRA checkpoint to load",
    )
    parser.add_argument("--versions_per_prompt", type=int, default=5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Reserved compatibility option; only 1 is supported.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="Execution device. FLUX generation is CUDA-only in this repository.",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Resume from prompt index")
    parser.add_argument("--end_idx", type=int, default=None, help="Stop at prompt index")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--save_latents",
        action="store_true",
        default=True,
        help="Save VAE-encoded latents for training",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        default=True,
        help="Save decoded PNGs for scoring",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=None,
        help="Optional per-run/shard generation manifest path.",
    )
    parser.add_argument(
        "--run_manifest_path",
        default="",
        help="Optional run-manifest/v1 path linked from the generation manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_generation(
        GenerationConfig(
            prompts=args.prompts,
            output_dir=args.output_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            lora_path=args.lora_path,
            versions_per_prompt=args.versions_per_prompt,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            resolution=args.resolution,
            seed=args.seed,
            device=args.device,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            shard_index=args.shard_index,
            shard_count=args.num_shards,
            save_latents=args.save_latents,
            save_png=args.save_png,
            manifest_path=args.manifest_path,
            run_manifest_path=args.run_manifest_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
