"""
Score generated images using VLM or OCR reward model.

Reads PNGs from the image generation output, computes scores,
and writes a scores CSV for the SFT/DPO pipeline.

Scorers:
  vlm  — Qwen3.5-9B P(yes) (default, ~2.5GB VRAM, ~500ms/img)
  ocr  — PaddleOCR v5 CER+Entropy (~50MB VRAM, ~10ms/img, needs paddle)
  both — runs both and writes all columns

Usage:
  python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv \
    --scorer vlm

  python -m scripts.score_images \
    --scorer ocr --entropy_lambda 1.5 ...

  python -m scripts.score_images \
    --scorer both ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.scoring.pipeline import (
    CANONICAL_SCORE_COLUMNS,
    ScoringConfig,
    build_canonical_score_row,
    run_scoring,
    write_score_schema_sidecar,
)

__all__ = [
    "CANONICAL_SCORE_COLUMNS",
    "ScoringConfig",
    "build_canonical_score_row",
    "main",
    "run_scoring",
    "write_score_schema_sidecar",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score generated images with VLM")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Dir with {prompt_id}/v{ver}.png images",
    )
    parser.add_argument(
        "--text_embeds_dir",
        type=str,
        required=True,
        help="Dir with {prompt_id}.pt (contains target_text)",
    )
    parser.add_argument("--output_csv", type=str, default="outputs/generated/scores.csv")
    parser.add_argument(
        "--scorer",
        type=str,
        default="vlm",
        choices=["vlm", "ocr", "both"],
        help="Scoring method: vlm (Qwen yes-prob), ocr (CER+entropy), both",
    )
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--entropy_lambda",
        type=float,
        default=1.0,
        help="Lambda for OCR entropy scaling (R = (1-CER)*exp(-λ*H))",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="VLM scoring batch size (1 is safest for memory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-scored entries if CSV exists",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Shard index for parallel scoring (0-based)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for parallel scoring",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Run/evaluation manifest path to link in each score row",
    )
    parser.add_argument(
        "--source_manifest",
        action="append",
        default=[],
        help="Source manifest path to include in the score schema sidecar",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = ScoringConfig(
        images_dir=Path(args.images_dir),
        text_embeds_dir=Path(args.text_embeds_dir),
        output_csv=Path(args.output_csv),
        scorer=args.scorer,
        vlm_model_id=args.vlm_model_id,
        entropy_lambda=args.entropy_lambda,
        batch_size=args.batch_size,
        resume=args.resume,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        manifest_path=args.manifest_path,
        source_manifests=tuple(args.source_manifest),
    )
    run_scoring(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
