"""Thin CLI wrapper for the synthetic Cyrillic dataset builder.

Usage:
    python -m scripts.synth.build_dataset \
        --num 25000 --workers 8 \
        --template scripts/synth/synthtiger_template.py \
        --template-name CyrillicScene \
        --config configs/synth/cyrillic.yaml \
        --raw-dir data/synth_cyrillic/raw \
        --out-masked data/synth_cyrillic/masked_sft \
        --out-anyword data/synth_cyrillic/anyword_format \
        --bake-latents --encode-text
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synthesis.dataset_builder import (  # noqa: E402
    SynthesisBuildConfig,
    bake_latents_phase,
    build_dataset,
    collate_records,
    encode_text_phase,
    fan_out,
    render_phase,
    write_anyword_json,
    write_masked_index,
)

__all__ = [
    "SynthesisBuildConfig",
    "bake_latents_phase",
    "build_dataset",
    "collate_records",
    "encode_text_phase",
    "fan_out",
    "main",
    "render_phase",
    "write_anyword_json",
    "write_masked_index",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("scripts/synth/synthtiger_template.py"),
    )
    parser.add_argument("--template-name", type=str, default="CyrillicScene")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synth/cyrillic.yaml")
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/synth/run_synthtiger.py"),
        help="wrapper that applies numpy/Pillow shims before invoking synthtiger",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/synth_cyrillic/raw"))
    parser.add_argument(
        "--out-masked",
        type=Path,
        default=Path("data/synth_cyrillic/masked_sft"),
    )
    parser.add_argument(
        "--out-anyword",
        type=Path,
        default=Path("data/synth_cyrillic/anyword_format"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="reuse existing raw/ dir (e.g. when rerunning bake/encode only)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete raw/out directories before rendering/linking",
    )
    parser.add_argument("--bake-latents", action="store_true")
    parser.add_argument("--encode-text", action="store_true")
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = SynthesisBuildConfig(
        num=args.num,
        workers=args.workers,
        template=args.template,
        template_name=args.template_name,
        config=args.config,
        runner=args.runner,
        raw_dir=args.raw_dir,
        out_masked=args.out_masked,
        out_anyword=args.out_anyword,
        seed=args.seed,
        skip_render=args.skip_render,
        clean=args.clean,
        bake_latents=args.bake_latents,
        encode_text=args.encode_text,
        model_id=args.model_id,
        device=args.device,
    )
    return build_dataset(config)


if __name__ == "__main__":
    raise SystemExit(main())
