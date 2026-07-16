"""Precompute FLUX text embeddings for a prompt JSONL on a CUDA host."""

from __future__ import annotations

import argparse

from src.runtime.capabilities import check_stage_support


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-id",
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)

    support = check_stage_support("generate")
    if not support.ok:
        parser.error("; ".join(support.errors))

    from src.training.flux2_utils import precompute_text_embeddings

    precompute_text_embeddings(
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
