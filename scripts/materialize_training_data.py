#!/usr/bin/env python3
"""Materialize selected SFT samples or DPO preference pairs from scores CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.selection import materialize_dpo_pairs, materialize_sft_samples  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize SFT selected samples or DPO preference pairs"
    )
    parser.add_argument("--kind", required=True, choices=["sft", "dpo"])
    parser.add_argument("--scores-csv", required=True, help="Input scores CSV path")
    parser.add_argument(
        "--output-dir",
        default="outputs/generated",
        help="Directory for selected_samples.jsonl or preference_pairs.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit JSONL output path; overrides --output-dir default naming",
    )
    parser.add_argument("--manifest", default=None, help="Optional summary/manifest JSON path")
    parser.add_argument("--mode", default=None, help="Selection or pair construction mode")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k-per-prompt", type=int, default=1)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--ambiguity-margin", type=float, default=0.0)
    args = parser.parse_args(argv)

    output_path = _resolve_output_path(args.kind, args.output_dir, args.output)
    if args.kind == "sft":
        summary = materialize_sft_samples(
            args.scores_csv,
            output_path,
            mode=args.mode or "threshold",
            score_column=args.score_column,
            threshold=0.3 if args.threshold is None else args.threshold,
            top_k_per_prompt=args.top_k_per_prompt,
            hard_negative_threshold=args.hard_negative_threshold,
            manifest_path=args.manifest,
        )
    else:
        summary = materialize_dpo_pairs(
            args.scores_csv,
            output_path,
            mode=args.mode or "best_vs_worst",
            score_column=args.score_column,
            threshold=0.5 if args.threshold is None else args.threshold,
            margin=args.margin,
            ambiguity_margin=args.ambiguity_margin,
            manifest_path=args.manifest,
        )

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _resolve_output_path(kind: str, output_dir: str, output: str | None) -> Path:
    if output is not None:
        return Path(output)
    filename = "selected_samples.jsonl" if kind == "sft" else "preference_pairs.jsonl"
    return Path(output_dir) / filename


if __name__ == "__main__":
    sys.exit(main())
