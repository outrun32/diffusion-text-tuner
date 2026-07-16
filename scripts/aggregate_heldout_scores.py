"""Aggregate canonical held-out score CSV files across fixed seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.score_aggregation import (
    ScoreAggregationError,
    aggregate_score_files,
    parse_seed_input,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="Per-seed canonical score CSV; repeat once for every seed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        inputs = [parse_seed_input(value) for value in args.input]
        aggregate_score_files(inputs, output_path=args.output)
    except (OSError, ScoreAggregationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
