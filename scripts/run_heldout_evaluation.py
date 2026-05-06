"""Materialize CPU-safe held-out evaluation plans without running models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.heldout import HeldoutEvaluationError, write_evaluation_plan


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        write_evaluation_plan(
            args.config,
            output_plan=args.output_plan,
            markdown_summary=args.markdown_summary,
        )
    except HeldoutEvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"could not write held-out evaluation report: {exc}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a held-out evaluation config and materialize JSON/Markdown "
            "plans without running FLUX generation or reward scoring."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Held-out evaluation config JSON.",
    )
    parser.add_argument(
        "--output-plan",
        required=True,
        type=Path,
        help="Path for the materialized held-out evaluation JSON plan.",
    )
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        help="Optional Markdown summary path for human review.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
