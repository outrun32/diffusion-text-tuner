#!/usr/bin/env python3
"""Thin CLI for CPU-safe reward disagreement diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.diagnostics import (  # noqa: E402
    DiagnosticsInputError,
    analyze_reward_disagreement,
    load_score_records,
    write_diagnostics_markdown,
    write_diagnostics_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Recorded score CSV/JSONL/JSON input")
    parser.add_argument("--gold", help="Optional gold diagnostic JSONL benchmark")
    parser.add_argument("--output-report", required=True, help="Destination JSON report path")
    parser.add_argument("--markdown-summary", help="Optional destination Markdown report path")
    parser.add_argument("--contact-sheet", help="Optional bounded PIL contact sheet path")
    parser.add_argument("--contact-sheet-limit", type=int, default=12)
    parser.add_argument("--positive-threshold", type=float, default=0.8)
    parser.add_argument("--negative-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = load_score_records(args.scores)
        report = analyze_reward_disagreement(
            records,
            gold_records=Path(args.gold) if args.gold else None,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            contact_sheet_path=Path(args.contact_sheet) if args.contact_sheet else None,
            contact_sheet_limit=args.contact_sheet_limit,
        )
        if report["input_errors"]:
            raise DiagnosticsInputError("; ".join(report["input_errors"]))
        write_diagnostics_report(report, args.output_report)
        if args.markdown_summary:
            write_diagnostics_markdown(report, args.markdown_summary)
    except DiagnosticsInputError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
