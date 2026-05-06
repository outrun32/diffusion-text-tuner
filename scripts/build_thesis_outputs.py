#!/usr/bin/env python3
"""Thin CLI for CPU-safe thesis output bundle generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.thesis_outputs import (  # noqa: E402
    ThesisOutputError,
    build_thesis_output_bundle,
    write_thesis_output_bundle,
    write_thesis_output_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Thesis output bundle config JSON")
    parser.add_argument("--output-bundle", required=True, help="Destination bundle JSON path")
    parser.add_argument("--markdown-summary", help="Optional destination Markdown summary path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        bundle = build_thesis_output_bundle(args.config, require_ready=False)
        write_thesis_output_bundle(bundle, args.output_bundle)
        if args.markdown_summary:
            write_thesis_output_markdown(bundle, args.markdown_summary)
        if not bundle["readiness"]["ready"]:
            raise ThesisOutputError(
                "thesis output bundle is not thesis-ready: "
                + "; ".join(bundle["readiness"]["blocking_errors"])
            )
    except ThesisOutputError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
