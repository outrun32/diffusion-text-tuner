"""CPU-safe CLI for comparing local run manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.runtime.manifest_diff import compare_run_manifests, format_manifest_diff_markdown
from src.runtime.manifests import ManifestError


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        diff = compare_run_manifests(args.left, args.right)
        if args.markdown:
            output = format_manifest_diff_markdown(diff)
        else:
            output = json.dumps(diff, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return 0
    except (ManifestError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two local run manifest JSON files.")
    parser.add_argument("--left", required=True, type=Path, help="Path to the left manifest JSON.")
    parser.add_argument("--right", required=True, type=Path, help="Path to the right manifest JSON.")
    parser.add_argument("--markdown", action="store_true", help="Render Markdown instead of JSON.")
    parser.add_argument("--output", type=Path, help="Optional output path for the diff report.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
