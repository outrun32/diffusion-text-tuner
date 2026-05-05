"""CPU-safe integrated CLI for comparing two recorded training runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.runtime.manifest_diff import compare_run_manifests, format_manifest_diff_markdown
from src.runtime.manifests import ManifestError
from src.training.comparability import (
    compare_training_manifests,
    format_comparability_report,
)

COMPARISON_SCHEMA_VERSION = "training-run-comparison/v1"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = _build_report(args.left_manifest, args.right_manifest)
        output = _render_report(report, markdown=args.markdown)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        has_blocking = bool(report["comparability"]["blocking_mismatches"])
        return 1 if has_blocking and not args.allow_blocking else 0
    except (ManifestError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two training run manifests with both manifest diff and "
            "controlled-field comparability checks."
        )
    )
    parser.add_argument(
        "--left-manifest",
        required=True,
        type=Path,
        help="Path to the left run manifest JSON.",
    )
    parser.add_argument(
        "--right-manifest",
        required=True,
        type=Path,
        help="Path to the right run manifest JSON.",
    )
    parser.add_argument("--markdown", action="store_true", help="Render Markdown instead of JSON.")
    parser.add_argument("--output", type=Path, help="Optional output path for the report.")
    parser.add_argument(
        "--allow-blocking",
        action="store_true",
        help="Exit 0 even when blocking comparability mismatches are present.",
    )
    return parser


def _build_report(left_manifest: Path, right_manifest: Path) -> dict[str, object]:
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "manifest_diff": compare_run_manifests(left_manifest, right_manifest),
        "comparability": compare_training_manifests(left_manifest, right_manifest),
    }


def _render_report(report: dict[str, object], *, markdown: bool) -> str:
    if not markdown:
        return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    manifest_diff = report["manifest_diff"]
    comparability = report["comparability"]
    return "\n".join(
        [
            "# Training run comparison",
            "",
            f"Schema: {report.get('schema_version', COMPARISON_SCHEMA_VERSION)}",
            "",
            "## Manifest diff",
            "",
            format_manifest_diff_markdown(manifest_diff).rstrip(),
            "",
            "## Comparability mismatches",
            "",
            format_comparability_report(comparability).rstrip(),
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
