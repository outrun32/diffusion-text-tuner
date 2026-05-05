"""CPU-safe CLI for checking controlled training comparability fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.runtime.config_io import RuntimeConfigError, load_stage_config, resolve_config_snapshot
from src.runtime.manifests import ManifestError
from src.training.comparability import (
    compare_training_configs,
    compare_training_manifests,
    format_comparability_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = _compare_from_args(args, parser)
        output = _render_report(report, markdown=args.markdown)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        has_blocking = bool(report["blocking_mismatches"])
        return 1 if has_blocking and not args.allow_blocking else 0
    except (RuntimeConfigError, ManifestError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare controlled fields before comparing training outputs."
    )
    parser.add_argument("--left-config", type=Path, help="Left training config JSON path.")
    parser.add_argument("--left-stage", help="Left config stage: sft, dpo, or masked_sft.")
    parser.add_argument("--right-config", type=Path, help="Right training config JSON path.")
    parser.add_argument("--right-stage", help="Right config stage: sft, dpo, or masked_sft.")
    parser.add_argument("--left-manifest", type=Path, help="Left run manifest JSON path.")
    parser.add_argument("--right-manifest", type=Path, help="Right run manifest JSON path.")
    parser.add_argument("--markdown", action="store_true", help="Render Markdown instead of JSON.")
    parser.add_argument("--output", type=Path, help="Optional output path for the report.")
    parser.add_argument(
        "--allow-blocking",
        action="store_true",
        help="Exit 0 even when blocking mismatches are present.",
    )
    return parser


def _compare_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, object]:
    config_args = [args.left_config, args.left_stage, args.right_config, args.right_stage]
    manifest_args = [args.left_manifest, args.right_manifest]
    config_mode = any(value is not None for value in config_args)
    manifest_mode = any(value is not None for value in manifest_args)
    if config_mode == manifest_mode:
        parser.error(
            "choose exactly one mode: --left-config/--left-stage/--right-config/--right-stage "
            "or --left-manifest/--right-manifest"
        )
    if config_mode:
        if not all(value is not None for value in config_args):
            parser.error("config mode requires both configs and both stages")
        left_config = load_stage_config(args.left_stage, args.left_config)
        right_config = load_stage_config(args.right_stage, args.right_config)
        return compare_training_configs(
            resolve_config_snapshot(left_config),
            resolve_config_snapshot(right_config),
            left_label=args.left_stage,
            right_label=args.right_stage,
        )
    if not all(value is not None for value in manifest_args):
        parser.error("manifest mode requires --left-manifest and --right-manifest")
    return compare_training_manifests(args.left_manifest, args.right_manifest)


def _render_report(report: dict[str, object], *, markdown: bool) -> str:
    if markdown:
        return format_comparability_report(report)
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
