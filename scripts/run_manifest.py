"""CPU-safe command line surface for local run manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.runtime.manifests import (
    ManifestError,
    create_run_manifest,
    load_run_manifest,
    print_manifest_summary,
    update_run_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command_name == "init":
            manifest = create_run_manifest(
                stage=args.stage,
                config_path=args.config,
                command=args.command,
                run_root=args.run_root,
            )
            print(manifest.run_dir)
            return 0
        if args.command_name == "inspect":
            print_manifest_summary(load_run_manifest(args.manifest))
            return 0
        if args.command_name == "note":
            update_run_manifest(args.manifest, note=args.note)
            return 0
        if args.command_name == "metrics":
            update_run_manifest(args.manifest, metrics=_load_metrics(args))
            return 0
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"metrics payload is not valid JSON: {exc}", file=sys.stderr)
        return 2
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local run manifest provenance files.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    init_parser = subparsers.add_parser("init", help="Create a run manifest directory.")
    init_parser.add_argument("--stage", required=True, choices=("sft", "dpo", "masked_sft"))
    init_parser.add_argument("--config", required=True, type=Path)
    init_parser.add_argument("--command", required=True)
    init_parser.add_argument("--run-root", default=Path("runs"), type=Path)

    inspect_parser = subparsers.add_parser("inspect", help="Print a manifest summary.")
    inspect_parser.add_argument("manifest", type=Path)

    note_parser = subparsers.add_parser("note", help="Append a timestamped note.")
    note_parser.add_argument("manifest", type=Path)
    note_parser.add_argument("note")

    metrics_parser = subparsers.add_parser("metrics", help="Merge metrics into a manifest.")
    metrics_parser.add_argument("manifest", type=Path)
    group = metrics_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", dest="json_payload")
    group.add_argument("--file", dest="json_file", type=Path)
    return parser


def _load_metrics(args: argparse.Namespace) -> dict[str, Any]:
    if args.json_file is not None:
        raw = args.json_file.read_text(encoding="utf-8")
    else:
        raw = args.json_payload
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ManifestError("metrics payload must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
