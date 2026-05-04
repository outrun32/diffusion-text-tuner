#!/usr/bin/env python3
"""Validate prompt JSONL quality and optionally emit a dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.data_quality.manifests import create_dataset_manifest, write_dataset_manifest
from src.data_quality.prompt_validation import validate_prompt_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate prompt dataset quality")
    parser.add_argument("--input", required=True, help="Prompt JSONL dataset to validate")
    parser.add_argument("--report", default=None, help="Optional JSON quality report output path")
    parser.add_argument("--manifest", default=None, help="Optional dataset manifest JSON output path")
    parser.add_argument("--config", default=None, help="Optional prompt generation config JSON")
    parser.add_argument("--strict-warnings", action="store_true", help="Return nonzero for warnings")
    parser.add_argument("--min-target-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument("--required-rare-characters", default=None)
    parser.add_argument("--min-rare-character-coverage", type=float, default=None)
    parser.add_argument("--max-duplicate-rate", type=float, default=None)
    parser.add_argument(
        "--allowed-scripts",
        default=None,
        help="Comma-separated scripts: cyrillic,latin,digits,punctuation",
    )
    args = parser.parse_args(argv)

    thresholds = _thresholds_from_args(args)
    input_path = Path(args.input)
    report = validate_prompt_dataset(input_path, thresholds=thresholds)
    report_payload = report.to_dict()

    if args.report:
        _write_json(Path(args.report), report_payload)
    else:
        print(json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True))

    if args.manifest:
        source_paths: list[Path] = [input_path]
        if args.config:
            source_paths.append(Path(args.config))
        manifest = create_dataset_manifest(
            dataset_kind="prompt",
            dataset_paths=[input_path],
            config_path=args.config,
            seed_strategy=_seed_strategy_from_config(args.config),
            source_paths=source_paths,
            filtering_stats={
                "valid_records": report.valid_records,
                "malformed_records": report.malformed_records,
                "missing_required_records": report.missing_required_records,
                "warnings": len(report.warnings),
                "errors": len(report.errors),
            },
            output_counts={"valid_prompts": report.valid_records},
            metadata={"quality_report_path": str(args.report) if args.report else None},
        )
        write_dataset_manifest(args.manifest, manifest)

    if report.errors:
        return 2
    if args.strict_warnings and report.warnings:
        return 1
    return 0


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    optional_values = {
        "min_target_length": args.min_target_length,
        "max_target_length": args.max_target_length,
        "min_rare_character_coverage": args.min_rare_character_coverage,
        "max_duplicate_rate": args.max_duplicate_rate,
    }
    for key, value in optional_values.items():
        if value is not None:
            thresholds[key] = value
    if args.required_rare_characters:
        thresholds["required_rare_characters"] = _split_csv(args.required_rare_characters)
    if args.allowed_scripts:
        thresholds["allowed_scripts"] = _split_csv(args.allowed_scripts)
    return thresholds


def _seed_strategy_from_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    seed = payload.get("seed")
    return {"prompt.seed": seed} if isinstance(seed, int) else {}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
