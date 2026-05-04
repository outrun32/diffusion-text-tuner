#!/usr/bin/env python3
"""Inspect synthetic masked-SFT dataset quality and optional contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_quality.manifests import create_dataset_manifest, write_dataset_manifest  # noqa: E402
from src.data_quality.synthetic_quality import (  # noqa: E402
    create_synthetic_contact_sheet,
    inspect_synthetic_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect synthetic masked-SFT dataset quality")
    parser.add_argument("--data-dir", required=True, help="masked_sft dataset directory")
    parser.add_argument("--raw-dir", default=None, help="raw SynthTIGER output directory")
    parser.add_argument("--report", required=True, help="JSON quality report output path")
    parser.add_argument(
        "--manifest", default=None, help="Optional dataset manifest JSON output path"
    )
    parser.add_argument(
        "--contact-sheet", default=None, help="Optional contact sheet PNG output path"
    )
    parser.add_argument("--contact-sheet-samples", type=int, default=12)
    parser.add_argument("--ocr-results", default=None, help="Optional OCR CSV/JSONL result path")
    parser.add_argument("--config", default=None, help="Synthetic build config path")
    parser.add_argument("--seed", type=int, default=None, help="Synthetic build seed")
    parser.add_argument("--template", default=None, help="SynthTIGER template path/name")
    parser.add_argument("--runner", default=None, help="Synthetic builder/runner path/name")
    parser.add_argument("--model-id", default=None, help="Model ID used for latent/text baking")
    parser.add_argument("--model-revision", default=None, help="Optional model revision")
    parser.add_argument("--word-source", action="append", default=[], help="Word/source text file")
    parser.add_argument("--font-source", action="append", default=[], help="Font source file/list")
    parser.add_argument("--scene-source", action="append", default=[], help="Scene source file")
    parser.add_argument(
        "--background-source", action="append", default=[], help="Background source file"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-mask-area-fraction", type=float, default=None)
    parser.add_argument("--max-mask-area-fraction", type=float, default=None)
    parser.add_argument("--min-bbox-height-fraction", type=float, default=None)
    parser.add_argument("--min-bbox-area-fraction", type=float, default=None)
    parser.add_argument("--min-contrast", type=float, default=None)
    parser.add_argument("--max-text-length", type=int, default=None)
    args = parser.parse_args(argv)

    thresholds = _thresholds_from_args(args)
    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    report = inspect_synthetic_dataset(
        data_dir,
        raw_dir=raw_dir,
        ocr_results=args.ocr_results,
        thresholds=thresholds,
        max_samples=args.max_samples,
    )
    report_payload = report.to_dict()
    _write_json(Path(args.report), report_payload)

    if args.contact_sheet:
        create_synthetic_contact_sheet(
            report,
            args.contact_sheet,
            max_samples=args.contact_sheet_samples,
        )

    if args.manifest:
        manifest = create_dataset_manifest(
            dataset_kind="synthetic",
            dataset_paths=[data_dir],
            config_path=args.config,
            config_snapshot=_config_snapshot(args),
            seed_strategy=_seed_strategy(args),
            source_paths=_source_paths(data_dir, raw_dir, args),
            filtering_stats={
                "accepted": report.accepted_count,
                "rejected": report.rejected_count,
                "rejection_reasons": report.rejection_reasons,
                "missing_files": report.missing_files,
            },
            output_counts={"samples": report.sample_count},
            model_metadata=_model_metadata(args),
            metadata={
                "quality_report_path": str(args.report),
                "contact_sheet_path": str(args.contact_sheet) if args.contact_sheet else None,
                "thresholds": thresholds,
                "template": args.template,
                "runner": args.runner,
            },
        )
        write_dataset_manifest(args.manifest, manifest)

    return 0 if report.ok else 1


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    optional_values = {
        "min_mask_area_fraction": args.min_mask_area_fraction,
        "max_mask_area_fraction": args.max_mask_area_fraction,
        "min_bbox_height_fraction": args.min_bbox_height_fraction,
        "min_bbox_area_fraction": args.min_bbox_area_fraction,
        "min_contrast": args.min_contrast,
        "max_text_length": args.max_text_length,
    }
    for key, value in optional_values.items():
        if value is not None:
            thresholds[key] = value
    return thresholds


def _seed_strategy(args: argparse.Namespace) -> dict[str, Any]:
    return {"seed": args.seed} if args.seed is not None else {}


def _config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if args.seed is not None:
        snapshot["seed"] = args.seed
    if args.template:
        snapshot["template"] = args.template
    if args.runner:
        snapshot["runner"] = args.runner
    if args.model_id:
        snapshot["model_id"] = args.model_id
    if args.model_revision:
        snapshot["model_revision"] = args.model_revision
    return snapshot


def _model_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if args.model_id:
        metadata["model_id"] = args.model_id
    if args.model_revision:
        metadata["model_revision"] = args.model_revision
    return metadata


def _source_paths(data_dir: Path, raw_dir: Path | None, args: argparse.Namespace) -> list[Path]:
    paths = [data_dir / "index.csv", data_dir / "prompts.jsonl", data_dir / "shapes.csv"]
    if raw_dir is not None:
        paths.append(raw_dir / "index.jsonl")
    if args.ocr_results:
        paths.append(Path(args.ocr_results))
    for group in (
        args.word_source,
        args.font_source,
        args.scene_source,
        args.background_source,
    ):
        paths.extend(Path(item) for item in group)
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
