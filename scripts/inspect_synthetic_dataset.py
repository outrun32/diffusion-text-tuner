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
            source_paths=_source_paths(data_dir, raw_dir, args.ocr_results),
            filtering_stats={
                "accepted": report.accepted_count,
                "rejected": report.rejected_count,
                "rejection_reasons": report.rejection_reasons,
                "missing_files": report.missing_files,
            },
            output_counts={"samples": report.sample_count},
            metadata={
                "quality_report_path": str(args.report),
                "contact_sheet_path": str(args.contact_sheet) if args.contact_sheet else None,
                "thresholds": thresholds,
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


def _source_paths(data_dir: Path, raw_dir: Path | None, ocr_results: str | None) -> list[Path]:
    paths = [data_dir / "index.csv", data_dir / "prompts.jsonl", data_dir / "shapes.csv"]
    if raw_dir is not None:
        paths.append(raw_dir / "index.jsonl")
    if ocr_results:
        paths.append(Path(ocr_results))
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
