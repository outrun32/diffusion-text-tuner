#!/usr/bin/env python3
"""Compare generated reward-filtered data with synthetic masked-SFT evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_quality.source_comparison import compare_data_sources  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare reward-filtered generated data against synthetic masked-SFT data"
    )
    parser.add_argument("--generated-prompt-quality-report", default=None)
    parser.add_argument("--selected-samples", default=None)
    parser.add_argument("--preference-pairs", default=None)
    parser.add_argument("--generated-dataset-manifest", default=None)
    parser.add_argument("--synthetic-quality-report", default=None)
    parser.add_argument("--synthetic-manifest", default=None)
    parser.add_argument(
        "--output-report", default=None, help="Optional JSON comparison report path"
    )
    parser.add_argument("--markdown-summary", default=None, help="Optional Markdown summary path")
    args = parser.parse_args(argv)

    comparison = compare_data_sources(
        generated_prompt_quality_report=args.generated_prompt_quality_report,
        selected_samples=args.selected_samples,
        preference_pairs=args.preference_pairs,
        generated_dataset_manifest=args.generated_dataset_manifest,
        synthetic_quality_report=args.synthetic_quality_report,
        synthetic_manifest=args.synthetic_manifest,
    )
    payload = comparison.to_dict()
    if args.output_report is not None:
        _write_json(args.output_report, payload)
    if args.markdown_summary is not None:
        _write_text(args.markdown_summary, _markdown_summary(payload))
    print(_stdout_summary(payload))
    return 0


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _stdout_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    generated_count = counts.get("generated_selected_samples")
    synthetic_count = counts.get("synthetic_samples")
    available = payload.get("evidence_available", [])
    missing = payload.get("evidence_missing", [])
    return (
        "Data source comparison: "
        f"generated selected samples: {_count_label(generated_count)}, "
        f"synthetic samples: {_count_label(synthetic_count)}, "
        f"evidence available: {len(available)}, evidence missing: {len(missing)}"
    )


def _markdown_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    rare = (
        payload.get("rare_character_coverage", {})
        if isinstance(payload.get("rare_character_coverage"), dict)
        else {}
    )
    return "\n".join(
        [
            "# Data Source Comparison",
            "",
            "Compare reward-filtered generated images with synthetic masked-SFT evidence.",
            "",
            "## Evidence",
            "",
            f"- Available: {', '.join(payload.get('evidence_available', [])) or 'none'}",
            f"- Missing: {', '.join(payload.get('evidence_missing', [])) or 'none'}",
            "",
            "## Counts",
            "",
            "- Generated selected samples: "
            f"{_count_label(counts.get('generated_selected_samples'))}",
            "- Generated preference pairs: "
            f"{_count_label(counts.get('generated_preference_pairs'))}",
            f"- Synthetic samples: {_count_label(counts.get('synthetic_samples'))}",
            f"- Synthetic accepted/rejected: {_count_label(counts.get('synthetic_accepted'))} / "
            f"{_count_label(counts.get('synthetic_rejected'))}",
            "",
            "## Rare-character coverage",
            "",
            f"- Overlap: {', '.join(rare.get('overlap', [])) or 'none'}",
            f"- Generated-only: {', '.join(rare.get('generated_only', [])) or 'none'}",
            f"- Synthetic-only: {', '.join(rare.get('synthetic_only', [])) or 'none'}",
            "",
            "## Interpretation",
            "",
            "- Generated reward-filtered data aligns training to FLUX outputs and rewards.",
            "- Synthetic masked-SFT data helps controlled reconstruction with masks.",
            "- Generated data can inherit reward/OCR false positives and prompt distribution gaps.",
            "- Synthetic data can miss natural scene realism and domain complexity.",
            "- Treat training loss or DPO accuracy as internal until held-out evaluation.",
            "",
        ]
    )


def _count_label(value: object) -> str:
    return "unavailable" if value is None else str(value)


if __name__ == "__main__":
    sys.exit(main())
