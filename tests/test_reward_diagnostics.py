from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from src.evaluation.diagnostics import (
    analyze_reward_disagreement,
    format_diagnostics_markdown,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_scores_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = sorted({field for record in records for field in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _make_fixture_records(tmp_path: Path) -> list[dict[str, object]]:
    image_paths: list[Path] = []
    for index, color in enumerate(["red", "green", "blue"], start=1):
        path = tmp_path / f"sample-{index}.png"
        Image.new("RGB", (8, 6), color=color).save(path)
        image_paths.append(path)

    return [
        {
            "sample_id": "ok-rare",
            "target_text": "ёжик",
            "detected_text": "ёжик",
            "image_path": str(image_paths[0]),
            "score_vlm": 0.95,
            "score_ocr": 0.90,
            "cer": 0.0,
            "exact_match": True,
            "char_accuracy": 1.0,
            "product_score": 0.92,
            "missing_components": "",
            "font": "serif",
        },
        {
            "sample_id": "false-positive",
            "target_text": "щит 42!",
            "detected_text": "шит 42!",
            "image_path": str(image_paths[1]),
            "score_vlm": 0.91,
            "score_ocr": 0.86,
            "cer": 0.25,
            "exact_match": False,
            "char_accuracy": 0.75,
            "product_score": 0.88,
            "missing_components": "",
            "scene": "street",
        },
        {
            "sample_id": "false-negative",
            "target_text": "Дом",
            "detected_text": "Дом",
            "image_path": str(image_paths[2]),
            "score_vlm": 0.32,
            "score_ocr": 0.35,
            "cer": 0.0,
            "exact_match": True,
            "char_accuracy": 1.0,
            "product_score": 0.34,
            "missing_components": "",
        },
        {
            "sample_id": "missing-ocr",
            "target_text": "строка\nдве",
            "detected_text": "строка две",
            "image_path": str(tmp_path / "missing-image.png"),
            "score_vlm": 0.70,
            "score_ocr": "",
            "cer": "",
            "exact_match": False,
            "product_score": "",
            "missing_components": "score_ocr,cer,product_score",
        },
    ]


def _make_gold_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "ok-rare",
            "target_text": "ёжик",
            "image_path": "fixtures/ok-rare.png",
            "expected_exact_match": True,
            "expected_ocr_detected": True,
            "human_label": "pass",
        },
        {
            "sample_id": "false-positive",
            "target_text": "щит 42!",
            "image_path": "fixtures/false-positive.png",
            "expected_exact_match": False,
            "expected_ocr_detected": True,
            "human_label": "fail",
        },
        {
            "sample_id": "false-negative",
            "target_text": "Дом",
            "image_path": "fixtures/false-negative.png",
            "expected_exact_match": True,
            "expected_ocr_detected": True,
            "human_label": "pass",
        },
    ]


def test_reward_disagreement_report_counts_correlation_false_rows_and_missing_evidence(
    tmp_path: Path,
) -> None:
    report = analyze_reward_disagreement(
        _make_fixture_records(tmp_path),
        gold_records=_make_gold_records(),
        positive_threshold=0.80,
        negative_threshold=0.50,
    )

    assert report["schema_version"] == "reward-diagnostics/v1"
    assert report["record_counts"] == {
        "total": 4,
        "with_vlm_and_ocr": 3,
        "missing_evidence": 1,
    }
    assert report["missing_evidence"]["count"] == 1
    assert report["missing_evidence"]["by_component"]["score_ocr"] == 1
    assert report["vlm_ocr_correlation"]["n"] == 3
    assert report["vlm_ocr_correlation"]["pearson"] > 0.99
    assert report["scatter_summary"]["points"][0] == {
        "sample_id": "false-negative",
        "score_vlm": 0.32,
        "score_ocr": 0.35,
        "product_score": 0.34,
    }
    assert [row["sample_id"] for row in report["false_positives"]] == ["false-positive"]
    assert [row["sample_id"] for row in report["false_negatives"]] == ["false-negative"]
    assert report["false_positives"][0]["gold_label"] == "fail"
    assert report["false_negatives"][0]["gold_label"] == "pass"
    assert report["thresholds"] == {"positive_threshold": 0.8, "negative_threshold": 0.5}


def test_character_confusions_and_slice_disagreement_counts_use_recorded_scores(
    tmp_path: Path,
) -> None:
    report = analyze_reward_disagreement(
        _make_fixture_records(tmp_path),
        gold_records=_make_gold_records(),
        positive_threshold=0.80,
        negative_threshold=0.50,
    )

    assert report["character_confusions"]["total_confusions"] >= 1
    assert {
        "expected": "щ",
        "observed": "ш",
        "count": 1,
        "sample_ids": ["false-positive"],
    } in report["character_confusions"]["pairs"]
    rare_summary = report["per_slice_disagreement_counts"]["rare_cyrillic"]
    assert rare_summary["records"] == 2
    assert rare_summary["false_positives"] == 1
    assert rare_summary["missing_evidence"] == 0
    assert report["per_slice_disagreement_counts"]["multiline"]["missing_evidence"] == 1


def test_optional_contact_sheet_is_bounded_and_lists_source_paths(tmp_path: Path) -> None:
    contact_sheet_path = tmp_path / "diagnostics" / "contact-sheet.png"

    report = analyze_reward_disagreement(
        _make_fixture_records(tmp_path),
        gold_records=_make_gold_records(),
        positive_threshold=0.80,
        negative_threshold=0.50,
        contact_sheet_path=contact_sheet_path,
        contact_sheet_limit=1,
    )

    assert contact_sheet_path.exists()
    assert report["contact_sheet"]["path"] == str(contact_sheet_path)
    assert report["contact_sheet"]["limit"] == 1
    assert report["contact_sheet"]["entry_count"] == 1
    assert report["contact_sheet"]["entries"] == [
        {
            "sample_id": "false-positive",
            "caption": "false_positive false-positive product=0.880",
            "source_path": str(tmp_path / "sample-2.png"),
            "kind": "false_positive",
        }
    ]
    with Image.open(contact_sheet_path) as image:
        assert image.size[0] > 0
        assert image.size[1] > 0


def test_cli_writes_deterministic_json_and_markdown_without_running_models(
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "scores.csv"
    gold_path = tmp_path / "gold.jsonl"
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    contact_sheet_path = tmp_path / "contact-sheet.png"
    _write_scores_csv(scores_path, _make_fixture_records(tmp_path))
    _write_jsonl(gold_path, _make_gold_records())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_reward_diagnostics.py",
            "--scores",
            str(scores_path),
            "--gold",
            str(gold_path),
            "--output-report",
            str(report_path),
            "--markdown-summary",
            str(markdown_path),
            "--contact-sheet",
            str(contact_sheet_path),
            "--contact-sheet-limit",
            "2",
            "--positive-threshold",
            "0.80",
            "--negative-threshold",
            "0.50",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert report["false_positives"][0]["sample_id"] == "false-positive"
    assert report["false_negatives"][0]["sample_id"] == "false-negative"
    assert report["contact_sheet"]["entry_count"] == 2
    assert "# Reward disagreement diagnostics" in markdown
    assert "VLM-vs-OCR correlation" in markdown
    assert "false-positive" in markdown
    assert contact_sheet_path.exists()


def test_format_diagnostics_markdown_includes_required_sections(tmp_path: Path) -> None:
    report = analyze_reward_disagreement(
        _make_fixture_records(tmp_path),
        gold_records=_make_gold_records(),
        positive_threshold=0.80,
        negative_threshold=0.50,
    )

    markdown = format_diagnostics_markdown(report)

    assert "# Reward disagreement diagnostics" in markdown
    assert "VLM-vs-OCR correlation" in markdown
    assert "False positives" in markdown
    assert "False negatives" in markdown
    assert "Per-character confusion" in markdown
    assert "Per-slice disagreement" in markdown
    assert "Missing evidence" in markdown


def test_evaluation_diagnostics_docs_cover_reward_disagreement_cli() -> None:
    docs = Path("docs/evaluation_diagnostics.md").read_text(encoding="utf-8")

    required_terms = [
        "analyze_reward_disagreement",
        "format_diagnostics_markdown",
        "scripts/analyze_reward_diagnostics.py",
        "--scores",
        "--gold",
        "--output-report",
        "--markdown-summary",
        "--contact-sheet",
        "VLM-vs-OCR scatter/correlation",
        "false-positive",
        "false-negative",
        "per-character confusion",
        "per-slice disagreement",
        "gold diagnostic benchmark",
        "recorded score outputs",
        "do not run reward models",
        "Do not commit generated diagnostic reports or contact sheets",
    ]
    for term in required_terms:
        assert term in docs
