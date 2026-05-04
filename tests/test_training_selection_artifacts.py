from __future__ import annotations

import csv
import json
from pathlib import Path

from src.training.selection import materialize_sft_samples


def _write_scores(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sft_materialization_defaults_to_existing_threshold_semantics(
    tmp_path: Path,
) -> None:
    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p2", "version": 2, "score": "0.900", "target_text": "Жук"},
            {"id": "p1", "version": 1, "score": "0.299", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.300", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.700", "target_text": "Жук"},
        ],
    )
    output = tmp_path / "selected_samples.jsonl"

    summary = materialize_sft_samples(scores, output, threshold=0.3)

    rows = _read_jsonl(output)
    assert summary["schema_version"] == "selected-samples/v1"
    assert summary["selection_mode"] == "threshold"
    assert summary["input_rows"] == 4
    assert summary["selected_count"] == 3
    assert summary["filtered_count"] == 1
    assert [(row["prompt_id"], row["version"]) for row in rows] == [
        ("p1", 2),
        ("p2", 1),
        ("p2", 2),
    ]
    assert rows[0]["schema_version"] == "selected-samples/v1"
    assert rows[0]["sample_id"] == "sft:p1:v2:score"
    assert rows[0]["target_text"] == "Ёж"
    assert rows[0]["selected_score"] == 0.3
    assert rows[0]["score_column"] == "score"
    assert rows[0]["selection_mode"] == "threshold"
    assert rows[0]["source_scores_path"] == str(scores)
    assert len(str(rows[0]["source_scores_sha256"])) == 64


def test_sft_top_k_materialization_selects_highest_versions_per_prompt(
    tmp_path: Path,
) -> None:
    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.60", "score_ocr": "0.80", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.95", "score_ocr": "0.70", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.40", "score_ocr": "0.40", "target_text": "Жук"},
            {"id": "p2", "version": 2, "score": "0.35", "score_ocr": "0.90", "target_text": "Жук"},
        ],
    )
    output = tmp_path / "selected_top1.jsonl"

    summary = materialize_sft_samples(
        scores,
        output,
        mode="top_k_per_prompt",
        score_column="score_ocr",
        threshold=0.75,
        top_k_per_prompt=1,
    )

    rows = _read_jsonl(output)
    assert summary["selection_mode"] == "top_k_per_prompt"
    assert summary["selected_count"] == 2
    assert summary["filtering_stats"] == {
        "below_threshold": 2,
        "selected": 2,
        "unselected_by_top_k": 0,
    }
    assert [(row["prompt_id"], row["version"], row["selected_score"]) for row in rows] == [
        ("p1", 1, 0.8),
        ("p2", 2, 0.9),
    ]
    assert {row["selection_mode"] for row in rows} == {"top_k_per_prompt"}
    assert {row["score_column"] for row in rows} == {"score_ocr"}


def test_sft_materialization_validates_required_columns_and_numeric_values(
    tmp_path: Path,
) -> None:
    missing_score = _write_scores(
        tmp_path / "missing-score.csv",
        [{"id": "p1", "version": 1, "target_text": "Ёж"}],
    )
    invalid_version = _write_scores(
        tmp_path / "invalid-version.csv",
        [{"id": "p1", "version": "one", "score": "0.7", "target_text": "Ёж"}],
    )

    for path, expected in [
        (missing_score, "missing required column: score"),
        (invalid_version, "invalid integer version"),
    ]:
        try:
            materialize_sft_samples(path, tmp_path / "out.jsonl")
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - documents required exception path
            raise AssertionError(f"{path} should have failed validation")
