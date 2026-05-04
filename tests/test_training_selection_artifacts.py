from __future__ import annotations

import csv
import json
from pathlib import Path

from src.training.selection import materialize_dpo_pairs, materialize_sft_samples


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


def test_dpo_materialization_defaults_to_best_vs_worst_pair_semantics(
    tmp_path: Path,
) -> None:
    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.20", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.80", "target_text": "Ёж"},
            {"id": "p1", "version": 3, "score": "0.50", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.30", "target_text": "Жук"},
            {"id": "p2", "version": 2, "score": "0.55", "target_text": "Жук"},
            {"id": "p3", "version": 1, "score": "0.10", "target_text": "Цех"},
        ],
    )
    output = tmp_path / "preference_pairs.jsonl"

    summary = materialize_dpo_pairs(scores, output, threshold=0.5, margin=0.1)

    rows = _read_jsonl(output)
    assert summary["schema_version"] == "preference-pairs/v1"
    assert summary["pair_construction_mode"] == "best_vs_worst"
    assert summary["prompt_count"] == 3
    assert summary["pair_count"] == 2
    assert summary["filtering_stats"] == {
        "ambiguous_below_margin": 0,
        "insufficient_versions": 1,
        "selected": 2,
        "winner_below_threshold": 0,
    }
    assert [(row["prompt_id"], row["winner_version"], row["loser_version"]) for row in rows] == [
        ("p1", 2, 1),
        ("p2", 2, 1),
    ]
    assert rows[0]["schema_version"] == "preference-pairs/v1"
    assert rows[0]["pair_id"] == "dpo:p1:w2:l1:score"
    assert rows[0]["target_text"] == "Ёж"
    assert rows[0]["winner_score"] == 0.8
    assert rows[0]["loser_score"] == 0.2
    assert rows[0]["margin"] == 0.6
    assert rows[0]["score_column"] == "score"
    assert rows[0]["pair_construction_mode"] == "best_vs_worst"
    assert rows[0]["source_scores_path"] == str(scores)
    assert len(str(rows[0]["source_scores_sha256"])) == 64


def test_dpo_materialization_excludes_ambiguous_pairs_and_low_winners(
    tmp_path: Path,
) -> None:
    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.51", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.58", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.10", "target_text": "Жук"},
            {"id": "p2", "version": 2, "score": "0.49", "target_text": "Жук"},
        ],
    )

    summary = materialize_dpo_pairs(scores, tmp_path / "pairs.jsonl", threshold=0.5, margin=0.1)

    assert _read_jsonl(tmp_path / "pairs.jsonl") == []
    assert summary["pair_count"] == 0
    assert summary["filtering_stats"] == {
        "ambiguous_below_margin": 1,
        "insufficient_versions": 0,
        "selected": 0,
        "winner_below_threshold": 1,
    }


def test_dpo_materialization_requires_winner_strictly_above_loser(tmp_path: Path) -> None:
    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.70", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.70", "target_text": "Ёж"},
        ],
    )

    summary = materialize_dpo_pairs(scores, tmp_path / "pairs.jsonl", threshold=0.5, margin=0.0)

    assert summary["pair_count"] == 0
    assert summary["filtering_stats"]["ambiguous_below_margin"] == 1


def test_materialization_cli_writes_sft_artifact_manifest_and_stdout_summary(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.materialize_training_data import main

    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.70", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.20", "target_text": "Жук"},
        ],
    )
    output_dir = tmp_path / "selection"
    manifest = tmp_path / "selection" / "selected_samples.manifest.json"

    exit_code = main(
        [
            "--kind",
            "sft",
            "--scores-csv",
            str(scores),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--threshold",
            "0.3",
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    rows = _read_jsonl(output_dir / "selected_samples.jsonl")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout["schema_version"] == "selected-samples/v1"
    assert stdout["selected_count"] == 1
    assert stdout["output_path"] == str(output_dir / "selected_samples.jsonl")
    assert manifest_payload["threshold"] == 0.3
    assert manifest_payload["score_column"] == "score"
    assert manifest_payload["source_scores_sha256"] == stdout["source_scores_sha256"]
    assert rows[0]["manifest_path"] == str(manifest)


def test_materialization_cli_writes_dpo_artifact_manifest_and_stdout_summary(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.materialize_training_data import main

    scores = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.10", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.70", "target_text": "Ёж"},
        ],
    )
    output_dir = tmp_path / "selection"
    manifest = output_dir / "preference_pairs.manifest.json"

    exit_code = main(
        [
            "--kind",
            "dpo",
            "--scores-csv",
            str(scores),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--threshold",
            "0.5",
            "--margin",
            "0.1",
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    rows = _read_jsonl(output_dir / "preference_pairs.jsonl")
    assert exit_code == 0
    assert stdout["schema_version"] == "preference-pairs/v1"
    assert stdout["pair_count"] == 1
    assert stdout["output_path"] == str(output_dir / "preference_pairs.jsonl")
    assert rows[0]["winner_version"] == 2
    assert rows[0]["loser_version"] == 1
    assert rows[0]["manifest_path"] == str(manifest)


def test_data_selection_docs_cover_artifact_schemas_and_runtime_contracts() -> None:
    docs = Path("docs/data_selection.md").read_text(encoding="utf-8")

    required_strings = [
        "selected_samples.jsonl",
        "preference_pairs.jsonl",
        "selected-samples/v1",
        "preference-pairs/v1",
        "materialize_training_data.py --kind sft",
        "materialize_training_data.py --kind dpo",
        "default equivalence",
        "docs/runtime_contracts.md",
        "Do not commit generated images or tensors",
    ]
    missing = [text for text in required_strings if text not in docs]
    assert not missing, missing
