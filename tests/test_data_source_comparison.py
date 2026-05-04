from __future__ import annotations

import json
from pathlib import Path

from src.data_quality.source_comparison import compare_data_sources


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_source_comparison_reports_metrics_interpretation_and_provenance(
    tmp_path: Path,
) -> None:
    generated_prompt_report = _write_json(
        tmp_path / "generated" / "prompt-quality.json",
        {
            "schema_version": "prompt-quality/v1",
            "valid_records": 3,
            "rare_character_coverage": {"counts": {"ё": 2, "ж": 1, "щ": 0}},
            "content_type_distribution": {"phrase": 1, "word": 2},
            "style_distribution": {"font": {"Serif": 2, "Sans": 1}},
        },
    )
    selected_samples = _write_jsonl(
        tmp_path / "generated" / "selected_samples.jsonl",
        [
            {
                "schema_version": "selected-samples/v1",
                "prompt_id": "p1",
                "target_text": "Ёж",
                "selected_score": 0.8,
                "score_column": "score",
            },
            {
                "schema_version": "selected-samples/v1",
                "prompt_id": "p2",
                "target_text": "Жук",
                "selected_score": 0.4,
                "score_column": "score",
            },
        ],
    )
    preference_pairs = _write_jsonl(
        tmp_path / "generated" / "preference_pairs.jsonl",
        [
            {
                "schema_version": "preference-pairs/v1",
                "prompt_id": "p1",
                "target_text": "Ёж",
                "winner_score": 0.9,
                "loser_score": 0.2,
                "margin": 0.7,
            }
        ],
    )
    generated_manifest = _write_json(
        tmp_path / "generated" / "manifest.json",
        {
            "schema_version": "dataset-manifest/v1",
            "dataset_kind": "selected_samples",
            "source_hashes": {"scores.csv": {"hashed": True, "sha256": "a" * 64}},
            "output_counts": {"selected_count": 2},
        },
    )
    synthetic_report = _write_json(
        tmp_path / "synthetic" / "synthetic-quality.json",
        {
            "schema_version": "synthetic-quality/v1",
            "sample_count": 2,
            "accepted_count": 1,
            "rejected_count": 1,
            "rejection_reasons": {"contrast_below_min": 1},
            "character_coverage": {"counts": {"ё": 1, "ц": 2}},
            "font_coverage": {"Sans": 2},
            "resolution_distribution": {"10x10": 2},
            "mask_area_fraction": {"count": 2, "min": 0.1, "max": 0.2, "mean": 0.15},
            "contrast": {"count": 2, "min": 12.0, "max": 60.0, "mean": 36.0},
            "ocr_summary": {"count": 2, "exact_match_rate": 0.5, "mean_cer": 0.25},
        },
    )
    synthetic_manifest = _write_json(
        tmp_path / "synthetic" / "manifest.json",
        {
            "schema_version": "dataset-manifest/v1",
            "dataset_kind": "synthetic",
            "source_hashes": {"index.csv": {"hashed": True, "sha256": "b" * 64}},
            "filtering_stats": {"accepted": 1, "rejected": 1},
        },
    )

    comparison = compare_data_sources(
        generated_prompt_quality_report=generated_prompt_report,
        selected_samples=selected_samples,
        preference_pairs=preference_pairs,
        generated_dataset_manifest=generated_manifest,
        synthetic_quality_report=synthetic_report,
        synthetic_manifest=synthetic_manifest,
    )
    payload = comparison.to_dict()

    assert payload["schema_version"] == "data-source-comparison/v1"
    assert payload["evidence_missing"] == []
    assert payload["counts"] == {
        "generated_prompt_records": 3,
        "generated_selected_samples": 2,
        "generated_preference_pairs": 1,
        "synthetic_samples": 2,
        "synthetic_accepted": 1,
        "synthetic_rejected": 1,
    }
    assert payload["rare_character_coverage"] == {
        "generated": {"ё": 2, "ж": 1},
        "synthetic": {"ё": 1, "ц": 2},
        "overlap": ["ё"],
        "generated_only": ["ж"],
        "synthetic_only": ["ц"],
    }
    assert payload["generated_score_summary"]["selected_score"] == {
        "count": 2,
        "min": 0.4,
        "max": 0.8,
        "mean": 0.6,
    }
    assert payload["generated_score_summary"]["preference_margin"] == {
        "count": 1,
        "min": 0.7,
        "max": 0.7,
        "mean": 0.7,
    }
    assert payload["synthetic_mask_contrast_health"]["rejection_reasons"] == {
        "contrast_below_min": 1
    }
    assert payload["distribution_differences"]["content_type"]["generated"] == {
        "phrase": 1,
        "word": 2,
    }
    assert payload["distribution_differences"]["resolution"]["synthetic"] == {"10x10": 2}
    assert payload["provenance"][str(selected_samples)]["sha256"]
    assert payload["expected_help"]["generated_reward_filtered"]
    assert payload["expected_failure"]["synthetic_masked_sft"]


def test_source_comparison_marks_missing_optional_evidence_without_fabricating_metrics(
    tmp_path: Path,
) -> None:
    synthetic_report = _write_json(
        tmp_path / "synthetic-quality.json",
        {
            "schema_version": "synthetic-quality/v1",
            "sample_count": 1,
            "accepted_count": 1,
            "rejected_count": 0,
            "character_coverage": {"counts": {"щ": 1}},
        },
    )

    payload = compare_data_sources(synthetic_quality_report=synthetic_report).to_dict()

    assert payload["evidence_available"] == ["synthetic_quality_report"]
    assert "selected_samples" in payload["evidence_missing"]
    assert "generated_prompt_quality_report" in payload["evidence_missing"]
    assert payload["counts"]["generated_selected_samples"] is None
    assert payload["rare_character_coverage"]["generated"] == {}
    assert payload["rare_character_coverage"]["synthetic"] == {"щ": 1}
