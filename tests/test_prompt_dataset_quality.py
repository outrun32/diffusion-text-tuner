from __future__ import annotations

import json
from pathlib import Path

from src.data_quality.prompt_validation import validate_prompt_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object] | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _prompt_record(
    record_id: str,
    target_text: str,
    *,
    content_type: str = "poster",
    style: dict[str, str] | None = None,
    lang: str = "ru",
) -> dict[str, object]:
    style_payload = style or {"font": "serif", "color": "white", "effect": "clean", "size": "large"}
    return {
        "id": record_id,
        "prompt": f"Render {target_text} on a poster",
        "target_text": target_text,
        "content_type": content_type,
        "style": style_payload,
        "lang": lang,
        "char_coverage": {ch.lower(): 1 for ch in target_text if ch.isalpha()},
    }


def test_prompt_quality_report_counts_valid_records_and_distributions(tmp_path: Path) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", "Ёж Цех", content_type="poster"),
            _prompt_record("p2", "Шрифт 42!", content_type="typography", style={"font": "gothic", "color": "red"}),
            _prompt_record("p3", "Hello Ж readable", content_type="product", lang="en"),
        ],
    )

    report = validate_prompt_dataset(
        prompts,
        thresholds={
            "required_rare_characters": ["ё", "ж", "ц", "ш"],
            "min_rare_character_coverage": 1.0,
            "allowed_scripts": ["cyrillic", "latin", "digits", "punctuation"],
        },
    )

    assert report.ok
    assert report.schema_version == "prompt-quality/v1"
    assert report.valid_records == 3
    assert report.length_buckets == {"1-4": 0, "5-12": 2, "13-24": 1, "25+": 0}
    assert report.script_coverage["cyrillic"] == 3
    assert report.script_coverage["latin"] == 1
    assert report.script_coverage["digits"] == 1
    assert report.script_coverage["punctuation"] == 1
    assert report.rare_character_coverage["coverage_ratio"] == 1.0
    assert report.content_type_distribution == {"poster": 1, "product": 1, "typography": 1}
    assert report.style_distribution["font"] == {"gothic": 1, "serif": 2}


def test_prompt_quality_report_aggregates_malformed_rows_missing_fields_and_duplicates(
    tmp_path: Path,
) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", "Привет"),
            "{not-json",
            {"id": "p2", "prompt": "missing target"},
            _prompt_record("p3", "Привет"),
        ],
    )

    report = validate_prompt_dataset(prompts, thresholds={"max_duplicate_rate": 0.1})

    assert not report.ok
    assert report.total_lines == 4
    assert report.valid_records == 2
    assert any("line 2" in error and "malformed JSON" in error for error in report.errors)
    assert any("line 3" in error and "missing required field: target_text" in error for error in report.errors)
    assert report.duplicate_rate == 0.5
    assert report.duplicate_examples == ["Привет"]
    assert any("duplicate_rate" in warning for warning in report.warnings)


def test_prompt_quality_report_warns_for_rare_character_gaps_and_distribution_drift(
    tmp_path: Path,
) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", "Афиша", content_type="poster"),
            _prompt_record("p2", "Баннер", content_type="poster"),
            _prompt_record("p3", "Логотип", content_type="poster"),
        ],
    )

    report = validate_prompt_dataset(
        prompts,
        thresholds={
            "required_rare_characters": ["ё", "щ", "ъ"],
            "min_rare_character_coverage": 0.67,
            "expected_content_distribution": {"poster": [0.0, 0.5], "product": [0.2, 1.0]},
        },
    )

    assert report.ok
    assert report.rare_character_coverage["missing"] == ["ё", "щ", "ъ"]
    assert any("rare_character_coverage" in warning for warning in report.warnings)
    assert any("content_type poster" in warning for warning in report.warnings)
    assert any("content_type product" in warning for warning in report.warnings)


def test_prompt_quality_report_flags_cpu_safe_naturalness_and_script_heuristics(
    tmp_path: Path,
) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", ""),
            _prompt_record("p2", "Ответь одной строкой: вывеска"),
            _prompt_record("p3", "слово слово слово слово слово"),
            _prompt_record("p4", '"Незакрытая цитата'),
            _prompt_record("p5", "Valid🙂"),
        ],
    )

    report = validate_prompt_dataset(prompts, thresholds={"allowed_scripts": ["cyrillic"]})

    assert not report.ok
    assert any("line 1" in error and "empty target_text" in error for error in report.errors)
    assert any("instruction-like" in warning for warning in report.warnings)
    assert any("repeated token" in warning for warning in report.warnings)
    assert any("unmatched quote" in warning for warning in report.warnings)
    assert any("illegal character" in error and "line 5" in error for error in report.errors)
    assert any("disallowed script latin" in error and "line 5" in error for error in report.errors)
