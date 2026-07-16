from __future__ import annotations

import importlib
import sys
from pathlib import Path

HEAVY_OPTIONAL_MODULES = {"transformers", "paddleocr", "diffusers", "torch", "vllm", "mlx"}
MODULES_BEFORE_EVALUATION_IMPORT = set(sys.modules)
FIXTURE_PATH = Path("tests/fixtures/evaluation/gold_diagnostic.jsonl")
DOC_PATH = Path("docs/evaluation_diagnostics.md")


def test_gold_benchmark_module_import_is_cpu_safe() -> None:
    importlib.import_module("src.evaluation.gold_benchmark")

    newly_loaded = {
        module_name
        for module_name in HEAVY_OPTIONAL_MODULES
        if module_name in sys.modules and module_name not in MODULES_BEFORE_EVALUATION_IMPORT
    }

    assert newly_loaded == set()


def test_slice_module_import_is_cpu_safe() -> None:
    importlib.import_module("src.evaluation.slices")

    newly_loaded = {
        module_name
        for module_name in HEAVY_OPTIONAL_MODULES
        if module_name in sys.modules and module_name not in MODULES_BEFORE_EVALUATION_IMPORT
    }

    assert newly_loaded == set()


def test_classify_text_slices_covers_russian_difficulty_dimensions() -> None:
    from src.evaluation.slices import classify_text_slices

    record = {
        "sample_id": "hard-001",
        "target_text": "Ёжик ЩУКА 2026!\nДлиннослово",
        "font": "serif-bold",
        "style": "neon sign",
        "scene": "busy street",
        "background": "complex texture",
    }

    slices = classify_text_slices(record)

    assert slices == {
        "rare_cyrillic",
        "long_word",
        "multi_word_phrase",
        "has_digits",
        "has_punctuation",
        "mixed_case",
        "multiline",
        "font_or_style",
        "scene_or_background",
    }


def test_classify_text_slices_returns_simple_short_word_slice() -> None:
    from src.evaluation.slices import classify_text_slices

    slices = classify_text_slices({"target_text": "мир"})

    assert slices == {"short_word"}


def test_summarize_slices_counts_records_and_missing_target_text() -> None:
    from src.evaluation.slices import summarize_slices

    records = [
        {"sample_id": "one", "target_text": "Привет, мир!"},
        {"sample_id": "two", "target_text": "Ёж 42", "scene": "forest"},
        {"sample_id": "three", "target_text": ""},
        {"sample_id": "four"},
    ]

    summary = summarize_slices(records)

    assert summary["total_records"] == 4
    assert summary["classified_records"] == 2
    assert summary["missing_target_text_records"] == ["three", "four"]
    assert summary["slice_counts"]["multi_word_phrase"] == 2
    assert summary["slice_counts"]["has_punctuation"] == 1
    assert summary["slice_counts"]["rare_cyrillic"] == 1
    assert summary["slice_counts"]["has_digits"] == 1
    assert summary["slice_counts"]["scene_or_background"] == 1


def test_load_gold_benchmark_validates_tiny_fixture_schema() -> None:
    from src.evaluation.gold_benchmark import load_gold_benchmark

    benchmark = load_gold_benchmark(FIXTURE_PATH)

    assert benchmark["schema_version"] == "gold-diagnostic-benchmark/v1"
    assert benchmark["source_path"] == str(FIXTURE_PATH)
    assert [record["sample_id"] for record in benchmark["records"]] == [
        "gold-cyrillic-rare",
        "gold-digits-punctuation",
        "gold-mixed-case",
        "gold-multiline",
    ]
    assert benchmark["slice_summary"]["slice_counts"]["rare_cyrillic"] == 1
    assert benchmark["slice_summary"]["slice_counts"]["has_digits"] == 1
    assert benchmark["slice_summary"]["slice_counts"]["mixed_case"] == 1
    assert benchmark["slice_summary"]["slice_counts"]["multiline"] == 1


def test_load_gold_benchmark_aggregates_schema_errors(tmp_path: Path) -> None:
    from src.evaluation.gold_benchmark import GoldBenchmarkError, load_gold_benchmark

    malformed_path = tmp_path / "bad.jsonl"
    malformed_path.write_text(
        "\n".join(
            [
                '{"sample_id": "missing-fields", "target_text": "Привет"}',
                '{"sample_id": 42, "target_text": "Мир", "image_path": "fixtures/mir.png", '
                '"expected_exact_match": true, "expected_ocr_detected": true, '
                '"human_label": "pass"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_gold_benchmark(malformed_path)
    except GoldBenchmarkError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failure output
        raise AssertionError("Expected schema validation to fail")

    assert "line 1: missing required field image_path" in message
    assert "line 1: missing required field expected_exact_match" in message
    assert "line 2: sample_id must be a non-empty string" in message


def test_evaluate_gold_predictions_reports_agreement_missing_and_slices() -> None:
    from src.evaluation.gold_benchmark import evaluate_gold_predictions, load_gold_benchmark

    benchmark = load_gold_benchmark(FIXTURE_PATH)
    predictions = [
        {
            "sample_id": "gold-cyrillic-rare",
            "exact_text_match": True,
            "ocr_detected": True,
            "detected_text": "Ёжик",
        },
        {
            "sample_id": "gold-digits-punctuation",
            "exact_text_match": False,
            "ocr_detected": False,
            "detected_text": "Счет 42",
        },
        {
            "sample_id": "gold-mixed-case",
            "exact_text_match": True,
            "ocr_detected": True,
            "detected_text": "Москва CITY",
        },
    ]

    report = evaluate_gold_predictions(benchmark, predictions)

    assert report["source_path"] == str(FIXTURE_PATH)
    assert report["total_gold_records"] == 4
    assert report["matched_prediction_count"] == 3
    assert report["missing_prediction_count"] == 1
    assert report["missing_prediction_sample_ids"] == ["gold-multiline"]
    assert report["exact_agreement"] == {"agree": 2, "disagree": 1}
    assert report["ocr_detection_agreement"] == {"agree": 2, "disagree": 1}
    assert report["ocr_text_agreement"] == {"agree": 3, "disagree": 0}
    assert report["per_slice"]["rare_cyrillic"]["records"] == 1
    assert report["per_slice"]["has_digits"]["exact_disagreements"] == 1
    assert report["per_slice"]["multiline"]["missing_predictions"] == 1
    assert report["findings"] == [
        "Missing predictions: 1 sample(s): gold-multiline",
        "Exact-match expectation disagreements: 1",
        "OCR-detection expectation disagreements: 1",
    ]


def test_format_gold_report_markdown_surfaces_diagnostic_evidence() -> None:
    from src.evaluation.gold_benchmark import (
        evaluate_gold_predictions,
        format_gold_report_markdown,
        load_gold_benchmark,
    )

    report = evaluate_gold_predictions(load_gold_benchmark(FIXTURE_PATH), [])
    markdown = format_gold_report_markdown(report)

    required_terms = [
        "# Gold diagnostic benchmark report",
        "gold-diagnostic-benchmark/v1",
        "source_path",
        "missing_prediction_count",
        "exact_agreement",
        "ocr_detection_agreement",
        "ocr_text_agreement",
        "rare_cyrillic",
        "multiline",
        "Missing predictions are diagnostic evidence",
    ]

    missing = [term for term in required_terms if term not in markdown]
    assert missing == []


def test_evaluation_diagnostics_docs_match_slice_and_gold_contracts() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    required_terms = [
        "classify_text_slices",
        "summarize_slices",
        "rare_cyrillic",
        "short_word",
        "long_word",
        "multi_word_phrase",
        "has_digits",
        "has_punctuation",
        "mixed_case",
        "multiline",
        "font_or_style",
        "scene_or_background",
        "load_gold_benchmark",
        "evaluate_gold_predictions",
        "format_gold_report_markdown",
        "gold-diagnostic-benchmark/v1",
        "sample_id",
        "target_text",
        "image_path",
        "expected_exact_match",
        "expected_ocr_detected",
        "human_label",
        "notes",
        "missing predictions/disagreements are explicit evidence",
        "metadata-only fixture",
        "Do not commit generated images, tensors, checkpoints, or logs",
    ]
    missing = [term for term in required_terms if term not in docs]

    assert missing == []
