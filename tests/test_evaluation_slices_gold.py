from __future__ import annotations

import importlib
import sys


HEAVY_OPTIONAL_MODULES = {"transformers", "paddleocr", "diffusers", "torch", "vllm", "mlx"}
MODULES_BEFORE_EVALUATION_IMPORT = set(sys.modules)


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
