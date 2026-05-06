from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from src.evaluation.reward_interface import (
    ProductScoreFormula,
    RewardResult,
    build_score_metadata,
    compute_product_score,
)

HEAVY_OPTIONAL_MODULES = {"transformers", "paddleocr", "diffusers", "torch", "vllm", "mlx"}
DOC_PATH = Path("docs/reward_evaluation.md")


def test_reward_interface_import_is_cpu_safe() -> None:
    newly_loaded = HEAVY_OPTIONAL_MODULES.intersection(sys.modules)

    assert newly_loaded == set()


def test_reward_result_serializes_canonical_safe_row() -> None:
    result = RewardResult(
        sample_id="prompt-0001",
        version=2,
        target_text="Привет",
        score=0.814219,
        components={
            "score_vlm": 0.91,
            "score_ocr": 0.76,
            "cer": 0.125,
            "entropy": 0.32,
            "exact_text_match": False,
        },
        text_metrics={"detected_text": "Прнвет", "char_accuracy": 0.875},
        scorer_metadata={"vlm": "qwen-fake@cpu", "ocr": "paddle-fake@cpu"},
        thresholds={"score_vlm_min": 0.7, "cer_max": 0.2},
        manifest_path="runs/eval-001/manifest.json",
        missing_components=("slice_label",),
    )

    row = result.to_row()

    assert row == {
        "sample_id": "prompt-0001",
        "version": 2,
        "target_text": "Привет",
        "score": 0.814219,
        "score_vlm": 0.91,
        "score_ocr": 0.76,
        "cer": 0.125,
        "entropy": 0.32,
        "exact_text_match": False,
        "text_metrics": json.dumps(
            {"char_accuracy": 0.875, "detected_text": "Прнвет"},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "scorer_metadata": json.dumps(
            {"ocr": "paddle-fake@cpu", "vlm": "qwen-fake@cpu"},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "thresholds": json.dumps(
            {"cer_max": 0.2, "score_vlm_min": 0.7},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "missing_components": "slice_label",
        "manifest_path": "runs/eval-001/manifest.json",
    }


def test_compute_product_score_combines_weighted_components_and_flags_thresholds() -> None:
    formula = ProductScoreFormula(
        name="thesis_product_v1",
        weights={
            "score_vlm": 0.35,
            "score_ocr": 0.25,
            "cer_quality": 0.2,
            "entropy_quality": 0.1,
            "exact_text_match": 0.1,
        },
        thresholds={"score_vlm_min": 0.7, "score_ocr_min": 0.6, "cer_max": 0.2},
        scorer_versions={"vlm": "qwen-fake@1", "ocr": "paddle-fake@2"},
        entropy_scale=0.5,
    )

    product = compute_product_score(
        {
            "score_vlm": 0.9,
            "score_ocr": 0.8,
            "cer": 0.1,
            "entropy": 0.4,
            "exact_text_match": True,
        },
        formula=formula,
    )

    expected = (
        (0.9**0.35)
        * (0.8**0.25)
        * ((1.0 - 0.1) ** 0.2)
        * (math.exp(-0.5 * 0.4) ** 0.1)
        * (1.0**0.1)
    )
    assert product.score == pytest.approx(expected)
    assert product.components == {
        "score_vlm": 0.9,
        "score_ocr": 0.8,
        "cer_quality": 0.9,
        "entropy_quality": pytest.approx(math.exp(-0.2)),
        "exact_text_match": 1.0,
    }
    assert product.threshold_flags == {
        "score_vlm_min": True,
        "score_ocr_min": True,
        "cer_max": True,
    }
    assert product.missing_components == ()
    assert product.formula_complete is True
    assert product.formula.name == "thesis_product_v1"


def test_missing_or_invalid_evidence_is_reported_not_imputed() -> None:
    formula = ProductScoreFormula(
        name="missing-aware",
        weights={"score_vlm": 0.5, "score_ocr": 0.3, "cer_quality": 0.2},
        thresholds={"score_vlm_min": 0.8, "score_ocr_min": 0.7},
        scorer_versions={"vlm": "qwen-fake@1", "ocr": "paddle-fake@2"},
    )

    product = compute_product_score(
        {"score_vlm": 0.82, "score_ocr": None, "cer": float("nan")},
        formula=formula,
    )

    assert product.score == pytest.approx(0.82)
    assert product.components == {"score_vlm": 0.82}
    assert product.missing_components == ("score_ocr", "cer")
    assert product.threshold_flags == {"score_vlm_min": True, "score_ocr_min": False}
    assert product.formula_complete is False


def test_score_metadata_records_formula_versions_thresholds_and_manifests() -> None:
    formula = ProductScoreFormula(
        name="metadata-product-v1",
        weights={"score_vlm": 0.6, "score_ocr": 0.4},
        thresholds={"score_vlm_min": 0.5},
        scorer_versions={"vlm": "qwen-fake@1", "ocr": "paddle-fake@2"},
    )

    metadata = build_score_metadata(
        formula=formula,
        source_manifest_paths=["runs/baseline/manifest.json", "runs/lora/manifest.json"],
        generated_at="2026-05-06T14:21:18Z",
    )

    assert metadata == {
        "schema_version": "reward-score-metadata/v1",
        "generated_at": "2026-05-06T14:21:18Z",
        "formula": {
            "name": "metadata-product-v1",
            "weights": {"score_vlm": 0.6, "score_ocr": 0.4},
            "thresholds": {"score_vlm_min": 0.5},
            "scorer_versions": {"ocr": "paddle-fake@2", "vlm": "qwen-fake@1"},
            "entropy_scale": 1.0,
        },
        "source_manifest_paths": ["runs/baseline/manifest.json", "runs/lora/manifest.json"],
    }


def test_reward_evaluation_docs_match_canonical_contract_names() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    required_terms = [
        "RewardResult",
        "ProductScoreFormula",
        "compute_product_score",
        "build_score_metadata",
        "sample_id",
        "version",
        "target_text",
        "score_vlm",
        "score_ocr",
        "cer_quality",
        "entropy_quality",
        "exact_text_match",
        "missing_components",
        "threshold_flags",
        "scorer_versions",
        "source_manifest_paths",
        "reward-score-metadata/v1",
        "Generated artifacts safety",
    ]
    missing = [term for term in required_terms if term not in docs]

    assert missing == []
