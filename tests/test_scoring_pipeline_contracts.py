from __future__ import annotations

import importlib
import inspect
import json
import sys

import torch

from src.evaluation.reward_interface import ProductScoreFormula

HEAVY_OPTIONAL_ROOTS = {
    "diffusers",
    "transformers",
    "paddleocr",
    "paddlex",
    "torchvision",
}


def test_pipeline_builds_canonical_row_with_product_missing_and_legacy_fields():
    from src.scoring.pipeline import build_canonical_score_row

    formula = ProductScoreFormula(
        thresholds={"score_vlm_min": 0.7, "cer_max": 0.2},
        scorer_versions={"vlm": "fake-vlm@1", "ocr": "fake-ocr@2"},
    )

    row = build_canonical_score_row(
        sample_id="prompt-001",
        version=2,
        target_text="ТЕСТ",
        evidence={
            "score_vlm": 0.8,
            "score_ocr": 0.75,
            "cer": 0.25,
            "entropy": 0.1,
            "ocr_detected": "ТЕСТ!",
        },
        formula=formula,
        manifest_path="runs/eval/manifest.json",
    )

    assert row["id"] == "prompt-001"
    assert row["sample_id"] == "prompt-001"
    assert row["version"] == 2
    assert row["target_text"] == "ТЕСТ"
    assert row["score_vlm"] == "0.800000"
    assert row["score_ocr"] == "0.750000"
    assert row["cer"] == "0.250000"
    assert row["entropy"] == "0.100000"
    assert row["detection_status"] == "detected_mismatch"
    assert row["exact_text_match"] == "false"
    assert row["char_matches"] == 4
    assert row["char_total"] == 5
    assert row["char_accuracy"] == "0.800000"
    assert row["product_score"] == row["score"]
    assert row["missing_components"] == ""
    assert row["formula_complete"] == "true"
    assert row["manifest_path"] == "runs/eval/manifest.json"
    assert json.loads(row["text_metrics"])["detected_text"] == "ТЕСТ!"
    assert json.loads(row["scorer_metadata"])["scorer_versions"] == {
        "ocr": "fake-ocr@2",
        "vlm": "fake-vlm@1",
    }
    assert json.loads(row["thresholds"])["cer_max"] is False


def test_pipeline_collects_scoring_tasks_from_tiny_text_embedding_metadata(tmp_path):
    from src.scoring.pipeline import ScoringTask, collect_scoring_tasks

    images_dir = tmp_path / "images"
    text_embeds_dir = tmp_path / "text_embeds"
    prompt_dir = images_dir / "prompt-001"
    prompt_dir.mkdir(parents=True)
    text_embeds_dir.mkdir()
    (prompt_dir / "v2.png").write_bytes(b"")
    (prompt_dir / "v0.png").write_bytes(b"")
    torch.save({"target_text": "ТЕСТ"}, text_embeds_dir / "prompt-001.pt")

    tasks = collect_scoring_tasks(images_dir=images_dir, text_embeds_dir=text_embeds_dir)

    assert tasks == [
        ScoringTask(
            sample_id="prompt-001",
            version=0,
            image_path=prompt_dir / "v0.png",
            target_text="ТЕСТ",
        ),
        ScoringTask(
            sample_id="prompt-001",
            version=2,
            image_path=prompt_dir / "v2.png",
            target_text="ТЕСТ",
        ),
    ]


def test_pipeline_skips_missing_embedding_metadata_with_warning(tmp_path, caplog):
    from src.scoring.pipeline import collect_scoring_tasks

    images_dir = tmp_path / "images"
    text_embeds_dir = tmp_path / "text_embeds"
    (images_dir / "missing-embed").mkdir(parents=True)
    text_embeds_dir.mkdir()
    (images_dir / "missing-embed" / "v0.png").write_bytes(b"")

    tasks = collect_scoring_tasks(images_dir=images_dir, text_embeds_dir=text_embeds_dir)

    assert tasks == []
    assert "No text embedding for missing-embed, skipping" in caplog.text


def test_pipeline_score_schema_sidecar_contains_phase6_fields(tmp_path):
    from src.scoring.pipeline import write_score_schema_sidecar

    output_csv = tmp_path / "scores.csv"
    formula = ProductScoreFormula(
        thresholds={"score_vlm_min": 0.7},
        scorer_versions={"vlm": "fake-vlm@1"},
    )

    sidecar = write_score_schema_sidecar(
        output_csv,
        formula=formula,
        source_manifest_paths=["runs/baseline/manifest.json", "runs/eval/manifest.json"],
        generated_at="2026-05-06T00:00:00Z",
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar == output_csv.with_suffix(".schema.json")
    assert payload["schema_version"] == "reward-score-metadata/v1"
    assert payload["score_file_schema_version"] == "phase6-score-file/v1"
    assert payload["formula"]["name"] == "vlm_ocr_cer_entropy_exact_product_v1"
    assert payload["formula"]["weights"]["score_vlm"] == 0.35
    assert payload["formula"]["thresholds"] == {"score_vlm_min": 0.7}
    assert payload["formula"]["scorer_versions"] == {"vlm": "fake-vlm@1"}
    assert payload["source_manifest_paths"] == [
        "runs/baseline/manifest.json",
        "runs/eval/manifest.json",
    ]
    assert "score" in payload["required_fields"]
    assert "product_score" in payload["required_phase6_fields"]


def test_importing_pipeline_does_not_import_model_or_ocr_stacks(monkeypatch):
    for module_name in ["src.scoring", "src.scoring.pipeline", "src.training.rewards"]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    before = set(sys.modules)

    pipeline = importlib.import_module("src.scoring.pipeline")

    assert pipeline.CANONICAL_SCORE_COLUMNS
    newly_imported = set(sys.modules) - before
    assert not (HEAVY_OPTIONAL_ROOTS & newly_imported)
    assert "src.training.rewards" not in sys.modules


def test_score_images_main_builds_config_and_delegates_to_pipeline(monkeypatch, tmp_path):
    import scripts.score_images as score_images
    from src.scoring.pipeline import ScoringConfig

    captured: list[ScoringConfig] = []

    def fake_run_scoring(config: ScoringConfig) -> None:
        captured.append(config)

    monkeypatch.setattr(score_images, "run_scoring", fake_run_scoring)

    result = score_images.main(
        [
            "--images_dir",
            str(tmp_path / "images"),
            "--text_embeds_dir",
            str(tmp_path / "text_embeds"),
            "--output_csv",
            str(tmp_path / "scores.csv"),
            "--scorer",
            "ocr",
            "--entropy_lambda",
            "1.5",
            "--batch_size",
            "3",
            "--resume",
            "--shard_idx",
            "1",
            "--num_shards",
            "4",
            "--manifest_path",
            "runs/eval/manifest.json",
            "--source_manifest",
            "runs/source-a/manifest.json",
            "--source_manifest",
            "runs/source-b/manifest.json",
        ]
    )

    assert result == 0
    assert captured == [
        ScoringConfig(
            images_dir=tmp_path / "images",
            text_embeds_dir=tmp_path / "text_embeds",
            output_csv=tmp_path / "scores.csv",
            scorer="ocr",
            vlm_model_id="Qwen/Qwen3.5-9B",
            entropy_lambda=1.5,
            batch_size=3,
            resume=True,
            shard_idx=1,
            num_shards=4,
            manifest_path="runs/eval/manifest.json",
            source_manifests=("runs/source-a/manifest.json", "runs/source-b/manifest.json"),
        )
    ]


def test_score_images_is_thin_wrapper_over_scoring_pipeline():
    import scripts.score_images as score_images

    module_source = inspect.getsource(score_images)
    main_source = inspect.getsource(score_images.main)

    assert "from src.scoring.pipeline import" in module_source
    assert "run_scoring(config)" in main_source
    assert "from src.training.rewards import" not in module_source
    assert "torch.load" not in module_source
