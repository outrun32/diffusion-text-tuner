from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.reward_interface import ProductScoreFormula


def _write_source_run_manifest(tmp_path: Path, slug: str) -> Path:
    from src.runtime.manifests import create_run_manifest

    return create_run_manifest(
        stage="evaluation",
        command=["pytest", slug],
        run_root=tmp_path / "runs",
        slug=slug,
        root=tmp_path,
    ).manifest_path


def test_score_images_builds_canonical_csv_row_with_product_and_missing_evidence():
    from scripts.score_images import build_canonical_score_row

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


def test_score_images_metadata_sidecar_contains_formula_versions_and_manifests(tmp_path):
    from scripts.score_images import write_score_schema_sidecar

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


def test_evaluate_rewards_emits_canonical_jsonl_record_with_metadata():
    from src.evaluation.evaluate_rewards import build_canonical_evaluation_record

    formula = ProductScoreFormula(scorer_versions={"vlm": "fake-vlm", "ocr": "fake-ocr"})

    record = build_canonical_evaluation_record(
        source_record={"id": "sample-a", "image": "images/a.png", "target_text": "КИТ"},
        reward_outputs={
            "reward_qwen_yes_prob": 0.9,
            "reward_paddleocr": 0.5,
            "ocr_detected": "КОТ",
            "cer": 1 / 3,
            "entropy": 0.2,
        },
        version=3,
        formula=formula,
        manifest_path="runs/eval/manifest.json",
    )

    assert record["schema_version"] == "reward-result/v1"
    assert record["sample_id"] == "sample-a"
    assert record["image"] == "images/a.png"
    assert record["version"] == 3
    assert record["score_vlm"] == 0.9
    assert record["score_ocr"] == 0.5
    assert record["product_score"] == record["score"]
    assert record["exact_text_match"] is False
    assert record["char_accuracy"] == 2 / 3
    assert record["detection_status"] == "detected_mismatch"
    assert record["manifest_path"] == "runs/eval/manifest.json"
    assert record["scorer_metadata"]["scorer_versions"] == {"ocr": "fake-ocr", "vlm": "fake-vlm"}
    assert record["missing_components"] == []


def test_artifact_validation_accepts_phase6_score_csv_and_sidecar(tmp_path):
    from src.runtime.artifacts import validate_artifacts
    from src.runtime.manifests import create_run_manifest
    from src.scoring.pipeline import write_score_schema_sidecar

    csv_path = tmp_path / "scores.csv"
    fieldnames = [
        "id",
        "sample_id",
        "version",
        "score",
        "product_score",
        "target_text",
        "score_vlm",
        "score_ocr",
        "cer",
        "entropy",
        "ocr_detected",
        "detection_status",
        "exact_text_match",
        "char_accuracy",
        "char_matches",
        "char_total",
        "missing_components",
        "formula_complete",
        "manifest_path",
        "text_metrics",
        "scorer_metadata",
        "thresholds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "id": "sample-a",
                "sample_id": "sample-a",
                "version": "1",
                "score": "0.5",
                "product_score": "0.5",
                "target_text": "ТЕСТ",
                "score_vlm": "0.8",
                "score_ocr": "0.7",
                "cer": "0.1",
                "entropy": "0.2",
                "ocr_detected": "ТЕСТ",
                "detection_status": "detected_exact",
                "exact_text_match": "true",
                "char_accuracy": "1.0",
                "char_matches": "4",
                "char_total": "4",
                "missing_components": "",
                "formula_complete": "true",
                "manifest_path": "runs/eval/manifest.json",
                "text_metrics": json.dumps({"detected_text": "ТЕСТ"}),
                "scorer_metadata": json.dumps({"scorer_versions": {"vlm": "fake"}}),
                "thresholds": json.dumps({"score_vlm_min": True}),
            }
        )
    source_manifest = create_run_manifest(
        stage="evaluation",
        command=["pytest", "phase6-artifact-fixture"],
        run_root=tmp_path / "runs",
        slug="phase6-artifact-fixture",
        outputs={"scores_csv": str(csv_path)},
        root=tmp_path,
    )
    write_score_schema_sidecar(
        csv_path,
        formula=ProductScoreFormula(
            thresholds={"score_vlm_min": 0.7},
            scorer_versions={"vlm": "fake"},
        ),
        source_manifest_paths=(str(source_manifest.manifest_path),),
        primary_score="product",
        execution_metadata={
            "status": "complete",
            "scored_row_count": 1,
            "scores_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        },
    )

    report = validate_artifacts("evaluation_scores", {"scores_csv": csv_path})

    assert report.ok
    assert report.metadata["score_file_schema_version"] == "phase6-score-file/v1"
    assert report.metadata["scores_rows"] == 1


def test_artifact_validation_rejects_missing_phase6_score_fields(tmp_path):
    from src.runtime.artifacts import validate_artifacts

    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("id,version,score,target_text\nsample-a,1,0.5,ТЕСТ\n", encoding="utf-8")
    csv_path.with_suffix(".schema.json").write_text(
        json.dumps(
            {
                "schema_version": "reward-score-metadata/v1",
                "score_file_schema_version": "phase6-score-file/v1",
                "formula": {"name": "vlm_ocr_cer_entropy_exact_product_v1"},
                "source_manifest_paths": [],
                "required_phase6_fields": ["product_score"],
            }
        ),
        encoding="utf-8",
    )

    report = validate_artifacts("evaluation_scores", {"scores_csv": csv_path})

    assert not report.ok
    assert any("missing required canonical score columns" in error for error in report.errors)


def test_reward_evaluation_docs_describe_score_files_sidecars_and_validation():
    docs = Path("docs/reward_evaluation.md").read_text(encoding="utf-8")

    required_phrases = [
        "Canonical score CSV/JSONL fields",
        "score_file_schema_version",
        "phase6-score-file/v1",
        "phase6-score-jsonl/v1",
        "product_score",
        "detection_status",
        "char_accuracy",
        "formula_complete",
        "--manifest_path",
        "--source_manifest",
        "--manifest-path",
        "--source-manifest",
        'validate_artifacts("evaluation_scores"',
        "Generated score files and `.schema.json` sidecars are runtime artifacts",
        "missing_components",
    ]
    for phrase in required_phrases:
        assert phrase in docs


def test_evaluate_rewards_is_atomic_idempotent_and_requires_explicit_overwrite(
    monkeypatch, tmp_path
):
    import src.evaluation.evaluate_rewards as evaluate_rewards

    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"id": "a", "image": str(image), "target_text": "ТЕСТ"}) + "\n",
        encoding="utf-8",
    )
    manifest = _write_source_run_manifest(tmp_path, "idempotent-source")
    output = tmp_path / "scores.jsonl"

    class FakeOcr:
        def score(self, image_path, target_text):
            assert image_path == str(image)
            assert target_text == "ТЕСТ"
            return {"reward_paddleocr": 1.0, "ocr_detected": "ТЕСТ"}

    monkeypatch.setattr(evaluate_rewards, "PaddleOCRReward", FakeOcr)
    monkeypatch.setattr(
        evaluate_rewards,
        "check_stage_support",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=()),
    )
    args = [
        "--metadata",
        str(metadata),
        "--output",
        str(output),
        "--reward",
        "paddleocr",
        "--source-manifest",
        str(manifest),
    ]

    assert evaluate_rewards.main(args) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    from src.runtime.artifacts import validate_artifacts

    assert validate_artifacts("evaluation_scores", {"scores_jsonl": output}).ok
    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluate_rewards.main(args)
    assert evaluate_rewards.main([*args, "--overwrite"]) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_evaluate_rewards_rejects_missing_inputs_before_platform_or_output_mutation(
    monkeypatch, tmp_path
):
    import src.evaluation.evaluate_rewards as evaluate_rewards

    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"id": "a", "image": str(tmp_path / "missing.png"), "target_text": "ТЕСТ"})
        + "\n",
        encoding="utf-8",
    )
    manifest = _write_source_run_manifest(tmp_path, "missing-input-source")
    output = tmp_path / "scores.jsonl"
    platform_checked = False

    def fail_if_checked(*args, **kwargs):
        nonlocal platform_checked
        platform_checked = True
        return SimpleNamespace(ok=True, errors=())

    monkeypatch.setattr(evaluate_rewards, "check_stage_support", fail_if_checked)

    with pytest.raises(FileNotFoundError, match="metadata image does not exist"):
        evaluate_rewards.main(
            [
                "--metadata",
                str(metadata),
                "--output",
                str(output),
                "--reward",
                "paddleocr",
                "--source-manifest",
                str(manifest),
            ]
        )

    assert not platform_checked
    assert not output.exists()
    assert not output.with_suffix(".schema.json").exists()


def test_evaluate_rewards_requires_real_source_manifest(tmp_path):
    from src.evaluation.evaluate_rewards import write_evaluation_score_metadata

    output = tmp_path / "scores.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one source manifest"):
        write_evaluation_score_metadata(output)
    with pytest.raises(ValueError, match="invalid source manifest"):
        write_evaluation_score_metadata(
            output, source_manifest_paths=(str(tmp_path / "missing.json"),)
        )
    arbitrary = tmp_path / "arbitrary.json"
    arbitrary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported source manifest schema"):
        write_evaluation_score_metadata(output, source_manifest_paths=(str(arbitrary),))

    strict_manifest = _write_source_run_manifest(tmp_path, "frozen-source")
    expected_hashes = {
        str(strict_manifest): hashlib.sha256(strict_manifest.read_bytes()).hexdigest()
    }
    from src.runtime.manifests import update_run_manifest

    update_run_manifest(strict_manifest, note="changed concurrently")
    with pytest.raises(ValueError, match="changed during evaluation"):
        write_evaluation_score_metadata(
            output,
            source_manifest_paths=(str(strict_manifest),),
            expected_source_manifest_sha256=expected_hashes,
        )


def test_evaluate_rewards_platform_guard_preserves_existing_artifacts(monkeypatch, tmp_path):
    import src.evaluation.evaluate_rewards as evaluate_rewards

    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"id": "a", "image": str(image), "target_text": "ТЕСТ"}) + "\n",
        encoding="utf-8",
    )
    manifest = _write_source_run_manifest(tmp_path, "platform-source")
    output = tmp_path / "scores.jsonl"
    sidecar = output.with_suffix(".schema.json")
    output.write_text("old scores\n", encoding="utf-8")
    sidecar.write_text("old sidecar\n", encoding="utf-8")

    monkeypatch.setattr(
        evaluate_rewards,
        "check_stage_support",
        lambda *args, **kwargs: SimpleNamespace(ok=False, errors=("unsupported host",)),
    )

    with pytest.raises(RuntimeError, match="unsupported host"):
        evaluate_rewards.main(
            [
                "--metadata",
                str(metadata),
                "--output",
                str(output),
                "--reward",
                "paddleocr",
                "--source-manifest",
                str(manifest),
                "--overwrite",
            ]
        )

    assert output.read_text(encoding="utf-8") == "old scores\n"
    assert sidecar.read_text(encoding="utf-8") == "old sidecar\n"


def test_evaluate_rewards_failure_does_not_publish_partial_rows(monkeypatch, tmp_path):
    import src.evaluation.evaluate_rewards as evaluate_rewards

    records = []
    for sample_id in ("a", "b"):
        image = tmp_path / f"{sample_id}.png"
        image.write_bytes(b"fixture")
        records.append({"id": sample_id, "image": str(image), "target_text": "ТЕСТ"})
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    manifest = _write_source_run_manifest(tmp_path, "failure-source")
    output = tmp_path / "scores.jsonl"

    class FailingOcr:
        def __init__(self):
            self.calls = 0

        def score(self, image_path, target_text):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("scorer failed")
            return {"reward_paddleocr": 1.0, "ocr_detected": target_text}

    monkeypatch.setattr(evaluate_rewards, "PaddleOCRReward", FailingOcr)
    monkeypatch.setattr(
        evaluate_rewards,
        "check_stage_support",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=()),
    )

    with pytest.raises(RuntimeError, match="scorer failed"):
        evaluate_rewards.main(
            [
                "--metadata",
                str(metadata),
                "--output",
                str(output),
                "--reward",
                "paddleocr",
                "--source-manifest",
                str(manifest),
            ]
        )

    assert not output.exists()
    assert not output.with_suffix(".schema.json").exists()
    assert not output.with_suffix(output.suffix + ".tmp").exists()
