from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.runtime.artifacts import ArtifactValidationError, validate_artifacts
from src.runtime.paths import assert_artifact_git_safety, resolve_stage_paths


def _write_jsonl(path: Path, rows: list[dict] | list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_scores_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "version", "score", "target_text", "score_ocr"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _torch_save(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def test_art_02_resolve_stage_paths_uses_canonical_runtime_roots(tmp_path: Path) -> None:
    generated = resolve_stage_paths("generated", root=tmp_path)
    masked = resolve_stage_paths("masked_sft", root=tmp_path)
    manifest = resolve_stage_paths("run_manifest", root=tmp_path, run_id="demo-run")

    assert generated.root == tmp_path
    assert generated.stage == "generated"
    assert generated.paths["prompts_jsonl"] == tmp_path / "data" / "prompts_simple.jsonl"
    assert generated.paths["images_dir"] == tmp_path / "outputs" / "generated" / "images"
    assert generated.paths["latents_dir"] == tmp_path / "outputs" / "generated" / "latents"
    assert generated.paths["text_embeds_dir"] == tmp_path / "outputs" / "generated" / "text_embeds"
    assert generated.paths["scores_csv"] == tmp_path / "outputs" / "generated" / "scores.csv"
    assert masked.paths["data_dir"] == tmp_path / "data" / "synth_cyrillic" / "masked_sft"
    assert masked.paths["shapes_csv"] == masked.paths["data_dir"] / "shapes.csv"
    assert manifest.paths["manifest_json"] == tmp_path / "runs" / "demo-run" / "manifest.json"


def test_art_01_prompt_jsonl_reports_required_fields_and_line_numbers(tmp_path: Path) -> None:
    prompts = _write_jsonl(
        tmp_path / "data" / "prompts.jsonl",
        [
            {"prompt": "Render Привет", "target_text": "Привет"},
            "{not-json",
            {"target_text": "missing prompt"},
        ],
    )

    report = validate_artifacts("prompts", {"prompts_jsonl": prompts})

    assert report.schema_version == "runtime-artifacts/v1"
    assert str(prompts) in report.checked_paths
    assert not report.ok
    assert any("line 2" in error and "malformed JSON" in error for error in report.errors)
    assert any("line 3" in error and "prompt" in error for error in report.errors)


def test_art_03_scores_csv_validates_columns_and_schema_metadata(tmp_path: Path) -> None:
    scores = _write_scores_csv(
        tmp_path / "outputs" / "generated" / "scores.csv",
        [{"id": "000001", "version": 0, "score": 0.75, "target_text": "Ж", "score_ocr": 0.6}],
    )
    scores.with_suffix(".schema.json").write_text(
        json.dumps({"schema_version": "scores/v1", "scorer": "ocr"}), encoding="utf-8"
    )

    report = validate_artifacts("scores", {"scores_csv": scores})

    assert report.ok
    assert report.metadata["scores_schema_version"] == "scores/v1"
    assert not report.errors


def test_art_01_generated_layout_matches_prompt_ids_and_versions(tmp_path: Path) -> None:
    generated = tmp_path / "outputs" / "generated"
    _write_jsonl(generated / "prompts.jsonl", [{"prompt": "Render A", "target_text": "A"}])
    _torch_save({"latent": torch.zeros(1, 2, 2)}, generated / "latents" / "000000" / "v0.pt")
    _torch_save(
        {"prompt_embeds": torch.zeros(2, 3), "target_text": "A", "prompt": "Render A"},
        generated / "text_embeds" / "000000.pt",
    )
    image_dir = generated / "images" / "000000"
    image_dir.mkdir(parents=True)
    (image_dir / "v0.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    report = validate_artifacts(
        "generated",
        {
            "prompts_jsonl": generated / "prompts.jsonl",
            "latents_dir": generated / "latents",
            "text_embeds_dir": generated / "text_embeds",
            "images_dir": generated / "images",
        },
    )

    assert report.ok
    assert report.metadata["prompt_count"] == 1
    assert report.metadata["generated_versions"] == {"000000": [0]}


def test_art_01_generated_layout_reports_missing_tensor_keys(tmp_path: Path) -> None:
    generated = tmp_path / "outputs" / "generated"
    _torch_save({"wrong": torch.zeros(1)}, generated / "latents" / "000000" / "v0.pt")
    _torch_save({"prompt_embeds": torch.zeros(2, 3)}, generated / "text_embeds" / "000000.pt")
    (generated / "images" / "000000").mkdir(parents=True)
    (generated / "images" / "000000" / "v0.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    report = validate_artifacts(
        "generated",
        {
            "latents_dir": generated / "latents",
            "text_embeds_dir": generated / "text_embeds",
            "images_dir": generated / "images",
        },
    )

    assert not report.ok
    assert any("000000/v0.pt" in error and "latent" in error for error in report.errors)


def test_scoring_input_validation_reports_corrupt_embeddings_and_bad_versions(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images" / "p1"
    embeds = tmp_path / "embeds"
    images.mkdir(parents=True)
    embeds.mkdir()
    (images / "vbad.png").write_bytes(b"image")
    (embeds / "p1.pt").write_bytes(b"not-a-torch-file")

    report = validate_artifacts(
        "scoring_inputs",
        {"images_dir": images.parent, "text_embeds_dir": embeds},
    )

    assert report.ok is False
    assert any("invalid image version" in error for error in report.errors)
    assert any("could not read text embedding" in error for error in report.errors)


def test_art_01_masked_sft_validates_latents_masks_embeds_and_shapes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "synth_cyrillic" / "masked_sft"
    _torch_save(
        {"latent": torch.zeros(1, 2, 2), "mask_lat": torch.ones(2, 2)},
        data_dir / "latents" / "sample-1.pt",
    )
    _torch_save({"prompt_embeds": torch.zeros(2, 3)}, data_dir / "text_embeds" / "sample-1.pt")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "shapes.csv").write_text("id,H,W\nsample-1,2,2\n", encoding="utf-8")

    report = validate_artifacts("masked_sft", {"data_dir": data_dir})

    assert report.ok
    assert report.metadata["masked_sft_samples"] == ["sample-1"]
    assert str(data_dir / "shapes.csv") in report.checked_paths


def test_art_04_git_safety_blocks_generated_artifacts_and_allows_fixtures() -> None:
    unsafe = assert_artifact_git_safety(
        [
            "outputs/generated/images/000001/v0.png",
            "runs/demo/manifest.json",
            "outputs/sft/checkpoints/step-1/adapter_model.safetensors",
            "logs/train.log",
            "data/synth_cyrillic/masked_sft/latents/sample.pt",
        ]
    )
    safe = assert_artifact_git_safety(
        ["experiments/assets/example.png", "tests/fixtures/readme.txt", "docs/runtime_contracts.md"]
    )

    assert not unsafe.ok
    assert len(unsafe.errors) == 5
    assert safe.ok


def test_blocking_required_inputs_raise_for_expensive_downstream_stages(tmp_path: Path) -> None:
    missing = tmp_path / "outputs" / "generated" / "scores.csv"

    with pytest.raises(ArtifactValidationError, match="scores_csv"):
        validate_artifacts("sft", {"scores_csv": missing, "require_ready": True})


def test_training_artifacts_require_configured_materialized_rows_and_checkpoints(
    tmp_path: Path,
) -> None:
    scores = _write_scores_csv(
        tmp_path / "outputs" / "generated" / "scores.csv",
        [{"id": "p1", "version": 0, "score": 0.8, "target_text": "ТЕСТ"}],
    )
    latents = tmp_path / "outputs" / "generated" / "latents"
    embeds = tmp_path / "outputs" / "generated" / "text_embeds"
    latents.mkdir(parents=True)
    embeds.mkdir(parents=True)

    sft = validate_artifacts(
        "sft",
        {
            "scores_csv": scores,
            "latents_dir": latents,
            "text_embeds_dir": embeds,
            "selected_samples": tmp_path / "missing-selected.jsonl",
            "resume_lora": tmp_path / "missing-resume",
        },
    )
    dpo = validate_artifacts(
        "dpo",
        {
            "scores_csv": scores,
            "latents_dir": latents,
            "text_embeds_dir": embeds,
            "preference_pairs": tmp_path / "missing-pairs.jsonl",
            "sft_lora": tmp_path / "empty-checkpoint",
        },
    )

    assert not sft.ok
    assert any("selected_samples file is missing" in error for error in sft.errors)
    assert any("resume_lora_path: checkpoint path does not exist" in error for error in sft.errors)
    assert not dpo.ok
    assert any("preference_pairs file is missing" in error for error in dpo.errors)
    assert any("sft_lora_path: checkpoint path does not exist" in error for error in dpo.errors)


def test_phase6_jsonl_validation_rejects_nonfinite_invalid_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    from src.evaluation.reward_interface import ProductScoreFormula
    from src.runtime.artifacts import PHASE6_REQUIRED_SCORE_FIELDS
    from src.runtime.manifests import create_run_manifest

    scores = tmp_path / "scores.jsonl"
    base = {
        "sample_id": "p1",
        "version": 0,
        "score": 0.5,
        "product_score": 0.5,
        "target_text": "ТЕСТ",
        "score_vlm": 0.8,
        "score_ocr": 0.7,
        "cer": 0.0,
        "entropy": 0.1,
        "ocr_detected": "ТЕСТ",
        "detection_status": "detected_exact",
        "exact_text_match": True,
        "char_accuracy": 1.0,
        "char_matches": 4,
        "char_total": 4,
        "missing_components": [],
        "formula_complete": True,
        "manifest_path": "runs/eval/manifest.json",
        "text_metrics": {},
        "scorer_metadata": {},
        "thresholds": {},
    }
    invalid = dict(base)
    invalid.update(score=float("nan"), detection_status="unknown", exact_text_match="true")
    _write_jsonl(scores, [invalid, dict(base)])
    source = create_run_manifest(
        stage="evaluation",
        command=["pytest", "jsonl-validation"],
        run_root=tmp_path / "runs",
        outputs={"scores_jsonl": str(scores)},
        root=tmp_path,
    )
    sidecar = {
        "schema_version": "reward-score-metadata/v1",
        "score_file_schema_version": "phase6-score-jsonl/v1",
        "formula": ProductScoreFormula().to_metadata(),
        "primary_score": "product",
        "required_phase6_fields": sorted(PHASE6_REQUIRED_SCORE_FIELDS),
        "source_manifest_paths": [str(source.manifest_path)],
        "source_manifest_sha256": {
            str(source.manifest_path): hashlib.sha256(source.manifest_path.read_bytes()).hexdigest()
        },
        "execution": {
            "status": "complete",
            "scored_row_count": 2,
            "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
        },
    }
    scores.with_suffix(".schema.json").write_text(json.dumps(sidecar), encoding="utf-8")

    report = validate_artifacts("evaluation_scores", {"scores_jsonl": scores})

    assert not report.ok
    assert any("score must be a finite number" in error for error in report.errors)
    assert any("detection_status is invalid" in error for error in report.errors)
    assert any("exact_text_match must be boolean" in error for error in report.errors)
    assert any("duplicate sample_id/version pair" in error for error in report.errors)


def test_runtime_contract_docs_cover_required_artifact_families() -> None:
    docs = Path("docs/runtime_contracts.md").read_text(encoding="utf-8")
    required_sections = [
        "## Canonical Runtime Roots",
        "## Artifact Contract Matrix",
        "## Local and SLURM Path Guidance",
        "## Git-Safety Classification",
        "## Preflight Validator Hooks",
    ]
    required_artifacts = [
        "prompts",
        "images",
        "latents",
        "text embeddings",
        "masks",
        "scores",
        "selected samples",
        "preference pairs",
        "checkpoints",
        "samples",
        "logs",
        "eval outputs",
        "run manifests",
        "schema_version",
    ]

    missing_sections = [section for section in required_sections if section not in docs]
    missing_artifacts = [artifact for artifact in required_artifacts if artifact not in docs]

    assert not missing_sections
    assert not missing_artifacts
