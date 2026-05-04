from __future__ import annotations

import csv
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
