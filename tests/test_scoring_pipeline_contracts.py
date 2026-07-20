from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_pipeline_primary_score_tracks_selected_reward() -> None:
    from src.evaluation.reward_interface import vlm_ocr_product_formula
    from src.scoring.pipeline import build_canonical_score_row

    evidence = {"score_vlm": 0.8, "score_ocr": 0.75, "ocr_detected": "ТЕСТ"}
    formula = vlm_ocr_product_formula()

    vlm_row = build_canonical_score_row(
        sample_id="p",
        version=0,
        target_text="ТЕСТ",
        evidence=evidence,
        formula=formula,
        primary_score="vlm",
    )
    ocr_row = build_canonical_score_row(
        sample_id="p",
        version=0,
        target_text="ТЕСТ",
        evidence=evidence,
        formula=formula,
        primary_score="ocr",
    )
    product_row = build_canonical_score_row(
        sample_id="p",
        version=0,
        target_text="ТЕСТ",
        evidence=evidence,
        formula=formula,
        primary_score="product",
    )

    assert vlm_row["score"] == "0.800000"
    assert ocr_row["score"] == "0.750000"
    assert product_row["score"] == "0.600000"
    assert product_row["product_score"] == "0.600000"


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


def test_pipeline_rejects_missing_embedding_metadata(tmp_path):
    from src.scoring.pipeline import collect_scoring_tasks

    images_dir = tmp_path / "images"
    text_embeds_dir = tmp_path / "text_embeds"
    (images_dir / "missing-embed").mkdir(parents=True)
    text_embeds_dir.mkdir()
    (images_dir / "missing-embed" / "v0.png").write_bytes(b"")

    with pytest.raises(ValueError, match="No text embedding"):
        collect_scoring_tasks(images_dir=images_dir, text_embeds_dir=text_embeds_dir)


def test_pipeline_rejects_noncanonical_or_aliased_image_versions(tmp_path):
    from src.scoring.pipeline import collect_scoring_tasks

    images_dir = tmp_path / "images"
    text_embeds_dir = tmp_path / "text_embeds"
    prompt_dir = images_dir / "000000"
    prompt_dir.mkdir(parents=True)
    text_embeds_dir.mkdir()
    (prompt_dir / "v0.png").write_bytes(b"canonical")
    (prompt_dir / "v00.png").write_bytes(b"alias")
    torch.save({"target_text": "ТЕСТ"}, text_embeds_dir / "000000.pt")

    with pytest.raises(ValueError, match="Invalid image version filename"):
        collect_scoring_tasks(images_dir=images_dir, text_embeds_dir=text_embeds_dir)


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
    assert payload["primary_score"] == "product"
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
            "1",
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
            vlm_model_revision=None,
            vlm_device="cuda",
            ocr_device="cpu",
            product_formula="product",
            entropy_lambda=1.5,
            batch_size=1,
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


def test_scoring_config_rejects_noop_batching_and_invalid_shards(tmp_path):
    from src.scoring.pipeline import ScoringConfig

    base = {
        "images_dir": tmp_path / "images",
        "text_embeds_dir": tmp_path / "embeds",
    }
    with pytest.raises(ValueError, match="batched scoring is not implemented"):
        ScoringConfig(**base, batch_size=2)
    with pytest.raises(ValueError, match="num_shards"):
        ScoringConfig(**base, num_shards=0)
    with pytest.raises(ValueError, match="shard_idx"):
        ScoringConfig(**base, num_shards=2, shard_idx=2)


def test_resume_rejects_formula_or_source_manifest_drift(tmp_path):
    from src.evaluation.reward_interface import vlm_ocr_product_formula
    from src.scoring.pipeline import _validate_resume_sidecar, write_score_schema_sidecar

    output = tmp_path / "scores.csv"
    output.write_text("id\n", encoding="utf-8")
    formula = vlm_ocr_product_formula(scorer_versions={"vlm": "qwen@a"})
    write_score_schema_sidecar(
        output,
        formula=formula,
        primary_score="product",
        source_manifest_paths=("runs/a/manifest.json",),
        execution_metadata={"shard_idx": 0, "num_shards": 1},
    )

    with pytest.raises(ValueError, match="does not match"):
        _validate_resume_sidecar(
            output,
            formula=vlm_ocr_product_formula(scorer_versions={"vlm": "qwen@b"}),
            primary_score="product",
            source_manifest_paths=("runs/a/manifest.json",),
            shard_idx=0,
            num_shards=1,
        )


def _write_complete_generation_source(tmp_path, targets=("ТЕСТ",)):
    from src.generation.pipeline import (
        GenerationConfig,
        begin_generation_attempt,
        complete_generation_attempt,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )
    from src.runtime.manifests import create_run_manifest

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {"prompt": f"Render {target}", "target_text": target},
                ensure_ascii=False,
            )
            + "\n"
            for target in targets
        ),
        encoding="utf-8",
    )
    run_manifest = create_run_manifest(
        stage="generate",
        command=["pytest", "generation-source"],
        run_root=tmp_path / "runs",
        root=tmp_path,
    ).manifest_path
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        model_revision="a" * 40,
        versions_per_prompt=1,
        end_idx=len(targets),
        run_manifest_path=str(run_manifest),
    )
    paths = resolve_generation_paths(config.output_dir)
    manifest_path = ensure_generation_resume_contract(config, paths, load_prompt_records(prompts))
    begin_generation_attempt(manifest_path, run_manifest_path=str(run_manifest))
    paths.text_embeds_dir.mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets):
        prompt_id = f"{index:06d}"
        torch.save({"target_text": target}, paths.text_embeds_dir / f"{prompt_id}.pt")
        image = paths.images_dir / prompt_id / "v0.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"not-loaded")
        latent = paths.latents_dir / prompt_id / "v0.pt"
        latent.parent.mkdir(parents=True, exist_ok=True)
        latent.write_bytes(b"fixture-latent")
    expected = {"text_embeddings": len(targets), "images": len(targets), "latents": len(targets)}
    complete_generation_attempt(
        manifest_path,
        generated=expected,
        skipped={"text_embeddings": 0, "images": 0, "latents": 0},
    )
    return paths, manifest_path


def _write_resume_fixture(tmp_path, *, status="in-progress"):
    from src.scoring.pipeline import (
        CANONICAL_SCORE_COLUMNS,
        ScoringConfig,
        _formula_for_config,
        build_canonical_score_row,
        write_score_schema_sidecar,
    )

    generation_paths, source_manifest = _write_complete_generation_source(tmp_path)
    images_dir = generation_paths.images_dir
    embeds_dir = generation_paths.text_embeds_dir
    output = tmp_path / "scores.csv"
    config = ScoringConfig(
        images_dir=images_dir,
        text_embeds_dir=embeds_dir,
        output_csv=output,
        scorer="ocr",
        resume=True,
        source_manifests=(str(source_manifest),),
    )
    formula, primary_score = _formula_for_config(config)
    row = build_canonical_score_row(
        sample_id="000000",
        version=0,
        target_text="ТЕСТ",
        evidence={"score_ocr": 0.75, "ocr_detected": "ТЕСТ"},
        formula=formula,
        primary_score=primary_score,
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    execution = {
        "status": status,
        "discovered_task_count": 1,
        "expected_shard_count": 1,
        "scored_row_count": 1,
        "shard_idx": 0,
        "num_shards": 1,
        "scores_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    write_score_schema_sidecar(
        output,
        formula=formula,
        primary_score=primary_score,
        source_manifest_paths=config.source_manifests,
        execution_metadata=execution,
    )
    return config, output


def test_resume_rejects_tampered_existing_row_content(tmp_path):
    from src.scoring.pipeline import run_scoring

    config, output = _write_resume_fixture(tmp_path)
    text = output.read_text(encoding="utf-8")
    output.write_text(text.replace("ТЕСТ,", "ДРИФТ,"), encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        run_scoring(config)


def test_resume_rejects_coherently_tampered_reward_evidence(tmp_path):
    from src.scoring.pipeline import run_scoring

    config, output = _write_resume_fixture(tmp_path)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["score_ocr"] = "0.990000"
    rows[0]["score"] = "0.990000"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="hash does not match"):
        run_scoring(config)


def test_resume_rejects_uncheckpointed_in_progress_rows(tmp_path):
    from src.scoring.pipeline import run_scoring

    config, output = _write_resume_fixture(tmp_path)
    sidecar_path = output.with_suffix(".schema.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["execution"].pop("scores_sha256")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint scores_sha256"):
        run_scoring(config)


def test_resume_rejects_complete_sidecar_hash_drift(tmp_path):
    from src.scoring.pipeline import run_scoring

    config, output = _write_resume_fixture(tmp_path, status="complete")
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        run_scoring(config)


def test_run_scoring_checks_platform_before_creating_artifacts(monkeypatch, tmp_path):
    from src.runtime import capabilities
    from src.scoring.pipeline import ScoringConfig, run_scoring

    generation_paths, source_manifest = _write_complete_generation_source(tmp_path)
    images_dir = generation_paths.images_dir
    embeds_dir = generation_paths.text_embeds_dir
    output = tmp_path / "results" / "scores.csv"
    monkeypatch.setattr(
        capabilities,
        "check_stage_support",
        lambda *args, **kwargs: SimpleNamespace(ok=False, errors=("unsupported host",)),
    )

    with pytest.raises(RuntimeError, match="unsupported host"):
        run_scoring(
            ScoringConfig(
                images_dir=images_dir,
                text_embeds_dir=embeds_dir,
                output_csv=output,
                scorer="ocr",
                source_manifests=(str(source_manifest),),
            )
        )

    assert not output.exists()
    assert not output.with_suffix(".schema.json").exists()


def test_run_scoring_requires_source_manifest_before_artifact_creation(tmp_path):
    from src.scoring.pipeline import ScoringConfig, run_scoring

    images_dir = tmp_path / "images"
    embeds_dir = tmp_path / "embeds"
    prompt_dir = images_dir / "sample-a"
    prompt_dir.mkdir(parents=True)
    embeds_dir.mkdir()
    (prompt_dir / "v0.png").write_bytes(b"not-loaded")
    torch.save({"target_text": "ТЕСТ"}, embeds_dir / "sample-a.pt")
    output = tmp_path / "results" / "scores.csv"

    with pytest.raises(ValueError, match="at least one source manifest"):
        run_scoring(
            ScoringConfig(
                images_dir=images_dir,
                text_embeds_dir=embeds_dir,
                output_csv=output,
                scorer="ocr",
            )
        )

    assert not output.exists()
    assert not output.with_suffix(".schema.json").exists()


def test_generation_source_manifest_must_cover_every_discovered_scoring_task(tmp_path):
    from src.scoring.pipeline import (
        ScoringConfig,
        _require_source_manifests,
        _validate_generation_task_coverage,
        collect_scoring_tasks,
    )

    paths, source_manifest = _write_complete_generation_source(tmp_path)
    stale_id = "000001"
    stale_image = paths.images_dir / stale_id / "v0.png"
    stale_image.parent.mkdir(parents=True)
    stale_image.write_bytes(b"stale-extra")
    torch.save({"target_text": "ЛИШНЕЕ"}, paths.text_embeds_dir / f"{stale_id}.pt")
    tasks = collect_scoring_tasks(
        images_dir=paths.images_dir,
        text_embeds_dir=paths.text_embeds_dir,
    )
    config = ScoringConfig(
        images_dir=paths.images_dir,
        text_embeds_dir=paths.text_embeds_dir,
        source_manifests=(str(source_manifest),),
    )

    with pytest.raises(ValueError, match="task coverage does not match"):
        _validate_generation_task_coverage(
            config,
            tasks,
            _require_source_manifests(config.source_manifests),
        )


def test_run_scoring_checkpoints_every_persisted_row(monkeypatch, tmp_path):
    import src.scoring.pipeline as pipeline
    from src.runtime import capabilities

    generation_paths, source_manifest = _write_complete_generation_source(
        tmp_path, targets=("ТЕСТ", "ТЕКСТ")
    )
    images_dir = generation_paths.images_dir
    embeds_dir = generation_paths.text_embeds_dir
    output = tmp_path / "scores.csv"

    class FakeOcr:
        def __init__(self, **_kwargs):
            pass

        def score(self, _image_path, target_text):
            return {
                "reward_ocr": 1.0,
                "ocr_detected": target_text,
                "cer": 0.0,
                "entropy": 0.0,
            }

    monkeypatch.setitem(
        sys.modules,
        "src.training.rewards",
        SimpleNamespace(OcrCerEntropyReward=FakeOcr),
    )
    monkeypatch.setattr(
        capabilities,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, errors=()),
    )
    original_write = pipeline.write_score_schema_sidecar
    checkpoints = []

    def record_checkpoint(output_csv, **kwargs):
        execution = dict(kwargs.get("execution_metadata") or {})
        if Path(output_csv).is_file() and execution:
            checkpoints.append(
                (
                    execution,
                    hashlib.sha256(Path(output_csv).read_bytes()).hexdigest(),
                )
            )
        return original_write(output_csv, **kwargs)

    monkeypatch.setattr(pipeline, "write_score_schema_sidecar", record_checkpoint)

    pipeline.run_scoring(
        pipeline.ScoringConfig(
            images_dir=images_dir,
            text_embeds_dir=embeds_dir,
            output_csv=output,
            scorer="ocr",
            source_manifests=(str(source_manifest),),
        )
    )

    assert [checkpoint[0]["scored_row_count"] for checkpoint in checkpoints] == [0, 1, 2, 2]
    assert [checkpoint[0]["status"] for checkpoint in checkpoints] == [
        "in-progress",
        "in-progress",
        "in-progress",
        "complete",
    ]
    assert all(execution["scores_sha256"] == actual_hash for execution, actual_hash in checkpoints)


def test_resume_repairs_full_in_progress_sidecar(monkeypatch, tmp_path):
    from src.runtime import capabilities
    from src.scoring.pipeline import run_scoring

    config, output = _write_resume_fixture(tmp_path)

    def unexpected_platform_check(*args, **kwargs):
        raise AssertionError("repairing a fully written shard must remain CPU-safe")

    monkeypatch.setattr(capabilities, "check_stage_support", unexpected_platform_check)

    run_scoring(config)

    sidecar = json.loads(output.with_suffix(".schema.json").read_text(encoding="utf-8"))
    assert sidecar["execution"]["status"] == "complete"
    assert sidecar["execution"]["scored_row_count"] == 1
    assert sidecar["execution"]["scores_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_resume_complete_noop_does_not_require_execution_platform(monkeypatch, tmp_path):
    from src.runtime import capabilities
    from src.scoring.pipeline import run_scoring

    config, output = _write_resume_fixture(tmp_path, status="complete")

    def unexpected_platform_check(*args, **kwargs):
        raise AssertionError("a verified complete resume must be a CPU-safe no-op")

    monkeypatch.setattr(capabilities, "check_stage_support", unexpected_platform_check)

    run_scoring(config)

    sidecar = json.loads(output.with_suffix(".schema.json").read_text(encoding="utf-8"))
    assert sidecar["execution"]["status"] == "complete"
