"""Tests for deterministic, contract-safe multi-seed score aggregation."""

from __future__ import annotations

import csv
import hashlib
import json

import pytest

from src.evaluation.reward_interface import thesis_product_formula
from src.scoring.pipeline import CANONICAL_SCORE_COLUMNS, write_score_schema_sidecar


def _write_manifest(path, name):
    from src.runtime.manifests import create_run_manifest

    return create_run_manifest(
        stage="evaluation",
        command=["pytest", name],
        run_root=path.parent / "runs",
        slug=name,
        root=path.parent,
    ).manifest_path


def _write_generation_manifest(path, *, seed, run_manifest):
    from src.generation.pipeline import (
        GenerationConfig,
        _contract_artifact_paths,
        begin_generation_attempt,
        complete_generation_attempt,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    prompts = path.parent / "prompts.jsonl"
    prompts.write_text('{"prompt":"Render ТЕСТ","target_text":"ТЕСТ"}\n', encoding="utf-8")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=path.parent / "generated",
        model_revision="a3b4f4849157f664bdbc776fd7453c2783562f4d",
        versions_per_prompt=1,
        seed=seed,
        end_idx=1,
        manifest_path=path,
        run_manifest_path=str(run_manifest),
    )
    records = load_prompt_records(prompts)
    ensure_generation_resume_contract(config, resolve_generation_paths(config.output_dir), records)
    begin_generation_attempt(path, run_manifest_path=str(run_manifest))
    contract = json.loads(path.read_text(encoding="utf-8"))["contract"]
    for artifact_paths in _contract_artifact_paths(contract).values():
        for artifact_path in artifact_paths:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"fixture")
    complete_generation_attempt(
        path,
        generated={"text_embeddings": 1, "images": 1, "latents": 1},
        skipped={"text_embeddings": 0, "images": 0, "latents": 0},
    )
    return path


def _write_scores(path, sample_ids, *, seed, common_manifest, formula=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in sample_ids:
        row = {field: "" for field in CANONICAL_SCORE_COLUMNS}
        row.update(
            {
                "id": sample_id,
                "sample_id": sample_id,
                "version": 0,
                "score": 0.8,
                "product_score": 0.8,
                "target_text": "ТЕСТ",
                "detection_status": "detected_exact",
                "exact_text_match": "true",
                "formula_complete": "true",
            }
        )
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    generation_manifest = _write_generation_manifest(
        path.parent / "generation.json",
        seed=seed,
        run_manifest=common_manifest,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    write_score_schema_sidecar(
        path,
        formula=formula or thesis_product_formula(),
        source_manifest_paths=(str(common_manifest), str(generation_manifest)),
        primary_score="product",
        execution_metadata={
            "status": "complete",
            "scored_row_count": len(rows),
            "scores_sha256": digest,
            "shard_idx": 0,
            "num_shards": 1,
        },
    )
    return path


def test_aggregate_score_files_adds_seed_hashes_and_canonical_sidecar(tmp_path):
    from src.evaluation.score_aggregation import aggregate_score_files

    common = _write_manifest(tmp_path / "training.json", "training")
    seed_2 = _write_scores(
        tmp_path / "seed-2" / "scores.csv", ["a", "b"], seed=2, common_manifest=common
    )
    seed_1 = _write_scores(
        tmp_path / "seed-1" / "scores.csv", ["a", "b"], seed=1, common_manifest=common
    )
    output = tmp_path / "aggregate.csv"

    metadata = aggregate_score_files([(2, seed_2), (1, seed_1)], output_path=output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["seed"], row["sample_id"]) for row in rows] == [
        ("1", "a"),
        ("1", "b"),
        ("2", "a"),
        ("2", "b"),
    ]
    assert metadata["score_file_schema_version"] == "heldout-score-aggregate/v1"
    assert metadata["execution"]["row_count"] == 4
    assert metadata["execution"]["seeds"] == [1, 2]
    assert metadata["execution"]["sample_count_per_seed"] == 2
    sidecar = json.loads(output.with_suffix(".schema.json").read_text(encoding="utf-8"))
    assert len(sidecar["execution"]["scores_sha256"]) == 64
    assert sidecar["source_manifest_paths"] == [str(common)]


def test_aggregate_rejects_coverage_and_formula_drift(tmp_path):
    from src.evaluation.reward_interface import ProductScoreFormula
    from src.evaluation.score_aggregation import ScoreAggregationError, aggregate_score_files

    common = _write_manifest(tmp_path / "training.json", "training")
    seed_1 = _write_scores(
        tmp_path / "seed-1" / "scores.csv", ["a", "b"], seed=1, common_manifest=common
    )
    missing = _write_scores(
        tmp_path / "seed-2" / "scores.csv", ["a"], seed=2, common_manifest=common
    )
    with pytest.raises(ScoreAggregationError, match="sample coverage mismatch"):
        aggregate_score_files([(1, seed_1), (2, missing)], output_path=tmp_path / "coverage.csv")

    diagnostic = ProductScoreFormula()
    drift = _write_scores(
        tmp_path / "seed-3" / "scores.csv",
        ["a", "b"],
        seed=3,
        common_manifest=common,
        formula=diagnostic,
    )
    with pytest.raises(ScoreAggregationError, match="formula/schema mismatch"):
        aggregate_score_files([(1, seed_1), (3, drift)], output_path=tmp_path / "formula.csv")


def test_aggregate_cli_rejects_bad_seed_syntax(tmp_path, capsys):
    from scripts.aggregate_heldout_scores import main

    result = main(["--input", "broken", "--output", str(tmp_path / "out.csv")])

    assert result == 2
    assert "SEED=PATH" in capsys.readouterr().err


def test_aggregate_verifies_source_manifests_against_real_files(tmp_path):
    from src.evaluation.score_aggregation import ScoreAggregationError, aggregate_score_files

    common = _write_manifest(tmp_path / "training.json", "training")
    scores = _write_scores(
        tmp_path / "seed-1" / "scores.csv", ["a"], seed=1, common_manifest=common
    )
    common.write_text(json.dumps({"name": "tampered"}), encoding="utf-8")

    with pytest.raises(ScoreAggregationError, match="invalid source manifest"):
        aggregate_score_files([(1, scores)], output_path=tmp_path / "aggregate.csv")

    common.unlink()
    with pytest.raises(
        ScoreAggregationError,
        match="source manifest does not exist|invalid source manifest",
    ):
        aggregate_score_files([(1, scores)], output_path=tmp_path / "missing.csv")


def test_aggregate_rejects_schema_less_json_even_when_its_hash_matches(tmp_path):
    from src.evaluation.score_aggregation import ScoreAggregationError, aggregate_score_files

    common = _write_manifest(tmp_path / "training.json", "training")
    scores = _write_scores(
        tmp_path / "seed-1" / "scores.csv", ["a"], seed=1, common_manifest=common
    )
    arbitrary = tmp_path / "arbitrary.json"
    arbitrary.write_text('{"name":"not-a-manifest"}\n', encoding="utf-8")
    sidecar_path = scores.with_suffix(".schema.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["source_manifest_paths"] = [str(arbitrary)]
    sidecar["source_manifest_sha256"] = {
        str(arbitrary): hashlib.sha256(arbitrary.read_bytes()).hexdigest()
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ScoreAggregationError, match="invalid source manifest"):
        aggregate_score_files([(1, scores)], output_path=tmp_path / "aggregate.csv")
