"""Coverage checks for held-out score reports."""

from __future__ import annotations

import csv
import hashlib
import json

import pytest


def _write_prompts(path):
    path.write_text(
        "".join(
            json.dumps(
                {"id": prompt_id, "prompt": f"Render {text}", "target_text": text},
                ensure_ascii=False,
            )
            + "\n"
            for prompt_id, text in [("p0", "ДОМ"), ("p1", "ЛЕС")]
        ),
        encoding="utf-8",
    )
    return path


def _write_scores(path, rows):
    from src.evaluation.reward_interface import ProductScoreFormula
    from src.runtime.manifests import create_run_manifest
    from src.scoring.pipeline import write_score_schema_sidecar

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "version", "target_text", "ocr_detected", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)
    source_manifest = create_run_manifest(
        stage="evaluation",
        command=["pytest", "benchmark-coverage-fixture"],
        run_root=path.parent / "runs",
        slug=f"{path.stem}-coverage-source",
        outputs={"scores_csv": str(path)},
        root=path.parent,
    )
    write_score_schema_sidecar(
        path,
        formula=ProductScoreFormula(
            name="ocr_score_v1",
            weights={"score_ocr": 1.0},
            scorer_versions={"ocr": "fixture-ocr@1"},
            aggregation="weighted_product",
            require_all=True,
        ),
        source_manifest_paths=(str(source_manifest.manifest_path),),
        primary_score="ocr",
        execution_metadata={
            "status": "complete",
            "scored_row_count": len(rows),
            "scores_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return path


def test_benchmark_report_requires_complete_unique_coverage(tmp_path):
    from src.evaluation.final_benchmark import ScoreSpec, benchmark_score_report

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    complete = _write_scores(
        tmp_path / "complete.csv",
        [
            {"id": "000000", "version": 0, "target_text": "ДОМ", "ocr_detected": "ДОМ", "score": 1},
            {"id": "000001", "version": 0, "target_text": "ЛЕС", "ocr_detected": "ЛЕС", "score": 1},
        ],
    )

    report = benchmark_score_report(
        prompts_jsonl=prompts,
        score_specs=[ScoreSpec(name="base", path=complete)],
    )

    assert report["runs"][0]["coverage"]["expected_prompt_count"] == 2
    assert report["runs"][0]["coverage"]["groups"]["single"]["row_count"] == 2


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_benchmark_report_rejects_missing_or_duplicate_rows(tmp_path, case):
    from src.evaluation.final_benchmark import ScoreSpec, benchmark_score_report

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    rows = [{"id": "000000", "version": 0, "target_text": "ДОМ", "ocr_detected": "ДОМ", "score": 1}]
    if case == "duplicate":
        rows.append(dict(rows[0]))
    scores = _write_scores(tmp_path / f"{case}.csv", rows)

    with pytest.raises(ValueError, match="invalid benchmark coverage"):
        benchmark_score_report(
            prompts_jsonl=prompts,
            score_specs=[ScoreSpec(name=case, path=scores)],
        )
