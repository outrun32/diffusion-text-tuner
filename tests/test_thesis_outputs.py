from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


def test_builds_traceable_tables_svgs_and_contact_sheets(tmp_path: Path) -> None:
    from src.evaluation.thesis_outputs import (
        build_thesis_output_bundle,
        format_thesis_output_markdown,
    )

    manifest_a = _write_manifest(
        tmp_path / "runs" / "sft" / "manifest.json",
        run_id="run-sft",
        git_commit="abc123",
        model_id="flux-schnell",
        outputs={"score_csv": "scores/sft.csv", "samples_dir": "outputs/sft"},
        metrics={"final_loss": 0.42},
    )
    manifest_b = _write_manifest(
        tmp_path / "runs" / "dpo" / "manifest.json",
        run_id="run-dpo",
        git_commit="def456",
        model_id="flux-dev",
        outputs={"score_csv": "scores/dpo.csv", "samples_dir": "outputs/dpo"},
        metrics={"accuracy": 0.61},
    )
    scores = _write_json(
        tmp_path / "reports" / "scores.json",
        {
            "schema_version": "phase6-score-report/v1",
            "records": [
                {
                    "sample_id": "a",
                    "run_id": "run-sft",
                    "product_score": 0.72,
                    "score_vlm": 0.8,
                    "score_ocr": 0.7,
                    "exact_match": True,
                },
                {
                    "sample_id": "b",
                    "run_id": "run-dpo",
                    "product_score": 0.66,
                    "score_vlm": 0.6,
                    "score_ocr": 0.8,
                    "exact_match": False,
                },
            ],
        },
    )
    diagnostics = _write_json(
        tmp_path / "reports" / "diagnostics.json",
        {
            "schema_version": "reward-diagnostics/v1",
            "record_counts": {"total": 2, "missing_evidence": 0},
            "vlm_ocr_correlation": {"n": 2, "pearson": -1.0},
            "false_positives": [],
            "false_negatives": [{"sample_id": "b"}],
        },
    )
    image_a = _write_image(tmp_path / "images" / "a.png", "red")
    image_b = _write_image(tmp_path / "images" / "b.png", "blue")
    config = {
        "schema_version": "thesis-output-config/v1",
        "source_manifests": [str(manifest_a), str(manifest_b)],
        "score_reports": [str(scores)],
        "diagnostic_reports": [str(diagnostics)],
        "output_dir": str(tmp_path / "bundle"),
        "table_specs": [
            {
                "name": "score_summary",
                "source": str(scores),
                "columns": ["sample_id", "run_id", "product_score", "exact_match"],
                "output_csv": "tables/score_summary.csv",
                "markdown_title": "Score summary",
            }
        ],
        "svg_plot_specs": [
            {
                "name": "product_scores",
                "source": str(scores),
                "x": "sample_id",
                "y": "product_score",
                "output_svg": "plots/product_scores.svg",
                "title": "Product scores",
            }
        ],
        "contact_sheet_specs": [
            {
                "name": "examples",
                "output_image": "contact_sheets/examples.png",
                "limit": 2,
                "images": [
                    {"sample_id": "a", "path": str(image_a), "caption": "SFT"},
                    {"sample_id": "b", "path": str(image_b), "caption": "DPO"},
                ],
            }
        ],
    }

    bundle = build_thesis_output_bundle(config)
    markdown = format_thesis_output_markdown(bundle)

    assert bundle["schema_version"] == "thesis-output-bundle/v1"
    assert [source["run_id"] for source in bundle["source_manifests"]] == ["run-sft", "run-dpo"]
    assert bundle["source_manifests"][0]["git"]["commit"] == "abc123"
    assert bundle["source_manifests"][1]["config_snapshot"]["model_id"] == "flux-dev"
    assert bundle["evidence"]["score_reports"][0]["path"] == str(scores)
    assert bundle["evidence"]["diagnostic_reports"][0]["schema_version"] == "reward-diagnostics/v1"
    assert bundle["readiness"]["ready"] is True
    assert bundle["readiness"]["blocking_errors"] == []
    assert bundle["readiness"]["warnings"] == []

    table = bundle["tables"][0]
    assert table["row_count"] == 2
    assert table["source_paths"] == [str(scores)]
    table_path = Path(table["path"])
    assert table_path.read_text(encoding="utf-8").splitlines() == [
        "sample_id,run_id,product_score,exact_match",
        "a,run-sft,0.72,True",
        "b,run-dpo,0.66,False",
    ]

    plot = bundle["svg_plots"][0]
    assert plot["source_paths"] == [str(scores)]
    svg_text = Path(plot["path"]).read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Product scores" in svg_text
    assert "data-sample-id=\"a\"" in svg_text

    sheet = bundle["contact_sheets"][0]
    assert sheet["entry_count"] == 2
    assert sheet["source_paths"] == [str(image_a), str(image_b)]
    with Image.open(sheet["path"]) as rendered_sheet:
        assert rendered_sheet.size == (192, 96)

    assert "| run-sft | evaluation | abc123 |" in markdown
    assert "Evidence warnings" in markdown
    assert "No warnings." in markdown
    assert "tables/score_summary.csv" in markdown
    assert "plots/product_scores.svg" in markdown
    assert "contact_sheets/examples.png" in markdown


def test_missing_or_malformed_provenance_blocks_thesis_readiness(tmp_path: Path) -> None:
    from src.evaluation.thesis_outputs import ThesisOutputError, build_thesis_output_bundle

    valid_manifest = _write_manifest(tmp_path / "runs" / "valid" / "manifest.json")
    malformed_report = _write_json(tmp_path / "reports" / "bad.json", {"records": "not-a-list"})
    config = {
        "schema_version": "thesis-output-config/v1",
        "source_manifests": [str(valid_manifest), str(tmp_path / "runs" / "missing.json")],
        "score_reports": [str(malformed_report), str(tmp_path / "reports" / "missing_scores.json")],
        "diagnostic_reports": [str(tmp_path / "reports" / "missing_diagnostics.json")],
        "output_dir": str(tmp_path / "bundle"),
        "table_specs": [
            {
                "name": "missing_table_source",
                "source": str(tmp_path / "reports" / "missing_scores.json"),
                "columns": ["sample_id", "product_score"],
                "output_csv": "tables/missing.csv",
            }
        ],
        "svg_plot_specs": [],
        "contact_sheet_specs": [],
    }

    bundle = build_thesis_output_bundle(config, require_ready=False)

    assert bundle["readiness"]["ready"] is False
    joined_errors = "\n".join(bundle["readiness"]["blocking_errors"])
    assert "missing manifest" in joined_errors
    assert "missing score report" in joined_errors
    assert "missing diagnostic report" in joined_errors
    assert "records must be a list" in joined_errors
    assert "table missing_table_source" in joined_errors
    assert bundle["tables"] == []
    assert bundle["svg_plots"] == []
    assert bundle["contact_sheets"] == []
    with pytest.raises(ThesisOutputError, match="not thesis-ready"):
        build_thesis_output_bundle(config, require_ready=True)


def test_cli_writes_bundle_and_markdown_and_fails_when_not_ready(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "runs" / "cli" / "manifest.json", run_id="cli-run")
    scores = _write_json(
        tmp_path / "reports" / "scores.json",
        {"records": [{"sample_id": "cli", "run_id": "cli-run", "product_score": 0.9}]},
    )
    config_path = _write_json(
        tmp_path / "thesis_config.json",
        {
            "schema_version": "thesis-output-config/v1",
            "source_manifests": [str(manifest)],
            "score_reports": [str(scores)],
            "diagnostic_reports": [],
            "output_dir": str(tmp_path / "bundle"),
            "table_specs": [
                {
                    "name": "cli_table",
                    "source": str(scores),
                    "columns": ["sample_id", "product_score"],
                    "output_csv": "tables/cli.csv",
                }
            ],
            "svg_plot_specs": [],
            "contact_sheet_specs": [],
        },
    )
    bundle_path = tmp_path / "bundle.json"
    markdown_path = tmp_path / "bundle.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_thesis_outputs.py",
            "--config",
            str(config_path),
            "--output-bundle",
            str(bundle_path),
            "--markdown-summary",
            str(markdown_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(bundle_path.read_text(encoding="utf-8"))["readiness"]["ready"] is True
    assert "cli-run" in markdown_path.read_text(encoding="utf-8")

    bad_config_path = _write_json(
        tmp_path / "bad_config.json",
        {
            "schema_version": "thesis-output-config/v1",
            "source_manifests": [str(tmp_path / "missing_manifest.json")],
            "score_reports": [],
            "diagnostic_reports": [],
            "output_dir": str(tmp_path / "bad-bundle"),
            "table_specs": [],
            "svg_plot_specs": [],
            "contact_sheet_specs": [],
        },
    )
    bad_result = subprocess.run(
        [
            sys.executable,
            "scripts/build_thesis_outputs.py",
            "--config",
            str(bad_config_path),
            "--output-bundle",
            str(tmp_path / "bad_bundle.json"),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert bad_result.returncode == 2
    assert "not thesis-ready" in bad_result.stderr


def test_thesis_output_docs_describe_evidence_workflow() -> None:
    docs_path = Path("docs/thesis_outputs.md")
    assert docs_path.exists()
    docs = docs_path.read_text(encoding="utf-8")
    required_terms = [
        "thesis-output-config/v1",
        "source_manifests",
        "score_reports",
        "diagnostic_reports",
        "table_specs",
        "svg_plot_specs",
        "contact_sheet_specs",
        "readiness blocking errors",
        "generated tables, SVG plots, contact sheets, bundle JSON, and Markdown",
        "runtime artifacts",
        "exact run manifests",
    ]
    for term in required_terms:
        assert term in docs


def _write_manifest(
    path: Path,
    *,
    run_id: str = "run-001",
    git_commit: str = "abc123",
    model_id: str = "flux-test",
    outputs: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "run-manifest/v1",
            "run_id": run_id,
            "stage": "evaluation",
            "created_at": "2026-05-06T00:00:00Z",
            "command": ["python", "evaluate.py"],
            "git": {"commit": git_commit, "dirty": False},
            "environment": {"python": "3.11"},
            "config_snapshot_path": "config_snapshot.json",
            "config_snapshot": {"model_id": model_id, "seed": 7},
            "seeds": {"seed": 7},
            "models": {"base": model_id},
            "inputs": {"prompts": "prompts/eval.jsonl"},
            "outputs": outputs or {"scores": "scores.csv"},
            "metrics": metrics or {"product_score_mean": 0.8},
            "notes": [],
            "artifact_schema_versions": {"runtime_artifacts": "runtime-artifacts/v1"},
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_image(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)
    return path
