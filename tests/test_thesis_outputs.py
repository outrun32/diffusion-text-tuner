from __future__ import annotations

import hashlib
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
    scores = _write_canonical_scores(
        tmp_path / "reports" / "scores.jsonl",
        [
            {
                "sample_id": "a",
                "run_id": "run-sft",
                "product_score": 0.72,
                "score_vlm": 0.8,
                "score_ocr": 0.7,
                "exact_text_match": True,
            },
            {
                "sample_id": "b",
                "run_id": "run-dpo",
                "product_score": 0.66,
                "score_vlm": 0.6,
                "score_ocr": 0.8,
                "exact_text_match": False,
            },
        ],
        source_manifests=[manifest_a, manifest_b],
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
                "columns": ["sample_id", "run_id", "product_score", "exact_text_match"],
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
    assert len(bundle["source_manifests"][0]["sha256"]) == 64
    assert len(bundle["source_manifests"][0]["config_snapshot_sha256"]) == 64
    assert bundle["evidence"]["score_reports"][0]["path"] == str(scores)
    assert len(bundle["evidence"]["score_reports"][0]["sha256"]) == 64
    assert len(bundle["evidence"]["score_reports"][0]["sidecar_sha256"]) == 64
    assert bundle["evidence"]["score_reports"][0]["source_manifests"] == [
        {"path": str(manifest_a), "sha256": _sha256(manifest_a)},
        {"path": str(manifest_b), "sha256": _sha256(manifest_b)},
    ]
    assert bundle["evidence"]["diagnostic_reports"][0]["schema_version"] == "reward-diagnostics/v1"
    assert bundle["readiness"]["ready"] is True
    assert bundle["readiness"]["blocking_errors"] == []
    assert bundle["readiness"]["warnings"] == []

    table = bundle["tables"][0]
    assert table["row_count"] == 2
    assert table["source_paths"] == [str(scores)]
    table_path = Path(table["path"])
    assert table_path.read_text(encoding="utf-8").splitlines() == [
        "sample_id,run_id,product_score,exact_text_match",
        "a,run-sft,0.72,True",
        "b,run-dpo,0.66,False",
    ]

    plot = bundle["svg_plots"][0]
    assert plot["source_paths"] == [str(scores)]
    svg_text = Path(plot["path"]).read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Product scores" in svg_text
    assert 'data-sample-id="a"' in svg_text

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
    assert "must be a canonical .csv or .jsonl artifact" in joined_errors
    assert "table missing_table_source" in joined_errors
    assert bundle["tables"] == []
    assert bundle["svg_plots"] == []
    assert bundle["contact_sheets"] == []
    with pytest.raises(ThesisOutputError, match="not thesis-ready"):
        build_thesis_output_bundle(config, require_ready=True)


def test_score_hash_and_manifest_link_must_match_before_thesis_readiness(
    tmp_path: Path,
) -> None:
    from src.evaluation.thesis_outputs import build_thesis_output_bundle

    configured_manifest = _write_manifest(
        tmp_path / "runs" / "configured" / "manifest.json", run_id="configured"
    )
    unrelated_manifest = _write_manifest(
        tmp_path / "runs" / "unrelated" / "manifest.json", run_id="unrelated"
    )
    tampered_scores = _write_canonical_scores(
        tmp_path / "reports" / "tampered.jsonl",
        [{"sample_id": "a", "product_score": 0.8}],
        source_manifests=[configured_manifest],
    )
    tampered_scores.write_text(tampered_scores.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    config = {
        "schema_version": "thesis-output-config/v1",
        "source_manifests": [str(configured_manifest)],
        "score_reports": [str(tampered_scores)],
        "diagnostic_reports": [],
        "output_dir": str(tmp_path / "tampered-bundle"),
        "table_specs": [],
        "svg_plot_specs": [],
        "contact_sheet_specs": [],
    }

    tampered_bundle = build_thesis_output_bundle(config, require_ready=False)

    assert tampered_bundle["readiness"]["ready"] is False
    assert "scores_sha256 does not match score file" in "\n".join(
        tampered_bundle["readiness"]["blocking_errors"]
    )

    unlinked_scores = _write_canonical_scores(
        tmp_path / "reports" / "unlinked.jsonl",
        [{"sample_id": "b", "product_score": 0.7}],
        source_manifests=[unrelated_manifest],
    )
    config["score_reports"] = [str(unlinked_scores)]
    config["output_dir"] = str(tmp_path / "unlinked-bundle")

    unlinked_bundle = build_thesis_output_bundle(config, require_ready=False)
    joined_errors = "\n".join(unlinked_bundle["readiness"]["blocking_errors"])
    assert unlinked_bundle["readiness"]["ready"] is False
    assert "does not link any configured source_manifests" in joined_errors
    assert "not linked by any score sidecar" in joined_errors


def test_score_sidecar_may_link_hashed_auxiliary_generation_manifest(tmp_path: Path) -> None:
    from src.evaluation.thesis_outputs import build_thesis_output_bundle

    run_manifest = _write_manifest(tmp_path / "runs" / "trained" / "manifest.json")
    generation_manifest = _write_generation_manifest(
        tmp_path / "runs" / "eval" / "generation.manifest.json",
        run_manifest=run_manifest,
        seed=101,
    )
    scores = _write_canonical_scores(
        tmp_path / "reports" / "scores.jsonl",
        [{"sample_id": "a", "product_score": 0.8}],
        source_manifests=[run_manifest, generation_manifest],
    )
    config = {
        "schema_version": "thesis-output-config/v1",
        "source_manifests": [str(run_manifest)],
        "score_reports": [str(scores)],
        "diagnostic_reports": [],
        "output_dir": str(tmp_path / "bundle"),
        "table_specs": [],
        "svg_plot_specs": [],
        "contact_sheet_specs": [],
    }

    bundle = build_thesis_output_bundle(config)

    assert bundle["readiness"]["ready"] is True
    assert bundle["evidence"]["score_reports"][0]["source_manifests"] == [
        {"path": str(run_manifest), "sha256": _sha256(run_manifest)},
        {"path": str(generation_manifest), "sha256": _sha256(generation_manifest)},
    ]


def test_tables_and_plots_must_use_declared_validated_evidence(tmp_path: Path) -> None:
    from src.evaluation.thesis_outputs import build_thesis_output_bundle

    manifest = _write_manifest(tmp_path / "runs" / "valid" / "manifest.json")
    scores = _write_canonical_scores(
        tmp_path / "reports" / "scores.jsonl",
        [{"sample_id": "a", "product_score": 0.8}],
        source_manifests=[manifest],
    )
    unrelated = _write_json(
        tmp_path / "reports" / "unrelated.json",
        {"records": [{"sample_id": "forged", "product_score": 1.0}]},
    )
    config = {
        "schema_version": "thesis-output-config/v1",
        "source_manifests": [str(manifest)],
        "score_reports": [str(scores)],
        "diagnostic_reports": [],
        "output_dir": str(tmp_path / "bundle"),
        "table_specs": [
            {
                "name": "forged_table",
                "source": str(unrelated),
                "columns": ["sample_id", "product_score"],
                "output_csv": "tables/forged.csv",
            }
        ],
        "svg_plot_specs": [
            {
                "name": "forged_plot",
                "source": str(unrelated),
                "x": "sample_id",
                "y": "product_score",
                "output_svg": "plots/forged.svg",
            }
        ],
        "contact_sheet_specs": [],
    }

    bundle = build_thesis_output_bundle(config, require_ready=False)

    errors = "\n".join(bundle["readiness"]["blocking_errors"])
    assert bundle["readiness"]["ready"] is False
    assert errors.count("source is not a validated score or diagnostic report") == 2
    assert bundle["tables"] == []
    assert bundle["svg_plots"] == []
    assert not (tmp_path / "bundle" / "tables" / "forged.csv").exists()


def test_empty_config_and_output_path_escape_are_rejected(tmp_path: Path) -> None:
    from src.evaluation.thesis_outputs import ThesisOutputError, build_thesis_output_bundle

    with pytest.raises(ThesisOutputError, match="schema_version"):
        build_thesis_output_bundle({})

    manifest = _write_manifest(tmp_path / "runs" / "valid" / "manifest.json")
    scores = _write_json(tmp_path / "scores.json", {"records": [{"score": 1.0}]})
    unsafe = {
        "schema_version": "thesis-output-config/v1",
        "source_manifests": [str(manifest)],
        "score_reports": [str(scores)],
        "diagnostic_reports": [],
        "output_dir": str(tmp_path / "bundle"),
        "table_specs": [
            {
                "source": str(scores),
                "columns": ["score"],
                "output_csv": "../escape.csv",
            }
        ],
        "svg_plot_specs": [],
        "contact_sheet_specs": [],
    }

    with pytest.raises(ThesisOutputError, match="stay inside output_dir"):
        build_thesis_output_bundle(unsafe)
    assert not (tmp_path / "escape.csv").exists()


def test_cli_writes_bundle_and_markdown_and_fails_when_not_ready(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "runs" / "cli" / "manifest.json", run_id="cli-run")
    scores = _write_canonical_scores(
        tmp_path / "reports" / "scores.jsonl",
        [{"sample_id": "cli", "run_id": "cli-run", "product_score": 0.9}],
        source_manifests=[manifest],
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
    assert "score_reports must contain" in bad_result.stderr


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
    snapshot = {"model_id": model_id, "seed": 7}
    _write_json(path.parent / "config_snapshot.json", snapshot)
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
            "config_snapshot_sha256": _json_sha256(snapshot),
            "config_snapshot": snapshot,
            "seeds": {"seed": 7},
            "models": {"base": model_id},
            "inputs": {"prompts": "prompts/eval.jsonl"},
            "outputs": outputs or {"scores": "scores.csv"},
            "metrics": metrics or {"product_score_mean": 0.8},
            "notes": [],
            "artifact_schema_versions": {"runtime_artifacts": "runtime-artifacts/v1"},
        },
    )


def _write_canonical_scores(
    path: Path,
    records: list[dict[str, object]],
    *,
    source_manifests: list[Path],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_records = []
    for index, record in enumerate(records):
        product_score = float(record.get("product_score", 0.8))
        exact_match = bool(record.get("exact_text_match", True))
        canonical = {
            "sample_id": str(record.get("sample_id", f"sample-{index}")),
            "version": 0,
            "score": product_score,
            "product_score": product_score,
            "target_text": "ТЕКСТ",
            "score_vlm": float(record.get("score_vlm", 0.8)),
            "score_ocr": float(record.get("score_ocr", 0.8)),
            "cer": 0.0 if exact_match else 0.2,
            "entropy": 0.1,
            "ocr_detected": True,
            "detection_status": "detected_exact" if exact_match else "detected_mismatch",
            "exact_text_match": exact_match,
            "char_accuracy": 1.0 if exact_match else 0.8,
            "char_matches": 5 if exact_match else 4,
            "char_total": 5,
            "missing_components": [],
            "formula_complete": True,
            "manifest_path": str(source_manifests[0]),
            "text_metrics": {},
            "scorer_metadata": {},
            "thresholds": {},
            **record,
        }
        canonical_records.append(canonical)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in canonical_records),
        encoding="utf-8",
    )
    source_paths = [str(manifest) for manifest in source_manifests]
    sidecar = {
        "schema_version": "reward-score-metadata/v1",
        "score_file_schema_version": "phase6-score-jsonl/v1",
        "formula": {
            "name": "thesis_vlm_ocr_product_v1",
            "weights": {"score_vlm": 1.0, "score_ocr": 1.0},
            "thresholds": {},
            "scorer_versions": {"vlm": "test@revision", "ocr": "test@revision"},
            "aggregation": "product",
            "require_all": True,
        },
        "primary_score": "product",
        "source_manifest_paths": source_paths,
        "source_manifest_sha256": {
            source_path: _sha256(Path(source_path)) for source_path in source_paths
        },
        "required_phase6_fields": sorted(canonical_records[0].keys()),
        "execution": {
            "status": "complete",
            "scored_row_count": len(canonical_records),
            "scores_sha256": _sha256(path),
        },
    }
    _write_json(path.with_suffix(".schema.json"), sidecar)
    return path


def _write_generation_manifest(path: Path, *, run_manifest: Path, seed: int) -> Path:
    from src.generation.pipeline import (
        GenerationConfig,
        _contract_artifact_paths,
        begin_generation_attempt,
        complete_generation_attempt,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_jsonl_prompt(path.parent / "prompts.jsonl")
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


def _write_jsonl_prompt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"prompt":"Render ТЕКСТ","target_text":"ТЕКСТ"}\n', encoding="utf-8")
    return path


def _json_sha256(payload: dict[str, object]) -> str:
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_image(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)
    return path
