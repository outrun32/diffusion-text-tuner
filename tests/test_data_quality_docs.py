from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runtime.artifacts import ArtifactValidationError, validate_artifacts
from src.runtime.paths import assert_artifact_git_safety, resolve_stage_paths


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_phase3_runtime_paths_expose_data_quality_artifacts(tmp_path: Path) -> None:
    prompt_paths = resolve_stage_paths("prompt_generation", root=tmp_path).paths
    synthetic_paths = resolve_stage_paths("synthetic", root=tmp_path).paths
    selection_paths = resolve_stage_paths("data_selection", root=tmp_path).paths
    comparison_paths = resolve_stage_paths("data_comparison", root=tmp_path).paths

    assert prompt_paths["prompt_quality_report"] == (
        tmp_path / "runs" / "prompt-quality" / "prompt-quality.json"
    )
    assert prompt_paths["dataset_manifest"] == (
        tmp_path / "runs" / "prompt-quality" / "dataset-manifest.json"
    )
    assert synthetic_paths["synthetic_quality_report"] == (
        tmp_path / "runs" / "synthetic-quality" / "synthetic-quality.json"
    )
    assert synthetic_paths["contact_sheet"] == (
        tmp_path / "runs" / "synthetic-quality" / "contact-sheet.png"
    )
    assert selection_paths["selected_samples"] == (
        tmp_path / "outputs" / "generated" / "selected_samples.jsonl"
    )
    assert selection_paths["preference_pairs"] == (
        tmp_path / "outputs" / "generated" / "preference_pairs.jsonl"
    )
    assert comparison_paths["data_source_comparison"] == (
        tmp_path / "runs" / "comparisons" / "generated-vs-synthetic.json"
    )


def test_phase3_runtime_artifact_validators_accept_json_reports_and_jsonl(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "runs" / "prompt-quality" / "dataset-manifest.json",
        {
            "schema_version": "dataset-manifest/v1",
            "dataset_kind": "prompt",
            "dataset_paths": ["data/prompts/simple.jsonl"],
        },
    )
    prompt_report = _write_json(
        tmp_path / "runs" / "prompt-quality" / "prompt-quality.json",
        {"schema_version": "prompt-quality/v1", "valid_records": 2, "errors": []},
    )
    synthetic_report = _write_json(
        tmp_path / "runs" / "synthetic-quality" / "synthetic-quality.json",
        {"schema_version": "synthetic-quality/v1", "sample_count": 2, "accepted_count": 2},
    )
    comparison_report = _write_json(
        tmp_path / "runs" / "comparisons" / "generated-vs-synthetic.json",
        {
            "schema_version": "data-source-comparison/v1",
            "evidence_available": [],
            "evidence_missing": [],
        },
    )
    selected_samples = _write_jsonl(
        tmp_path / "outputs" / "generated" / "selected_samples.jsonl",
        [
            {
                "schema_version": "selected-samples/v1",
                "sample_id": "sft:p1:v0:score",
                "prompt_id": "p1",
                "version": 0,
                "selected_score": 0.8,
            }
        ],
    )
    preference_pairs = _write_jsonl(
        tmp_path / "outputs" / "generated" / "preference_pairs.jsonl",
        [
            {
                "schema_version": "preference-pairs/v1",
                "pair_id": "dpo:p1:w1:l0:score",
                "prompt_id": "p1",
                "winner_version": 1,
                "loser_version": 0,
                "winner_score": 0.9,
                "loser_score": 0.2,
            }
        ],
    )

    checks = [
        validate_artifacts(
            "dataset_manifest", {"dataset_manifest": manifest, "require_ready": True}
        ),
        validate_artifacts(
            "prompt_quality_report",
            {"prompt_quality_report": prompt_report, "require_ready": True},
        ),
        validate_artifacts(
            "synthetic_quality_report",
            {"synthetic_quality_report": synthetic_report, "require_ready": True},
        ),
        validate_artifacts(
            "data_source_comparison",
            {"data_source_comparison": comparison_report, "require_ready": True},
        ),
        validate_artifacts(
            "selected_samples", {"selected_samples": selected_samples, "require_ready": True}
        ),
        validate_artifacts(
            "preference_pairs", {"preference_pairs": preference_pairs, "require_ready": True}
        ),
    ]

    assert all(report.ok for report in checks)
    assert checks[0].metadata["schema_version"] == "dataset-manifest/v1"
    assert checks[4].metadata["record_count"] == 1
    assert checks[5].metadata["record_count"] == 1


def test_phase3_runtime_artifact_validators_block_missing_or_wrong_schema(tmp_path: Path) -> None:
    wrong_schema = _write_json(
        tmp_path / "runs" / "prompt-quality" / "prompt-quality.json",
        {"schema_version": "wrong/v1"},
    )

    with pytest.raises(ArtifactValidationError, match="prompt-quality/v1"):
        validate_artifacts(
            "prompt_quality_report",
            {"prompt_quality_report": wrong_schema, "require_ready": True},
        )
    with pytest.raises(ArtifactValidationError, match="selected_samples"):
        validate_artifacts(
            "selected_samples",
            {
                "selected_samples": tmp_path
                / "outputs"
                / "generated"
                / "missing.jsonl",
                "require_ready": True,
            },
        )


def test_phase3_generated_reports_and_contact_sheets_are_non_committable() -> None:
    report = assert_artifact_git_safety(
        [
            "runs/prompt-quality/prompt-quality.json",
            "runs/synthetic-quality/contact-sheet.png",
            "outputs/generated/selected_samples.jsonl",
            "outputs/generated/preference_pairs.jsonl",
            "runs/comparisons/generated-vs-synthetic.json",
        ]
    )

    assert not report.ok
    assert len(report.errors) == 5


def _read_repo_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_phase3_command_docs_publish_data_quality_workflows() -> None:
    docs = _read_repo_file("docs/commands.md")

    required_strings = [
        "## Phase 3 data curriculum and quality",
        "configs/prompts/simple.json",
        "configs/prompts/full.json",
        "configs/prompts/curriculum.json",
        "python -m src.prompt_pipeline.generate --config configs/prompts/curriculum.json",
        "uv run python scripts/validate_prompt_dataset.py",
        "uv run python scripts/inspect_synthetic_dataset.py",
        "uv run python scripts/materialize_training_data.py --kind sft",
        "uv run python scripts/materialize_training_data.py --kind dpo",
        "uv run python scripts/compare_data_sources.py",
        "OCR/model-heavy checks are opt-in",
        "SLURM-compatible",
        "generated reports, images, tensors, contact sheets, selections, and comparisons",
    ]

    missing = [value for value in required_strings if value not in docs]
    assert missing == []


def test_makefile_exposes_phase3_cpu_safe_aliases() -> None:
    makefile = _read_repo_file("Makefile")

    required_strings = [
        "phase3-generate-prompts:",
        "phase3-validate-prompts:",
        "phase3-inspect-synthetic:",
        "phase3-materialize-sft:",
        "phase3-materialize-dpo:",
        "phase3-compare-sources:",
        "scripts/validate_prompt_dataset.py",
        "scripts/inspect_synthetic_dataset.py",
        "scripts/materialize_training_data.py --kind sft",
        "scripts/materialize_training_data.py --kind dpo",
        "scripts/compare_data_sources.py",
    ]

    missing = [value for value in required_strings if value not in makefile]
    assert missing == []


def test_readme_links_phase3_data_quality_docs_and_artifact_safety() -> None:
    readme = _read_repo_file("README.md")

    required_strings = [
        "docs/data_curriculum.md",
        "docs/dataset_quality.md",
        "docs/synthetic_quality.md",
        "docs/data_selection.md",
        "docs/data_source_comparison.md",
        "Phase 3 data curriculum and quality",
        "generated reports, images, tensors, contact sheets, selections, and comparisons",
        "remain out of git",
    ]

    missing = [value for value in required_strings if value not in readme]
    assert missing == []
