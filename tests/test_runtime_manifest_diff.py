from __future__ import annotations

import json
from pathlib import Path

from src.runtime.manifest_diff import compare_run_manifests, format_manifest_diff_markdown
from scripts import compare_run_manifests as compare_cli


def test_compare_run_manifests_categorizes_changes_and_redacts_sensitive_metadata(tmp_path):
    left = _write_manifest(
        tmp_path / "runs" / "left" / "manifest.json",
        run_id="run-left",
        config_snapshot={
            "stage": "sft",
            "score_threshold": 0.3,
            "pair_construction_mode": "best_vs_worst",
            "seed": 11,
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "scores_csv": "outputs/generated/scores-vlm.csv",
            "reward_model": "vlm-v1",
            "output_dir": "outputs/sft-left",
        },
        outputs={"samples_dir": "outputs/sft-left/samples"},
        metrics={"loss": 0.7},
        environment={
            "env_presence": {"HF_TOKEN": True},
            "cache": {"HF_HOME": {"present": True, "path": "/private/hf-cache"}},
        },
    )
    right = _write_manifest(
        tmp_path / "runs" / "right" / "manifest.json",
        run_id="run-right",
        config_snapshot={
            "stage": "dpo",
            "score_threshold": 0.5,
            "pair_construction_mode": "margin_weighted",
            "seed": 22,
            "num_inference_steps": 28,
            "guidance_scale": 4.0,
            "scores_csv": "outputs/generated/scores-ocr.csv",
            "reward_model": "ocr-v2",
            "output_dir": "outputs/dpo-right",
        },
        outputs={"samples_dir": "outputs/dpo-right/samples"},
        metrics={"loss": 0.4, "accuracy": 0.8},
        environment={
            "env_presence": {"HF_TOKEN": False},
            "cache": {"HF_HOME": {"present": False, "path": "/private/other-cache"}},
        },
    )

    diff = compare_run_manifests(left, right)

    assert diff["schema_version"] == "run-manifest-diff/v1"
    assert diff["left_run_id"] == "run-left"
    assert diff["right_run_id"] == "run-right"
    assert diff["config_changes"] == {
        "output_dir": {"left": "outputs/sft-left", "right": "outputs/dpo-right"},
        "pair_construction_mode": {"left": "best_vs_worst", "right": "margin_weighted"},
        "score_threshold": {"left": 0.3, "right": 0.5},
        "stage": {"left": "sft", "right": "dpo"},
    }
    assert diff["data_source_changes"] == {
        "scores_csv": {
            "left": "outputs/generated/scores-vlm.csv",
            "right": "outputs/generated/scores-ocr.csv",
        }
    }
    assert diff["reward_changes"] == {
        "reward_model": {"left": "vlm-v1", "right": "ocr-v2"}
    }
    assert diff["seed_changes"] == {"seed": {"left": 11, "right": 22}}
    assert diff["inference_changes"] == {
        "guidance_scale": {"left": 3.5, "right": 4.0},
        "num_inference_steps": {"left": 20, "right": 28},
    }
    assert diff["metric_changes"] == {
        "accuracy": {"left": None, "right": 0.8},
        "loss": {"left": 0.7, "right": 0.4},
    }
    assert diff["artifact_changes"] == {
        "samples_dir": {
            "left": "outputs/sft-left/samples",
            "right": "outputs/dpo-right/samples",
        }
    }

    serialized = json.dumps(diff, sort_keys=True)
    assert "/private/hf-cache" not in serialized
    assert "/private/other-cache" not in serialized
    assert diff["environment_changes"] == {
        "cache.HF_HOME.present": {"left": True, "right": False},
        "env_presence.HF_TOKEN": {"left": True, "right": False},
    }


def test_format_manifest_diff_markdown_renders_expected_headings(tmp_path):
    left = _write_manifest(
        tmp_path / "runs" / "left" / "manifest.json",
        run_id="run-left",
        config_snapshot={"stage": "sft", "seed": 1, "scores_csv": "scores-a.csv"},
        outputs={"samples_dir": "outputs/a"},
        metrics={"loss": 1.0},
    )
    right = _write_manifest(
        tmp_path / "runs" / "right" / "manifest.json",
        run_id="run-right",
        config_snapshot={
            "stage": "dpo",
            "seed": 2,
            "scores_csv": "scores-b.csv",
            "reward_model": "vlm",
            "num_inference_steps": 12,
        },
        outputs={"samples_dir": "outputs/b"},
        metrics={"loss": 0.5},
    )

    markdown = format_manifest_diff_markdown(compare_run_manifests(left, right))

    assert "# Run manifest diff" in markdown
    assert "Left run: run-left" in markdown
    assert "Right run: run-right" in markdown
    assert "## Config changes" in markdown
    assert "## Data source changes" in markdown
    assert "## Reward changes" in markdown
    assert "## Seed changes" in markdown
    assert "## Inference changes" in markdown
    assert "## Metric changes" in markdown
    assert "## Artifact changes" in markdown
    assert "| Key | Left | Right |" in markdown


def test_compare_run_manifests_cli_prints_json_and_writes_markdown(tmp_path, capsys):
    left = _write_manifest(
        tmp_path / "runs" / "left" / "manifest.json",
        run_id="run-left",
        config_snapshot={"stage": "sft", "seed": 1},
    )
    right = _write_manifest(
        tmp_path / "runs" / "right" / "manifest.json",
        run_id="run-right",
        config_snapshot={"stage": "dpo", "seed": 2},
    )

    assert compare_cli.main(["--left", str(left), "--right", str(right)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["left_run_id"] == "run-left"
    assert payload["right_run_id"] == "run-right"
    assert payload["seed_changes"] == {"seed": {"left": 1, "right": 2}}

    output = tmp_path / "run-diff.md"
    assert (
        compare_cli.main(
            ["--left", str(left), "--right", str(right), "--markdown", "--output", str(output)]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    assert output.read_text(encoding="utf-8").startswith("# Run manifest diff")


def test_compare_run_manifests_cli_reports_malformed_input(tmp_path, capsys):
    missing = tmp_path / "runs" / "missing" / "manifest.json"

    assert compare_cli.main(["--left", str(missing), "--right", str(missing)]) == 2
    assert "could not read manifest" in capsys.readouterr().err


def _write_manifest(
    path: Path,
    *,
    run_id: str,
    config_snapshot: dict[str, object],
    outputs: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    environment: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "run-manifest/v1",
        "run_id": run_id,
        "stage": str(config_snapshot.get("stage", "sft")),
        "created_at": "2026-05-05T00:00:00Z",
        "command": ["python", "train.py"],
        "git": {"commit": "abc1234"},
        "environment": environment or {},
        "config_snapshot_path": "config_snapshot.json",
        "config_snapshot": config_snapshot,
        "seeds": {},
        "models": {},
        "inputs": {},
        "outputs": outputs or {},
        "metrics": metrics or {},
        "notes": [],
        "artifact_schema_versions": {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path
