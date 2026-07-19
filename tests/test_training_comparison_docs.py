from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import compare_training_runs


def test_compare_training_runs_cli_writes_integrated_json_and_blocks_on_mismatch(tmp_path, capsys):
    left = _write_manifest(
        tmp_path / "left" / "manifest.json",
        run_id="baseline-run",
        config_snapshot=_base_config(seed=1),
    )
    right = _write_manifest(
        tmp_path / "right" / "manifest.json",
        run_id="sft-run",
        config_snapshot=_base_config(seed=2),
    )
    output = tmp_path / "comparison.json"

    exit_code = compare_training_runs.main(
        [
            "--left-manifest",
            str(left),
            "--right-manifest",
            str(right),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert capsys.readouterr().out == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "training-run-comparison/v1"
    assert payload["manifest_diff"]["left_run_id"] == "baseline-run"
    assert payload["manifest_diff"]["right_run_id"] == "sft-run"
    assert payload["comparability"]["left_label"] == "baseline-run"
    assert payload["comparability"]["right_label"] == "sft-run"
    assert payload["comparability"]["summary"]["is_comparable"] is False
    assert {item["field"] for item in payload["comparability"]["blocking_mismatches"]} == {"seed"}


def test_compare_training_runs_cli_allows_blocking_and_writes_markdown(tmp_path, capsys):
    left = _write_manifest(
        tmp_path / "left" / "manifest.json",
        run_id="dpo-run",
        config_snapshot=_base_config(num_inference_steps=20),
    )
    right = _write_manifest(
        tmp_path / "right" / "manifest.json",
        run_id="masked-sft-run",
        config_snapshot=_base_config(num_inference_steps=28),
    )
    output = tmp_path / "comparison.md"

    exit_code = compare_training_runs.main(
        [
            "--left-manifest",
            str(left),
            "--right-manifest",
            str(right),
            "--markdown",
            "--output",
            str(output),
            "--allow-blocking",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Training run comparison\n")
    assert "## Manifest diff" in markdown
    assert "## Comparability mismatches" in markdown
    assert "# Run manifest diff" in markdown
    assert "# Training comparability report" in markdown
    assert "| num_inference_steps | inference | 20 | 28 | value_mismatch |" in markdown


def test_comparison_docs_and_readme_publish_exact_command_strings():
    commands = Path("docs/commands.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "## Training comparability" in commands
    assert "compare-training-runs" in commands
    assert "compare-training-runs" not in readme
    assert "compare-training-runs" in makefile
    assert "uv run python -m scripts.compare_training_runs" in commands
    assert "python -m scripts.compare_training_runs" in makefile
    assert (
        "uv run python -m scripts.compare_run_manifests --left runs/<a>/manifest.json "
        "--right runs/<b>/manifest.json"
    ) in commands
    assert (
        "uv run python -m scripts.check_training_comparability "
        "--left-manifest runs/<a>/manifest.json "
        "--right-manifest runs/<b>/manifest.json"
    ) in commands
    assert (
        "uv run python -m scripts.compare_training_runs "
        "--left-manifest runs/<a>/manifest.json "
        "--right-manifest runs/<b>/manifest.json --markdown "
        "--output runs/comparisons/training-run-comparison.md"
    ) in commands
    assert "docs/training_comparability.md" in commands
    assert "docs/commands.md" in readme
    for approach in ("baseline", "SFT", "DPO", "masked-SFT", "combined", "curriculum"):
        assert approach in commands


def test_compare_training_runs_make_alias_dry_run_uses_manifest_variables():
    result = subprocess.run(
        [
            "make",
            "-n",
            "compare-training-runs",
            "LEFT_MANIFEST=runs/a/manifest.json",
            "RIGHT_MANIFEST=runs/b/manifest.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "python -m scripts.compare_training_runs" in result.stdout
    assert "--left-manifest runs/a/manifest.json" in result.stdout
    assert "--right-manifest runs/b/manifest.json" in result.stdout


def _base_config(**overrides):
    config = {
        "model_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "num_training_steps": 1000,
        "num_inference_steps": 28,
        "guidance_scale": 4.0,
        "prompt_embedding_padding": "max_length",
        "seed": 42,
        "sample_prompt": "Render text A",
        "sample_target_text": "ТЕКСТ A",
        "latents_dir": "outputs/generated/latents",
        "text_embeds_dir": "outputs/generated/text_embeds",
        "scores_csv": "outputs/generated/scores.csv",
        "data_dir": "data/synth_cyrillic/masked_sft",
        "score_column": "vlm_score",
        "reward_model": "qwen-vlm-v1",
        "scorer": "vlm",
        "metric_columns": ["vlm_score"],
        "samples_dir": "outputs/sft/samples",
    }
    config.update(overrides)
    return config


def _write_manifest(path: Path, *, run_id: str, config_snapshot: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": "run-manifest/v1",
        "run_id": run_id,
        "stage": str(config_snapshot.get("stage", "sft")),
        "created_at": "2026-05-05T00:00:00Z",
        "command": ["python", "train.py"],
        "git": {"commit": "abc1234"},
        "environment": {},
        "config_snapshot_path": "config_snapshot.json",
        "config_snapshot_sha256": _json_sha256(config_snapshot),
        "config_snapshot": config_snapshot,
        "seeds": {},
        "models": {},
        "inputs": {},
        "outputs": {},
        "metrics": {},
        "notes": [],
        "artifact_schema_versions": {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _json_sha256(payload: dict[str, object]) -> str:
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()
