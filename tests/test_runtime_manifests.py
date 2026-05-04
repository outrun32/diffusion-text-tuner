from __future__ import annotations

import json
from pathlib import Path

from src.runtime import manifests
from src.runtime.reproducibility import collect_environment_summary


def test_create_run_manifest_writes_complete_secret_safe_snapshot(tmp_path, monkeypatch):
    config_path = _write_sft_config(tmp_path / "configs" / "sft.json")
    run_root = tmp_path / "runs"

    monkeypatch.setattr(
        manifests,
        "collect_git_state",
        lambda root: {
            "commit": "abc1234",
            "dirty": True,
            "untracked_count": 2,
            "untracked_paths": ["notes.txt", "outputs/sample.png"],
        },
    )
    monkeypatch.setattr(
        manifests,
        "collect_environment_summary",
        lambda: {
            "python": "3.11.test",
            "platform": "linux-test",
            "packages": {"pytest": "8.test"},
            "cuda": {"available": False},
            "cache": {"HF_HOME": {"present": True}},
            "env_presence": {"HF_TOKEN": True, "AWS_SECRET_ACCESS_KEY": True},
        },
    )

    manifest = manifests.create_run_manifest(
        stage="sft",
        config_path=config_path,
        command=["accelerate", "launch", "scripts/train_sft.py"],
        run_root=run_root,
        slug="unit-test",
        outputs={"samples_dir": "outputs/sft/samples"},
    )

    assert manifest.schema_version == "run-manifest/v1"
    assert manifest.stage == "sft"
    assert manifest.run_id.endswith("-sft-unit-test")
    assert manifest.run_dir == run_root / manifest.run_id
    assert manifest.manifest_path.is_file()
    assert manifest.config_snapshot_path.is_file()

    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert payload["command"] == ["accelerate", "launch", "scripts/train_sft.py"]
    assert payload["git"] == {
        "commit": "abc1234",
        "dirty": True,
        "untracked_count": 2,
        "untracked_paths": ["notes.txt", "outputs/sample.png"],
    }
    assert payload["environment"]["env_presence"] == {
        "AWS_SECRET_ACCESS_KEY": True,
        "HF_TOKEN": True,
    }
    assert "secret-value" not in json.dumps(payload)
    assert payload["config_snapshot_path"] == "config_snapshot.json"
    assert payload["config_snapshot"]["schema_version"] == "runtime-config/v1"
    assert payload["seeds"] == {"seed": 123}
    assert payload["models"] == {
        "model_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "model_revision": None,
    }
    assert payload["inputs"] == {
        "config_path": str(config_path),
        "latents_dir": "outputs/generated/latents",
        "scores_csv": "outputs/generated/scores.csv",
        "text_embeds_dir": "outputs/generated/text_embeds",
    }
    assert payload["outputs"] == {"samples_dir": "outputs/sft/samples"}
    assert payload["metrics"] == {}
    assert payload["notes"] == []
    assert payload["artifact_schema_versions"] == {"runtime_artifacts": "runtime-artifacts/v1"}


def test_load_update_and_inspect_manifest_preserve_prior_provenance(tmp_path, monkeypatch, capsys):
    config_path = _write_sft_config(tmp_path / "configs" / "sft.json")
    monkeypatch.setattr(manifests, "collect_git_state", lambda root: {"commit": "abc1234"})

    manifest = manifests.create_run_manifest(
        stage="sft",
        config_path=config_path,
        command=["python", "train.py"],
        run_root=tmp_path / "runs",
        metrics={"loss": 1.0},
    )
    original = manifests.load_run_manifest(manifest.manifest_path)

    updated = manifests.update_run_manifest(
        manifest.manifest_path,
        note="first checkpoint finished",
        metrics={"accuracy": 0.75},
    )

    assert updated.command == original.command
    assert updated.git == original.git
    assert updated.config_snapshot == original.config_snapshot
    assert updated.metrics == {"accuracy": 0.75, "loss": 1.0}
    assert len(updated.notes) == 1
    assert updated.notes[0]["text"] == "first checkpoint finished"
    assert "timestamp" in updated.notes[0]

    manifests.print_manifest_summary(updated, file=capsys.disabled() if False else None)
    summary = manifests.format_manifest_summary(updated)
    assert updated.run_id in summary
    assert "Stage: sft" in summary
    assert "Command: python train.py" in summary
    assert "Config snapshot: config_snapshot.json" in summary
    assert "Outputs: none" in summary
    assert "Metrics: accuracy, loss" in summary
    assert "Notes: 1" in summary


def test_environment_summary_records_secret_presence_without_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "another-secret")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")

    summary = collect_environment_summary(package_names=("pytest", "definitely-not-installed"))

    serialized = json.dumps(summary, sort_keys=True)
    assert "secret-value" not in serialized
    assert "another-secret" not in serialized
    assert summary["env_presence"]["HF_TOKEN"] is True
    assert summary["env_presence"]["AWS_SECRET_ACCESS_KEY"] is True
    assert summary["cache"]["HF_HOME"] == {"present": True}
    assert "pytest" in summary["packages"]


def test_run_manifest_cli_init_inspect_note_and_metrics(tmp_path, monkeypatch, capsys):
    from scripts import run_manifest

    config_path = _write_sft_config(tmp_path / "configs" / "sft.json")
    monkeypatch.setattr(manifests, "collect_git_state", lambda root: {"commit": "abc1234"})

    exit_code = run_manifest.main(
        [
            "init",
            "--stage",
            "sft",
            "--config",
            str(config_path),
            "--command",
            "accelerate launch scripts/train_sft.py",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 0
    init_stdout = capsys.readouterr().out.strip()
    run_dir = Path(init_stdout)
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()

    assert run_manifest.main(["note", str(manifest_path), "checkpoint complete"]) == 0
    assert run_manifest.main(["metrics", str(manifest_path), '--json', '{"loss": 0.4}']) == 0
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text('{"accuracy": 0.9}', encoding="utf-8")
    assert run_manifest.main(["metrics", str(manifest_path), "--file", str(metrics_file)]) == 0
    assert run_manifest.main(["inspect", str(manifest_path)]) == 0

    stdout = capsys.readouterr().out
    assert "Run ID:" in stdout
    assert "Stage: sft" in stdout
    assert "Command: accelerate launch scripts/train_sft.py" in stdout
    assert "Config snapshot: config_snapshot.json" in stdout
    assert "Outputs: none" in stdout
    assert "Metrics: accuracy, loss" in stdout
    assert "Notes: 1" in stdout


def test_create_manifest_for_non_training_stage_without_trainer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(manifests, "collect_git_state", lambda root: {"commit": "abc1234"})

    manifest = manifests.create_run_manifest(
        stage="generate",
        config_path=None,
        command=["python", "-m", "scripts.generate_images"],
        run_root=tmp_path / "runs",
        inputs={"prompts": "data/prompts_simple.jsonl"},
        outputs={"output_dir": "outputs/generated"},
    )

    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert manifest.stage == "generate"
    assert payload["config_snapshot"] == {
        "schema_version": "runtime-config/v1",
        "stage": "generate",
    }
    assert payload["inputs"] == {"prompts": "data/prompts_simple.jsonl"}
    assert payload["outputs"] == {"output_dir": "outputs/generated"}
    assert payload["models"] == {}
    assert payload["seeds"] == {}


def test_create_manifest_for_non_training_stage_with_raw_config_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(manifests, "collect_git_state", lambda root: {"commit": "abc1234"})
    config_path = tmp_path / "configs" / "generate.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "model_id": "black-forest-labs/FLUX.2-klein-base-4B",
                "seed": 42,
                "prompts": "data/prompts_simple.jsonl",
            }
        ),
        encoding="utf-8",
    )

    manifest = manifests.create_run_manifest(
        stage="generate",
        config_path=config_path,
        command="python -m scripts.generate_images",
        run_root=tmp_path / "runs",
    )

    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert payload["config_snapshot"]["stage"] == "generate"
    assert payload["config_snapshot"]["raw_config"]["seed"] == 42
    assert payload["inputs"] == {
        "config_path": str(config_path),
        "prompts": "data/prompts_simple.jsonl",
    }
    assert payload["models"] == {
        "model_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "model_revision": None,
    }
    assert payload["seeds"] == {"seed": 42}


def test_run_manifest_cli_accepts_generation_stage_without_config(tmp_path, monkeypatch, capsys):
    from scripts import run_manifest

    monkeypatch.setattr(manifests, "collect_git_state", lambda root: {"commit": "abc1234"})

    exit_code = run_manifest.main(
        [
            "init",
            "--stage",
            "generate",
            "--command",
            "python -m scripts.generate_images",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    assert exit_code == 0
    manifest_path = Path(capsys.readouterr().out.strip()) / "manifest.json"
    assert manifest_path.is_file()
    assert run_manifest.main(["inspect", str(manifest_path)]) == 0
    stdout = capsys.readouterr().out
    assert "Stage: generate" in stdout
    assert "Command: python -m scripts.generate_images" in stdout


def test_run_manifest_cli_reports_missing_or_corrupt_manifests(tmp_path, capsys):
    from scripts import run_manifest

    missing = tmp_path / "runs" / "missing" / "manifest.json"
    assert run_manifest.main(["inspect", str(missing)]) == 2
    assert "could not read manifest" in capsys.readouterr().err

    corrupt = tmp_path / "manifest.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    assert run_manifest.main(["inspect", str(corrupt)]) == 2
    assert "malformed manifest JSON" in capsys.readouterr().err


def test_run_manifest_cli_reports_invalid_metrics_payloads(tmp_path, monkeypatch, capsys):
    from scripts import run_manifest

    config_path = _write_sft_config(tmp_path / "configs" / "sft.json")
    monkeypatch.setattr(manifests, "collect_git_state", lambda root: {"commit": "abc1234"})
    manifest = manifests.create_run_manifest(
        stage="sft",
        config_path=config_path,
        command=["python", "train.py"],
        run_root=tmp_path / "runs",
    ).manifest_path

    assert run_manifest.main(["metrics", str(manifest), "--json", "[]"]) == 2
    assert "metrics payload must be a JSON object" in capsys.readouterr().err

    missing_metrics = tmp_path / "missing-metrics.json"
    assert run_manifest.main(["metrics", str(manifest), "--file", str(missing_metrics)]) == 2
    assert "could not read metrics file" in capsys.readouterr().err



def _write_sft_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "latents_dir": "outputs/generated/latents",
        "text_embeds_dir": "outputs/generated/text_embeds",
        "scores_csv": "outputs/generated/scores.csv",
        "score_threshold": 0.3,
        "num_training_steps": 10,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 0,
        "seed": 123,
        "resume_lora_path": None,
        "resume_step": 0,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "resolution": 512,
        "lora": {"r": 4, "lora_alpha": 8, "target_modules": ["to_q"]},
        "sample_prompt": "render text",
        "sample_target_text": "тест",
        "sample_interval": 0,
        "num_inference_steps": 4,
        "log_interval": 1,
        "save_interval": 5,
        "output_dir": "outputs/sft",
        "experiment_name": "unit-test",
        "gradient_checkpointing": False,
        "mixed_precision": "no",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
