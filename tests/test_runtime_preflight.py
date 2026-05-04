from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.runtime import config_io
from src.training.config import DPOConfig, MaskedSFTConfig, SFTConfig

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
HEAVY_RUNTIME_MODULES = {
    "diffusers",
    "transformers",
    "paddleocr",
    "vllm",
    "mlx_lm",
}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_scores_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,version,score,target_text\n000001,0,0.8,Ж\n",
        encoding="utf-8",
    )
    path.with_suffix(".schema.json").write_text(
        json.dumps({"schema_version": "scores/v1"}),
        encoding="utf-8",
    )
    return path


def _stub_refl_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep trainer imports focused on load_config without importing reward diagnostics."""

    monkeypatch.setitem(
        sys.modules,
        "src.training.refl_trainer",
        SimpleNamespace(FlowMatchScheduler=object),
    )


def _sft_payload() -> dict:
    return {
        "model_id": MODEL_ID,
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
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "resolution": 512,
        "lora": {"r": 4, "lora_alpha": 8, "target_modules": ["to_q"]},
        "sample_interval": 0,
        "num_inference_steps": 4,
        "log_interval": 1,
        "save_interval": 5,
        "output_dir": "outputs/sft",
        "experiment_name": "unit-test",
        "gradient_checkpointing": False,
        "mixed_precision": "no",
    }


def _dpo_payload() -> dict:
    payload = _sft_payload()
    payload.update(
        {
            "sft_lora_path": None,
            "score_threshold": 0.5,
            "score_diff_min": 0.1,
            "batch_size": 1,
            "beta": 5000.0,
            "output_dir": "outputs/dpo",
            "experiment_name": "dpo-unit-test",
        }
    )
    return payload


def _masked_sft_payload() -> dict:
    return {
        "model_id": MODEL_ID,
        "data_dir": "data/synth_cyrillic/masked_sft",
        "val_n_samples": 0,
        "num_training_steps": 10,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "lr": 0.0001,
        "lr_min": 0.000001,
        "lr_schedule": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 0,
        "seed": 123,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "masked_lambda": 0.65,
        "resolution": 512,
        "lora": {
            "attn_r": 4,
            "attn_alpha": 8,
            "ffn_r": 0,
            "ffn_alpha": 0,
            "joint_attn_r": 4,
            "joint_attn_alpha": 8,
            "dropout": 0.0,
        },
        "sample_interval": 0,
        "num_inference_steps": 4,
        "validation_interval": 5,
        "val_t_anchors": [100],
        "eval_suite_path": None,
        "eval_suite_n_per_step": 1,
        "log_interval": 1,
        "save_interval": 5,
        "output_dir": "outputs/masked_sft",
        "experiment_name": "masked-unit-test",
        "progress_bar_mininterval": 1.0,
        "gradient_checkpointing": False,
        "mixed_precision": "no",
    }


@pytest.mark.parametrize(
    ("module_name", "stage", "expected_config"),
    [
        ("src.training.sft_trainer", "sft", SFTConfig()),
        ("src.training.dpo_trainer", "dpo", DPOConfig()),
        ("src.training.masked_sft_trainer", "masked_sft", MaskedSFTConfig()),
    ],
)
def test_trainer_load_config_delegates_to_shared_runtime_validation(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    stage: str,
    expected_config: SFTConfig | DPOConfig | MaskedSFTConfig,
) -> None:
    _stub_refl_trainer(monkeypatch)
    calls: list[tuple[str, str | Path]] = []

    def fake_load_stage_config(actual_stage: str, path: str | Path):
        calls.append((actual_stage, path))
        return expected_config

    monkeypatch.setattr(config_io, "load_stage_config", fake_load_stage_config)
    module = __import__(module_name, fromlist=["load_config"])

    assert module.load_config("configs/example.json") is expected_config
    assert calls == [(stage, "configs/example.json")]


def test_invalid_trainer_config_fails_before_heavy_runtime_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_refl_trainer(monkeypatch)
    from src.training import sft_trainer

    before = set(sys.modules)
    invalid_config = _write_json(tmp_path / "bad-sft.json", _sft_payload() | {"batch_size": 0})

    with pytest.raises(config_io.RuntimeConfigError):
        sft_trainer.load_config(str(invalid_config))

    imported = set(sys.modules) - before
    assert not (HEAVY_RUNTIME_MODULES & imported)


@pytest.mark.parametrize(
    "stage",
    ["generate", "score", "sft", "dpo", "masked-sft", "synthetic", "evaluation"],
)
def test_preflight_supports_phase_two_stages_with_json_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    stage: str,
) -> None:
    from scripts import preflight_runtime

    args = ["--stage", stage, "--root", str(tmp_path), "--json"]
    if stage in {"sft", "dpo", "masked-sft"}:
        config_payload = {
            "sft": _sft_payload,
            "dpo": _dpo_payload,
            "masked-sft": _masked_sft_payload,
        }[stage]()
        args.extend(["--config", str(_write_json(tmp_path / "configs" / f"{stage}.json", config_payload))])

    exit_code = preflight_runtime.main(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == stage
    assert set(payload) >= {"stage", "config", "artifacts", "manifest", "blocking_errors", "warnings"}
    assert exit_code in {0, 1}


def test_preflight_validates_config_artifacts_and_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import preflight_runtime

    config_path = _write_json(tmp_path / "configs" / "sft.json", _sft_payload())
    _write_scores_csv(tmp_path / "outputs" / "generated" / "scores.csv")
    (tmp_path / "outputs" / "generated" / "latents").mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / "text_embeds").mkdir(parents=True)
    manifest_path = _write_json(
        tmp_path / "runs" / "sft-test" / "manifest.json",
        {
            "schema_version": "run-manifest/v1",
            "run_id": "sft-test",
            "stage": "sft",
            "created_at": "2026-05-04T00:00:00Z",
            "command": ["python", "-m", "src.training.sft_trainer"],
            "git": {},
            "environment": {},
            "config_snapshot_path": "config_snapshot.json",
            "config_snapshot": {},
            "seeds": {},
            "models": {},
            "inputs": {},
            "outputs": {"checkpoints_dir": "outputs/sft/checkpoints"},
            "metrics": {},
            "notes": [],
            "artifact_schema_versions": {"runtime_artifacts": "runtime-artifacts/v1"},
        },
    )

    exit_code = preflight_runtime.main(
        [
            "--stage",
            "sft",
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["config"]["ok"] is True
    assert payload["artifacts"]["ok"] is True
    assert payload["manifest"]["ok"] is True
    assert payload["manifest"]["resume_ready"] is True
    assert payload["blocking_errors"] == []


def test_preflight_returns_nonzero_for_blocking_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import preflight_runtime

    bad_config = _write_json(tmp_path / "configs" / "sft.json", _sft_payload() | {"lr": -1})

    exit_code = preflight_runtime.main(
        ["--stage", "sft", "--root", str(tmp_path), "--config", str(bad_config), "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["config"]["ok"] is False
    assert payload["blocking_errors"]
