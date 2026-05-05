from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import src.runtime.artifacts as artifacts_module
from src.runtime.artifacts import ArtifactValidationError, validate_artifacts
from src.runtime.config_io import RuntimeConfigError, load_stage_config, resolve_config_snapshot
from src.training.config import DPOConfig, MaskedSFTConfig, SFTConfig

CONFIG_CASES = [
    pytest.param("sft", Path("configs/sft.json"), SFTConfig, id="sft"),
    pytest.param("dpo", Path("configs/dpo.json"), DPOConfig, id="dpo"),
    pytest.param("masked_sft", Path("configs/masked_sft.json"), MaskedSFTConfig, id="masked-sft"),
]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_scores_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,version,score,target_text\n000001,0,0.75,Привет\n",
        encoding="utf-8",
    )
    return path


def _torch_save(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


@pytest.mark.parametrize(("stage", "config_path", "expected_type"), CONFIG_CASES)
def test_committed_training_configs_load_as_existing_dataclasses(
    stage: str, config_path: Path, expected_type: type[object]
) -> None:
    config = load_stage_config(stage, config_path)

    assert isinstance(config, expected_type)
    assert config.model_id == "black-forest-labs/FLUX.2-klein-base-4B"
    assert config.num_training_steps > 0


def test_invalid_config_error_names_path_and_field_without_secret_value(tmp_path: Path) -> None:
    payload = _load_json(Path("configs/sft.json"))
    secret_like_value = "../private/sk-live-secret-token-123"
    payload["resume_lora_path"] = secret_like_value
    config_path = _write_json(tmp_path, "invalid-secret-path.json", payload)

    with pytest.raises(RuntimeConfigError) as exc_info:
        load_stage_config("sft", config_path)

    message = str(exc_info.value)
    assert str(config_path) in message
    assert "resume_lora_path" in message
    assert "path traversal" in message
    assert secret_like_value not in message
    assert "sk-live-secret-token-123" not in message


def test_unknown_config_fields_remain_rejected(tmp_path: Path) -> None:
    payload = _load_json(Path("configs/dpo.json"))
    payload["silent_future_toggle"] = True
    config_path = _write_json(tmp_path, "unknown-field.json", payload)

    with pytest.raises(RuntimeConfigError) as exc_info:
        load_stage_config("dpo", config_path)

    message = str(exc_info.value)
    assert str(config_path) in message
    assert "silent_future_toggle" in message
    assert "extra" in message.lower() or "forbidden" in message.lower()


def test_explicit_training_choices_appear_in_config_snapshots(tmp_path: Path) -> None:
    sft_payload = _load_json(Path("configs/sft.json")) | {
        "selection_mode": "score_weighted",
        "selected_samples_path": "outputs/generated/selected_samples.jsonl",
        "score_column": "score",
        "hard_negative_threshold": 0.2,
        "sample_weighting": "score_normalized",
    }
    dpo_payload = _load_json(Path("configs/dpo.json")) | {
        "pair_construction_mode": "margin_weighted",
        "preference_pairs_path": "outputs/generated/preference_pairs.jsonl",
        "score_column": "score",
        "ambiguity_margin": 0.05,
        "pair_weighting": "margin_normalized",
    }
    masked_payload = _load_json(Path("configs/masked_sft.json")) | {
        "masked_lambda": 0.65,
        "eval_suite_path": "configs/eval_suite.json",
        "validation_interval": 250,
        "eval_suite_n_per_step": 4,
    }

    sft_snapshot = resolve_config_snapshot(
        load_stage_config("sft", _write_json(tmp_path, "sft_explicit.json", sft_payload))
    )
    dpo_snapshot = resolve_config_snapshot(
        load_stage_config("dpo", _write_json(tmp_path, "dpo_explicit.json", dpo_payload))
    )
    masked_snapshot = resolve_config_snapshot(
        load_stage_config(
            "masked_sft", _write_json(tmp_path, "masked_explicit.json", masked_payload)
        )
    )

    assert sft_snapshot["selection_mode"] == "score_weighted"
    assert sft_snapshot["selected_samples_path"] == "outputs/generated/selected_samples.jsonl"
    assert dpo_snapshot["pair_construction_mode"] == "margin_weighted"
    assert dpo_snapshot["preference_pairs_path"] == "outputs/generated/preference_pairs.jsonl"
    assert masked_snapshot["masked_lambda"] == 0.65
    assert masked_snapshot["lora"]["attn_r"] == 64
    assert masked_snapshot["lora"]["joint_attn_r"] == 16
    assert masked_snapshot["data_dir"] == "data/synth_cyrillic/masked_sft"
    assert masked_snapshot["eval_suite_path"] == "configs/eval_suite.json"
    assert masked_snapshot["validation_interval"] == 250
    assert masked_snapshot["eval_suite_n_per_step"] == 4


def test_tiny_prompt_scores_and_generated_layout_validate_with_schema_warning(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "outputs" / "generated"
    prompts = _write_jsonl(
        generated / "prompts.jsonl",
        [{"prompt": "Render Привет", "target_text": "Привет"}],
    )
    scores = _write_scores_csv(generated / "scores.csv")
    _torch_save({"latent": torch.zeros(1, 2, 2)}, generated / "latents" / "000001" / "v0.pt")
    _torch_save(
        {
            "prompt_embeds": torch.zeros(2, 3),
            "target_text": "Привет",
            "prompt": "Render Привет",
        },
        generated / "text_embeds" / "000001.pt",
    )
    image_path = generated / "images" / "000001" / "v0.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    prompt_report = validate_artifacts("prompts", {"prompts_jsonl": prompts})
    score_report = validate_artifacts("scores", {"scores_csv": scores})
    generated_report = validate_artifacts(
        "generated",
        {
            "prompts_jsonl": prompts,
            "latents_dir": generated / "latents",
            "text_embeds_dir": generated / "text_embeds",
            "images_dir": generated / "images",
        },
    )

    assert prompt_report.ok
    assert prompt_report.metadata["prompt_count"] == 1
    assert score_report.ok
    assert any("optional score schema metadata" in warning for warning in score_report.warnings)
    assert generated_report.ok
    assert generated_report.metadata["generated_versions"] == {"000001": [0]}


def test_tiny_masked_sft_tensor_layout_validates_with_weights_only_cpu_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_calls: list[dict[str, object]] = []
    real_torch_load = torch.load

    def spy_torch_load(path: Path, *args: object, **kwargs: object) -> object:
        load_calls.append({"path": path, "kwargs": dict(kwargs)})
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(artifacts_module.torch, "load", spy_torch_load)
    data_dir = tmp_path / "data" / "synth_cyrillic" / "masked_sft"
    _torch_save(
        {"latent": torch.zeros(1, 2, 2), "mask_lat": torch.ones(2, 2)},
        data_dir / "latents" / "sample-1.pt",
    )
    _torch_save({"prompt_embeds": torch.zeros(2, 3)}, data_dir / "text_embeds" / "sample-1.pt")
    (data_dir / "shapes.csv").write_text("id,H,W\nsample-1,2,2\n", encoding="utf-8")

    report = validate_artifacts("masked_sft", {"data_dir": data_dir})

    assert report.ok
    assert report.metadata["masked_sft_samples"] == ["sample-1"]
    assert load_calls
    assert all(call["kwargs"]["map_location"] == "cpu" for call in load_calls)
    assert all(call["kwargs"]["weights_only"] is True for call in load_calls)


def test_require_ready_keyword_raises_aggregate_artifact_validation_context(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-generated"

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_artifacts(
            "generated",
            {
                "latents_dir": missing_root / "latents",
                "text_embeds_dir": missing_root / "text_embeds",
                "images_dir": missing_root / "images",
            },
            require_ready=True,
        )

    message = str(exc_info.value)
    assert "latents" in message
    assert "text_embeds" in message
    assert "images" in message
    assert ";" in message
