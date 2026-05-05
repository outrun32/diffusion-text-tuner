from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runtime.config_io import RuntimeConfigError, load_stage_config
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
