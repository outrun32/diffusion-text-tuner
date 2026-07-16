from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.runtime.config_io import (
    RuntimeConfigError,
    load_stage_config,
    resolve_config_snapshot,
    validate_path_policy,
)
from src.training.config import (
    DPOConfig,
    MaskedSFTConfig,
    MultiRankLoraConfig,
    SFTConfig,
)

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _sft_payload() -> dict:
    return {
        "model_id": MODEL_ID,
        "latents_dir": "outputs/generated/latents",
        "text_embeds_dir": "outputs/generated/text_embeds",
        "scores_csv": "outputs/generated/scores.csv",
        "score_threshold": 0.3,
        "num_training_steps": 1000,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "lr": 2e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "seed": 42,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "resolution": 512,
        "lora": {"r": 64, "lora_alpha": 64, "target_modules": ["to_k", "to_q"]},
        "log_interval": 10,
        "save_interval": 200,
        "output_dir": "outputs/sft",
        "experiment_name": "sft_v1",
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
    }


def _dpo_payload() -> dict:
    payload = _sft_payload()
    payload.update(
        {
            "sft_lora_path": "outputs/sft/checkpoints/final",
            "score_threshold": 0.5,
            "score_diff_min": 0.1,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "lr": 1e-4,
            "beta": 5000.0,
            "save_interval": 100,
            "output_dir": "outputs/dpo",
            "experiment_name": "dpo_v1",
        }
    )
    return payload


def _masked_sft_payload() -> dict:
    return {
        "model_id": MODEL_ID,
        "data_dir": "data/synth_cyrillic/masked_sft",
        "val_n_samples": 200,
        "num_training_steps": 5000,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "lr": 2e-5,
        "lr_min": 1e-6,
        "lr_schedule": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "seed": 42,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "masked_lambda": 0.65,
        "resolution": 512,
        "lora": {
            "attn_r": 64,
            "attn_alpha": 64,
            "ffn_r": 24,
            "ffn_alpha": 24,
            "joint_attn_r": 32,
            "joint_attn_alpha": 32,
            "dropout": 0.0,
        },
        "sample_interval": 0,
        "num_inference_steps": 28,
        "validation_interval": 250,
        "val_t_anchors": [100, 300, 500, 700, 900],
        "eval_suite_path": "configs/eval_suite.json",
        "eval_suite_n_per_step": 4,
        "log_interval": 10,
        "save_interval": 500,
        "output_dir": "outputs/masked_sft",
        "experiment_name": "masked_sft_cyrillic_v1",
        "progress_bar_mininterval": 30.0,
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
    }


@pytest.mark.parametrize(
    ("stage", "payload_factory", "expected_type", "lora_type"),
    [
        ("sft", _sft_payload, SFTConfig, None),
        ("dpo", _dpo_payload, DPOConfig, None),
        ("masked_sft", _masked_sft_payload, MaskedSFTConfig, MultiRankLoraConfig),
    ],
)
def test_load_stage_config_returns_existing_training_dataclasses(
    tmp_path: Path,
    stage: str,
    payload_factory,
    expected_type: type,
    lora_type: type | None,
) -> None:
    path = _write_json(tmp_path, f"{stage}.json", payload_factory())

    cfg = load_stage_config(stage, path)

    assert isinstance(cfg, expected_type)
    if lora_type is not None:
        assert isinstance(cfg.lora, lora_type)
    assert cfg.model_id == MODEL_ID


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("num_training_steps", 0),
        ("batch_size", -1),
        ("gradient_accumulation_steps", 0),
        ("warmup_steps", -1),
        ("mixed_precision", "fp32"),
        ("score_threshold", -0.1),
        ("score_threshold", 1.1),
        ("model_id", "black-forest-labs/FLUX.2-dev"),
    ],
)
def test_invalid_sft_values_raise_runtime_config_error_with_path_and_field(
    tmp_path: Path, field_name: str, bad_value: object
) -> None:
    payload = _sft_payload()
    payload[field_name] = bad_value
    path = _write_json(tmp_path, "sft_bad.json", payload)

    with pytest.raises(RuntimeConfigError) as exc_info:
        load_stage_config("sft", path)

    message = str(exc_info.value)
    assert str(path) in message
    assert field_name in message
    assert "TOKEN" not in message


def test_unknown_and_missing_fields_raise_field_level_runtime_errors(tmp_path: Path) -> None:
    unknown_payload = _sft_payload() | {"unexpected_field": True}
    missing_payload = _sft_payload()
    missing_payload.pop("latents_dir")

    unknown_path = _write_json(tmp_path, "unknown.json", unknown_payload)
    missing_path = _write_json(tmp_path, "missing.json", missing_payload)

    with pytest.raises(RuntimeConfigError) as unknown_error:
        load_stage_config("sft", unknown_path)
    with pytest.raises(RuntimeConfigError) as missing_error:
        load_stage_config("sft", missing_path)

    assert str(unknown_path) in str(unknown_error.value)
    assert "unexpected_field" in str(unknown_error.value)
    assert str(missing_path) in str(missing_error.value)
    assert "latents_dir" in str(missing_error.value)


def test_nonlegacy_selection_modes_require_materialized_artifacts(tmp_path: Path) -> None:
    sft_payload = _sft_payload() | {"selection_mode": "top_k_per_prompt"}
    dpo_payload = _dpo_payload() | {"pair_construction_mode": "margin_weighted"}

    with pytest.raises(RuntimeConfigError, match="selected_samples_path"):
        load_stage_config("sft", _write_json(tmp_path, "sft-no-selection.json", sft_payload))
    with pytest.raises(RuntimeConfigError, match="preference_pairs_path"):
        load_stage_config("dpo", _write_json(tmp_path, "dpo-no-pairs.json", dpo_payload))


def test_weighted_mode_names_require_matching_weight_semantics(tmp_path: Path) -> None:
    bad_sft = _sft_payload() | {"selection_mode": "score_weighted"}
    bad_dpo = _dpo_payload() | {
        "pair_construction_mode": "margin_weighted",
        "preference_pairs_path": "outputs/generated/pairs.jsonl",
    }

    with pytest.raises(RuntimeConfigError, match="sample_weighting"):
        load_stage_config("sft", _write_json(tmp_path, "sft-weight.json", bad_sft))
    with pytest.raises(RuntimeConfigError, match="pair_weighting"):
        load_stage_config("dpo", _write_json(tmp_path, "dpo-weight.json", bad_dpo))


@pytest.mark.parametrize(
    ("stage", "payload_factory"),
    [("sft", _sft_payload), ("masked_sft", _masked_sft_payload)],
)
def test_resume_step_requires_resume_checkpoint(
    tmp_path: Path, stage: str, payload_factory
) -> None:
    payload = payload_factory() | {"resume_step": 5, "resume_lora_path": None}

    with pytest.raises(RuntimeConfigError, match="resume_lora_path"):
        load_stage_config(stage, _write_json(tmp_path, f"{stage}-resume.json", payload))


def test_malformed_json_raises_runtime_config_error_with_path(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text('{"model_id": ', encoding="utf-8")

    with pytest.raises(RuntimeConfigError) as exc_info:
        load_stage_config("sft", path)

    assert str(path) in str(exc_info.value)
    assert "json" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "allowed",
    ["configs/sft.json", "data/input.jsonl", "outputs/sft", "runs/demo"],
)
def test_validate_path_policy_allows_relative_runtime_roots(tmp_path: Path, allowed: str) -> None:
    config_path = tmp_path / "config.json"

    assert (
        validate_path_policy(allowed, field_name="output_dir", config_path=config_path) == allowed
    )


@pytest.mark.parametrize("rejected", ["/home/user/private", "~/private", "../outside"])
def test_validate_path_policy_rejects_unsafe_committed_paths(tmp_path: Path, rejected: str) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(RuntimeConfigError) as exc_info:
        validate_path_policy(rejected, field_name="output_dir", config_path=config_path)

    message = str(exc_info.value)
    assert str(config_path) in message
    assert "output_dir" in message


def test_validate_path_policy_allows_explicit_environment_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    assert (
        validate_path_policy(
            "~/private",
            field_name="hf_cache_dir",
            config_path=config_path,
            allow_environment_input=True,
        )
        == "~/private"
    )


def test_path_policy_is_applied_during_stage_loading(tmp_path: Path) -> None:
    payload = _dpo_payload()
    payload["sft_lora_path"] = "../outside/checkpoint"
    path = _write_json(tmp_path, "dpo_bad_path.json", payload)

    with pytest.raises(RuntimeConfigError) as exc_info:
        load_stage_config("dpo", path)

    assert str(path) in str(exc_info.value)
    assert "sft_lora_path" in str(exc_info.value)


def test_sft_explicit_selection_fields_snapshot_for_manifest_provenance(tmp_path: Path) -> None:
    payload = _sft_payload() | {
        "selection_mode": "score_weighted",
        "selected_samples_path": "outputs/generated/selected_samples.jsonl",
        "score_column": "reward_score",
        "hard_negative_threshold": 0.15,
        "sample_weighting": "score_normalized",
    }
    path = _write_json(tmp_path, "sft_explicit_selection.json", payload)

    snapshot = resolve_config_snapshot(load_stage_config("sft", path))

    assert snapshot["selection_mode"] == "score_weighted"
    assert snapshot["selected_samples_path"] == "outputs/generated/selected_samples.jsonl"
    assert snapshot["score_column"] == "reward_score"
    assert snapshot["hard_negative_threshold"] == 0.15
    assert snapshot["sample_weighting"] == "score_normalized"


def test_dpo_explicit_pair_fields_snapshot_for_manifest_provenance(tmp_path: Path) -> None:
    payload = _dpo_payload() | {
        "pair_construction_mode": "margin_weighted",
        "preference_pairs_path": "outputs/generated/preference_pairs.jsonl",
        "score_column": "reward_score",
        "ambiguity_margin": 0.05,
        "pair_weighting": "margin_normalized",
    }
    path = _write_json(tmp_path, "dpo_explicit_pairs.json", payload)

    snapshot = resolve_config_snapshot(load_stage_config("dpo", path))

    assert snapshot["pair_construction_mode"] == "margin_weighted"
    assert snapshot["preference_pairs_path"] == "outputs/generated/preference_pairs.jsonl"
    assert snapshot["score_column"] == "reward_score"
    assert snapshot["ambiguity_margin"] == 0.05
    assert snapshot["pair_weighting"] == "margin_normalized"


def test_invalid_explicit_mode_strings_are_secret_safe(tmp_path: Path) -> None:
    secret_like_value = "score_weighted_sk-live-secret-token-123"  # gitleaks:allow
    payload = _sft_payload() | {"selection_mode": secret_like_value}
    path = _write_json(tmp_path, "sft_bad_mode.json", payload)

    with pytest.raises(RuntimeConfigError) as exc_info:
        load_stage_config("sft", path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "selection_mode" in message
    assert secret_like_value not in message
    assert "sk-live-secret-token-123" not in message


def test_resolve_config_snapshot_is_json_serializable_sorted_and_immutable(
    tmp_path: Path,
) -> None:
    payload = _masked_sft_payload()
    path = _write_json(tmp_path, "masked_sft.json", payload)
    cfg = load_stage_config("masked_sft", path)
    before = deepcopy(cfg)

    snapshot = resolve_config_snapshot(cfg)

    assert snapshot["schema_version"] == "runtime-config/v1"
    assert snapshot["stage"] == "masked_sft"
    assert snapshot["lora"]["attn_r"] == 64
    assert cfg == before
    assert json.loads(json.dumps(snapshot, sort_keys=True)) == snapshot
    assert list(snapshot) == sorted(snapshot)


def test_experiment_config_docs_guard_explicit_training_choice_fields() -> None:
    docs = {
        "sft": Path("configs/experiments/sft/README.md").read_text(encoding="utf-8"),
        "dpo": Path("configs/experiments/dpo/README.md").read_text(encoding="utf-8"),
        "masked_sft": Path("configs/experiments/masked_sft/README.md").read_text(encoding="utf-8"),
    }

    for field in [
        "selection_mode",
        "selected_samples_path",
        "score_column",
        "score_threshold",
        "hard_negative_threshold",
        "sample_weighting",
    ]:
        assert field in docs["sft"]

    for field in [
        "pair_construction_mode",
        "preference_pairs_path",
        "score_column",
        "score_threshold",
        "score_diff_min",
        "ambiguity_margin",
        "pair_weighting",
        "beta",
    ]:
        assert field in docs["dpo"]

    for field in [
        "masked_lambda",
        "lora.attn_r",
        "lora.joint_attn_r",
        "data_dir",
        "eval_suite_path",
        "validation_interval",
        "eval_suite_n_per_step",
    ]:
        assert field in docs["masked_sft"]
