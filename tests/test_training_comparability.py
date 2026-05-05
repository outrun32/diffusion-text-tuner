from __future__ import annotations

import json

from src.training.comparability import compare_training_configs, format_comparability_report


def test_compare_training_configs_reports_blocking_controlled_field_mismatches():
    left = _base_config(
        num_training_steps=1000,
        num_inference_steps=28,
        guidance_scale=3.5,
        prompt_embedding_padding="max_length",
        seed=42,
        sample_prompt="Render text A",
        sample_target_text="ТЕКСТ A",
        latents_dir="outputs/generated-a/latents",
        text_embeds_dir="outputs/generated-a/text_embeds",
        scores_csv="outputs/generated-a/scores.csv",
        data_dir="data/synth-a/masked_sft",
        score_column="vlm_score",
        reward_model="qwen-vlm-v1",
        scorer="vlm",
        metric_columns=["vlm_score", "ocr_cer"],
        samples_dir="outputs/sft-a/samples",
    )
    right = _base_config(
        model_id="black-forest-labs/FLUX.2-klein-dev-4B",
        num_training_steps=1200,
        num_inference_steps=32,
        guidance_scale=4.0,
        prompt_embedding_padding="do_not_pad",
        seed=7,
        sample_prompt="Render text B",
        sample_target_text="ТЕКСТ B",
        latents_dir="outputs/generated-b/latents",
        text_embeds_dir="outputs/generated-b/text_embeds",
        scores_csv="outputs/generated-b/scores.csv",
        data_dir="data/synth-b/masked_sft",
        score_column="ocr_score",
        reward_model="ocr-v2",
        scorer="ocr",
        metric_columns=["ocr_score"],
        samples_dir="outputs/dpo-b/samples",
    )

    report = compare_training_configs(left, right, left_label="sft", right_label="dpo")

    assert set(report) == {
        "schema_version",
        "left_label",
        "right_label",
        "blocking_mismatches",
        "warnings",
        "controlled_fields",
        "summary",
    }
    assert report["left_label"] == "sft"
    assert report["right_label"] == "dpo"
    blocking_by_field = {item["field"]: item for item in report["blocking_mismatches"]}
    warning_by_field = {item["field"]: item for item in report["warnings"]}

    assert blocking_by_field["model_id"] == {
        "field": "model_id",
        "group": "model",
        "left": "black-forest-labs/FLUX.2-klein-base-4B",
        "right": "black-forest-labs/FLUX.2-klein-dev-4B",
        "reason": "value_mismatch",
        "severity": "blocking",
    }
    assert blocking_by_field["seed"]["left"] == 42
    assert blocking_by_field["seed"]["right"] == 7
    assert blocking_by_field["num_inference_steps"]["group"] == "inference"
    assert blocking_by_field["guidance_scale"]["group"] == "inference"
    assert blocking_by_field["prompt_embedding_padding"]["group"] == "inference"
    assert blocking_by_field["sample_prompt"]["group"] == "prompt"
    assert blocking_by_field["sample_target_text"]["group"] == "prompt"
    assert blocking_by_field["latents_dir"]["group"] == "data_source"
    assert blocking_by_field["text_embeds_dir"]["group"] == "data_source"
    assert blocking_by_field["scores_csv"]["group"] == "data_source"
    assert blocking_by_field["data_dir"]["group"] == "data_source"
    assert blocking_by_field["score_column"]["group"] == "reward"
    assert blocking_by_field["reward_model"]["group"] == "reward"
    assert blocking_by_field["scorer"]["group"] == "reward"

    assert warning_by_field["num_training_steps"]["severity"] == "warning"
    assert warning_by_field["metric_columns"]["group"] == "metrics"
    assert warning_by_field["samples_dir"]["group"] == "artifacts"
    assert report["controlled_fields"]["inference"] == [
        "num_inference_steps",
        "guidance_scale",
        "prompt_embedding_padding",
    ]
    assert report["summary"] == {
        "blocking_count": len(report["blocking_mismatches"]),
        "warning_count": len(report["warnings"]),
        "is_comparable": False,
    }


def test_compare_training_configs_reports_missing_controlled_fields_explicitly():
    left = _base_config(num_inference_steps=28, guidance_scale=4.0)
    right = _base_config(num_inference_steps=28)
    del right["guidance_scale"]
    del left["prompt_embedding_padding"]

    report = compare_training_configs(left, right)

    mismatches = {item["field"]: item for item in report["blocking_mismatches"]}
    assert mismatches["guidance_scale"]["reason"] == "missing_right"
    assert mismatches["guidance_scale"]["left"] == 4.0
    assert mismatches["guidance_scale"]["right"] is None
    assert mismatches["prompt_embedding_padding"]["reason"] == "missing_left"
    assert mismatches["prompt_embedding_padding"]["left"] is None
    assert mismatches["prompt_embedding_padding"]["right"] == "max_length"


def test_compare_training_configs_identical_controlled_fields_has_no_blockers():
    left = _base_config(metric_columns=["vlm_score"], samples_dir="outputs/left/samples")
    right = _base_config(metric_columns=["vlm_score", "ocr_cer"], samples_dir="outputs/right/samples")

    report = compare_training_configs(left, right)

    assert report["blocking_mismatches"] == []
    assert report["summary"]["is_comparable"] is True
    assert {item["field"] for item in report["warnings"]} == {"metric_columns", "samples_dir"}


def test_format_comparability_report_renders_deterministic_markdown():
    report = compare_training_configs(
        _base_config(seed=42, num_inference_steps=28),
        _base_config(seed=99, num_inference_steps=32),
        left_label="baseline",
        right_label="masked_sft",
    )

    markdown = format_comparability_report(report)

    assert markdown.startswith("# Training comparability report\n")
    assert "Left: baseline" in markdown
    assert "Right: masked_sft" in markdown
    assert "## Blocking mismatches" in markdown
    assert "## Warnings" in markdown
    assert "| Field | Group | Left | Right | Reason |" in markdown
    assert "| num_inference_steps | inference | 28 | 32 | value_mismatch |" in markdown
    assert "| seed | prompt | 42 | 99 | value_mismatch |" in markdown
    assert markdown == format_comparability_report(json.loads(json.dumps(report)))


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
