from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_prompts(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "p-001", "prompt": "Render the word Привет", "target_text": "Привет"},
        {"id": "p-002", "prompt": "Render number 42", "target_text": "42"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_manifest(path: Path, run_id: str) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "run-manifest/v1",
            "run_id": run_id,
            "stage": "generate",
            "created_at": "2026-05-06T14:30:00Z",
            "command": ["python", "-m", "src.evaluation.generate_baseline"],
            "git": {},
            "environment": {},
            "config_snapshot_path": "config_snapshot.json",
            "config_snapshot": {"schema_version": "runtime-config/v1"},
            "seeds": {"seed": 123},
            "models": {"model": "black-forest-labs/FLUX.2-klein-4B"},
            "inputs": {},
            "outputs": {},
            "metrics": {},
            "notes": [],
            "artifact_schema_versions": {},
        },
    )


def _valid_config(tmp_path: Path) -> dict[str, object]:
    prompts_path = _write_prompts(tmp_path / "fixtures" / "heldout_prompts.jsonl")
    baseline_manifest = _write_manifest(tmp_path / "runs" / "baseline" / "manifest.json", "baseline")
    lora_manifest = _write_manifest(tmp_path / "runs" / "lora" / "manifest.json", "lora")
    output_root = tmp_path / "heldout" / "eval-001"
    return {
        "schema_version": "heldout-evaluation-config/v1",
        "fixed_prompts_path": str(prompts_path),
        "fixed_seeds": [101, 202],
        "inference_settings": {
            "model": "black-forest-labs/FLUX.2-klein-4B",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
        },
        "output_root": str(output_root),
        "targets": [
            {
                "name": "baseline",
                "lora_checkpoint_path": None,
                "source_run_manifest_path": str(baseline_manifest),
                "generation_output_path": str(output_root / "baseline" / "generated"),
                "score_output_path": str(output_root / "baseline" / "scores.jsonl"),
                "notes": ["unadapted baseline checkpoint"],
            },
            {
                "name": "dpo-product-lora",
                "lora_checkpoint_path": "runs/dpo-product/checkpoints/final",
                "source_run_manifest_path": str(lora_manifest),
                "generation_output_path": str(output_root / "dpo-product-lora" / "generated"),
                "score_output_path": str(output_root / "dpo-product-lora" / "scores.jsonl"),
                "notes": ["trained LoRA comparison target"],
            },
        ],
    }


def test_valid_config_builds_deterministic_heldout_evaluation_plan(tmp_path: Path) -> None:
    from src.evaluation.heldout import build_evaluation_plan

    config_path = _write_json(tmp_path / "heldout_config.json", _valid_config(tmp_path))

    plan = build_evaluation_plan(config_path)

    assert plan["schema_version"] == "heldout-evaluation-plan/v1"
    assert plan["source_config_path"] == str(config_path)
    assert plan["fixed_prompts_path"].endswith("heldout_prompts.jsonl")
    assert plan["fixed_seeds"] == [101, 202]
    assert plan["inference_settings"] == {
        "guidance_scale": 1.0,
        "height": 1024,
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "num_inference_steps": 4,
        "width": 1024,
    }
    assert [target["name"] for target in plan["targets"]] == [
        "baseline",
        "dpo-product-lora",
    ]
    assert plan["targets"][0]["lora_checkpoint_path"] is None
    assert plan["targets"][1]["lora_checkpoint_path"] == "runs/dpo-product/checkpoints/final"
    assert all(target["source_run_manifest_path"].endswith("manifest.json") for target in plan["targets"])
    assert len(plan["planned_generation_commands"]) == 4
    assert len(plan["planned_scoring_commands"]) == 2
    assert "src.evaluation.generate_baseline" in plan["planned_generation_commands"][0]["command"]
    assert "scripts.score_images" in plan["planned_scoring_commands"][0]["command"]


def test_evaluation_target_records_manifest_outputs_and_notes() -> None:
    from src.evaluation.heldout import EvaluationTarget

    target = EvaluationTarget(
        name="masked-sft-lora",
        lora_checkpoint_path="runs/masked/checkpoints/final",
        source_run_manifest_path="runs/masked/manifest.json",
        generation_output_path="runs/eval/masked/generated",
        score_output_path="runs/eval/masked/scores.jsonl",
        notes=("synthetic masked-SFT target",),
    )

    assert target.to_plan_entry() == {
        "generation_output_path": "runs/eval/masked/generated",
        "lora_checkpoint_path": "runs/masked/checkpoints/final",
        "name": "masked-sft-lora",
        "notes": ["synthetic masked-SFT target"],
        "score_output_path": "runs/eval/masked/scores.jsonl",
        "source_run_manifest_path": "runs/masked/manifest.json",
    }


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("fixed_prompts_path", "", "fixed_prompts_path"),
        ("fixed_seeds", [], "fixed_seeds"),
        ("inference_settings", {}, "inference_settings"),
    ],
)
def test_missing_fixed_prompts_seeds_or_settings_are_blocking_validation_errors(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    from src.evaluation.heldout import HeldoutEvaluationConfig, HeldoutEvaluationError

    payload = _valid_config(tmp_path)
    payload[field] = invalid_value
    config_path = _write_json(tmp_path / f"invalid_{field}.json", payload)

    with pytest.raises(HeldoutEvaluationError, match=message):
        HeldoutEvaluationConfig.from_file(config_path)
