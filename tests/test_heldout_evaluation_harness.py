from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

DOC_PATH = Path("docs/evaluation_harness.md")
BASE_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
VLM_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    snapshot = {"schema_version": "runtime-config/v1"}
    _write_json(path.parent / "config_snapshot.json", snapshot)
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
            "config_snapshot_sha256": _json_sha256(snapshot),
            "config_snapshot": snapshot,
            "seeds": {"seed": 123},
            "models": {"model": "black-forest-labs/FLUX.2-klein-base-4B"},
            "inputs": {},
            "outputs": {},
            "metrics": {},
            "notes": [],
            "artifact_schema_versions": {},
        },
    )


def _valid_config(tmp_path: Path) -> dict[str, object]:
    prompts_path = _write_prompts(tmp_path / "fixtures" / "heldout_prompts.jsonl")
    baseline_manifest = _write_manifest(
        tmp_path / "runs" / "baseline" / "manifest.json",
        "baseline",
    )
    lora_manifest = _write_manifest(tmp_path / "runs" / "lora" / "manifest.json", "lora")
    output_root = tmp_path / "heldout" / "eval-001"
    return {
        "schema_version": "heldout-evaluation-config/v1",
        "fixed_prompts_path": str(prompts_path),
        "fixed_seeds": [101, 202],
        "inference_settings": {
            "model": "black-forest-labs/FLUX.2-klein-base-4B",
            "model_revision": BASE_REVISION,
            "vlm_model": "Qwen/Qwen3.5-9B",
            "vlm_model_revision": VLM_REVISION,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "scorer": "both",
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
        "model": "black-forest-labs/FLUX.2-klein-base-4B",
        "model_revision": BASE_REVISION,
        "num_inference_steps": 4,
        "scorer": "both",
        "vlm_model": "Qwen/Qwen3.5-9B",
        "vlm_model_revision": VLM_REVISION,
        "width": 1024,
    }
    assert [target["name"] for target in plan["targets"]] == [
        "baseline",
        "dpo-product-lora",
    ]
    assert plan["targets"][0]["lora_checkpoint_path"] is None
    assert plan["targets"][1]["lora_checkpoint_path"] == "runs/dpo-product/checkpoints/final"
    assert all(
        target["source_run_manifest_path"].endswith("manifest.json") for target in plan["targets"]
    )
    assert len(plan["planned_generation_commands"]) == 4
    assert len(plan["planned_scoring_commands"]) == 4
    assert len(plan["planned_aggregation_commands"]) == 2
    assert "scripts.generate_images" in plan["planned_generation_commands"][0]["command"]
    assert "scripts.score_images" in plan["planned_scoring_commands"][0]["command"]
    assert "scripts.aggregate_heldout_scores" in plan["planned_aggregation_commands"][0]["command"]
    assert "--lora_path" in plan["planned_generation_commands"][2]["command"]
    assert f"--model_revision {BASE_REVISION}" in plan["planned_generation_commands"][0]["command"]
    assert "generation.manifest.json" in plan["planned_scoring_commands"][0]["command"]
    assert f"--vlm_model_revision {VLM_REVISION}" in plan["planned_scoring_commands"][0]["command"]
    assert "seed-101/text_embeds" in plan["planned_scoring_commands"][0]["command"]


def test_write_evaluation_plan_materializes_json_and_markdown_reports(tmp_path: Path) -> None:
    from src.evaluation.heldout import write_evaluation_plan

    config_path = _write_json(tmp_path / "heldout_config.json", _valid_config(tmp_path))
    output_plan = tmp_path / "reports" / "heldout_plan.json"
    markdown_summary = tmp_path / "reports" / "heldout_plan.md"

    plan = write_evaluation_plan(
        config_path,
        output_plan=output_plan,
        markdown_summary=markdown_summary,
    )

    assert json.loads(output_plan.read_text(encoding="utf-8")) == plan
    markdown = markdown_summary.read_text(encoding="utf-8")
    assert "# Held-out evaluation plan" in markdown
    assert "fixed_prompts_path" in markdown
    assert "baseline" in markdown
    assert "dpo-product-lora" in markdown
    assert "source_run_manifest_path" in markdown
    assert "Plan only: generation and scoring commands are not executed" in markdown


def test_cli_materializes_plan_and_markdown_without_running_generation(tmp_path: Path) -> None:
    from scripts import run_heldout_evaluation

    config_path = _write_json(tmp_path / "heldout_config.json", _valid_config(tmp_path))
    output_plan = tmp_path / "reports" / "heldout_plan.json"
    markdown_summary = tmp_path / "reports" / "heldout_plan.md"

    exit_code = run_heldout_evaluation.main(
        [
            "--config",
            str(config_path),
            "--output-plan",
            str(output_plan),
            "--markdown-summary",
            str(markdown_summary),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_plan.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "heldout-evaluation-plan/v1"
    assert payload["execution_mode"] == "materialize-only"
    assert all(
        command["status"] == "planned-not-run" for command in payload["planned_generation_commands"]
    )
    assert markdown_summary.is_file()


def test_cli_returns_nonzero_for_invalid_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_heldout_evaluation

    payload = _valid_config(tmp_path)
    payload["fixed_seeds"] = []
    config_path = _write_json(tmp_path / "invalid_config.json", payload)

    exit_code = run_heldout_evaluation.main(
        ["--config", str(config_path), "--output-plan", str(tmp_path / "plan.json")]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "fixed_seeds" in captured.err


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("output_root", "../outside-eval"),
        ("output_root", "~/private-eval"),
        ("generation_output_path", "../outside-generated"),
        ("score_output_path", "~/outside-scores.jsonl"),
    ],
)
def test_output_paths_reject_traversal_and_home_expansion(
    tmp_path: Path,
    field: str,
    unsafe_value: str,
) -> None:
    from src.evaluation.heldout import HeldoutEvaluationConfig, HeldoutEvaluationError

    payload = _valid_config(tmp_path)
    if field == "output_root":
        payload[field] = unsafe_value
    else:
        targets = list(payload["targets"])
        first_target = dict(targets[0])
        first_target[field] = unsafe_value
        payload["targets"] = [first_target, *targets[1:]]
    config_path = _write_json(tmp_path / f"unsafe_{field}.json", payload)

    with pytest.raises(HeldoutEvaluationError, match=field):
        HeldoutEvaluationConfig.from_file(config_path)


def test_evaluation_harness_docs_match_config_and_command_contracts() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    required_terms = [
        "HeldoutEvaluationConfig",
        "EvaluationTarget",
        "build_evaluation_plan",
        "write_evaluation_plan",
        "heldout-evaluation-config/v1",
        "heldout-evaluation-plan/v1",
        "fixed_prompts_path",
        "fixed_seeds",
        "inference_settings",
        "output_root",
        "baseline",
        "lora_checkpoint_path",
        "source_run_manifest_path",
        "generation_output_path",
        "score_output_path",
        "python -m scripts.run_heldout_evaluation",
        "--config",
        "--output-plan",
        "--markdown-summary",
        "SLURM",
        "materialize-only",
        "does not run FLUX, Qwen, PaddleOCR, CUDA, or model weights",
        "generated images, tensors, checkpoints, logs",
        "## Comparison prerequisites",
    ]
    missing = [term for term in required_terms if term not in docs]

    assert missing == []


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


@pytest.mark.parametrize("seeds", [[101, 101], [-1, 101]])
def test_fixed_seeds_must_be_unique_and_nonnegative(tmp_path: Path, seeds: list[int]) -> None:
    from src.evaluation.heldout import HeldoutEvaluationConfig, HeldoutEvaluationError

    payload = _valid_config(tmp_path)
    payload["fixed_seeds"] = seeds

    with pytest.raises(HeldoutEvaluationError, match="fixed_seeds"):
        HeldoutEvaluationConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("model_revision", None),
        ("model_revision", "main"),
        ("vlm_model_revision", None),
        ("vlm_model_revision", "latest"),
    ],
)
def test_model_revisions_must_be_immutable_commit_shas(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    from src.evaluation.heldout import HeldoutEvaluationConfig, HeldoutEvaluationError

    payload = _valid_config(tmp_path)
    settings = dict(payload["inference_settings"])
    settings[field] = invalid_value
    payload["inference_settings"] = settings

    with pytest.raises(HeldoutEvaluationError, match=field):
        HeldoutEvaluationConfig.from_mapping(payload)


def test_manifest_link_uses_strict_snapshot_hash_validation(tmp_path: Path) -> None:
    from src.evaluation.heldout import HeldoutEvaluationConfig, HeldoutEvaluationError

    payload = _valid_config(tmp_path)
    target = dict(payload["targets"][0])
    manifest_path = Path(str(target["source_run_manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["config_snapshot_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HeldoutEvaluationError, match="config_snapshot_sha256"):
        HeldoutEvaluationConfig.from_mapping(payload)


def _json_sha256(payload: dict[str, object]) -> str:
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()
