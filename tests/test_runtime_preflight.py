from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
    ["generate", "score", "sft", "dpo", "masked-sft", "refl", "synthetic", "evaluation"],
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
        args.extend(
            ["--config", str(_write_json(tmp_path / "configs" / f"{stage}.json", config_payload))]
        )

    exit_code = preflight_runtime.main(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == stage
    assert set(payload) >= {
        "stage",
        "config",
        "artifacts",
        "manifest",
        "blocking_errors",
        "warnings",
    }
    assert exit_code in {0, 1}


def test_preflight_validates_config_artifacts_and_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import preflight_runtime

    monkeypatch.setattr(
        preflight_runtime,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "ok": True,
                "stage": "sft",
                "runtime": "cuda",
                "capabilities": {},
                "errors": [],
                "warnings": [],
            }
        ),
    )

    config_path = _write_json(tmp_path / "configs" / "sft.json", _sft_payload())
    _write_scores_csv(tmp_path / "outputs" / "generated" / "scores.csv")
    (tmp_path / "outputs" / "generated" / "latents").mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / "text_embeds").mkdir(parents=True)
    (tmp_path / "outputs" / "sft" / "checkpoints").mkdir(parents=True)
    (tmp_path / "outputs" / "sft" / "checkpoints" / "adapter_model.safetensors").write_bytes(
        b"fixture"
    )
    from src.runtime.manifests import create_run_manifest

    manifest_path = create_run_manifest(
        stage="sft",
        config_path=config_path,
        command=["python", "-m", "src.training.sft_trainer"],
        run_root=tmp_path / "runs",
        outputs={"checkpoints_dir": "outputs/sft/checkpoints"},
        root=tmp_path,
    ).manifest_path

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


def test_generate_and_score_preflight_validate_inputs_not_future_outputs(
    tmp_path: Path,
) -> None:
    from scripts import preflight_runtime

    prompts = tmp_path / "data" / "prompts.jsonl"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(
        json.dumps({"prompt": "Render ТЕСТ", "target_text": "ТЕСТ"}) + "\n",
        encoding="utf-8",
    )
    generate_args = preflight_runtime._build_parser().parse_args(
        [
            "--stage",
            "generate",
            "--root",
            str(tmp_path),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(tmp_path / "not-created-yet"),
        ]
    )
    generate_report = preflight_runtime.build_preflight_report(generate_args)
    assert generate_report["artifacts"]["ok"] is True

    image_dir = tmp_path / "images" / "000000"
    embeds = tmp_path / "embeds"
    image_dir.mkdir(parents=True)
    embeds.mkdir()
    (image_dir / "v0.png").write_bytes(b"fixture")
    torch.save({"target_text": "ТЕСТ"}, embeds / "000000.pt")
    score_args = preflight_runtime._build_parser().parse_args(
        [
            "--stage",
            "score",
            "--root",
            str(tmp_path),
            "--images-dir",
            str(image_dir.parent),
            "--text-embeds-dir",
            str(embeds),
            "--scores-csv",
            str(tmp_path / "future-scores.csv"),
        ]
    )
    score_report = preflight_runtime.build_preflight_report(score_args)
    assert score_report["artifacts"]["ok"] is True
    assert score_report["artifacts"]["metadata"]["scoring_prompt_count"] == 1

    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    blocked_args = preflight_runtime._build_parser().parse_args(
        [
            "--stage",
            "score",
            "--root",
            str(tmp_path),
            "--images-dir",
            str(image_dir.parent),
            "--text-embeds-dir",
            str(embeds),
            "--scores-csv",
            str(blocked_parent / "scores.csv"),
        ]
    )
    blocked_report = preflight_runtime.build_preflight_report(blocked_args)
    assert blocked_report["artifacts"]["ok"] is False
    assert any("not a directory" in error for error in blocked_report["blocking_errors"])


def test_product_training_preflight_requires_canonical_thesis_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import preflight_runtime
    from src.evaluation.reward_interface import ProductScoreFormula, vlm_ocr_product_formula
    from src.runtime.manifests import create_run_manifest
    from src.scoring.pipeline import CANONICAL_SCORE_COLUMNS, write_score_schema_sidecar

    monkeypatch.setattr(
        preflight_runtime,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "ok": True,
                "stage": "sft",
                "runtime": "cuda",
                "capabilities": {},
                "errors": [],
                "warnings": [],
            }
        ),
    )
    scores = tmp_path / "outputs" / "generated" / "scores_product.csv"
    scores.parent.mkdir(parents=True)
    row = {field: "" for field in CANONICAL_SCORE_COLUMNS}
    row.update(
        {
            "id": "p1",
            "sample_id": "p1",
            "version": 0,
            "score": 0.6,
            "product_score": 0.6,
            "target_text": "ТЕСТ",
            "score_vlm": 0.8,
            "score_ocr": 0.75,
            "detection_status": "detected_exact",
            "exact_text_match": "true",
            "char_accuracy": 1.0,
            "char_matches": 4,
            "char_total": 4,
            "formula_complete": "true",
            "manifest_path": "runs/eval/manifest.json",
            "text_metrics": "{}",
            "scorer_metadata": "{}",
            "thresholds": "{}",
        }
    )
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    source_manifest = create_run_manifest(
        stage="evaluation",
        command=["pytest", "product-score-source"],
        run_root=tmp_path / "runs",
        outputs={"scores_csv": str(scores)},
        root=tmp_path,
    )
    execution = {
        "status": "complete",
        "scored_row_count": 1,
        "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
    }
    write_score_schema_sidecar(
        scores,
        formula=vlm_ocr_product_formula(),
        source_manifest_paths=(str(source_manifest.manifest_path),),
        primary_score="product",
        execution_metadata=execution,
    )
    (tmp_path / "outputs" / "generated" / "latents").mkdir()
    (tmp_path / "outputs" / "generated" / "text_embeds").mkdir()
    config = _sft_payload() | {
        "scores_csv": "outputs/generated/scores_product.csv",
        "experiment_name": "sft_product_test",
    }
    config_path = _write_json(tmp_path / "configs" / "product.json", config)
    args = preflight_runtime._build_parser().parse_args(
        ["--stage", "sft", "--root", str(tmp_path), "--config", str(config_path)]
    )

    assert preflight_runtime.build_preflight_report(args)["artifacts"]["ok"] is True

    write_score_schema_sidecar(
        scores,
        formula=ProductScoreFormula(),
        source_manifest_paths=(str(source_manifest.manifest_path),),
        primary_score="product",
        execution_metadata=execution,
    )
    blocked = preflight_runtime.build_preflight_report(args)

    assert blocked["artifacts"]["ok"] is False
    assert any("vlm_ocr_product_v1" in error for error in blocked["blocking_errors"])

    neutral_scores = scores.with_name("scores.csv")
    scores.replace(neutral_scores)
    scores.with_suffix(".schema.json").replace(neutral_scores.with_suffix(".schema.json"))
    neutral_config = _sft_payload() | {
        "scores_csv": "outputs/generated/scores.csv",
        "experiment_name": "sft_neutral_test",
        "score_column": "product_score",
    }
    neutral_config_path = _write_json(tmp_path / "configs" / "neutral.json", neutral_config)
    neutral_args = preflight_runtime._build_parser().parse_args(
        ["--stage", "sft", "--root", str(tmp_path), "--config", str(neutral_config_path)]
    )

    neutral_blocked = preflight_runtime.build_preflight_report(neutral_args)

    assert neutral_blocked["artifacts"]["ok"] is False
    assert any("vlm_ocr_product_v1" in error for error in neutral_blocked["blocking_errors"])


def test_product_detection_reads_materialized_artifact_metadata(tmp_path: Path) -> None:
    from scripts.preflight_runtime import _requires_vlm_ocr_product

    scores = tmp_path / "outputs" / "generated" / "scores.csv"
    selected = tmp_path / "outputs" / "generated" / "selected_samples.jsonl"
    selected.parent.mkdir(parents=True)
    selected.write_text(
        json.dumps(
            {
                "schema_version": "selected-samples/v1",
                "sample_id": "sft:p1:v0:product_score",
                "prompt_id": "p1",
                "version": 0,
                "selected_score": 0.8,
                "score_column": "product_score",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        experiment_name="sft_neutral",
        scores_csv="outputs/generated/scores.csv",
        score_column="score",
    )

    assert _requires_vlm_ocr_product(
        config,
        {"scores_csv": scores, "selected_samples": selected},
    )

    selected.unlink()
    scores.with_suffix(".schema.json").write_text(
        json.dumps(
            {
                "primary_score": "vlm",
                "formula": {"name": "vlm_ocr_product_v1"},
            }
        ),
        encoding="utf-8",
    )

    assert not _requires_vlm_ocr_product(config, {"scores_csv": scores})


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_preflight_blocks_missing_configured_selection_and_initialization_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    from scripts import preflight_runtime

    monkeypatch.setattr(
        preflight_runtime,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "ok": True,
                "stage": stage,
                "runtime": "cuda",
                "capabilities": {},
                "errors": [],
                "warnings": [],
            }
        ),
    )
    _write_scores_csv(tmp_path / "outputs" / "generated" / "scores.csv")
    (tmp_path / "outputs" / "generated" / "latents").mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / "text_embeds").mkdir()
    if stage == "sft":
        config = _sft_payload() | {
            "selection_mode": "top_k_per_prompt",
            "selected_samples_path": "outputs/generated/missing-selected.jsonl",
            "resume_lora_path": "outputs/missing-resume",
        }
    else:
        config = _dpo_payload() | {
            "pair_construction_mode": "margin_weighted",
            "pair_weighting": "margin_normalized",
            "preference_pairs_path": "outputs/generated/missing-pairs.jsonl",
            "sft_lora_path": "outputs/missing-sft-init",
        }
    config_path = _write_json(tmp_path / "configs" / f"{stage}.json", config)
    args = preflight_runtime._build_parser().parse_args(
        ["--stage", stage, "--root", str(tmp_path), "--config", str(config_path)]
    )

    report = preflight_runtime.build_preflight_report(args)

    assert report["artifacts"]["ok"] is False
    if stage == "sft":
        assert any(
            "selected_samples file is missing" in error for error in report["blocking_errors"]
        )
        assert any("resume_lora_path" in error for error in report["blocking_errors"])
    else:
        assert any(
            "preference_pairs file is missing" in error for error in report["blocking_errors"]
        )
        assert any("sft_lora_path" in error for error in report["blocking_errors"])
