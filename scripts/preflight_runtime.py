"""CPU-safe runtime preflight checks for configs, artifacts, and manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.runtime.artifacts import ArtifactReport, validate_artifacts
from src.runtime.capabilities import check_stage_support
from src.runtime.config_io import RuntimeConfigError, load_stage_config
from src.runtime.manifests import ManifestError, load_run_manifest
from src.runtime.paths import resolve_stage_paths

VLM_OCR_PRODUCT_FORMULA_NAME = "vlm_ocr_product_v1"

CLI_STAGES = (
    "generate",
    "score",
    "sft",
    "dpo",
    "masked-sft",
    "refl",
    "synthetic",
    "evaluation",
)
CONFIG_STAGES = {"sft": "sft", "dpo": "dpo", "masked-sft": "masked_sft"}
HELPER_STAGES = {
    "generate": "generated",
    "score": "score",
    "masked-sft": "masked_sft",
    "refl": "generated",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = build_preflight_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 1 if report["blocking_errors"] else 0


def build_preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    root = Path(args.root).resolve()

    config_report, config = _check_config(stage, args.config)
    paths = _resolve_paths(args, root, config=config)
    artifact_report = _check_artifacts(stage, paths, config=config)
    manifest_report = _check_manifest(
        args.manifest,
        expected_stage=CONFIG_STAGES.get(stage, stage),
        root=root,
    )
    runtime_report = check_stage_support(
        stage,
        scorer=args.scorer,
        ocr_device=args.ocr_device,
        mixed_precision=getattr(config, "mixed_precision", None),
    ).to_dict()

    warnings = [
        *config_report.get("warnings", []),
        *artifact_report["warnings"],
        *manifest_report["warnings"],
        *runtime_report["warnings"],
    ]
    blocking_errors = [
        *config_report["errors"],
        *artifact_report["errors"],
        *manifest_report["errors"],
        *runtime_report["errors"],
    ]
    return {
        "stage": stage,
        "config": config_report,
        "artifacts": artifact_report,
        "manifest": manifest_report,
        "runtime": runtime_report,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-safe runtime preflight for pipeline configs and artifacts"
    )
    parser.add_argument("--stage", choices=CLI_STAGES, required=True)
    parser.add_argument("--root", default=Path.cwd(), help="Repository/runtime root to inspect")
    parser.add_argument("--config", help="Optional stage config JSON to validate")
    parser.add_argument("--manifest", help="Optional run manifest to inspect for resume readiness")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--prompts", dest="prompts_jsonl", help="Prompt JSONL path")
    parser.add_argument("--output-dir", dest="output_dir", help="Output directory path")
    parser.add_argument("--images-dir", dest="images_dir", help="Generated images directory")
    parser.add_argument("--latents-dir", dest="latents_dir", help="Latent tensor directory")
    parser.add_argument(
        "--text-embeds-dir",
        dest="text_embeds_dir",
        help="Text embedding tensor directory",
    )
    parser.add_argument("--scores-csv", dest="scores_csv", help="Scores CSV path")
    parser.add_argument("--data-dir", dest="data_dir", help="Masked-SFT data directory")
    parser.add_argument("--run-dir", dest="run_dir", help="Run directory path")
    parser.add_argument(
        "--scorer",
        choices=("vlm", "ocr", "both"),
        default="both",
        help="Scorer runtime to validate for the score stage",
    )
    parser.add_argument(
        "--ocr-device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="PaddleOCR device to validate for the score stage",
    )
    return parser


def _resolve_paths(
    args: argparse.Namespace,
    root: Path,
    *,
    config: Any | None = None,
) -> dict[str, Path]:
    config_overrides: dict[str, Path] = {}
    if config is not None:
        field_mapping = {
            "latents_dir": "latents_dir",
            "text_embeds_dir": "text_embeds_dir",
            "scores_csv": "scores_csv",
            "data_dir": "data_dir",
            "selected_samples_path": "selected_samples",
            "preference_pairs_path": "preference_pairs",
            "resume_lora_path": "resume_lora",
            "sft_lora_path": "sft_lora",
            "output_dir": "output_dir",
        }
        for config_field, path_key in field_mapping.items():
            value = getattr(config, config_field, None)
            if value:
                config_overrides[path_key] = _rooted_path(root, value)
        output_dir = getattr(config, "output_dir", None)
        if output_dir:
            output_root = _rooted_path(root, output_dir)
            config_overrides.update(
                {
                    "checkpoints_dir": output_root / "checkpoints",
                    "samples_dir": output_root / "samples",
                }
            )
        data_dir = getattr(config, "data_dir", None)
        if data_dir:
            data_root = _rooted_path(root, data_dir)
            config_overrides.update(
                {
                    "latents_dir": data_root / "latents",
                    "text_embeds_dir": data_root / "text_embeds",
                    "shapes_csv": data_root / "shapes.csv",
                }
            )

    overrides = {
        key: value
        for key, value in {
            "prompts_jsonl": args.prompts_jsonl,
            "output_dir": args.output_dir,
            "images_dir": args.images_dir,
            "latents_dir": args.latents_dir,
            "text_embeds_dir": args.text_embeds_dir,
            "scores_csv": args.scores_csv,
            "data_dir": args.data_dir,
            "run_dir": args.run_dir,
            "manifest_json": args.manifest,
        }.items()
        if value is not None
    }
    rooted_cli_overrides = {key: _rooted_path(root, value) for key, value in overrides.items()}
    resolved_overrides = {**config_overrides, **rooted_cli_overrides}
    helper_stage = HELPER_STAGES.get(args.stage, args.stage)
    return dict(
        resolve_stage_paths(
            helper_stage,
            root=root,
            **resolved_overrides,
        ).paths
    )


def _check_config(stage: str, config_path: str | None) -> tuple[dict[str, Any], Any | None]:
    if config_path is None:
        return (
            {
                "ok": True,
                "path": None,
                "stage": CONFIG_STAGES.get(stage),
                "errors": [],
                "warnings": [],
            },
            None,
        )
    config_stage = CONFIG_STAGES.get(stage)
    if config_stage is None:
        return (
            {
                "ok": False,
                "path": config_path,
                "stage": None,
                "errors": [f"{stage}: --config is only supported for sft, dpo, or masked-sft"],
                "warnings": [],
            },
            None,
        )
    try:
        cfg = load_stage_config(config_stage, config_path)
    except RuntimeConfigError as exc:
        return (
            {
                "ok": False,
                "path": config_path,
                "stage": config_stage,
                "errors": [str(exc)],
                "warnings": [],
            },
            None,
        )
    return (
        {
            "ok": True,
            "path": config_path,
            "stage": config_stage,
            "type": type(cfg).__name__,
            "errors": [],
            "warnings": (
                []
                if getattr(cfg, "model_revision", None)
                else [
                    "model_revision is unset; the model ID may resolve to different weights later"
                ]
            ),
        },
        cfg,
    )


def _check_artifacts(
    stage: str,
    paths: dict[str, Path],
    *,
    config: Any | None = None,
) -> dict[str, Any]:
    if stage in {"generate", "refl"}:
        report = validate_artifacts("prompts", {"prompts_jsonl": paths["prompts_jsonl"]})
    elif stage == "score":
        report = validate_artifacts(
            "scoring_inputs",
            {
                "images_dir": paths["images_dir"],
                "text_embeds_dir": paths["text_embeds_dir"],
            },
        )
    else:
        report = validate_artifacts(HELPER_STAGES.get(stage, stage), paths)
    payload = _artifact_report_to_json(report)
    if stage in {"sft", "dpo"} and config is not None and _requires_vlm_ocr_product(config, paths):
        product_report = validate_artifacts(
            "evaluation_scores",
            {"scores_csv": paths["scores_csv"]},
        )
        payload["checked_paths"] = list(
            dict.fromkeys([*payload["checked_paths"], *product_report.checked_paths])
        )
        payload["warnings"].extend(product_report.warnings)
        payload["errors"].extend(product_report.errors)
        payload["metadata"]["product_score_contract"] = product_report.metadata
        payload["errors"].extend(_product_formula_errors(paths["scores_csv"]))
        payload["ok"] = not payload["errors"]
    destination = paths.get("output_dir") if stage in {"generate", "refl"} else None
    if stage == "score":
        destination = paths.get("scores_csv")
    if destination is not None:
        error = _output_destination_error(destination)
        if error:
            payload["errors"].append(error)
            payload["ok"] = False
    return payload


def _requires_vlm_ocr_product(config: Any, paths: dict[str, Path]) -> bool:
    experiment_name = str(getattr(config, "experiment_name", "")).casefold()
    scores_name = Path(str(getattr(config, "scores_csv", ""))).stem.casefold()
    score_column = str(getattr(config, "score_column", "")).casefold()
    if any("product" in value for value in (experiment_name, scores_name, score_column)):
        return True

    scores_sidecar = paths["scores_csv"].with_suffix(".schema.json")
    if _json_metadata_indicates_product(scores_sidecar):
        return True

    for key in ("selected_samples", "preference_pairs"):
        artifact_path = paths.get(key)
        if artifact_path is None:
            continue
        if _jsonl_metadata_indicates_product(artifact_path):
            return True
        if _json_metadata_indicates_product(artifact_path.with_suffix(".manifest.json")):
            return True
    return False


def _json_metadata_indicates_product(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    formula = payload.get("formula")
    direct_values = (
        payload.get("primary_score"),
        payload.get("score_column"),
        payload.get("reward"),
        payload.get("reward_name"),
    )
    if any("product" in str(value).casefold() for value in direct_values if value is not None):
        return True
    return (
        payload.get("primary_score") is None
        and isinstance(formula, dict)
        and "product" in str(formula.get("name") or "").casefold()
    )


def _jsonl_metadata_indicates_product(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    return False
                values = (
                    payload.get("score_column"),
                    payload.get("reward"),
                    payload.get("reward_name"),
                    payload.get("source_scores_path"),
                )
                return any(
                    "product" in str(value).casefold() for value in values if value is not None
                )
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _product_formula_errors(scores_csv: Path) -> list[str]:
    sidecar = scores_csv.with_suffix(".schema.json")
    if not sidecar.is_file():
        return [f"Product training requires canonical score sidecar: {sidecar}"]
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"Product training score sidecar is unreadable: {sidecar}"]
    if not isinstance(payload, dict):
        return [f"Product training score sidecar must be a JSON object: {sidecar}"]
    formula = payload.get("formula")
    expected_weights = {"score_ocr": 1.0, "score_vlm": 1.0}
    errors: list[str] = []
    if payload.get("primary_score") != "product":
        errors.append("Product training requires primary_score='product'")
    if not isinstance(formula, dict) or formula.get("name") != VLM_OCR_PRODUCT_FORMULA_NAME:
        errors.append(f"Product training requires formula {VLM_OCR_PRODUCT_FORMULA_NAME!r}")
        return errors
    if formula.get("aggregation") != "weighted_product":
        errors.append("Product training requires weighted_product aggregation")
    if formula.get("require_all") is not True:
        errors.append("Product training requires both VLM and OCR components")
    if formula.get("weights") != expected_weights:
        errors.append("Product training requires unit VLM and OCR Product weights")
    return errors


def _output_destination_error(path: Path) -> str | None:
    candidate = path if path.is_dir() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        return f"output destination has no existing parent: {path}"
    if not candidate.is_dir():
        return f"output destination parent is not a directory: {candidate}"
    if not os.access(candidate, os.W_OK):
        return f"output destination parent is not writable: {candidate}"
    return None


def _check_manifest(
    manifest_path: str | None,
    *,
    expected_stage: str,
    root: Path,
) -> dict[str, Any]:
    if manifest_path is None:
        return {"ok": True, "path": None, "resume_ready": False, "warnings": [], "errors": []}
    try:
        manifest = load_run_manifest(manifest_path)
    except ManifestError as exc:
        return {
            "ok": False,
            "path": manifest_path,
            "resume_ready": False,
            "warnings": [],
            "errors": [str(exc)],
        }
    errors = []
    warnings = []
    if manifest.stage != expected_stage:
        errors.append(
            f"manifest stage {manifest.stage!r} does not match requested stage {expected_stage!r}"
        )
    if manifest.git.get("dirty") is True:
        errors.append(
            "manifest was created from a dirty Git tree; comparison-grade runs require "
            "a clean commit"
        )
    existing_outputs = []
    for name, value in manifest.outputs.items():
        if not isinstance(value, str):
            continue
        output_path = _rooted_path(root, value)
        if output_path.exists():
            existing_outputs.append(name)
    if expected_stage in {"sft", "dpo", "masked_sft"}:
        checkpoint_root_value = manifest.outputs.get("checkpoints_dir")
        if not isinstance(checkpoint_root_value, str):
            output_root_value = manifest.outputs.get("output_dir")
            checkpoint_root_value = (
                str(Path(output_root_value) / "checkpoints")
                if isinstance(output_root_value, str)
                else None
            )
        checkpoint_files = []
        if checkpoint_root_value:
            checkpoint_root = _rooted_path(root, checkpoint_root_value)
            if checkpoint_root.is_dir():
                checkpoint_files = [
                    path
                    for path in checkpoint_root.rglob("*")
                    if path.is_file() and path.suffix in {".bin", ".pt", ".safetensors"}
                ]
        resume_ready = not errors and bool(checkpoint_files)
        if not checkpoint_files:
            warnings.append("no checkpoint weight files were found; resume is not ready")
    else:
        resume_ready = not errors and bool(existing_outputs)
    if not manifest.outputs:
        warnings.append("manifest has no declared outputs; it cannot prove resume readiness")
    elif not existing_outputs:
        warnings.append("none of the manifest's declared outputs exist under the runtime root")
    return {
        "ok": not errors,
        "path": str(manifest.manifest_path),
        "run_id": manifest.run_id,
        "stage": manifest.stage,
        "resume_ready": resume_ready,
        "existing_outputs": existing_outputs,
        "outputs": manifest.outputs,
        "warnings": warnings,
        "errors": errors,
    }


def _rooted_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _artifact_report_to_json(report: ArtifactReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["ok"] = report.ok
    return payload


def _print_human_report(report: dict[str, Any]) -> None:
    print(f"Runtime preflight: {report['stage']}")
    print(f"Config: {'ok' if report['config']['ok'] else 'blocked'}")
    print(f"Artifacts: {'ok' if report['artifacts']['ok'] else 'blocked'}")
    print(f"Manifest: {'ok' if report['manifest']['ok'] else 'blocked'}")
    print(f"Runtime: {'ok' if report['runtime']['ok'] else 'blocked'}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["blocking_errors"]:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
