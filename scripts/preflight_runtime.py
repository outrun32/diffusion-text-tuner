"""CPU-safe runtime preflight checks for configs, artifacts, and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.runtime.artifacts import ArtifactReport, validate_artifacts
from src.runtime.config_io import RuntimeConfigError, load_stage_config
from src.runtime.manifests import ManifestError, load_run_manifest
from src.runtime.paths import resolve_stage_paths

CLI_STAGES = ("generate", "score", "sft", "dpo", "masked-sft", "synthetic", "evaluation")
CONFIG_STAGES = {"sft": "sft", "dpo": "dpo", "masked-sft": "masked_sft"}
HELPER_STAGES = {"generate": "generated", "score": "score", "masked-sft": "masked_sft"}


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
    paths = _resolve_paths(args, root)

    config_report = _check_config(stage, args.config)
    artifact_report = _check_artifacts(stage, paths)
    manifest_report = _check_manifest(args.manifest)

    warnings = [*artifact_report["warnings"], *manifest_report["warnings"]]
    blocking_errors = [
        *config_report["errors"],
        *artifact_report["errors"],
        *manifest_report["errors"],
    ]
    return {
        "stage": stage,
        "config": config_report,
        "artifacts": artifact_report,
        "manifest": manifest_report,
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
    parser.add_argument("--text-embeds-dir", dest="text_embeds_dir", help="Text embedding tensor directory")
    parser.add_argument("--scores-csv", dest="scores_csv", help="Scores CSV path")
    parser.add_argument("--data-dir", dest="data_dir", help="Masked-SFT data directory")
    parser.add_argument("--run-dir", dest="run_dir", help="Run directory path")
    return parser


def _resolve_paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
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
    helper_stage = HELPER_STAGES.get(args.stage, args.stage)
    return dict(resolve_stage_paths(helper_stage, root=root, **overrides).paths)


def _check_config(stage: str, config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {"ok": True, "path": None, "stage": CONFIG_STAGES.get(stage), "errors": []}
    config_stage = CONFIG_STAGES.get(stage)
    if config_stage is None:
        return {
            "ok": False,
            "path": config_path,
            "stage": None,
            "errors": [f"{stage}: --config is only supported for sft, dpo, or masked-sft"],
        }
    try:
        cfg = load_stage_config(config_stage, config_path)
    except RuntimeConfigError as exc:
        return {"ok": False, "path": config_path, "stage": config_stage, "errors": [str(exc)]}
    return {
        "ok": True,
        "path": config_path,
        "stage": config_stage,
        "type": type(cfg).__name__,
        "errors": [],
    }


def _check_artifacts(stage: str, paths: dict[str, Path]) -> dict[str, Any]:
    report = validate_artifacts(HELPER_STAGES.get(stage, stage), paths)
    return _artifact_report_to_json(report)


def _check_manifest(manifest_path: str | None) -> dict[str, Any]:
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
    return {
        "ok": True,
        "path": str(manifest.manifest_path),
        "run_id": manifest.run_id,
        "stage": manifest.stage,
        "resume_ready": True,
        "outputs": manifest.outputs,
        "warnings": [],
        "errors": [],
    }


def _artifact_report_to_json(report: ArtifactReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["ok"] = report.ok
    return payload


def _print_human_report(report: dict[str, Any]) -> None:
    print(f"Runtime preflight: {report['stage']}")
    print(f"Config: {'ok' if report['config']['ok'] else 'blocked'}")
    print(f"Artifacts: {'ok' if report['artifacts']['ok'] else 'blocked'}")
    print(f"Manifest: {'ok' if report['manifest']['ok'] else 'blocked'}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["blocking_errors"]:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
