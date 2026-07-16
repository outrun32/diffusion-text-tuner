"""Build or verify SHA-256 provenance for the committed public evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_ARTIFACTS = (
    "reports/final/README.md",
    "reports/final/prompt_dataset_source.manifest.json",
    "reports/final/prompt_dataset_quality_v1.json",
    "reports/final/prompt_training_target_hashes_v1.json",
    "reports/final/benchmark_prompts_v2.jsonl",
    "reports/final/benchmark_prompts_v2.manifest.json",
    "reports/final/historical_benchmark_summary.csv",
    "reports/final/historical_selection_bias.json",
    "reports/final/current_model_sources.json",
    "docs/project-page/assets/teaser_success.webp",
    "docs/project-page/assets/method_candidates.webp",
    "docs/project-page/assets/heldout_cer.webp",
    "docs/project-page/assets/dpo_tradeoff.webp",
    "docs/project-page/assets/product_bias.webp",
    "docs/project-page/assets/failure_cases.webp",
)


def build_manifest(root: Path) -> dict[str, object]:
    artifacts = []
    for relative_path in DEFAULT_ARTIFACTS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"evidence artifact is missing: {relative_path}")
        artifacts.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "evidence_class": _evidence_class(relative_path, path),
            }
        )
    return {
        "schema_version": "public-evidence-manifest/v1",
        "historical_results_status": "aggregate-only; raw per-sample evidence unavailable",
        "checkpoint_status": "not published",
        "artifacts": artifacts,
    }


def verify_manifest(root: Path, manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_manifest = build_manifest(root)
    if payload.get("schema_version") != "public-evidence-manifest/v1":
        errors.append("invalid evidence manifest schema_version")
    for field in ("historical_results_status", "checkpoint_status"):
        if payload.get(field) != expected_manifest[field]:
            errors.append(f"evidence status mismatch: {field}")

    recorded_artifacts = payload.get("artifacts")
    if not isinstance(recorded_artifacts, list):
        return [*errors, "artifacts must be a list"]
    expected = {artifact["path"]: artifact for artifact in expected_manifest["artifacts"]}
    if not all(isinstance(artifact, dict) for artifact in recorded_artifacts):
        return [*errors, "every evidence artifact entry must be an object"]
    recorded_paths = [str(artifact.get("path") or "") for artifact in recorded_artifacts]
    if len(recorded_paths) != len(set(recorded_paths)):
        errors.append("evidence manifest contains duplicate artifact paths")
    missing_paths = sorted(set(expected) - set(recorded_paths))
    unexpected_paths = sorted(set(recorded_paths) - set(expected))
    if missing_paths:
        errors.append("missing required evidence artifacts: " + ", ".join(missing_paths))
    if unexpected_paths:
        errors.append("unexpected evidence artifacts: " + ", ".join(unexpected_paths))

    for artifact in recorded_artifacts:
        relative_path = str(artifact.get("path") or "")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe evidence path: {relative_path}")
            continue
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing: {relative_path}")
            continue
        actual = _sha256(path)
        if actual != artifact.get("sha256"):
            errors.append(f"hash mismatch: {relative_path}")
        expected_artifact = expected.get(relative_path)
        if expected_artifact is not None:
            if artifact.get("bytes") != expected_artifact["bytes"]:
                errors.append(f"size mismatch: {relative_path}")
            if artifact.get("evidence_class") != expected_artifact["evidence_class"]:
                errors.append(f"evidence class mismatch: {relative_path}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/final/evidence_manifest.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    if args.verify:
        errors = verify_manifest(root, output)
        for error in errors:
            print(error, file=sys.stderr)
        return 1 if errors else 0

    payload = build_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_class(relative_path: str, path: Path) -> str:
    if path.suffix == ".webp":
        return "static-defense-figure"
    if relative_path == "reports/final/historical_selection_bias.json":
        return "historical-aggregate-only"
    return "public-source"


if __name__ == "__main__":
    raise SystemExit(main())
