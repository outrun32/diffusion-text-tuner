"""Local run manifest helpers for reproducible CPU-safe provenance."""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from src.runtime.artifacts import ARTIFACT_SCHEMA_VERSION
from src.runtime.config_io import load_stage_config, resolve_config_snapshot
from src.runtime.reproducibility import (
    collect_environment_summary,
    collect_git_state,
    collect_model_revisions,
    collect_seeds,
)

MANIFEST_SCHEMA_VERSION = "run-manifest/v1"
CONFIG_SNAPSHOT_FILENAME = "config_snapshot.json"
MANIFEST_FILENAME = "manifest.json"
TRAINING_STAGES = frozenset({"sft", "dpo", "masked_sft"})
MANIFEST_STAGES = frozenset(
    {"generate", "score", "sft", "dpo", "masked_sft", "synthetic", "evaluation"}
)


class ManifestError(ValueError):
    """Raised when a run manifest cannot be created, loaded, or updated safely."""


@dataclass(frozen=True)
class RunManifest:
    """Loaded run manifest with filesystem convenience paths."""

    schema_version: str
    run_id: str
    stage: str
    created_at: str
    command: list[str]
    git: dict[str, Any]
    environment: dict[str, Any]
    config_snapshot_path: Path
    config_snapshot: dict[str, Any]
    seeds: dict[str, Any]
    models: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metrics: dict[str, Any]
    notes: list[dict[str, Any]]
    artifact_schema_versions: dict[str, str]
    manifest_path: Path
    run_dir: Path

    def to_json_payload(self) -> dict[str, Any]:
        """Return the serializable manifest payload using relative file references."""

        return _sort_mapping(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "stage": self.stage,
                "created_at": self.created_at,
                "command": list(self.command),
                "git": self.git,
                "environment": self.environment,
                "config_snapshot_path": self.config_snapshot_path.name,
                "config_snapshot": self.config_snapshot,
                "seeds": self.seeds,
                "models": self.models,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "metrics": self.metrics,
                "notes": self.notes,
                "artifact_schema_versions": self.artifact_schema_versions,
            }
        )


def create_run_manifest(
    *,
    stage: str,
    config_path: str | Path | None = None,
    command: str | list[str],
    run_root: str | Path = "runs",
    slug: str | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: list[dict[str, Any]] | None = None,
    root: str | Path | None = None,
) -> RunManifest:
    """Create a local ignored run directory with manifest and immutable config snapshot."""

    stage = _normalize_stage(stage)
    root_path = (Path.cwd() if root is None else Path(root)).resolve()
    run_root_path = _validate_run_root(run_root)
    normalized_config_path = Path(config_path) if config_path is not None else None
    config_snapshot = _resolve_manifest_config_snapshot(stage, normalized_config_path)
    run_id = _allocate_run_id(stage=stage, run_root=run_root_path, slug=slug)
    run_dir = run_root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_snapshot_path = run_dir / CONFIG_SNAPSHOT_FILENAME
    manifest_path = run_dir / MANIFEST_FILENAME

    merged_inputs = _sort_mapping(
        {
            **({"config_path": str(normalized_config_path)} if normalized_config_path else {}),
            **_path_like_config_values(config_snapshot, include_outputs=False),
            **(inputs or {}),
        }
    )
    manifest = RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        stage=stage,
        created_at=_utc_now(),
        command=_normalize_command(command),
        git=collect_git_state(root_path),
        environment=collect_environment_summary(),
        config_snapshot_path=config_snapshot_path,
        config_snapshot=config_snapshot,
        seeds=collect_seeds(config_snapshot),
        models=collect_model_revisions(config_snapshot),
        inputs=merged_inputs,
        outputs=_sort_mapping(outputs or {}),
        metrics=_sort_mapping(metrics or {}),
        notes=list(notes or []),
        artifact_schema_versions={"runtime_artifacts": ARTIFACT_SCHEMA_VERSION},
        manifest_path=manifest_path,
        run_dir=run_dir,
    )
    _atomic_write_json(config_snapshot_path, config_snapshot)
    _atomic_write_json(manifest_path, manifest.to_json_payload())
    return manifest


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load an existing manifest for resume/inspection."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path}: malformed manifest JSON") from exc
    except OSError as exc:
        raise ManifestError(f"{manifest_path}: could not read manifest") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{manifest_path}: manifest must be a JSON object")

    run_dir = manifest_path.parent
    snapshot_path = run_dir / str(payload.get("config_snapshot_path", CONFIG_SNAPSHOT_FILENAME))
    config_snapshot = payload.get("config_snapshot")
    if not isinstance(config_snapshot, dict) and snapshot_path.is_file():
        try:
            config_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"{snapshot_path}: could not read config snapshot") from exc
    if not isinstance(config_snapshot, dict):
        raise ManifestError(f"{manifest_path}: config_snapshot must be an object")

    return RunManifest(
        schema_version=_required_str(payload, "schema_version", manifest_path),
        run_id=_required_str(payload, "run_id", manifest_path),
        stage=_required_str(payload, "stage", manifest_path),
        created_at=_required_str(payload, "created_at", manifest_path),
        command=_required_list(payload, "command", manifest_path),
        git=_dict_field(payload, "git"),
        environment=_dict_field(payload, "environment"),
        config_snapshot_path=snapshot_path,
        config_snapshot=config_snapshot,
        seeds=_dict_field(payload, "seeds"),
        models=_dict_field(payload, "models"),
        inputs=_dict_field(payload, "inputs"),
        outputs=_dict_field(payload, "outputs"),
        metrics=_dict_field(payload, "metrics"),
        notes=_notes_field(payload),
        artifact_schema_versions=_dict_field(payload, "artifact_schema_versions"),
        manifest_path=manifest_path,
        run_dir=run_dir,
    )


def update_run_manifest(
    path: str | Path,
    *,
    note: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> RunManifest:
    """Append a timestamped note and/or merge metrics without deleting provenance fields."""

    manifest = load_run_manifest(path)
    updated_notes = list(manifest.notes)
    if note is not None:
        updated_notes.append({"timestamp": _utc_now(), "text": note})
    updated_metrics = _sort_mapping({**manifest.metrics, **(metrics or {})})
    updated = RunManifest(
        schema_version=manifest.schema_version,
        run_id=manifest.run_id,
        stage=manifest.stage,
        created_at=manifest.created_at,
        command=list(manifest.command),
        git=dict(manifest.git),
        environment=dict(manifest.environment),
        config_snapshot_path=manifest.config_snapshot_path,
        config_snapshot=dict(manifest.config_snapshot),
        seeds=dict(manifest.seeds),
        models=dict(manifest.models),
        inputs=dict(manifest.inputs),
        outputs=dict(manifest.outputs),
        metrics=updated_metrics,
        notes=updated_notes,
        artifact_schema_versions=dict(manifest.artifact_schema_versions),
        manifest_path=manifest.manifest_path,
        run_dir=manifest.run_dir,
    )
    _atomic_write_json(updated.manifest_path, updated.to_json_payload())
    return updated


def format_manifest_summary(manifest: RunManifest) -> str:
    """Return a concise human-readable manifest inspection summary."""

    outputs = ", ".join(sorted(manifest.outputs)) if manifest.outputs else "none"
    metrics = ", ".join(sorted(manifest.metrics)) if manifest.metrics else "none"
    return "\n".join(
        [
            f"Run ID: {manifest.run_id}",
            f"Stage: {manifest.stage}",
            f"Command: {' '.join(manifest.command)}",
            f"Config snapshot: {manifest.config_snapshot_path.name}",
            f"Outputs: {outputs}",
            f"Metrics: {metrics}",
            f"Notes: {len(manifest.notes)}",
        ]
    )


def print_manifest_summary(manifest: RunManifest, file: TextIO | None = None) -> None:
    """Print a concise manifest inspection summary."""

    print(format_manifest_summary(manifest), file=file or sys.stdout)


def _validate_run_root(run_root: str | Path) -> Path:
    run_root_path = Path(run_root)
    if ".." in run_root_path.parts or run_root_path.expanduser() != run_root_path:
        raise ManifestError("run_root must not use traversal or home-directory expansion")
    resolved = run_root_path.resolve()
    if "runs" not in resolved.parts:
        raise ManifestError("run_root must be a runs/ directory or temporary runs root")
    return resolved


def _normalize_stage(stage: str) -> str:
    normalized = stage.replace("-", "_")
    if normalized not in MANIFEST_STAGES:
        allowed = ", ".join(sorted(MANIFEST_STAGES))
        raise ManifestError(f"unsupported manifest stage {stage!r}; expected one of: {allowed}")
    return normalized


def _resolve_manifest_config_snapshot(
    stage: str, config_path: Path | None
) -> dict[str, Any]:
    if stage in TRAINING_STAGES:
        if config_path is None:
            raise ManifestError(f"{stage}: --config is required for training-stage manifests")
        config = load_stage_config(stage, config_path)
        return resolve_config_snapshot(config)
    if config_path is None:
        return {"schema_version": "runtime-config/v1", "stage": stage}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{config_path}: malformed config JSON") from exc
    except OSError as exc:
        raise ManifestError(f"{config_path}: could not read config") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{config_path}: config snapshot must be a JSON object")
    return _sort_mapping(
        {
            "schema_version": "runtime-config/v1",
            "stage": stage,
            "raw_config": payload,
        }
    )


def _allocate_run_id(*, stage: str, run_root: Path, slug: str | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = _slugify(slug or stage)
    base = f"{timestamp}-{stage}-{suffix}"
    candidate = base
    counter = 2
    while (run_root / candidate).exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def _normalize_command(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _path_like_config_values(
    config_snapshot: dict[str, Any], *, include_outputs: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    output_keys = {"output_dir", "outputs_dir", "samples_dir", "checkpoints_dir", "logs_dir"}
    input_keys = {"prompts", "images", "latents", "text_embeds", "scores"}
    source = config_snapshot.get("raw_config")
    if not isinstance(source, dict):
        source = config_snapshot
    for key, value in source.items():
        if not isinstance(value, str):
            continue
        if key in output_keys:
            if include_outputs:
                result[key] = value
            continue
        if key in input_keys or key.endswith(("_dir", "_path", "_csv", "_jsonl")):
            result[key] = value
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(_sort_mapping(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mapping(item) for item in value]
    return value


def _required_str(payload: dict[str, Any], key: str, manifest_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{manifest_path}: {key} must be a non-empty string")
    return value


def _required_list(payload: dict[str, Any], key: str, manifest_path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"{manifest_path}: {key} must be a list of strings")
    return list(value)


def _dict_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _notes_field(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("notes", [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
