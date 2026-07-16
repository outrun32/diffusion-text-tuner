"""Local run manifest helpers for reproducible CPU-safe provenance."""

from __future__ import annotations

import hashlib
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
from src.training.runtime import training_run_inputs, training_run_outputs

MANIFEST_SCHEMA_VERSION = "run-manifest/v1"
GENERATION_MANIFEST_SCHEMA_VERSION = "generation-manifest/v4"
CONFIG_SNAPSHOT_FILENAME = "config_snapshot.json"
MANIFEST_FILENAME = "manifest.json"
TRAINING_STAGES = frozenset({"sft", "dpo", "masked_sft"})
MANIFEST_STAGES = frozenset(
    {"generate", "score", "sft", "dpo", "masked_sft", "refl", "synthetic", "evaluation"}
)
_SENSITIVE_KEY_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "AUTH",
    "BEARER",
    "COOKIE",
)
_SENSITIVE_COMMAND_FLAGS = frozenset(
    {
        "--api-key",
        "--password",
        "--secret",
        "--token",
        "--access-key",
    }
)
_SECRET_PATTERNS = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_ACCESS_KEY]"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENAI_TOKEN]"),
    (re.compile(r"(?:gh[pousr]|github_pat)_[A-Za-z0-9_-]{20,}"), "[REDACTED_TOKEN]"),
)
_AUTHORIZATION_VALUE_PATTERN = re.compile(r"(?i)\bauthorization\s*:\s*(?:(?:bearer|basic)\s+)?\S+")
_AUTH_SCHEME_VALUE_PATTERN = re.compile(r"(?i)\b(?:bearer|basic)\s+\S+")
_SENSITIVE_POSITIONAL_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:token|secret|password|credential|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|auth)\s*:\s*\S+"
)
_AUTH_SCHEMES = frozenset({"bearer", "basic"})


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
    config_snapshot_sha256: str
    config_snapshot: dict[str, Any]
    seeds: dict[str, Any]
    models: dict[str, Any]
    inputs: dict[str, Any]
    input_hashes: dict[str, str]
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
                "config_snapshot_sha256": self.config_snapshot_sha256,
                "config_snapshot": self.config_snapshot,
                "seeds": self.seeds,
                "models": self.models,
                "inputs": self.inputs,
                "input_hashes": self.input_hashes,
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
    config_snapshot = _redact_sensitive(
        _resolve_manifest_config_snapshot(stage, normalized_config_path)
    )
    run_id = _allocate_run_id(stage=stage, run_root=run_root_path, slug=slug)
    run_dir = run_root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_snapshot_path = run_dir / CONFIG_SNAPSHOT_FILENAME
    manifest_path = run_dir / MANIFEST_FILENAME

    config_inputs = (
        training_run_inputs(config_snapshot)
        if stage in TRAINING_STAGES
        else _path_like_config_values(config_snapshot, include_outputs=False)
    )
    config_outputs = training_run_outputs(config_snapshot) if stage in TRAINING_STAGES else {}
    merged_inputs = _sort_mapping(
        _redact_sensitive(
            {
                **({"config_path": str(normalized_config_path)} if normalized_config_path else {}),
                **config_inputs,
                **(inputs or {}),
            }
        )
    )
    manifest = RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        stage=stage,
        created_at=_utc_now(),
        command=_redact_command(_normalize_command(command)),
        git=collect_git_state(root_path),
        environment=collect_environment_summary(),
        config_snapshot_path=config_snapshot_path,
        config_snapshot_sha256=_json_payload_sha256(config_snapshot),
        config_snapshot=config_snapshot,
        seeds=collect_seeds(config_snapshot),
        models=collect_model_revisions(config_snapshot),
        inputs=merged_inputs,
        input_hashes=_collect_input_hashes(merged_inputs, root_path),
        outputs=_sort_mapping(_redact_sensitive({**config_outputs, **(outputs or {})})),
        metrics=_sort_mapping(_redact_sensitive(metrics or {})),
        notes=list(_redact_sensitive(notes or [])),
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
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"{manifest_path}: schema_version must be {MANIFEST_SCHEMA_VERSION!r}")

    run_dir = manifest_path.parent
    snapshot_reference_value = payload.get("config_snapshot_path")
    if not isinstance(snapshot_reference_value, str) or not snapshot_reference_value.strip():
        raise ManifestError(f"{manifest_path}: config_snapshot_path must be a non-empty string")
    snapshot_reference = Path(snapshot_reference_value)
    if snapshot_reference.is_absolute() or ".." in snapshot_reference.parts:
        raise ManifestError(
            f"{manifest_path}: config_snapshot_path must stay inside the run directory"
        )
    snapshot_path = (run_dir / snapshot_reference).resolve()
    try:
        snapshot_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ManifestError(
            f"{manifest_path}: config_snapshot_path must stay inside the run directory"
        ) from exc
    if not snapshot_path.is_file():
        raise ManifestError(f"{manifest_path}: config snapshot file is missing: {snapshot_path}")
    try:
        file_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{snapshot_path}: could not read config snapshot") from exc
    if not isinstance(file_snapshot, dict):
        raise ManifestError(f"{snapshot_path}: config snapshot must be an object")

    embedded_snapshot = payload.get("config_snapshot")
    if embedded_snapshot is not None and not isinstance(embedded_snapshot, dict):
        raise ManifestError(f"{manifest_path}: embedded config_snapshot must be an object")
    if isinstance(embedded_snapshot, dict) and embedded_snapshot != file_snapshot:
        raise ManifestError(
            f"{manifest_path}: embedded config_snapshot differs from {snapshot_path.name}"
        )
    config_snapshot = file_snapshot
    snapshot_sha256 = _sha256_file(snapshot_path)
    declared_snapshot_sha256 = payload.get("config_snapshot_sha256")
    if not isinstance(declared_snapshot_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", declared_snapshot_sha256
    ):
        raise ManifestError(
            f"{manifest_path}: config_snapshot_sha256 must be a declared SHA-256 digest"
        )
    if declared_snapshot_sha256 != snapshot_sha256:
        raise ManifestError(f"{manifest_path}: config snapshot SHA-256 mismatch")
    safe_config_snapshot = _redact_sensitive(config_snapshot)

    return RunManifest(
        schema_version=_required_str(payload, "schema_version", manifest_path),
        run_id=_required_str(payload, "run_id", manifest_path),
        stage=_required_str(payload, "stage", manifest_path),
        created_at=_required_str(payload, "created_at", manifest_path),
        command=_redact_command(_required_list(payload, "command", manifest_path)),
        git=_dict_field(payload, "git"),
        environment=_redact_environment(_dict_field(payload, "environment")),
        config_snapshot_path=snapshot_path,
        config_snapshot_sha256=declared_snapshot_sha256,
        config_snapshot=safe_config_snapshot,
        seeds=_dict_field(payload, "seeds"),
        models=_redact_sensitive(_dict_field(payload, "models")),
        inputs=_redact_sensitive(_dict_field(payload, "inputs")),
        input_hashes={
            str(key): str(value) for key, value in _dict_field(payload, "input_hashes").items()
        },
        outputs=_redact_sensitive(_dict_field(payload, "outputs")),
        metrics=_redact_sensitive(_dict_field(payload, "metrics")),
        notes=list(_redact_sensitive(_notes_field(payload))),
        artifact_schema_versions=_dict_field(payload, "artifact_schema_versions"),
        manifest_path=manifest_path,
        run_dir=run_dir,
    )


def validate_source_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a recognized provenance manifest and return its JSON payload.

    Canonical score artifacts may link either a strict ``run-manifest/v1`` or the
    current immutable ``generation-manifest/v4``. Arbitrary JSON and legacy
    generation schemas are not evidence manifests.
    """

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path}: malformed source manifest JSON") from exc
    except OSError as exc:
        raise ManifestError(f"{manifest_path}: could not read source manifest") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{manifest_path}: source manifest must be a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version == MANIFEST_SCHEMA_VERSION:
        load_run_manifest(manifest_path)
        return payload
    if schema_version == GENERATION_MANIFEST_SCHEMA_VERSION:
        _validate_generation_source_manifest(manifest_path, payload)
        return payload
    raise ManifestError(
        f"{manifest_path}: unsupported source manifest schema {schema_version!r}; expected "
        f"{MANIFEST_SCHEMA_VERSION!r} or {GENERATION_MANIFEST_SCHEMA_VERSION!r}"
    )


def _validate_generation_source_manifest(path: Path, payload: dict[str, Any]) -> None:
    from src.generation.pipeline import GenerationResumeError, validate_generation_manifest

    try:
        validated = validate_generation_manifest(path, require_complete=True, verify_artifacts=True)
    except GenerationResumeError as exc:
        raise ManifestError(str(exc)) from exc
    model_revision = str(validated["contract"].get("model_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ManifestError(f"{path}: generation model_revision must be an immutable commit SHA")
    run_manifest = str(validated["completion"].get("run_manifest_path") or "")
    if not run_manifest:
        raise ManifestError(f"{path}: complete generation manifest must link a run manifest")
    load_run_manifest(_resolve_source_link(run_manifest, manifest_path=path))


def _resolve_source_link(raw_path: str, *, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.is_file():
        return path
    sibling_relative = manifest_path.parent / path
    return sibling_relative if sibling_relative.is_file() else path


def _compact_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generation_records_sha256(records: list[Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(serialized).to_bytes(8, "big"))
        digest.update(serialized)
    return digest.hexdigest()


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
        updated_notes.append({"timestamp": _utc_now(), "text": _redact_string(note)})
    updated_metrics = _sort_mapping(_redact_sensitive({**manifest.metrics, **(metrics or {})}))
    updated = RunManifest(
        schema_version=manifest.schema_version,
        run_id=manifest.run_id,
        stage=manifest.stage,
        created_at=manifest.created_at,
        command=_redact_command(list(manifest.command)),
        git=dict(manifest.git),
        environment=dict(manifest.environment),
        config_snapshot_path=manifest.config_snapshot_path,
        config_snapshot_sha256=manifest.config_snapshot_sha256,
        config_snapshot=_redact_sensitive(dict(manifest.config_snapshot)),
        seeds=dict(manifest.seeds),
        models=dict(manifest.models),
        inputs=_redact_sensitive(dict(manifest.inputs)),
        input_hashes=dict(manifest.input_hashes),
        outputs=_redact_sensitive(dict(manifest.outputs)),
        metrics=updated_metrics,
        notes=list(_redact_sensitive(updated_notes)),
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


def _resolve_manifest_config_snapshot(stage: str, config_path: Path | None) -> dict[str, Any]:
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


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for part in command:
        if hide_next:
            if part.strip().lower().rstrip(":") in _AUTH_SCHEMES:
                redacted.append(part)
            else:
                redacted.append("[REDACTED]")
                hide_next = False
            continue
        safe_part = _redact_string(part)
        if safe_part != part:
            redacted.append(safe_part)
            continue
        normalized_part = part.strip().lower().rstrip(":")
        if normalized_part in _AUTH_SCHEMES or normalized_part == "authorization":
            redacted.append(part)
            hide_next = True
            continue
        flag, separator, _value = part.partition("=")
        normalized_flag = flag.lower().replace("_", "-")
        flag_key = normalized_flag.lstrip("-").replace("-", "_")
        if normalized_flag in _SENSITIVE_COMMAND_FLAGS or _is_sensitive_key(flag_key):
            if separator:
                redacted.append(f"{flag}=[REDACTED]")
            else:
                redacted.append(part)
                hide_next = True
            continue
        redacted.append(_redact_string(part))
    return redacted


def _redact_sensitive(value: Any, *, key: str = "") -> Any:
    if key and _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): _redact_sensitive(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _redact_environment(environment: dict[str, Any]) -> dict[str, Any]:
    env_presence = environment.get("env_presence")
    safe = _redact_sensitive(
        {key: value for key, value in environment.items() if key != "env_presence"}
    )
    if isinstance(env_presence, dict):
        safe["env_presence"] = {str(key): bool(value) for key, value in env_presence.items()}
    return safe


def _redact_string(value: str) -> str:
    redacted = _AUTHORIZATION_VALUE_PATTERN.sub("Authorization: [REDACTED]", value)
    redacted = _AUTH_SCHEME_VALUE_PATTERN.sub("[REDACTED_AUTH]", redacted)
    redacted = _SENSITIVE_POSITIONAL_VALUE_PATTERN.sub("[REDACTED]", redacted)
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _is_sensitive_key(key: str) -> bool:
    normalized = key.upper().replace("-", "_")
    return any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS)


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


def _json_payload_sha256(payload: dict[str, Any]) -> str:
    serialized = (
        json.dumps(
            _sort_mapping(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _collect_input_hashes(inputs: dict[str, Any], root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in inputs.items():
        if not isinstance(value, str):
            continue
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        hashes[name] = _sha256_file(path)
        sidecar = path.with_suffix(".schema.json")
        if sidecar.is_file() and sidecar.stat().st_size <= 64 * 1024 * 1024:
            hashes[f"{name}_schema"] = _sha256_file(sidecar)
    return _sort_mapping(hashes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
