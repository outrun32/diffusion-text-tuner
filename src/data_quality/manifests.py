"""CPU-safe dataset manifest helpers for prompt and data-quality artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.runtime.reproducibility import collect_git_state, collect_model_revisions

DATASET_MANIFEST_SCHEMA_VERSION = "dataset-manifest/v1"
SAFE_TEXT_SUFFIXES = frozenset({".txt", ".csv", ".json", ".jsonl", ".md", ".yaml", ".yml"})
UNSAFE_GENERATED_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".png", ".jpg", ".jpeg"}
)
DEFAULT_MAX_HASH_BYTES = 10 * 1024 * 1024


class DatasetManifestError(ValueError):
    """Raised when dataset manifest creation or loading fails."""


@dataclass(frozen=True)
class DatasetManifest:
    """Serializable dataset manifest provenance contract."""

    dataset_kind: str
    dataset_paths: list[str]
    created_at: str
    git: dict[str, Any]
    config: dict[str, Any]
    seed_strategy: dict[str, Any]
    source_hashes: dict[str, dict[str, Any]]
    filtering_stats: dict[str, Any]
    output_counts: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = DATASET_MANIFEST_SCHEMA_VERSION

    def to_json_payload(self) -> dict[str, Any]:
        return _sort_mapping(
            {
                "schema_version": self.schema_version,
                "dataset_kind": self.dataset_kind,
                "created_at": self.created_at,
                "dataset_paths": list(self.dataset_paths),
                "source_hashes": self.source_hashes,
                "config": self.config,
                "seed_strategy": self.seed_strategy,
                "git": self.git,
                "models": self.models,
                "filtering_stats": self.filtering_stats,
                "output_counts": self.output_counts,
                "metadata": self.metadata,
            }
        )


def hash_source_file(
    path: str | Path,
    *,
    safe_hash_inputs: Iterable[str | Path] | None = None,
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
) -> dict[str, Any]:
    """Hash a small safe text source, or reference unsafe/generated artifacts by path."""

    source_path = Path(path)
    safe_paths = {Path(item).resolve() for item in safe_hash_inputs or []}
    payload_path = str(source_path)
    if not source_path.is_file():
        return {"path": payload_path, "hashed": False, "reason": "missing file"}

    forced_safe = source_path.resolve() in safe_paths
    if not forced_safe and _is_unsafe_generated_or_binary(source_path):
        return {
            "path": payload_path,
            "hashed": False,
            "reason": "unsafe generated or binary artifact",
        }
    file_size = source_path.stat().st_size
    if not forced_safe and file_size > max_hash_bytes:
        return {"path": payload_path, "hashed": False, "reason": "file exceeds safe hash size"}

    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": payload_path,
        "hashed": True,
        "sha256": digest.hexdigest(),
        "size_bytes": file_size,
    }


def create_dataset_manifest(
    *,
    dataset_kind: str,
    dataset_paths: Iterable[str | Path],
    config_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    seed_strategy: Mapping[str, Any] | None = None,
    source_paths: Iterable[str | Path] | None = None,
    safe_hash_inputs: Iterable[str | Path] | None = None,
    filtering_stats: Mapping[str, Any] | None = None,
    output_counts: Mapping[str, Any] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    """Create a dataset manifest payload with provenance and source hashes."""

    if not dataset_kind:
        raise DatasetManifestError("dataset_kind must be a non-empty string")
    normalized_dataset_paths = sorted(str(Path(path)) for path in dataset_paths)
    if not normalized_dataset_paths:
        raise DatasetManifestError("dataset_paths must contain at least one path")

    config_payload = _build_config_payload(config_path, config_snapshot)
    snapshot = (
        config_payload.get("snapshot")
        if isinstance(config_payload.get("snapshot"), dict)
        else {}
    )
    all_source_paths = _merge_source_paths(config_path, source_paths)
    source_hashes = {
        str(Path(source)): hash_source_file(source, safe_hash_inputs=safe_hash_inputs)
        for source in sorted(all_source_paths, key=lambda value: str(Path(value)))
    }
    models = _sort_mapping({**collect_model_revisions(snapshot), **dict(model_metadata or {})})
    return DatasetManifest(
        dataset_kind=dataset_kind,
        dataset_paths=normalized_dataset_paths,
        created_at=_utc_now(),
        git=collect_git_state(Path.cwd() if root is None else root),
        config=config_payload,
        seed_strategy=_sort_mapping(dict(seed_strategy or {})),
        source_hashes=source_hashes,
        filtering_stats=_sort_mapping(dict(filtering_stats or {})),
        output_counts=_sort_mapping(dict(output_counts or {})),
        models=models,
        metadata=_sort_mapping(dict(metadata or {})),
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate a dataset manifest JSON file."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"{manifest_path}: malformed JSON manifest") from exc
    except OSError as exc:
        raise DatasetManifestError(f"{manifest_path}: could not read manifest") from exc
    if not isinstance(payload, dict):
        raise DatasetManifestError(f"{manifest_path}: manifest must be a JSON object")

    schema_version = _required_str(payload, "schema_version", manifest_path)
    if schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
        raise DatasetManifestError(
            f"{manifest_path}: schema_version must be {DATASET_MANIFEST_SCHEMA_VERSION}"
        )
    return DatasetManifest(
        schema_version=schema_version,
        dataset_kind=_required_str(payload, "dataset_kind", manifest_path),
        created_at=_required_str(payload, "created_at", manifest_path),
        dataset_paths=_required_str_list(payload, "dataset_paths", manifest_path),
        source_hashes=_required_dict(payload, "source_hashes", manifest_path),
        config=_required_dict(payload, "config", manifest_path),
        seed_strategy=_required_dict(payload, "seed_strategy", manifest_path),
        git=_required_dict(payload, "git", manifest_path),
        models=_required_dict(payload, "models", manifest_path),
        filtering_stats=_required_dict(payload, "filtering_stats", manifest_path),
        output_counts=_required_dict(payload, "output_counts", manifest_path),
        metadata=_dict_field(payload, "metadata"),
    )


def write_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> Path:
    """Write a dataset manifest JSON file with deterministic key ordering."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_json_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_config_payload(
    config_path: str | Path | None,
    config_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    snapshot = dict(config_snapshot or {})
    payload: dict[str, Any] = {"path": None, "sha256": None, "snapshot": _sort_mapping(snapshot)}
    if config_path is None:
        return payload
    path = Path(config_path)
    config_hash = hash_source_file(path, safe_hash_inputs={path})
    if not snapshot:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetManifestError(f"{path}: malformed config JSON") from exc
        except OSError as exc:
            raise DatasetManifestError(f"{path}: could not read config") from exc
        if not isinstance(loaded, dict):
            raise DatasetManifestError(f"{path}: config snapshot must be a JSON object")
        snapshot = loaded
    payload.update(
        {
            "path": str(path),
            "sha256": config_hash.get("sha256"),
            "snapshot": _sort_mapping(snapshot),
        }
    )
    return payload


def _merge_source_paths(
    config_path: str | Path | None,
    source_paths: Iterable[str | Path] | None,
) -> list[str | Path]:
    merged = list(source_paths or [])
    if config_path is not None:
        merged.append(config_path)
    return list(dict.fromkeys(merged))


def _is_unsafe_generated_or_binary(path: Path) -> bool:
    if path.suffix.lower() in UNSAFE_GENERATED_SUFFIXES:
        return True
    if path.suffix.lower() in SAFE_TEXT_SUFFIXES:
        return False
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\0" in sample


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _required_str(payload: Mapping[str, Any], key: str, manifest_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetManifestError(f"{manifest_path}: {key} must be a non-empty string")
    return value


def _required_str_list(payload: Mapping[str, Any], key: str, manifest_path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DatasetManifestError(f"{manifest_path}: {key} must be a list of strings")
    return list(value)


def _required_dict(payload: Mapping[str, Any], key: str, manifest_path: Path) -> dict[str, Any]:
    if key not in payload or not isinstance(payload[key], dict):
        raise DatasetManifestError(f"{manifest_path}: {key} must be an object")
    return dict(payload[key])


def _dict_field(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mapping(item) for item in value]
    return value
