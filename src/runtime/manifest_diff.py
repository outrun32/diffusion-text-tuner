"""CPU-safe local run manifest diff helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.runtime.manifests import RunManifest, load_run_manifest

DIFF_SCHEMA_VERSION = "run-manifest-diff/v1"

_CHANGE_SECTIONS = (
    "config_changes",
    "data_source_changes",
    "reward_changes",
    "seed_changes",
    "inference_changes",
    "metric_changes",
    "artifact_changes",
)
_DATA_SOURCE_KEYS = frozenset(
    {
        "scores_csv",
        "latents_dir",
        "text_embeds_dir",
        "data_dir",
        "selected_samples_path",
        "preference_pairs_path",
    }
)
_INFERENCE_KEYS = frozenset(
    {
        "num_inference_steps",
        "guidance_scale",
        "resolution",
        "sample_prompt",
        "sample_target_text",
        "prompt_embedding_padding",
    }
)


def compare_run_manifests(
    left: str | Path | RunManifest,
    right: str | Path | RunManifest,
) -> dict[str, Any]:
    """Compare two local run manifests without loading model or training dependencies."""

    left_manifest = _coerce_manifest(left)
    right_manifest = _coerce_manifest(right)
    sections = {section: {} for section in _CHANGE_SECTIONS}

    for key, change in _diff_mappings(
        _comparison_config(left_manifest.config_snapshot),
        _comparison_config(right_manifest.config_snapshot),
    ).items():
        sections[_config_section(key)][key] = change

    sections["metric_changes"] = _diff_mappings(left_manifest.metrics, right_manifest.metrics)
    sections["artifact_changes"] = _diff_mappings(left_manifest.outputs, right_manifest.outputs)

    payload: dict[str, Any] = {
        "schema_version": DIFF_SCHEMA_VERSION,
        "left_run_id": left_manifest.run_id,
        "right_run_id": right_manifest.run_id,
        "left_manifest_path": str(left_manifest.manifest_path),
        "right_manifest_path": str(right_manifest.manifest_path),
        **sections,
        "environment_changes": _diff_mappings(
            _presence_only_environment(left_manifest.environment),
            _presence_only_environment(right_manifest.environment),
        ),
    }
    return _sort_mapping(payload)


def format_manifest_diff_markdown(diff: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown summary for a manifest diff payload."""

    lines = [
        "# Run manifest diff",
        "",
        f"Schema: {diff.get('schema_version', DIFF_SCHEMA_VERSION)}",
        f"Left run: {diff.get('left_run_id', 'unknown')}",
        f"Right run: {diff.get('right_run_id', 'unknown')}",
    ]
    left_path = diff.get("left_manifest_path")
    right_path = diff.get("right_manifest_path")
    if left_path is not None:
        lines.append(f"Left manifest: {left_path}")
    if right_path is not None:
        lines.append(f"Right manifest: {right_path}")

    for section in _CHANGE_SECTIONS:
        changes = diff.get(section, {})
        if not changes:
            continue
        lines.extend(
            ["", f"## {_section_title(section)}", "", "| Key | Left | Right |", "|---|---|---|"]
        )
        for key, change in sorted(dict(changes).items()):
            if not isinstance(change, Mapping):
                continue
            lines.append(
                "| "
                f"{_markdown_cell(key)} | "
                f"{_markdown_cell(_json_value(change.get('left')))} | "
                f"{_markdown_cell(_json_value(change.get('right')))} |"
            )
    return "\n".join(lines) + "\n"


def _coerce_manifest(value: str | Path | RunManifest) -> RunManifest:
    if isinstance(value, RunManifest):
        return value
    return load_run_manifest(value)


def _comparison_config(config_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    raw_config = config_snapshot.get("raw_config")
    source = raw_config if isinstance(raw_config, Mapping) else config_snapshot
    return _flatten_mapping(source)


def _diff_mappings(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    left_flat = _flatten_mapping(left)
    right_flat = _flatten_mapping(right)
    changes: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left_flat) | set(right_flat)):
        left_value = left_flat.get(key)
        right_value = right_flat.get(key)
        if left_value != right_value:
            changes[key] = {"left": _json_safe(left_value), "right": _json_safe(right_value)}
    return changes


def _config_section(key: str) -> str:
    leaf_key = key.rsplit(".", maxsplit=1)[-1]
    if leaf_key in _DATA_SOURCE_KEYS:
        return "data_source_changes"
    if "reward" in key or "scorer" in key or "score_column" in key:
        return "reward_changes"
    if leaf_key == "seeds" or key.endswith("seed"):
        return "seed_changes"
    if leaf_key in _INFERENCE_KEYS:
        return "inference_changes"
    return "config_changes"


def _presence_only_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    env_presence = environment.get("env_presence")
    if isinstance(env_presence, Mapping):
        for name, value in env_presence.items():
            safe[f"env_presence.{name}"] = bool(value)

    cache = environment.get("cache")
    if isinstance(cache, Mapping):
        for name, value in cache.items():
            if isinstance(value, Mapping):
                safe[f"cache.{name}.present"] = bool(value.get("present"))
            else:
                safe[f"cache.{name}.present"] = bool(value)
    return safe


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, next_key))
        else:
            flattened[next_key] = item
    return flattened


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sort_mapping({str(key): _json_safe(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mapping(item) for item in value]
    return value


def _section_title(section: str) -> str:
    return section.replace("_", " ").removesuffix(" changes").capitalize() + " changes"


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
