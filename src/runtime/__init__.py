"""Shared runtime helpers with lazy public exports.

Importing a lightweight submodule such as ``src.runtime.reproducibility`` must not
pull in PyTorch. Tensor validation remains available through ``validate_artifacts``
and loads PyTorch only when a tensor file actually needs inspection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ArtifactReport": (".artifacts", "ArtifactReport"),
    "ArtifactValidationError": (".artifacts", "ArtifactValidationError"),
    "RuntimeConfigError": (".config_io", "RuntimeConfigError"),
    "RuntimePaths": (".paths", "RuntimePaths"),
    "RunManifest": (".manifests", "RunManifest"),
    "assert_artifact_git_safety": (".paths", "assert_artifact_git_safety"),
    "collect_environment_summary": (".reproducibility", "collect_environment_summary"),
    "collect_git_state": (".reproducibility", "collect_git_state"),
    "collect_model_revisions": (".reproducibility", "collect_model_revisions"),
    "create_run_manifest": (".manifests", "create_run_manifest"),
    "load_stage_config": (".config_io", "load_stage_config"),
    "load_run_manifest": (".manifests", "load_run_manifest"),
    "resolve_config_snapshot": (".config_io", "resolve_config_snapshot"),
    "resolve_stage_paths": (".paths", "resolve_stage_paths"),
    "update_run_manifest": (".manifests", "update_run_manifest"),
    "validate_artifacts": (".artifacts", "validate_artifacts"),
    "validate_path_policy": (".config_io", "validate_path_policy"),
    "validate_source_manifest": (".manifests", "validate_source_manifest"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public helper on first access while preserving the package API."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
