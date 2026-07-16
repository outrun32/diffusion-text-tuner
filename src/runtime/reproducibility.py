"""CPU-safe reproducibility metadata collectors for local run manifests."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

SECRET_ENV_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASS", "CREDENTIAL")
DEFAULT_SECRET_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)
DEFAULT_CACHE_ENV_NAMES = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME")
DEFAULT_PACKAGE_NAMES = (
    "accelerate",
    "diffusers",
    "mlx",
    "mlx-lm",
    "paddleocr",
    "paddlepaddle",
    "peft",
    "pydantic",
    "pytest",
    "torch",
    "transformers",
)


def collect_git_state(root: str | Path | None = None) -> dict[str, Any]:
    """Collect local git commit and concise dirty/untracked state without failing hard."""

    root_path = Path.cwd() if root is None else Path(root)
    commit = _run_git(root_path, "rev-parse", "HEAD")
    status = _run_git(root_path, "status", "--porcelain")
    if commit is None:
        return {"available": False, "commit": None, "dirty": None, "untracked_count": None}

    status_lines = [] if status is None else [line for line in status.splitlines() if line]
    diff = _run_git(root_path, "diff", "--binary", "HEAD") or ""
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status_lines),
        "untracked_count": sum(1 for line in status_lines if line.startswith("?? ")),
        "working_tree_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def collect_environment_summary(
    *,
    package_names: tuple[str, ...] = DEFAULT_PACKAGE_NAMES,
    secret_env_names: tuple[str, ...] = DEFAULT_SECRET_ENV_NAMES,
    cache_env_names: tuple[str, ...] = DEFAULT_CACHE_ENV_NAMES,
) -> dict[str, Any]:
    """Collect Python/platform/package/cache metadata without serializing secret values."""

    env_presence = {
        name: name in os.environ
        for name in sorted(_secret_environment_names(secret_env_names=secret_env_names))
    }
    cache = {name: {"present": name in os.environ} for name in sorted(cache_env_names)}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "packages": _collect_package_versions(package_names),
        "cuda": _collect_cuda_presence(),
        "accelerators": _collect_accelerator_presence(),
        "cache": cache,
        "env_presence": env_presence,
    }


def collect_model_revisions(config_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract configured model IDs and optional revisions from a resolved config snapshot."""

    models: dict[str, Any] = {}
    source = _metadata_source(config_snapshot)
    for key, value in sorted(_flatten_mapping(source).items()):
        if key.endswith("model_id") and isinstance(value, str):
            models[key] = value
        if key.endswith("model_revision") and (isinstance(value, str) or value is None):
            models[key] = value
    if "model_id" in models and "model_revision" not in models:
        models["model_revision"] = None
    return models


def collect_seeds(config_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract seed-like scalar fields from a resolved config snapshot."""

    seeds: dict[str, Any] = {}
    source = _metadata_source(config_snapshot)
    for key, value in sorted(_flatten_mapping(source).items()):
        if key.endswith("seed") and isinstance(value, int):
            seeds[key] = value
    return seeds


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _secret_environment_names(*, secret_env_names: tuple[str, ...]) -> set[str]:
    names = set(secret_env_names)
    for name in os.environ:
        upper = name.upper()
        if any(marker in upper for marker in SECRET_ENV_MARKERS):
            names.add(name)
    return names


def _collect_package_versions(package_names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in sorted(package_names):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _collect_cuda_presence() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415 - optional runtime dependency, only metadata queried.
    except Exception:  # noqa: BLE001 - absence of torch is valid for CPU-safe metadata.
        return {"torch_importable": False, "available": None, "device_count": None}
    return {
        "torch_importable": True,
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "bf16_supported": (
            bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
        ),
    }


def _collect_accelerator_presence() -> dict[str, Any]:
    from src.runtime.capabilities import inspect_runtime_capabilities

    capabilities = inspect_runtime_capabilities(probe_torch=True)
    return {
        "apple_silicon": capabilities.apple_silicon,
        "mlx_available": capabilities.mlx_available,
        "mps_available": capabilities.mps_available,
        "cuda_bf16_supported": capabilities.cuda_bf16_supported,
    }


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value} if prefix else {}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_mapping(item, next_key))
        else:
            flattened[next_key] = item
    return flattened


def _metadata_source(config_snapshot: dict[str, Any]) -> dict[str, Any]:
    raw_config = config_snapshot.get("raw_config")
    return raw_config if isinstance(raw_config, dict) else config_snapshot
