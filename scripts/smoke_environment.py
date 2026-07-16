"""Import-safe environment smoke checks for Diffusion Text Tuner.

The default/listing paths avoid importing heavyweight ML packages. Optional checks use
package discovery where possible and keep CUDA/model/OCR probing behind explicit check names.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

CHECKS = ("imports", "platform", "mlx", "mps", "cuda", "cache", "model-access", "ocr")

_LOCAL_IMPORT_MODULES = (
    "src",
    "scripts",
    "src.training.losses",
    "src.training.config",
    "src.prompt_pipeline.generate",
    "scripts.generate_images",
)
_CACHE_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
_HF_AUTH_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN")
_HF_PACKAGES = ("huggingface_hub", "diffusers", "transformers")


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-check CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run explicit environment smoke checks without default model loading.",
    )
    parser.add_argument("--list", action="store_true", help="List available smoke checks.")
    parser.add_argument("--check", choices=CHECKS, help="Run one explicit smoke check.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing optional packages/paths but return success.",
    )
    return parser


def list_checks() -> str:
    """Return a newline-delimited list of available smoke checks."""
    return "\n".join(CHECKS)


def run_check(name: str, *, allow_missing: bool = False) -> int:
    """Run a named smoke check and return a process-style status code."""
    if name == "imports":
        return _check_imports(allow_missing=allow_missing)
    if name == "platform":
        return _check_platform()
    if name == "mlx":
        return _check_mlx(allow_missing=allow_missing)
    if name == "mps":
        return _check_mps(allow_missing=allow_missing)
    if name == "cuda":
        return _check_cuda(allow_missing=allow_missing)
    if name == "cache":
        return _check_cache(allow_missing=allow_missing)
    if name == "model-access":
        return _check_model_access(allow_missing=allow_missing)
    if name == "ocr":
        return _check_ocr(allow_missing=allow_missing)

    print(f"Unknown smoke check: {name}", file=sys.stderr)
    print(f"Available checks: {', '.join(CHECKS)}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """Run the smoke-check CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print(list_checks())
        return 0

    if args.check:
        return run_check(args.check, allow_missing=args.allow_missing)

    parser.print_help()
    return 0


def _check_imports(*, allow_missing: bool) -> int:
    failures: list[str] = []
    python_ok = sys.version_info[:2] == (3, 11)
    _print_status("python", f"{sys.version_info.major}.{sys.version_info.minor}", python_ok)
    if not python_ok:
        failures.append("Python 3.11 is required")

    for module_name in _LOCAL_IMPORT_MODULES:
        available = _is_module_discoverable(module_name)
        _print_status("module", module_name, available)
        if not available:
            failures.append(f"Missing importable module: {module_name}")

    return _result(failures, allow_missing=allow_missing)


def _check_cuda(*, allow_missing: bool) -> int:
    failures: list[str] = []
    try:
        import torch  # noqa: PLC0415  # Explicit CUDA check only.
    except ImportError:
        _print_status("package", "torch", False)
        failures.append("torch is not importable")
        return _result(failures, allow_missing=allow_missing)

    cuda_available = bool(torch.cuda.is_available())
    _print_status("package", "torch", True)
    _print_status("cuda", "torch.cuda.is_available()", cuda_available)
    if not cuda_available:
        failures.append("CUDA is not available through torch")
    return _result(failures, allow_missing=allow_missing)


def _check_platform() -> int:
    from src.runtime.capabilities import inspect_runtime_capabilities

    capabilities = inspect_runtime_capabilities(probe_torch=False)
    print(f"system: {capabilities.system}")
    print(f"machine: {capabilities.machine}")
    print(f"python: {capabilities.python}")
    print(f"apple-silicon: {'yes' if capabilities.apple_silicon else 'no'}")
    return 0


def _check_mlx(*, allow_missing: bool) -> int:
    from src.runtime.capabilities import check_stage_support

    support = check_stage_support("prompt-mlx")
    _print_status("platform", "Apple Silicon", support.capabilities.apple_silicon)
    _print_status("package", "mlx + mlx-lm", support.capabilities.mlx_available)
    failures = list(support.errors)
    if support.ok:
        try:
            import mlx.core as mx
            import mlx_lm

            total = int(mx.sum(mx.array([1, 2])).item())
            _print_status("runtime", "mlx tiny array operation", total == 3)
            _print_status("runtime", "mlx_lm import", bool(mlx_lm))
            if total != 3:
                failures.append("MLX tiny array operation returned an unexpected result")
        except Exception as exc:
            failures.append(f"MLX runtime import/operation failed: {exc}")
    return _result(failures, allow_missing=allow_missing)


def _check_mps(*, allow_missing: bool) -> int:
    from src.runtime.capabilities import inspect_runtime_capabilities

    capabilities = inspect_runtime_capabilities(probe_torch=True)
    _print_status("package", "torch", bool(capabilities.torch_importable))
    _print_status("mps", "torch.backends.mps.is_available()", bool(capabilities.mps_available))
    failures = [] if capabilities.mps_available else ["PyTorch MPS is not available"]
    return _result(failures, allow_missing=allow_missing)


def _check_cache(*, allow_missing: bool) -> int:
    del allow_missing  # Cache reporting is informational unless paths are inaccessible.
    for env_name in _CACHE_ENV_VARS:
        env_value = os.environ.get(env_name)
        if env_value:
            path = Path(env_value).expanduser()
            status = _path_status(path)
            print(f"env {env_name}: set ({status})")
        else:
            print(f"env {env_name}: unset")

    for runtime_path in (Path("outputs"), Path("runs")):
        print(f"path {runtime_path}/: {_path_status(runtime_path)}")
    return 0


def _check_model_access(*, allow_missing: bool) -> int:
    failures: list[str] = []
    for package_name in _HF_PACKAGES:
        available = _is_module_discoverable(package_name)
        _print_status("package", package_name, available)
        if not available:
            failures.append(f"Missing Hugging Face package: {package_name}")

    auth_present = any(os.environ.get(env_name) for env_name in _HF_AUTH_ENV_VARS)
    _print_status("huggingface-auth", "token environment", auth_present)
    for env_name in _CACHE_ENV_VARS:
        _print_status("cache-env", env_name, bool(os.environ.get(env_name)))

    return _result(failures, allow_missing=allow_missing)


def _check_ocr(*, allow_missing: bool) -> int:
    available = _is_module_discoverable("paddleocr")
    _print_status("package", "paddleocr", available)
    failures = [] if available else ["paddleocr is not discoverable"]
    return _result(failures, allow_missing=allow_missing)


def _is_module_discoverable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _path_status(path: Path) -> str:
    if path.exists():
        return "exists" if path.is_dir() else "exists (not a directory)"
    parent = path.parent if path.parent != Path("") else Path(".")
    return "missing; parent exists" if parent.exists() else "missing; parent missing"


def _print_status(kind: str, name: str, ok: bool) -> None:
    status = "ok" if ok else "missing"
    print(f"{kind} {name}: {status}")


def _result(failures: list[str], *, allow_missing: bool) -> int:
    if not failures:
        return 0

    for failure in failures:
        print(f"warning: {failure}", file=sys.stderr)
    return 0 if allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
