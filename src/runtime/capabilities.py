"""Platform capability checks for local and remote toolkit stages.

The project deliberately separates cheap, platform-neutral metadata work from
FLUX execution.  Importing this module never imports model libraries.  Torch is
only imported when a caller explicitly asks for an execution-readiness probe.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Literal

CUDA_ONLY_STAGES = frozenset({"generate", "sft", "dpo", "masked-sft", "refl"})


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Small, JSON-safe description of the current execution host."""

    system: str
    machine: str
    python: str
    apple_silicon: bool
    mlx_available: bool
    torch_importable: bool | None
    cuda_available: bool | None
    cuda_bf16_supported: bool | None
    mps_available: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageSupport:
    """Execution support result for one concrete pipeline stage."""

    stage: str
    runtime: str
    capabilities: RuntimeCapabilities
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "runtime": self.runtime,
            "capabilities": self.capabilities.to_dict(),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def inspect_runtime_capabilities(*, probe_torch: bool = False) -> RuntimeCapabilities:
    """Inspect platform and optional accelerator availability.

    ``probe_torch=False`` keeps discovery import-safe.  Execution preflights use
    ``probe_torch=True`` so a missing CUDA runtime fails before model loading.
    """

    system = platform.system()
    machine = platform.machine().lower()
    apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}
    mlx_available = _module_available("mlx") and _module_available("mlx_lm")

    torch_importable: bool | None = None
    cuda_available: bool | None = None
    cuda_bf16_supported: bool | None = None
    mps_available: bool | None = None
    if probe_torch:
        try:
            import torch
        except Exception:
            torch_importable = False
        else:
            torch_importable = True
            cuda_available = bool(torch.cuda.is_available())
            cuda_bf16_supported = bool(torch.cuda.is_bf16_supported()) if cuda_available else False
            mps_backend = getattr(torch.backends, "mps", None)
            mps_available = bool(mps_backend and mps_backend.is_available())

    return RuntimeCapabilities(
        system=system,
        machine=machine,
        python=sys.version.split()[0],
        apple_silicon=apple_silicon,
        mlx_available=mlx_available,
        torch_importable=torch_importable,
        cuda_available=cuda_available,
        cuda_bf16_supported=cuda_bf16_supported,
        mps_available=mps_available,
    )


def check_stage_support(
    stage: str,
    *,
    scorer: Literal["vlm", "ocr", "both"] = "both",
    ocr_device: Literal["cpu", "gpu"] = "cpu",
    mixed_precision: str | None = None,
) -> StageSupport:
    """Return whether the current host can execute a stage as configured."""

    capabilities = inspect_runtime_capabilities(probe_torch=True)
    errors: list[str] = []
    warnings: list[str] = []
    runtime = "cpu"

    needs_cuda = stage in CUDA_ONLY_STAGES
    if stage == "score":
        needs_cuda = scorer in {"vlm", "both"} or ocr_device == "gpu"
        runtime = "cuda" if needs_cuda else "cpu-ocr"
        if scorer in {"ocr", "both"} and not _module_available("paddleocr"):
            errors.append(
                "OCR scoring requires the optional PaddleOCR runtime; install the ocr extra"
            )
    elif needs_cuda:
        runtime = "cuda"
    elif stage == "prompt-mlx":
        runtime = "mlx"
        if not capabilities.apple_silicon:
            errors.append("The MLX prompt backend requires Apple Silicon macOS")
        if not capabilities.mlx_available:
            errors.append("The MLX prompt backend requires the mlx extra")

    if needs_cuda and not capabilities.cuda_available:
        errors.append(
            f"{stage} requires a CUDA host in this repository; MLX and MPS are not FLUX "
            "training backends here"
        )
        if capabilities.apple_silicon:
            warnings.append(
                "Run this stage on a Linux/CUDA machine and use the Mac for tests, prompt "
                "generation, planning, and recorded-result analysis"
            )
    elif needs_cuda:
        if capabilities.system != "Linux":
            errors.append(f"{stage} is supported only on Linux/CUDA hosts in this repository")
        requires_bf16 = (
            stage in {"generate", "score"} and (stage != "score" or scorer in {"vlm", "both"})
        ) or mixed_precision == "bf16"
        if requires_bf16 and not capabilities.cuda_bf16_supported:
            errors.append(
                f"{stage} is configured for BF16, but the selected CUDA device does not support it"
            )
        if stage == "score" and ocr_device == "gpu" and not _paddle_cuda_available():
            errors.append("OCR GPU scoring requires a PaddlePaddle build compiled with CUDA")

    return StageSupport(
        stage=stage,
        runtime=runtime,
        capabilities=capabilities,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _paddle_cuda_available() -> bool:
    try:
        import paddle
    except Exception:
        return False
    try:
        return bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False
