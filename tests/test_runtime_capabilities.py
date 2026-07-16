"""Tests for explicit Mac/MLX and CUDA execution boundaries."""

from __future__ import annotations

from src.runtime import capabilities


def _mac_capabilities(*, mlx_available: bool = True) -> capabilities.RuntimeCapabilities:
    return capabilities.RuntimeCapabilities(
        system="Darwin",
        machine="arm64",
        python="3.11.15",
        apple_silicon=True,
        mlx_available=mlx_available,
        torch_importable=True,
        cuda_available=False,
        cuda_bf16_supported=False,
        mps_available=True,
    )


def test_cuda_training_is_rejected_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "inspect_runtime_capabilities",
        lambda **_kwargs: _mac_capabilities(),
    )

    support = capabilities.check_stage_support("sft")

    assert support.ok is False
    assert support.runtime == "cuda"
    assert "MLX and MPS are not FLUX training backends" in support.errors[0]


def test_mlx_prompt_generation_is_supported_on_prepared_mac(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "inspect_runtime_capabilities",
        lambda **_kwargs: _mac_capabilities(),
    )

    support = capabilities.check_stage_support("prompt-mlx")

    assert support.ok is True
    assert support.runtime == "mlx"


def test_cpu_ocr_does_not_require_cuda(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "inspect_runtime_capabilities",
        lambda **_kwargs: _mac_capabilities(),
    )
    monkeypatch.setattr(capabilities, "_module_available", lambda _name: True)

    support = capabilities.check_stage_support("score", scorer="ocr", ocr_device="cpu")

    assert support.ok is True
    assert support.runtime == "cpu-ocr"


def test_vlm_scoring_is_rejected_without_cuda(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "inspect_runtime_capabilities",
        lambda **_kwargs: _mac_capabilities(),
    )

    support = capabilities.check_stage_support("score", scorer="vlm")

    assert support.ok is False
    assert support.runtime == "cuda"


def test_bf16_training_is_rejected_on_older_cuda_device(monkeypatch):
    older_cuda = capabilities.RuntimeCapabilities(
        system="Linux",
        machine="x86_64",
        python="3.11.15",
        apple_silicon=False,
        mlx_available=False,
        torch_importable=True,
        cuda_available=True,
        cuda_bf16_supported=False,
        mps_available=False,
    )
    monkeypatch.setattr(
        capabilities,
        "inspect_runtime_capabilities",
        lambda **_kwargs: older_cuda,
    )

    support = capabilities.check_stage_support("sft", mixed_precision="bf16")

    assert support.ok is False
    assert "does not support" in support.errors[0]


def test_windows_cuda_does_not_satisfy_linux_execution_contract(monkeypatch):
    windows_cuda = capabilities.RuntimeCapabilities(
        system="Windows",
        machine="amd64",
        python="3.11.15",
        apple_silicon=False,
        mlx_available=False,
        torch_importable=True,
        cuda_available=True,
        cuda_bf16_supported=True,
        mps_available=False,
    )
    monkeypatch.setattr(
        capabilities,
        "inspect_runtime_capabilities",
        lambda **_kwargs: windows_cuda,
    )

    support = capabilities.check_stage_support("generate")

    assert support.ok is False
    assert any("only on Linux" in error for error in support.errors)
