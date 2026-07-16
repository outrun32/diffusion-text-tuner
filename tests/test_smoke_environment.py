"""CPU-safe tests for the import-safe smoke environment CLI."""

from __future__ import annotations

import importlib
import sys

HEAVY_MODULES = ("torch", "diffusers", "transformers", "paddleocr", "vllm", "mlx_lm")
EXPECTED_CHECKS = (
    "imports",
    "platform",
    "mlx",
    "mps",
    "cuda",
    "cache",
    "model-access",
    "ocr",
)


def test_list_outputs_all_checks():
    from scripts.smoke_environment import list_checks

    output = list_checks()

    for check_name in EXPECTED_CHECKS:
        assert check_name in output


def test_main_list_prints_checks(capsys):
    from scripts.smoke_environment import main

    result = main(["--list"])

    captured = capsys.readouterr()
    assert result == 0
    for check_name in EXPECTED_CHECKS:
        assert check_name in captured.out


def test_unknown_check_returns_nonzero():
    from scripts.smoke_environment import run_check

    assert run_check("not-a-check") != 0


def test_import_has_no_heavy_side_effects():
    removed_modules = {}
    for module_name in ("scripts.smoke_environment", *HEAVY_MODULES):
        if module_name in sys.modules:
            removed_modules[module_name] = sys.modules.pop(module_name)

    try:
        importlib.import_module("scripts.smoke_environment")

        for module_name in HEAVY_MODULES:
            assert module_name not in sys.modules
    finally:
        sys.modules.pop("scripts.smoke_environment", None)
        sys.modules.update(removed_modules)


def test_platform_check_is_safe_and_reports_host(capsys):
    from scripts.smoke_environment import run_check

    assert run_check("platform") == 0
    output = capsys.readouterr().out
    assert "system:" in output
    assert "machine:" in output
    assert "python:" in output


def test_manual_profiler_wrapper_has_no_import_time_model_side_effects(monkeypatch):
    for module_name in ("scripts.profile_step", "diffusers", "peft"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("scripts.profile_step")

    assert "diffusers" not in sys.modules
    assert "peft" not in sys.modules
