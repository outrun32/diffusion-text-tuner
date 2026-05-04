"""CPU-safe tests for the import-safe smoke environment CLI."""

from __future__ import annotations

import importlib
import sys


HEAVY_MODULES = ("torch", "diffusers", "transformers", "paddleocr", "vllm", "mlx_lm")
EXPECTED_CHECKS = ("imports", "cuda", "cache", "model-access", "ocr")


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
    for module_name in ("scripts.smoke_environment", *HEAVY_MODULES):
        sys.modules.pop(module_name, None)

    importlib.import_module("scripts.smoke_environment")

    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules
