from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_NAMES = (
    "prompt generation",
    "image generation",
    "scoring",
    "synthesis",
    "training",
    "evaluation",
    "plotting",
    "run comparison",
    "diagnostics",
    "thesis outputs",
)
HEAVY_OPTIONAL_MODULES = frozenset(
    {
        "diffusers",
        "mlx_lm",
        "paddle",
        "paddleocr",
        "qwen_vl_utils",
        "synthtiger",
        "torchvision",
        "transformers",
    }
)
LEGACY_STRUCTURE_TARGET = "phase7-structure-tests"
EXTENSION_CHECKLIST_STRINGS = (
    "config",
    "artifact/manifest contract",
    "importable module",
    "thin CLI wrapper",
    "CPU-safe tests",
    "command docs",
    "generated-artifact safety",
)


def _read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_extension_registry_lists_all_supported_extension_points() -> None:
    from src.toolkit.extension_points import ExtensionPoint, list_extension_points

    entries = list_extension_points()

    assert isinstance(entries, tuple)
    assert tuple(entry.name for entry in entries) == EXPECTED_STAGE_NAMES
    assert all(isinstance(entry, ExtensionPoint) for entry in entries)

    for entry in entries:
        assert entry.purpose
        assert entry.implementation_module
        assert entry.config_home
        assert entry.docs_path
        assert entry.test_target
        assert entry.generated_artifact_notes


def test_extension_registry_config_homes_exist() -> None:
    from src.toolkit.extension_points import list_extension_points

    entries = {entry.name: entry for entry in list_extension_points()}
    path_homes = {
        "prompt generation": "configs/prompts/",
        "image generation": "configs/experiments/evaluation/",
        "scoring": "configs/experiments/reward/",
        "synthesis": "configs/experiments/synthesis/",
        "training": "configs/experiments/",
        "evaluation": "configs/experiments/evaluation/",
        "thesis outputs": "configs/experiments/evaluation/",
    }

    for name, expected_home in path_homes.items():
        assert entries[name].config_home == expected_home
        assert (ROOT / expected_home).is_dir()


def test_get_extension_point_returns_scoring_and_rejects_unknown_names() -> None:
    from src.toolkit.extension_points import get_extension_point

    scoring = get_extension_point("scoring")

    assert scoring.name == "scoring"
    assert scoring.implementation_module == "src.scoring.pipeline"
    assert scoring.thin_script == "scripts/score_images.py"

    try:
        get_extension_point("unknown stage")
    except KeyError as exc:
        assert "unknown extension point: unknown stage" in str(exc)
    else:  # pragma: no cover - defensive clarity for assertion failures
        raise AssertionError("get_extension_point should reject unknown stage names")


def test_registered_modules_import_without_new_heavy_optional_stacks() -> None:
    from src.toolkit.extension_points import list_extension_points

    before = set(sys.modules)

    for entry in list_extension_points():
        importlib.import_module(entry.implementation_module)

    newly_imported = set(sys.modules) - before
    newly_imported_heavy = {
        module_name
        for module_name in newly_imported
        if module_name.split(".", maxsplit=1)[0] in HEAVY_OPTIONAL_MODULES
    }

    assert newly_imported_heavy == set()


def test_public_structure_checks_do_not_advertise_legacy_alias() -> None:
    guide = _read_repo_file("docs/structure_and_extension.md")
    commands = _read_repo_file("docs/commands.md")
    readme = _read_repo_file("README.md")

    assert "make check" in guide
    assert "## Structure and extension checks" in commands
    assert (
        "uv run pytest tests/test_structure_extension_docs.py "
        "tests/test_generation_pipeline_contracts.py"
    ) in commands
    assert all(LEGACY_STRUCTURE_TARGET not in content for content in (guide, commands, readme))


def test_extension_checklist_names_required_future_pipeline_steps() -> None:
    guide = _read_repo_file("docs/structure_and_extension.md")

    required_strings = (
        "## Extension checklist",
        *EXTENSION_CHECKLIST_STRINGS,
        "list_extension_points",
        "src/toolkit/extension_points.py",
    )
    missing = [value for value in required_strings if value not in guide]

    assert missing == []


def test_structure_guide_mirrors_registry_entries() -> None:
    from src.toolkit.extension_points import list_extension_points

    guide = _read_repo_file("docs/structure_and_extension.md")

    missing = []
    for entry in list_extension_points():
        for value in (
            entry.name,
            entry.implementation_module,
            entry.config_home,
            entry.docs_path,
            entry.test_target,
        ):
            if value not in guide:
                missing.append(value)
        if entry.thin_script and entry.thin_script not in guide:
            missing.append(entry.thin_script)

    assert missing == []


def test_makefile_legacy_structure_alias_selects_cpu_safe_tests() -> None:
    makefile = _read_repo_file("Makefile")

    assert ".PHONY:" in makefile
    assert LEGACY_STRUCTURE_TARGET in makefile
    assert "uv run pytest" in makefile
    assert "tests/test_structure_extension_docs.py" in makefile
    assert "tests/test_generation_pipeline_contracts.py" in makefile
    assert "tests/test_scoring_pipeline_contracts.py" in makefile
    assert "tests/test_synthesis_pipeline_contracts.py" in makefile
    assert "tests/test_plotting_pipeline_contracts.py" in makefile
    assert "tests/test_extension_points_docs.py" in makefile
