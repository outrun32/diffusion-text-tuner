"""Prevent undocumented script/config entry points from accumulating again."""

from __future__ import annotations

from pathlib import Path


def _corpus(paths: list[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _is_named(path: Path, corpus: str, *, module_prefix: str | None = None) -> bool:
    candidates = {path.as_posix(), path.name, path.stem}
    if module_prefix is not None:
        candidates.add(module_prefix + "." + ".".join(path.with_suffix("").parts[1:]))
    return any(candidate in corpus for candidate in candidates)


def test_every_script_entrypoint_is_classified() -> None:
    corpus = _corpus(
        [
            Path("README.md"),
            Path("scripts/README.md"),
            Path("docs/pipeline_inventory.md"),
            Path("docs/commands.md"),
            Path("experiments/README.md"),
        ]
    )
    scripts = [
        path
        for path in sorted(Path("scripts").rglob("*"))
        if path.is_file() and path.name != "__init__.py" and "__pycache__" not in path.parts
    ]

    undocumented = [
        path.as_posix() for path in scripts if not _is_named(path, corpus, module_prefix="scripts")
    ]

    assert undocumented == []


def test_every_config_file_is_classified() -> None:
    corpus = _corpus(
        [
            Path("README.md"),
            *sorted(Path("configs").rglob("README.md")),
            Path("docs/commands.md"),
            Path("docs/pipeline_inventory.md"),
        ]
    )
    configs = [
        path
        for path in sorted(Path("configs").rglob("*"))
        if path.is_file() and path.name != "README.md"
    ]

    undocumented = [path.as_posix() for path in configs if not _is_named(path, corpus)]

    assert undocumented == []
