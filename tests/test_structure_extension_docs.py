from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_structure_guide_names_required_repository_homes_and_boundaries() -> None:
    guide = _read_repo_file("docs/structure_and_extension.md")
    normalized_guide = " ".join(guide.split())

    required_strings = [
        "# Structure and Extension Guide",
        "## Canonical repository homes",
        "Reusable source modules: `src/`",
        "Thin CLI wrappers: `scripts/`",
        "Cluster launchers: `scripts/cluster/`",
        "Synthesis helpers: `scripts/synth/`",
        "Thesis plotting helpers: `scripts.plot_metrics` and `scripts.build_thesis_outputs`",
        "Configuration contracts: `configs/`",
        "Experiment config variants: `configs/experiments/`",
        "CPU-safe tests: `tests/`",
        "Historical experiments: `experiments/`",
        "Generated runtime outputs: ignored `outputs/`, `runs/`, and generated `data/` subtrees",
        "Thesis evidence artifacts: recorded manifests, reports, plots, contact sheets, and bundles",
        "generated images, tensors, checkpoints, logs, reports, contact sheets",
        "thesis bundles, and private run outputs remain runtime artifacts",
        "Do not commit generated runtime outputs unless they are intentionally tiny reviewed fixtures or documentation assets.",
    ]

    missing = [value for value in required_strings if value not in normalized_guide]
    assert missing == []


def test_structure_guide_publishes_extension_rules_for_future_work() -> None:
    guide = _read_repo_file("docs/structure_and_extension.md")

    required_strings = [
        "## Extension rules before adding new work",
        "Before adding a new experiment, trainer, reward variant, dataset, pipeline, plot, or thesis-output step",
        "New experiment",
        "New trainer",
        "New reward variant",
        "New dataset or data pipeline",
        "New generation, scoring, synthesis, evaluation, or plotting pipeline",
        "New thesis-output step",
        "prefer importable implementation modules under `src/` with thin wrappers in `scripts/`",
        "document CPU-safe verification and explicit GPU/model/OCR/SLURM prerequisites",
    ]

    missing = [value for value in required_strings if value not in guide]
    assert missing == []


def test_scripts_readme_classifies_script_families_and_runtime_boundaries() -> None:
    scripts_readme = _read_repo_file("scripts/README.md")

    required_strings = [
        "# Script Navigation",
        "## Supported wrappers versus diagnostics",
        "Thin CLI wrappers",
        "Supported wrappers backed by importable modules",
        "Manual diagnostics",
        "Cluster jobs",
        "Synthesis helpers",
        "Plotting helpers",
        "Historical/experiment scripts",
        "diagnostics and generated outputs are local/runtime surfaces that can contain private prompt text, scores, paths, or run metadata",
        "Generated runtime outputs are not source",
        "docs/structure_and_extension.md",
    ]

    missing = [value for value in required_strings if value not in scripts_readme]
    assert missing == []


def test_readme_links_phase7_structure_extension_guide() -> None:
    readme = _read_repo_file("README.md")

    required_strings = [
        "docs/structure_and_extension.md",
        "repository boundaries",
        "structure_and_extension.md",
    ]

    missing = [value for value in required_strings if value not in readme]
    assert missing == []
