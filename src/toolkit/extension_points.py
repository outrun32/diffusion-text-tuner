"""Import-safe registry of documented toolkit extension points.

The registry is descriptive metadata only. It does not discover plugins,
execute commands, import registered modules, or validate runtime artifacts at
module import time. CPU-safe tests import the listed modules separately to catch
drift between this index, the docs, and the preserved command surface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtensionPoint:
    """One documented seam for extending a toolkit stage."""

    name: str
    purpose: str
    implementation_module: str
    thin_script: str | None
    config_home: str
    docs_path: str
    test_target: str
    generated_artifact_notes: str


_EXTENSION_POINTS: tuple[ExtensionPoint, ...] = (
    ExtensionPoint(
        name="prompt generation",
        purpose="Create multilingual/Cyrillic text-rendering prompt JSONL datasets from config-driven curricula.",
        implementation_module="src.prompt_pipeline.generate",
        thin_script="python -m src.prompt_pipeline.generate",
        config_home="configs/prompts/",
        docs_path="docs/data_curriculum.md",
        test_target="tests/test_prompt_curriculum.py",
        generated_artifact_notes="Prompt JSONL outputs are generated data; keep full datasets under ignored data/runs roots unless intentionally tiny fixtures.",
    ),
    ExtensionPoint(
        name="image generation",
        purpose="Generate FLUX images, latents, and text embeddings through an importable orchestration seam.",
        implementation_module="src.generation.pipeline",
        thin_script="scripts/generate_images.py",
        config_home="configs/experiments/evaluation/ or runtime generation arguments",
        docs_path="docs/commands.md",
        test_target="tests/test_generation_pipeline_contracts.py",
        generated_artifact_notes="Generated images, latents, embeddings, metadata, and manifests stay in ignored outputs/ or runs/ roots.",
    ),
    ExtensionPoint(
        name="scoring",
        purpose="Score generated images and write canonical reward rows, sidecars, and scorer provenance.",
        implementation_module="src.scoring.pipeline",
        thin_script="scripts/score_images.py",
        config_home="configs/experiments/reward/",
        docs_path="docs/reward_evaluation.md",
        test_target="tests/test_scoring_pipeline_contracts.py",
        generated_artifact_notes="Score CSV/JSONL files and sidecars may contain prompt text, local paths, and reward evidence; keep runtime score outputs ignored by default.",
    ),
    ExtensionPoint(
        name="synthesis",
        purpose="Build synthetic masked-SFT datasets through reusable render, collation, latent, and text-encoding phases.",
        implementation_module="src.synthesis.dataset_builder",
        thin_script="scripts/synth/build_dataset.py",
        config_home="configs/experiments/synthesis/ and configs/synth/",
        docs_path="docs/synthetic_quality.md",
        test_target="tests/test_synthesis_pipeline_contracts.py",
        generated_artifact_notes="Synthetic images, masks, latents, embeddings, indexes, and quality reports remain generated artifacts under ignored data/ or runs/ roots.",
    ),
    ExtensionPoint(
        name="training",
        purpose="Extend SFT, DPO, masked-SFT, runtime metadata, objective helpers, sampling, schedulers, and checkpointing without hiding trainer choices.",
        implementation_module="src.training.runtime",
        thin_script="accelerate launch -m src.training.sft_trainer / src.training.dpo_trainer; python -m src.training.masked_sft_trainer",
        config_home="configs/experiments/sft/, configs/experiments/dpo/, and configs/experiments/masked_sft/",
        docs_path="docs/training_comparability.md",
        test_target="tests/test_training_shared_utilities.py",
        generated_artifact_notes="Checkpoints, logs, sample images, tensors, and run manifests are runtime artifacts; record provenance but do not commit generated outputs.",
    ),
    ExtensionPoint(
        name="evaluation",
        purpose="Materialize held-out evaluation plans and validate recorded reward/evaluation evidence without launching expensive work by default.",
        implementation_module="src.evaluation.heldout",
        thin_script="scripts/run_heldout_evaluation.py",
        config_home="configs/experiments/evaluation/",
        docs_path="docs/evaluation_harness.md",
        test_target="tests/test_heldout_evaluation_harness.py",
        generated_artifact_notes="Held-out plans, generated eval outputs, scores, and manifests belong under ignored runs/ or outputs/ roots.",
    ),
    ExtensionPoint(
        name="plotting",
        purpose="Load, summarize, smooth, and plot training metrics through import-safe helpers with lazy plotting backends.",
        implementation_module="src.plotting.training_metrics",
        thin_script="scripts/plot_metrics.py",
        config_home="runtime metric CSV paths and configs/experiments/ metadata",
        docs_path="docs/commands.md",
        test_target="tests/test_plotting_pipeline_contracts.py",
        generated_artifact_notes="Training curves and thesis plots are generated artifacts; keep figures under ignored run/output roots unless reviewed as tiny docs assets.",
    ),
    ExtensionPoint(
        name="run comparison",
        purpose="Compare local run manifests and controlled training settings before interpreting experiments as thesis evidence.",
        implementation_module="src.runtime.manifest_diff",
        thin_script="scripts/compare_training_runs.py",
        config_home="runs/<run_id>/manifest.json and configs/experiments/",
        docs_path="docs/training_comparability.md",
        test_target="tests/test_runtime_manifest_diff.py",
        generated_artifact_notes="Comparison Markdown/JSON reports should be written under ignored runs/comparisons/ unless intentionally tiny reviewed evidence.",
    ),
    ExtensionPoint(
        name="diagnostics",
        purpose="Inspect reward disagreements, gold slices, gradients, and manual checks through explicit opt-in diagnostics outside default heavy jobs.",
        implementation_module="src.evaluation.diagnostics",
        thin_script="scripts/analyze_reward_diagnostics.py and scripts/diagnose_*.py",
        config_home="configs/experiments/evaluation/ and explicit diagnostic arguments",
        docs_path="docs/evaluation_diagnostics.md",
        test_target="tests/test_reward_diagnostics.py",
        generated_artifact_notes="Diagnostic reports, contact sheets, local paths, prompt text, scores, and logs are runtime/private artifacts by default.",
    ),
    ExtensionPoint(
        name="thesis outputs",
        purpose="Build thesis-ready tables, SVG plots, contact sheets, bundles, and Markdown summaries from recorded manifests and reports.",
        implementation_module="src.evaluation.thesis_outputs",
        thin_script="scripts/build_thesis_outputs.py",
        config_home="configs/thesis/ and configs/experiments/evaluation/",
        docs_path="docs/thesis_outputs.md",
        test_target="tests/test_thesis_outputs.py",
        generated_artifact_notes="Thesis tables, plots, bundles, and contact sheets stay in ignored outputs/thesis/ or runs/ roots unless a future plan approves fixtures.",
    ),
)


def list_extension_points() -> tuple[ExtensionPoint, ...]:
    """Return the immutable extension-point registry."""

    return _EXTENSION_POINTS


def get_extension_point(name: str) -> ExtensionPoint:
    """Return an extension point by exact stage name.

    Raises
    ------
    KeyError
        If ``name`` is not one of the registered extension-point names.
    """

    for entry in _EXTENSION_POINTS:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown extension point: {name}")


__all__ = ["ExtensionPoint", "get_extension_point", "list_extension_points"]
