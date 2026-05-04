"""Canonical runtime path helpers and generated-artifact git safety checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved filesystem contract for a pipeline stage."""

    stage: str
    root: Path
    data_root: Path
    outputs_root: Path
    runs_root: Path
    configs_root: Path
    paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class GitSafetyReport:
    """Result of classifying candidate paths for source-control safety."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    checked_paths: list[str]


GENERATED_ROOTS = ("outputs", "runs")
GENERATED_DATA_ROOTS = (
    Path("data/synth_cyrillic"),
    Path("data/backgrounds"),
    Path("data/fonts"),
)
GENERATED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".log",
    ".png",
    ".jpg",
    ".jpeg",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_GENERATED_EXCEPTIONS = (Path("experiments/assets"), Path("tests/fixtures"))


def resolve_stage_paths(
    stage: str, root: str | Path | None = None, **overrides: object
) -> RuntimePaths:
    """Return canonical local/SLURM-compatible path defaults for a pipeline stage.

    The resolver intentionally returns repository-relative layouts under a supplied root rather than
    embedding personal absolute paths. Callers may pass explicit path overrides, but default roots
    remain `data/`, `outputs/`, `runs/`, and `configs/`.
    """

    root_path = Path.cwd() if root is None else Path(root)
    root_path = root_path.resolve()
    data_root = root_path / "data"
    outputs_root = root_path / "outputs"
    runs_root = root_path / "runs"
    configs_root = root_path / "configs"
    stage_key = _normalize_stage(stage)

    paths = _default_paths(
        stage_key,
        data_root=data_root,
        outputs_root=outputs_root,
        runs_root=runs_root,
        configs_root=configs_root,
        overrides=overrides,
    )
    return RuntimePaths(
        stage=stage_key,
        root=root_path,
        data_root=data_root,
        outputs_root=outputs_root,
        runs_root=runs_root,
        configs_root=configs_root,
        paths=paths,
    )


def assert_artifact_git_safety(paths: Iterable[str | Path]) -> GitSafetyReport:
    """Classify generated artifacts that should not be committed by default."""

    checked = [str(path) for path in paths]
    errors = []
    for raw_path in checked:
        path = _as_relative_path(raw_path)
        if _is_allowed_exception(path):
            continue
        reason = _git_safety_reason(path)
        if reason:
            errors.append(f"{raw_path}: {reason}")

    return GitSafetyReport(ok=not errors, errors=errors, warnings=[], checked_paths=checked)


def _normalize_stage(stage: str) -> str:
    aliases = {
        "comparison": "data_comparison",
        "data-source-comparison": "data_comparison",
        "data_quality": "prompt_quality",
        "selection": "data_selection",
        "image_generation": "generated",
        "generation": "generated",
        "score": "scoring",
        "scores": "scores",
        "manifest": "run_manifest",
        "manifests": "run_manifest",
        "masked-sft": "masked_sft",
        "synthetic_masked_sft": "masked_sft",
        "eval": "evaluation",
        "plots": "plotting",
    }
    return aliases.get(stage, stage)


def _default_paths(
    stage: str,
    *,
    data_root: Path,
    outputs_root: Path,
    runs_root: Path,
    configs_root: Path,
    overrides: dict[str, object],
) -> dict[str, Path]:
    generated_root = outputs_root / "generated"
    masked_root = data_root / "synth_cyrillic" / "masked_sft"
    prompt_quality_root = runs_root / "prompt-quality"
    synthetic_quality_root = runs_root / "synthetic-quality"
    comparisons_root = runs_root / "comparisons"
    run_id = str(overrides.get("run_id", "<run_id>"))
    paths_by_stage: dict[str, dict[str, Path]] = {
        "prompts": {
            "prompts_jsonl": data_root / "prompts_simple.jsonl",
        },
        "prompt_generation": {
            "prompts_jsonl": data_root / "prompts" / "simple.jsonl",
            "config": configs_root / "prompts" / "simple.json",
            "simple_config": configs_root / "prompts" / "simple.json",
            "full_config": configs_root / "prompts" / "full.json",
            "curriculum_config": configs_root / "prompts" / "curriculum.json",
            "prompt_quality_report": prompt_quality_root / "prompt-quality.json",
            "dataset_manifest": prompt_quality_root / "dataset-manifest.json",
        },
        "generated": {
            "prompts_jsonl": data_root / "prompts_simple.jsonl",
            "output_dir": generated_root,
            "images_dir": generated_root / "images",
            "latents_dir": generated_root / "latents",
            "text_embeds_dir": generated_root / "text_embeds",
            "scores_csv": generated_root / "scores.csv",
            "manifest_json": generated_root / "manifest.json",
            "selected_samples": generated_root / "selected_samples.jsonl",
            "preference_pairs": generated_root / "preference_pairs.jsonl",
            "dataset_manifest": generated_root / "dataset-manifest.json",
        },
        "scoring": {
            "images_dir": generated_root / "images",
            "text_embeds_dir": generated_root / "text_embeds",
            "scores_csv": generated_root / "scores.csv",
        },
        "scores": {
            "scores_csv": generated_root / "scores.csv",
        },
        "sft": {
            "config": configs_root / "sft.json",
            "latents_dir": generated_root / "latents",
            "text_embeds_dir": generated_root / "text_embeds",
            "scores_csv": generated_root / "scores.csv",
            "output_dir": outputs_root / "sft",
            "checkpoints_dir": outputs_root / "sft" / "checkpoints",
            "samples_dir": outputs_root / "sft" / "samples",
            "logs_dir": runs_root / "sft",
        },
        "dpo": {
            "config": configs_root / "dpo.json",
            "latents_dir": generated_root / "latents",
            "text_embeds_dir": generated_root / "text_embeds",
            "scores_csv": generated_root / "scores.csv",
            "preference_pairs": generated_root / "preference_pairs.jsonl",
            "output_dir": outputs_root / "dpo",
            "checkpoints_dir": outputs_root / "dpo" / "checkpoints",
            "samples_dir": outputs_root / "dpo" / "samples",
            "logs_dir": runs_root / "dpo",
        },
        "masked_sft": {
            "config": configs_root / "masked_sft.json",
            "data_dir": masked_root,
            "latents_dir": masked_root / "latents",
            "text_embeds_dir": masked_root / "text_embeds",
            "masks_dir": masked_root / "raw_masks",
            "index_csv": masked_root / "index.csv",
            "prompts_jsonl": masked_root / "prompts.jsonl",
            "shapes_csv": masked_root / "shapes.csv",
            "output_dir": outputs_root / "masked_sft",
            "checkpoints_dir": outputs_root / "masked_sft" / "checkpoints",
            "samples_dir": outputs_root / "masked_sft" / "samples",
            "logs_dir": runs_root / "masked_sft",
        },
        "synthetic": {
            "raw_dir": data_root / "synth_cyrillic" / "raw",
            "masked_dir": masked_root,
            "anyword_dir": data_root / "synth_cyrillic" / "anyword_format",
            "index_csv": masked_root / "index.csv",
            "selected_samples": masked_root / "selected_samples.jsonl",
            "synthetic_quality_report": synthetic_quality_root / "synthetic-quality.json",
            "dataset_manifest": synthetic_quality_root / "dataset-manifest.json",
            "contact_sheet": synthetic_quality_root / "contact-sheet.png",
        },
        "data_selection": {
            "scores_csv": generated_root / "scores.csv",
            "selected_samples": generated_root / "selected_samples.jsonl",
            "selected_samples_manifest": generated_root / "selected_samples.manifest.json",
            "preference_pairs": generated_root / "preference_pairs.jsonl",
            "preference_pairs_manifest": generated_root / "preference_pairs.manifest.json",
        },
        "data_comparison": {
            "prompt_quality_report": prompt_quality_root / "prompt-quality.json",
            "selected_samples": generated_root / "selected_samples.jsonl",
            "preference_pairs": generated_root / "preference_pairs.jsonl",
            "generated_dataset_manifest": generated_root / "dataset-manifest.json",
            "synthetic_quality_report": synthetic_quality_root / "synthetic-quality.json",
            "synthetic_manifest": synthetic_quality_root / "dataset-manifest.json",
            "data_source_comparison": comparisons_root / "generated-vs-synthetic.json",
            "markdown_summary": comparisons_root / "generated-vs-synthetic.md",
        },
        "evaluation": {
            "config": configs_root / "eval_suite.json",
            "outputs_dir": outputs_root / "evaluation",
            "scores_csv": outputs_root / "evaluation" / "scores.csv",
            "samples_dir": outputs_root / "evaluation" / "samples",
        },
        "plotting": {
            "inputs_dir": outputs_root,
            "figures_dir": outputs_root / "thesis_plots",
        },
        "run_manifest": {
            "run_dir": runs_root / run_id,
            "manifest_json": runs_root / run_id / "manifest.json",
            "config_snapshot": runs_root / run_id / "config.snapshot.json",
        },
    }
    if stage not in paths_by_stage:
        valid = ", ".join(sorted(paths_by_stage))
        raise ValueError(f"unknown stage {stage!r}; expected one of {valid}")

    paths = dict(paths_by_stage[stage])
    for key, value in overrides.items():
        if key == "run_id":
            continue
        if value is not None:
            paths[key] = Path(value)
    return paths


def _as_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            return path
    return path


def _is_allowed_exception(path: Path) -> bool:
    if any(_is_relative_to(path, allowed) for allowed in ALLOWED_GENERATED_EXCEPTIONS):
        return True
    return False


def _git_safety_reason(path: Path) -> str | None:
    parts = path.parts
    if parts and parts[0] in GENERATED_ROOTS:
        return "generated runtime root is non-committable by default"
    if any(_is_relative_to(path, root) for root in GENERATED_DATA_ROOTS):
        return "generated data root is non-committable by default"
    if "checkpoints" in parts:
        return "checkpoint artifacts are non-committable by default"
    if path.suffix.lower() in GENERATED_SUFFIXES:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return "generated images are non-committable outside allowed fixture roots"
        return "generated tensor, weight, or log artifacts are non-committable by default"
    return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
