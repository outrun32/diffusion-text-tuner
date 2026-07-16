"""CPU-safe held-out evaluation planning contracts.

This module validates JSON evaluation configs and materializes deterministic
plans. It never launches generation, scoring, CUDA, OCR, or model-loading code.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.runtime.manifests import ManifestError, load_run_manifest

CONFIG_SCHEMA_VERSION = "heldout-evaluation-config/v1"
PLAN_SCHEMA_VERSION = "heldout-evaluation-plan/v1"
_IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class HeldoutEvaluationError(ValueError):
    """Raised when a held-out evaluation config cannot be trusted."""


@dataclass(frozen=True)
class EvaluationTarget:
    """One baseline or trained checkpoint target in a held-out comparison."""

    name: str
    lora_checkpoint_path: str | None
    source_run_manifest_path: str
    generation_output_path: str
    score_output_path: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise HeldoutEvaluationError("target.name must be a non-empty string")
        if not self.source_run_manifest_path.strip():
            raise HeldoutEvaluationError(f"{self.name}: source_run_manifest_path is required")
        _validate_writable_path(
            self.generation_output_path,
            field="generation_output_path",
            target_name=self.name,
        )
        _validate_writable_path(
            self.score_output_path,
            field="score_output_path",
            target_name=self.name,
        )
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    def to_plan_entry(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe target entry for reports."""
        return {
            "generation_output_path": self.generation_output_path,
            "lora_checkpoint_path": self.lora_checkpoint_path,
            "name": self.name,
            "notes": list(self.notes),
            "score_output_path": self.score_output_path,
            "source_run_manifest_path": self.source_run_manifest_path,
        }


@dataclass(frozen=True)
class HeldoutEvaluationConfig:
    """Validated held-out comparison configuration."""

    fixed_prompts_path: str
    fixed_seeds: tuple[int, ...]
    inference_settings: dict[str, Any]
    output_root: str
    targets: tuple[EvaluationTarget, ...]
    source_config_path: str
    schema_version: str = CONFIG_SCHEMA_VERSION

    @classmethod
    def from_file(cls, path: str | Path) -> HeldoutEvaluationConfig:
        """Load and validate a held-out evaluation JSON config."""
        config_path = Path(path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HeldoutEvaluationError(f"{config_path}: malformed JSON") from exc
        except OSError as exc:
            raise HeldoutEvaluationError(f"{config_path}: could not read config") from exc
        if not isinstance(payload, dict):
            raise HeldoutEvaluationError(f"{config_path}: config must be a JSON object")
        return cls.from_mapping(payload, source_config_path=str(config_path))

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        source_config_path: str = "<memory>",
    ) -> HeldoutEvaluationConfig:
        """Validate a config mapping without doing expensive runtime work."""
        schema_version = str(payload.get("schema_version", CONFIG_SCHEMA_VERSION))
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise HeldoutEvaluationError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        fixed_prompts_path = _required_str(payload, "fixed_prompts_path")
        _validate_prompts_jsonl(Path(fixed_prompts_path), field="fixed_prompts_path")
        fixed_seeds = _validate_fixed_seeds(payload.get("fixed_seeds"))
        inference_settings = _validate_inference_settings(payload.get("inference_settings"))
        output_root = _required_str(payload, "output_root")
        _validate_writable_path(output_root, field="output_root")
        targets = _validate_targets(payload.get("targets"), output_root=output_root)
        return cls(
            fixed_prompts_path=fixed_prompts_path,
            fixed_seeds=fixed_seeds,
            inference_settings=inference_settings,
            output_root=output_root,
            targets=targets,
            source_config_path=source_config_path,
            schema_version=schema_version,
        )


def build_evaluation_plan(config: str | Path | HeldoutEvaluationConfig) -> dict[str, Any]:
    """Build a deterministic, materialize-only held-out evaluation plan."""
    heldout_config = (
        HeldoutEvaluationConfig.from_file(config)
        if not isinstance(config, HeldoutEvaluationConfig)
        else config
    )
    target_entries = [target.to_plan_entry() for target in heldout_config.targets]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "execution_mode": "materialize-only",
        "source_config_path": heldout_config.source_config_path,
        "fixed_prompts_path": heldout_config.fixed_prompts_path,
        "fixed_seeds": list(heldout_config.fixed_seeds),
        "inference_settings": dict(sorted(heldout_config.inference_settings.items())),
        "output_root": heldout_config.output_root,
        "targets": target_entries,
        "manifest_links": [target.source_run_manifest_path for target in heldout_config.targets],
        "planned_generation_commands": _build_generation_commands(heldout_config),
        "planned_scoring_commands": _build_scoring_commands(heldout_config),
        "planned_aggregation_commands": _build_aggregation_commands(heldout_config),
    }


def write_evaluation_plan(
    config: str | Path | HeldoutEvaluationConfig,
    *,
    output_plan: str | Path,
    markdown_summary: str | Path | None = None,
) -> dict[str, Any]:
    """Write a held-out JSON plan and optional Markdown summary atomically."""
    plan = build_evaluation_plan(config)
    _atomic_write_text(
        Path(output_plan),
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if markdown_summary is not None:
        _atomic_write_text(Path(markdown_summary), format_markdown_summary(plan))
    return plan


def format_markdown_summary(plan: dict[str, Any]) -> str:
    """Render a concise Markdown report for a held-out evaluation plan."""
    target_lines = [
        "| Target | LoRA checkpoint | source_run_manifest_path | Generation output | Score output |"
    ]
    target_lines.append("| --- | --- | --- | --- | --- |")
    for target in plan["targets"]:
        lora_path = target["lora_checkpoint_path"] or "baseline/no LoRA"
        target_lines.append(
            "| {name} | {lora} | {manifest} | {generation} | {score} |".format(
                name=target["name"],
                lora=lora_path,
                manifest=target["source_run_manifest_path"],
                generation=target["generation_output_path"],
                score=target["score_output_path"],
            )
        )
    generation_lines = [f"- `{entry['command']}`" for entry in plan["planned_generation_commands"]]
    scoring_lines = [f"- `{entry['command']}`" for entry in plan["planned_scoring_commands"]]
    aggregation_lines = [
        f"- `{entry['command']}`" for entry in plan["planned_aggregation_commands"]
    ]
    return "\n".join(
        [
            "# Held-out evaluation plan",
            "",
            "Plan only: generation and scoring commands are not executed by this harness.",
            "",
            f"- Schema: `{plan['schema_version']}`",
            f"- Source config: `{plan['source_config_path']}`",
            f"- fixed_prompts_path: `{plan['fixed_prompts_path']}`",
            f"- fixed_seeds: `{', '.join(str(seed) for seed in plan['fixed_seeds'])}`",
            f"- output_root: `{plan['output_root']}`",
            f"- inference_settings: `{json.dumps(plan['inference_settings'], sort_keys=True)}`",
            "",
            "## Targets",
            "",
            *target_lines,
            "",
            "## Planned generation commands",
            "",
            *generation_lines,
            "",
            "## Planned scoring commands",
            "",
            *scoring_lines,
            "",
            "## Planned score aggregation commands",
            "",
            *aggregation_lines,
            "",
        ]
    )


def _build_generation_commands(config: HeldoutEvaluationConfig) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    settings = config.inference_settings
    for target in config.targets:
        for seed in config.fixed_seeds:
            output_root = Path(target.generation_output_path) / f"seed-{seed}"
            output_dir = str(output_root)
            generation_manifest_path = str(output_root / "generation.manifest.json")
            argv = [
                "python",
                "-m",
                "scripts.generate_images",
                "--prompts",
                config.fixed_prompts_path,
                "--output_dir",
                output_dir,
                "--versions_per_prompt",
                "1",
                "--batch_size",
                "1",
                "--seed",
                str(seed),
                "--device",
                "cuda",
                "--manifest_path",
                generation_manifest_path,
                "--run_manifest_path",
                target.source_run_manifest_path,
            ]
            if settings.get("model") is not None:
                argv.extend(["--model_id", str(settings["model"])])
            if settings.get("model_revision") is not None:
                argv.extend(["--model_revision", str(settings["model_revision"])])
            if settings.get("num_inference_steps") is not None:
                argv.extend(["--num_inference_steps", str(settings["num_inference_steps"])])
            if settings.get("guidance_scale") is not None:
                argv.extend(["--guidance_scale", str(settings["guidance_scale"])])
            if settings.get("height") is not None:
                argv.extend(["--resolution", str(settings["height"])])
            if target.lora_checkpoint_path is not None:
                argv.extend(["--lora_path", target.lora_checkpoint_path])
            commands.append(
                {
                    "target_name": target.name,
                    "seed": seed,
                    "status": "planned-not-run",
                    "argv": argv,
                    "command": shlex.join(argv),
                    "output_dir": output_dir,
                    "generation_manifest_path": generation_manifest_path,
                    "source_run_manifest_path": target.source_run_manifest_path,
                }
            )
    return commands


def _build_scoring_commands(config: HeldoutEvaluationConfig) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    scorer = str(config.inference_settings.get("scorer", "both"))
    ocr_device = str(config.inference_settings.get("ocr_device", "cpu"))
    for target in config.targets:
        for seed in config.fixed_seeds:
            seed_root = Path(target.generation_output_path) / f"seed-{seed}"
            seed_score_path = seed_root / "scores.csv"
            generation_manifest_path = seed_root / "generation.manifest.json"
            argv = [
                "python",
                "-m",
                "scripts.score_images",
                "--images_dir",
                str(seed_root / "images"),
                "--text_embeds_dir",
                str(seed_root / "text_embeds"),
                "--output_csv",
                str(seed_score_path),
                "--scorer",
                scorer,
                "--ocr_device",
                ocr_device,
                "--product_formula",
                "thesis",
                "--source_manifest",
                target.source_run_manifest_path,
                "--source_manifest",
                str(generation_manifest_path),
            ]
            if scorer in {"vlm", "both"}:
                argv.extend(
                    [
                        "--vlm_model_id",
                        str(config.inference_settings["vlm_model"]),
                        "--vlm_model_revision",
                        str(config.inference_settings["vlm_model_revision"]),
                    ]
                )
            commands.append(
                {
                    "target_name": target.name,
                    "seed": seed,
                    "status": "planned-not-run",
                    "argv": argv,
                    "command": shlex.join(argv),
                    "source_run_manifest_path": target.source_run_manifest_path,
                    "score_output_path": str(seed_score_path),
                    "generation_manifest_path": str(generation_manifest_path),
                }
            )
    return commands


def _build_aggregation_commands(config: HeldoutEvaluationConfig) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for target in config.targets:
        argv = ["python", "-m", "scripts.aggregate_heldout_scores"]
        for seed in config.fixed_seeds:
            score_path = Path(target.generation_output_path) / f"seed-{seed}" / "scores.csv"
            argv.extend(["--input", f"{seed}={score_path}"])
        argv.extend(["--output", target.score_output_path])
        commands.append(
            {
                "target_name": target.name,
                "status": "planned-not-run",
                "argv": argv,
                "command": shlex.join(argv),
                "score_output_path": target.score_output_path,
            }
        )
    return commands


def _required_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HeldoutEvaluationError(f"{field} must be a non-empty string")
    return value


def _validate_prompts_jsonl(path: Path, *, field: str) -> None:
    if path.expanduser() != path:
        raise HeldoutEvaluationError(f"{field} must not use home-directory expansion")
    if not path.is_file():
        raise HeldoutEvaluationError(f"{field} does not exist or is not a file")
    rows = 0
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise HeldoutEvaluationError(f"{field}:{line_number} must be a JSON object")
            if not record.get("prompt") or not record.get("target_text"):
                raise HeldoutEvaluationError(
                    f"{field}:{line_number} requires prompt and target_text fields"
                )
    except json.JSONDecodeError as exc:
        raise HeldoutEvaluationError(f"{field}: malformed JSONL") from exc
    except OSError as exc:
        raise HeldoutEvaluationError(f"{field}: could not read prompts") from exc
    if rows == 0:
        raise HeldoutEvaluationError(f"{field} must contain at least one prompt row")


def _validate_fixed_seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise HeldoutEvaluationError("fixed_seeds must be a non-empty list of integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise HeldoutEvaluationError("fixed_seeds must be a non-empty list of integers")
    if any(seed < 0 for seed in value):
        raise HeldoutEvaluationError("fixed_seeds must contain only nonnegative integers")
    if len(set(value)) != len(value):
        raise HeldoutEvaluationError("fixed_seeds must not contain duplicates")
    return tuple(value)


def _validate_inference_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise HeldoutEvaluationError("inference_settings must be a non-empty object")
    for key in (
        "model",
        "model_revision",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "scorer",
    ):
        if key not in value:
            raise HeldoutEvaluationError(f"inference_settings.{key} is required")
    if value["scorer"] not in {"vlm", "ocr", "both"}:
        raise HeldoutEvaluationError("inference_settings.scorer must be vlm, ocr, or both")
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise HeldoutEvaluationError("inference_settings.model is required")
    _validate_immutable_revision(value.get("model_revision"), "inference_settings.model_revision")
    if value["scorer"] in {"vlm", "both"}:
        if not isinstance(value.get("vlm_model"), str) or not value["vlm_model"].strip():
            raise HeldoutEvaluationError("inference_settings.vlm_model is required")
        _validate_immutable_revision(
            value.get("vlm_model_revision"), "inference_settings.vlm_model_revision"
        )
    if value["height"] != value["width"]:
        raise HeldoutEvaluationError(
            "inference_settings.height and width must match because the supported generation "
            "pipeline uses one square resolution"
        )
    return dict(sorted(value.items()))


def _validate_targets(value: Any, *, output_root: str) -> tuple[EvaluationTarget, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise HeldoutEvaluationError(
            "targets must include at least baseline plus one trained target"
        )
    targets: list[EvaluationTarget] = []
    names: set[str] = set()
    for index, raw_target in enumerate(value):
        if not isinstance(raw_target, dict):
            raise HeldoutEvaluationError(f"targets[{index}] must be a JSON object")
        target = EvaluationTarget(
            name=_required_str(raw_target, "name"),
            lora_checkpoint_path=_optional_str(raw_target.get("lora_checkpoint_path")),
            source_run_manifest_path=_required_str(raw_target, "source_run_manifest_path"),
            generation_output_path=_required_str(raw_target, "generation_output_path"),
            score_output_path=_required_str(raw_target, "score_output_path"),
            notes=_validate_notes(raw_target.get("notes", [])),
        )
        if target.name in names:
            raise HeldoutEvaluationError(f"targets contain duplicate name {target.name!r}")
        names.add(target.name)
        _validate_manifest_link(target.source_run_manifest_path, target_name=target.name)
        _validate_under_root(
            target.generation_output_path,
            output_root,
            field="generation_output_path",
        )
        _validate_under_root(target.score_output_path, output_root, field="score_output_path")
        targets.append(target)
    baseline_targets = [target for target in targets if target.name.lower() == "baseline"]
    if len(baseline_targets) != 1 or baseline_targets[0].lora_checkpoint_path is not None:
        raise HeldoutEvaluationError(
            "targets must include exactly one baseline target without LoRA"
        )
    trained_targets = [target for target in targets if target.name.lower() != "baseline"]
    if not any(target.lora_checkpoint_path for target in trained_targets):
        raise HeldoutEvaluationError("targets must include at least one trained target with LoRA")
    return tuple(targets)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise HeldoutEvaluationError("lora_checkpoint_path must be null or a non-empty string")
    if Path(value).expanduser() != Path(value):
        raise HeldoutEvaluationError("lora_checkpoint_path must not use home-directory expansion")
    if ".." in Path(value).parts:
        raise HeldoutEvaluationError("lora_checkpoint_path must not use traversal")
    return value


def _validate_notes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HeldoutEvaluationError("notes must be a list of strings")
    return tuple(value)


def _validate_manifest_link(path: str, *, target_name: str) -> None:
    manifest_path = Path(path)
    if manifest_path.expanduser() != manifest_path:
        raise HeldoutEvaluationError(f"{target_name}: source_run_manifest_path uses home expansion")
    if not manifest_path.is_file():
        raise HeldoutEvaluationError(f"{target_name}: source_run_manifest_path does not exist")
    try:
        load_run_manifest(manifest_path)
    except ManifestError as exc:
        raise HeldoutEvaluationError(
            f"{target_name}: source_run_manifest_path is not a valid run manifest: {exc}"
        ) from exc


def _validate_immutable_revision(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _IMMUTABLE_REVISION_PATTERN.fullmatch(value):
        raise HeldoutEvaluationError(f"{field} must be an immutable 40-character commit SHA")


def _validate_writable_path(value: str, *, field: str, target_name: str | None = None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise HeldoutEvaluationError(f"{field} must be a non-empty path")
    path = Path(value)
    if path.expanduser() != path:
        raise HeldoutEvaluationError(
            _field_message(field, target_name, "must not use home expansion")
        )
    if ".." in path.parts:
        raise HeldoutEvaluationError(_field_message(field, target_name, "must not use traversal"))


def _validate_under_root(value: str, output_root: str, *, field: str) -> None:
    path = Path(value).resolve(strict=False)
    root = Path(output_root).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HeldoutEvaluationError(f"{field} must be inside output_root") from exc


def _field_message(field: str, target_name: str | None, detail: str) -> str:
    prefix = f"{target_name}: " if target_name else ""
    return f"{prefix}{field} {detail}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
