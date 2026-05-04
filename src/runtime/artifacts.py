"""CPU-safe artifact schema validators for runtime pipeline preflights."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

ARTIFACT_SCHEMA_VERSION = "runtime-artifacts/v1"
REQUIRED_SCORE_COLUMNS = frozenset({"id", "version", "score", "target_text"})
PHASE3_JSON_SCHEMAS = {
    "dataset_manifest": {
        "key": "dataset_manifest",
        "schema": "dataset-manifest/v1",
        "required_fields": ("dataset_kind", "dataset_paths"),
    },
    "prompt_quality_report": {
        "key": "prompt_quality_report",
        "schema": "prompt-quality/v1",
        "required_fields": ("valid_records",),
    },
    "synthetic_quality_report": {
        "key": "synthetic_quality_report",
        "schema": "synthetic-quality/v1",
        "required_fields": ("sample_count",),
    },
    "data_source_comparison": {
        "key": "data_source_comparison",
        "schema": "data-source-comparison/v1",
        "required_fields": ("evidence_available", "evidence_missing"),
    },
}
PHASE3_JSONL_SCHEMAS = {
    "selected_samples": {
        "schema": "selected-samples/v1",
        "required_fields": ("sample_id", "prompt_id", "version", "selected_score"),
    },
    "preference_pairs": {
        "schema": "preference-pairs/v1",
        "required_fields": (
            "pair_id",
            "prompt_id",
            "winner_version",
            "loser_version",
            "winner_score",
            "loser_score",
        ),
    },
}


class ArtifactValidationError(ValueError):
    """Raised when required preflight inputs are not ready for an expensive stage."""


@dataclass(frozen=True)
class ArtifactReport:
    """Collected artifact validation findings.

    Validators aggregate errors so users can fix contracts before launching GPU-heavy stages. A
    caller can request fail-fast behavior with `require_ready=True` in the paths mapping.
    """

    stage: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    checked_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_artifacts(stage: str, paths: Mapping[str, str | Path | bool]) -> ArtifactReport:
    """Validate artifact contracts without importing models, CUDA, OCR, or diffusers."""

    require_ready = bool(paths.get("require_ready", False))
    context = _ValidationContext(stage=stage)
    normalized = {
        key: Path(value)
        for key, value in paths.items()
        if key != "require_ready" and isinstance(value, str | Path)
    }

    stage_key = _normalize_stage(stage)
    if stage_key == "prompts":
        _validate_prompt_jsonl(context, _required_path(context, normalized, "prompts_jsonl"))
    elif stage_key == "scores":
        _validate_scores_csv(context, _required_path(context, normalized, "scores_csv"))
    elif stage_key == "generated":
        _validate_generated_layout(context, normalized)
    elif stage_key == "masked_sft":
        _validate_masked_sft_layout(context, normalized)
    elif stage_key in {"sft", "dpo"}:
        _validate_training_inputs(context, normalized)
    elif stage_key == "synthetic":
        _validate_synthetic_outputs(context, normalized)
    elif stage_key == "checkpoints":
        _validate_optional_directory(context, normalized, "checkpoints_dir", required=require_ready)
    elif stage_key == "logs":
        _validate_optional_directory(context, normalized, "logs_dir", required=require_ready)
    elif stage_key == "evaluation":
        _validate_evaluation_outputs(context, normalized, require_ready=require_ready)
    elif stage_key == "run_manifest":
        _validate_manifest(context, normalized, require_ready=require_ready)
    elif stage_key in PHASE3_JSON_SCHEMAS:
        _validate_phase3_json(context, normalized, stage_key, required=require_ready)
    elif stage_key in PHASE3_JSONL_SCHEMAS:
        _validate_phase3_jsonl(context, normalized, stage_key, required=require_ready)
    else:
        context.error(f"stage: unsupported artifact stage {stage!r}")

    report = context.to_report()
    if require_ready and report.errors:
        raise ArtifactValidationError("; ".join(report.errors))
    return report


@dataclass
class _ValidationContext:
    stage: str
    checked_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def checked(self, path: Path) -> None:
        self.checked_paths.append(str(path))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def to_report(self) -> ArtifactReport:
        return ArtifactReport(
            stage=self.stage,
            checked_paths=list(dict.fromkeys(self.checked_paths)),
            warnings=self.warnings,
            errors=self.errors,
            metadata=self.metadata,
        )


def _normalize_stage(stage: str) -> str:
    aliases = {
        "prompt_generation": "prompts",
        "image_generation": "generated",
        "generation": "generated",
        "scoring": "scores",
        "score": "scores",
        "masked-sft": "masked_sft",
        "dataset-manifest": "dataset_manifest",
        "prompt-quality": "prompt_quality_report",
        "prompt_quality": "prompt_quality_report",
        "synthetic-quality": "synthetic_quality_report",
        "synthetic_quality": "synthetic_quality_report",
        "source-comparison": "data_source_comparison",
        "data-comparison": "data_source_comparison",
        "manifests": "run_manifest",
        "manifest": "run_manifest",
    }
    return aliases.get(stage, stage)


def _required_path(context: _ValidationContext, paths: Mapping[str, Path], key: str) -> Path:
    path = paths.get(key)
    if path is None:
        context.error(f"{key}: required path is missing from validation input")
        return Path("<missing>")
    return path


def _validate_prompt_jsonl(context: _ValidationContext, path: Path) -> None:
    context.checked(path)
    if not path.is_file():
        context.error(f"{path}: prompt JSONL file is missing")
        return

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                context.error(f"{path}: line {line_number}: malformed JSON")
                continue
            if not isinstance(payload, dict):
                context.error(f"{path}: line {line_number}: prompt record must be an object")
                continue
            if not isinstance(payload.get("prompt"), str) or not payload.get("prompt"):
                context.error(f"{path}: line {line_number}: missing required prompt field")
            if "target_text" in payload and not isinstance(payload["target_text"], str):
                context.error(
                    f"{path}: line {line_number}: target_text must be a string when present"
                )
    context.metadata["prompt_count"] = count


def _validate_scores_csv(context: _ValidationContext, path: Path) -> None:
    context.checked(path)
    if not path.is_file():
        context.error(f"{path}: scores CSV file is missing")
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SCORE_COLUMNS - fieldnames)
        if missing:
            context.error(f"{path}: scores CSV missing required columns: {', '.join(missing)}")
            return
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        _validate_float(context, path, row_number, "score", row.get("score"))
        _validate_int(context, path, row_number, "version", row.get("version"))
        if not row.get("id"):
            context.error(f"{path}: row {row_number}: id is required")
        if row.get("target_text") is None:
            context.error(f"{path}: row {row_number}: target_text is required")
    context.metadata["scores_rows"] = len(rows)

    sidecar = path.with_suffix(".schema.json")
    if sidecar.is_file():
        context.checked(sidecar)
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            context.error(f"{sidecar}: malformed JSON schema metadata")
        else:
            version = metadata.get("schema_version") if isinstance(metadata, dict) else None
            if isinstance(version, str) and version:
                context.metadata["scores_schema_version"] = version
            else:
                context.error(f"{sidecar}: missing schema_version")
    else:
        context.warn(f"{sidecar}: optional score schema metadata sidecar is missing")


def _validate_generated_layout(context: _ValidationContext, paths: Mapping[str, Path]) -> None:
    prompts = paths.get("prompts_jsonl")
    if prompts is not None:
        _validate_prompt_jsonl(context, prompts)

    latents_dir = _required_path(context, paths, "latents_dir")
    embeds_dir = _required_path(context, paths, "text_embeds_dir")
    images_dir = _required_path(context, paths, "images_dir")
    latent_versions = _collect_versioned_tensor_layout(context, latents_dir, "latent", ("latent",))
    image_versions = _collect_image_layout(context, images_dir)
    embed_ids = _collect_flat_tensors(
        context,
        embeds_dir,
        required_keys=("prompt_embeds", "target_text", "prompt"),
    )

    for prompt_id, versions in sorted(image_versions.items()):
        if prompt_id not in embed_ids:
            context.error(f"{embeds_dir / f'{prompt_id}.pt'}: missing text embedding for images")
        for version in versions:
            if version not in latent_versions.get(prompt_id, set()):
                context.error(
                    f"{latents_dir / prompt_id / f'v{version}.pt'}: "
                    "missing latent for image version"
                )
    for prompt_id, versions in sorted(latent_versions.items()):
        for version in versions:
            if version not in image_versions.get(prompt_id, set()):
                context.error(
                    f"{images_dir / prompt_id / f'v{version}.png'}: "
                    "missing image for latent version"
                )

    context.metadata["generated_versions"] = {
        prompt_id: sorted(versions) for prompt_id, versions in image_versions.items()
    }


def _validate_masked_sft_layout(context: _ValidationContext, paths: Mapping[str, Path]) -> None:
    data_dir = paths.get("data_dir")
    latents_dir = paths.get("latents_dir") or (data_dir / "latents" if data_dir else None)
    embeds_dir = paths.get("text_embeds_dir") or (data_dir / "text_embeds" if data_dir else None)
    shapes_csv = paths.get("shapes_csv") or (data_dir / "shapes.csv" if data_dir else None)

    if data_dir is not None:
        _validate_directory(context, data_dir, "data_dir")
    if latents_dir is None or embeds_dir is None:
        context.error("masked_sft: data_dir or latents_dir/text_embeds_dir paths are required")
        return

    sample_ids = _collect_flat_tensors(context, latents_dir, required_keys=("latent", "mask_lat"))
    embed_ids = _collect_flat_tensors(context, embeds_dir, required_keys=("prompt_embeds",))
    for sample_id in sorted(sample_ids - embed_ids):
        context.error(f"{embeds_dir / f'{sample_id}.pt'}: missing text embedding for latent sample")
    for sample_id in sorted(embed_ids - sample_ids):
        context.error(
            f"{latents_dir / f'{sample_id}.pt'}: missing latent for text embedding sample"
        )

    if shapes_csv is not None:
        _validate_shapes_csv(context, shapes_csv, required=False)
    context.metadata["masked_sft_samples"] = sorted(sample_ids & embed_ids)


def _validate_training_inputs(context: _ValidationContext, paths: Mapping[str, Path]) -> None:
    for key in ("scores_csv", "latents_dir", "text_embeds_dir"):
        path = _required_path(context, paths, key)
        if key == "scores_csv":
            if not path.is_file():
                context.checked(path)
                context.error(f"scores_csv: {path}: required scores CSV file is missing")
                continue
            _validate_scores_csv(context, path)
        else:
            _validate_directory(context, path, key)
    if context.stage == "dpo" and "preference_pairs" in paths:
        _validate_optional_index_file(context, paths, "preference_pairs", required=False)


def _validate_synthetic_outputs(context: _ValidationContext, paths: Mapping[str, Path]) -> None:
    for key in ("raw_dir", "masked_dir", "anyword_dir"):
        _validate_optional_directory(context, paths, key, required=False)
    if "selected_samples" in paths:
        _validate_phase3_jsonl(context, paths, "selected_samples", required=False)
    if "synthetic_quality_report" in paths:
        _validate_phase3_json(context, paths, "synthetic_quality_report", required=False)
    if "dataset_manifest" in paths:
        _validate_phase3_json(context, paths, "dataset_manifest", required=False)


def _validate_evaluation_outputs(
    context: _ValidationContext, paths: Mapping[str, Path], *, require_ready: bool
) -> None:
    _validate_optional_directory(context, paths, "outputs_dir", required=require_ready)
    if "scores_csv" in paths:
        _validate_scores_csv(context, paths["scores_csv"])


def _validate_manifest(
    context: _ValidationContext, paths: Mapping[str, Path], *, require_ready: bool
) -> None:
    manifest = paths.get("manifest_json")
    if manifest is None:
        if require_ready:
            context.error("manifest_json: required path is missing from validation input")
        return
    context.checked(manifest)
    if not manifest.is_file():
        if require_ready:
            context.error(f"{manifest}: manifest file is missing")
        else:
            context.warn(f"{manifest}: manifest file is not present yet")
        return
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        context.error(f"{manifest}: malformed JSON manifest")
        return
    if not isinstance(payload, dict):
        context.error(f"{manifest}: manifest must be a JSON object")
        return
    if isinstance(payload.get("schema_version"), str):
        context.metadata["manifest_schema_version"] = payload["schema_version"]


def _validate_optional_index_file(
    context: _ValidationContext,
    paths: Mapping[str, Path],
    key: str,
    *,
    required: bool,
) -> None:
    path = paths.get(key)
    if path is None:
        if required:
            context.error(f"{key}: required path is missing from validation input")
        return
    context.checked(path)
    if not path.is_file():
        message = f"{path}: {key} file is missing"
        context.error(message) if required else context.warn(message)


def _validate_phase3_json(
    context: _ValidationContext,
    paths: Mapping[str, Path],
    stage_key: str,
    *,
    required: bool,
) -> None:
    spec = PHASE3_JSON_SCHEMAS[stage_key]
    key = str(spec["key"])
    path = paths.get(key) or paths.get(stage_key)
    if path is None:
        if required:
            context.error(f"{key}: required path is missing from validation input")
        return
    context.checked(path)
    if not path.is_file():
        message = f"{path}: {key} JSON file is missing"
        context.error(message) if required else context.warn(message)
        return
    payload = _load_json_object(context, path, label=key)
    if payload is None:
        return
    expected_schema = str(spec["schema"])
    _validate_schema_version(context, path, payload, expected_schema)
    for field_name in spec["required_fields"]:
        if field_name not in payload:
            context.error(f"{path}: missing required field {field_name!r}")
    context.metadata["schema_version"] = payload.get("schema_version")
    context.metadata[f"{stage_key}_path"] = str(path)


def _validate_phase3_jsonl(
    context: _ValidationContext,
    paths: Mapping[str, Path],
    stage_key: str,
    *,
    required: bool,
) -> None:
    path = paths.get(stage_key)
    if path is None:
        if required:
            context.error(f"{stage_key}: required path is missing from validation input")
        return
    context.checked(path)
    if not path.is_file():
        message = f"{path}: {stage_key} file is missing"
        context.error(message) if required else context.warn(message)
        return
    spec = PHASE3_JSONL_SCHEMAS[stage_key]
    expected_schema = str(spec["schema"])
    record_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record_count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                context.error(f"{path}: line {line_number}: malformed JSONL")
                continue
            if not isinstance(payload, dict):
                context.error(f"{path}: line {line_number}: record must be an object")
                continue
            _validate_schema_version(
                context,
                path,
                payload,
                expected_schema,
                line_number=line_number,
            )
            for field_name in spec["required_fields"]:
                if field_name not in payload:
                    context.error(
                        f"{path}: line {line_number}: missing required field {field_name!r}"
                    )
    context.metadata["schema_version"] = expected_schema
    context.metadata["record_count"] = record_count


def _load_json_object(
    context: _ValidationContext, path: Path, *, label: str
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        context.error(f"{path}: malformed {label} JSON")
        return None
    if not isinstance(payload, dict):
        context.error(f"{path}: {label} JSON must be an object")
        return None
    return payload


def _validate_schema_version(
    context: _ValidationContext,
    path: Path,
    payload: Mapping[str, Any],
    expected_schema: str,
    *,
    line_number: int | None = None,
) -> None:
    actual_schema = payload.get("schema_version")
    location = f"{path}: line {line_number}" if line_number is not None else str(path)
    if actual_schema != expected_schema:
        context.error(
            f"{location}: schema_version must be {expected_schema}, got {actual_schema!r}"
        )


def _validate_optional_directory(
    context: _ValidationContext,
    paths: Mapping[str, Path],
    key: str,
    *,
    required: bool,
) -> None:
    path = paths.get(key)
    if path is None:
        if required:
            context.error(f"{key}: required path is missing from validation input")
        return
    if not _validate_directory(context, path, key) and not required:
        if context.errors:
            context.warnings.append(context.errors.pop())


def _validate_directory(context: _ValidationContext, path: Path, key: str) -> bool:
    context.checked(path)
    if not path.is_dir():
        context.error(f"{path}: {key} directory is missing")
        return False
    return True


def _validate_shapes_csv(context: _ValidationContext, path: Path, *, required: bool) -> None:
    context.checked(path)
    if not path.is_file():
        message = f"{path}: optional shapes.csv is missing"
        context.error(message) if required else context.warn(message)
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted({"id", "H", "W"} - set(reader.fieldnames or []))
        if missing:
            context.error(f"{path}: shapes.csv missing required columns: {', '.join(missing)}")
            return
        for row_number, row in enumerate(reader, start=2):
            if not row.get("id"):
                context.error(f"{path}: row {row_number}: id is required")
            _validate_int(context, path, row_number, "H", row.get("H"))
            _validate_int(context, path, row_number, "W", row.get("W"))


def _collect_image_layout(context: _ValidationContext, images_dir: Path) -> dict[str, set[int]]:
    context.checked(images_dir)
    versions: dict[str, set[int]] = {}
    if not images_dir.is_dir():
        context.error(f"{images_dir}: images_dir directory is missing")
        return versions
    for prompt_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        prompt_versions: set[int] = set()
        for image_path in sorted(prompt_dir.glob("v*.png")):
            context.checked(image_path)
            version = _parse_version(image_path)
            if version is None:
                context.error(f"{image_path}: image filename must match v{{version}}.png")
                continue
            prompt_versions.add(version)
        versions[prompt_dir.name] = prompt_versions
    return versions


def _collect_versioned_tensor_layout(
    context: _ValidationContext, root: Path, label: str, required_keys: Iterable[str]
) -> dict[str, set[int]]:
    context.checked(root)
    versions: dict[str, set[int]] = {}
    if not root.is_dir():
        context.error(f"{root}: {label} directory is missing")
        return versions
    for prompt_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        prompt_versions: set[int] = set()
        for tensor_path in sorted(prompt_dir.glob("v*.pt")):
            version = _parse_version(tensor_path)
            if version is None:
                context.error(f"{tensor_path}: tensor filename must match v{{version}}.pt")
                continue
            _validate_torch_dict(context, tensor_path, required_keys)
            prompt_versions.add(version)
        versions[prompt_dir.name] = prompt_versions
    return versions


def _collect_flat_tensors(
    context: _ValidationContext, root: Path, *, required_keys: Iterable[str]
) -> set[str]:
    context.checked(root)
    sample_ids: set[str] = set()
    if not root.is_dir():
        context.error(f"{root}: tensor directory is missing")
        return sample_ids
    for tensor_path in sorted(root.glob("*.pt")):
        _validate_torch_dict(context, tensor_path, required_keys)
        sample_ids.add(tensor_path.stem)
    return sample_ids


def _validate_torch_dict(
    context: _ValidationContext, path: Path, required_keys: Iterable[str]
) -> None:
    context.checked(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - report validation context instead of leaking tracebacks.
        context.error(f"{path}: could not load trusted local tensor artifact: {exc}")
        return
    if not isinstance(payload, dict):
        context.error(f"{path}: tensor artifact must contain a dictionary")
        return
    for key in required_keys:
        if key not in payload:
            context.error(f"{path}: missing required key {key!r}")


def _parse_version(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("v"):
        return None
    try:
        return int(stem[1:])
    except ValueError:
        return None


def _validate_float(
    context: _ValidationContext, path: Path, row_number: int, field_name: str, value: str | None
) -> None:
    try:
        float(value or "")
    except ValueError:
        context.error(f"{path}: row {row_number}: {field_name} must be numeric")


def _validate_int(
    context: _ValidationContext, path: Path, row_number: int, field_name: str, value: str | None
) -> None:
    try:
        int(value or "")
    except ValueError:
        context.error(f"{path}: row {row_number}: {field_name} must be an integer")
