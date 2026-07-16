"""Importable reward scoring pipeline implementation.

This module owns task discovery, canonical score-row conversion,
schema sidecar writing, resume/shard behavior, scorer selection, and CSV
writing for ``python -m scripts.score_images`` while keeping optional model
and OCR stacks inside ``run_scoring`` execution branches.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm import tqdm

from src.evaluation.reward_interface import (
    ProductScoreFormula,
    build_score_metadata,
    compute_product_score,
    thesis_product_formula,
)

logger = logging.getLogger(__name__)

PHASE6_SCORE_FILE_SCHEMA_VERSION = "phase6-score-file/v1"
CANONICAL_SCORE_COLUMNS = [
    "id",
    "sample_id",
    "version",
    "score",
    "product_score",
    "target_text",
    "score_vlm",
    "score_ocr",
    "cer",
    "entropy",
    "official_conf",
    "min_p",
    "frac_unc",
    "ocr_detected",
    "detection_status",
    "exact_text_match",
    "char_accuracy",
    "char_matches",
    "char_total",
    "missing_components",
    "formula_complete",
    "manifest_path",
    "text_metrics",
    "scorer_metadata",
    "thresholds",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Runtime configuration for generated-image reward scoring."""

    images_dir: Path
    text_embeds_dir: Path
    output_csv: Path = Path("outputs/generated/scores.csv")
    scorer: Literal["vlm", "ocr", "both"] = "vlm"
    vlm_model_id: str = "Qwen/Qwen3.5-9B"
    vlm_model_revision: str | None = None
    vlm_device: Literal["cuda"] = "cuda"
    ocr_device: Literal["cpu", "gpu"] = "cpu"
    product_formula: Literal["thesis", "diagnostic"] = "thesis"
    entropy_lambda: float = 1.0
    batch_size: int = 1
    resume: bool = False
    shard_idx: int = 0
    num_shards: int = 1
    manifest_path: str = ""
    source_manifests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError(
                "batch_size currently supports only 1; batched scoring is not implemented"
            )
        if self.num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if not 0 <= self.shard_idx < self.num_shards:
            raise ValueError("shard_idx must satisfy 0 <= shard_idx < num_shards")
        if self.entropy_lambda < 0:
            raise ValueError("entropy_lambda must be non-negative")


@dataclass(frozen=True)
class ScoringTask:
    """One image/target pair to score."""

    sample_id: str
    version: int
    image_path: Path
    target_text: str


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _format_optional_float(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.6f}" if number == number and abs(number) != float("inf") else ""


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _character_metrics(detected_text: str, target_text: str) -> dict[str, int | float | bool | str]:
    detected = _normalize_text(detected_text)
    target = _normalize_text(target_text)
    compared_total = max(len(detected), len(target))
    matches = sum(1 for left, right in zip(detected, target, strict=False) if left == right)
    accuracy = 1.0 if compared_total == 0 else matches / compared_total
    exact = detected == target and bool(target)
    if not detected:
        detection_status = "not_detected"
    elif exact:
        detection_status = "detected_exact"
    else:
        detection_status = "detected_mismatch"
    return {
        "detected_text": detected_text,
        "exact_text_match": exact,
        "char_accuracy": accuracy,
        "char_matches": matches,
        "char_total": compared_total,
        "detection_status": detection_status,
    }


def build_canonical_score_row(
    *,
    sample_id: str,
    version: int,
    target_text: str,
    evidence: dict[str, Any],
    formula: ProductScoreFormula | None = None,
    manifest_path: str = "",
    primary_score: Literal["vlm", "ocr", "product"] = "product",
) -> dict[str, Any]:
    """Convert scorer evidence into a canonical score CSV row."""

    active_formula = formula or ProductScoreFormula()
    detected_text = str(evidence.get("ocr_detected") or "")
    text_metrics = _character_metrics(detected_text, target_text)
    product_evidence = {
        **evidence,
        "exact_text_match": text_metrics["exact_text_match"],
    }
    product = compute_product_score(product_evidence, formula=active_formula)
    threshold_flags = dict(product.threshold_flags)
    scorer_metadata = {
        "formula_name": active_formula.name,
        "scorer_versions": dict(active_formula.scorer_versions),
        "primary_score": primary_score,
    }

    primary_values = {
        "vlm": _format_optional_float(evidence.get("score_vlm")),
        "ocr": _format_optional_float(evidence.get("score_ocr")),
        "product": f"{product.score:.6f}",
    }
    primary_value = primary_values[primary_score] or "0.000000"

    row: dict[str, Any] = {
        "id": sample_id,
        "sample_id": sample_id,
        "version": version,
        "score": primary_value,
        "product_score": f"{product.score:.6f}",
        "target_text": target_text,
        "score_vlm": _format_optional_float(evidence.get("score_vlm")),
        "score_ocr": _format_optional_float(evidence.get("score_ocr")),
        "cer": _format_optional_float(evidence.get("cer")),
        "entropy": _format_optional_float(evidence.get("entropy")),
        "official_conf": _format_optional_float(evidence.get("official_conf")),
        "min_p": _format_optional_float(evidence.get("min_p")),
        "frac_unc": _format_optional_float(evidence.get("frac_unc")),
        "ocr_detected": detected_text,
        "detection_status": text_metrics["detection_status"],
        "exact_text_match": "true" if text_metrics["exact_text_match"] else "false",
        "char_accuracy": f"{float(text_metrics['char_accuracy']):.6f}",
        "char_matches": text_metrics["char_matches"],
        "char_total": text_metrics["char_total"],
        "missing_components": ",".join(product.missing_components),
        "formula_complete": "true" if product.formula_complete else "false",
        "manifest_path": manifest_path,
        "text_metrics": _json_dumps(text_metrics),
        "scorer_metadata": _json_dumps(scorer_metadata),
        "thresholds": _json_dumps(threshold_flags),
    }
    return row


def write_score_schema_sidecar(
    output_csv: str | Path,
    *,
    formula: ProductScoreFormula | None = None,
    source_manifest_paths: list[str] | tuple[str, ...] = (),
    generated_at: str | None = None,
    primary_score: Literal["vlm", "ocr", "product"] = "product",
    execution_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write canonical score-file metadata next to a score CSV."""

    output_path = Path(output_csv)
    sidecar_path = output_path.with_suffix(".schema.json")
    metadata = build_score_metadata(
        formula=formula,
        source_manifest_paths=source_manifest_paths,
        generated_at=generated_at,
    )
    metadata["source_manifest_sha256"] = _source_manifest_hashes(source_manifest_paths)
    metadata.update(
        {
            "score_file_schema_version": PHASE6_SCORE_FILE_SCHEMA_VERSION,
            "required_fields": ["id", "version", "score", "target_text"],
            "required_phase6_fields": list(CANONICAL_SCORE_COLUMNS),
            "primary_score": primary_score,
            "execution": execution_metadata or {},
        }
    )
    temporary_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(sidecar_path)
    return sidecar_path


def collect_scoring_tasks(*, images_dir: Path, text_embeds_dir: Path) -> list[ScoringTask]:
    """Collect deterministic generated-image scoring tasks from local metadata."""

    tasks: list[ScoringTask] = []
    for prompt_dir in sorted(images_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_id = prompt_dir.name
        embed_path = text_embeds_dir / f"{prompt_id}.pt"
        if not embed_path.exists():
            raise ValueError(f"No text embedding for {prompt_id}: {embed_path}")
        embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
        target_text = embed_data.get("target_text", "")
        if not target_text:
            raise ValueError(f"No target_text in {embed_path}")

        seen_versions: set[int] = set()
        for img_file in sorted(prompt_dir.glob("v*.png")):
            match = re.fullmatch(r"v(0|[1-9]\d*)\.png", img_file.name)
            if match is None:
                raise ValueError(f"Invalid image version filename: {img_file}")
            if img_file.is_symlink():
                raise ValueError(f"Image artifact must not be a symlink: {img_file}")
            version = int(match.group(1))
            if version in seen_versions:
                raise ValueError(f"Duplicate image version for {prompt_id}: v{version}")
            seen_versions.add(version)
            tasks.append(
                ScoringTask(
                    sample_id=prompt_id,
                    version=version,
                    image_path=img_file,
                    target_text=target_text,
                )
            )
    if not tasks:
        raise ValueError(f"No scoreable images found under {images_dir}")
    return tasks


def _sharded_output_csv(config: ScoringConfig) -> Path:
    if config.num_shards <= 1:
        return config.output_csv
    base, ext = os.path.splitext(str(config.output_csv))
    return Path(f"{base}_shard{config.shard_idx:03d}{ext}")


def _apply_sharding(tasks: list[ScoringTask], config: ScoringConfig) -> list[ScoringTask]:
    if config.num_shards <= 1:
        return tasks
    sharded = [
        task for index, task in enumerate(tasks) if index % config.num_shards == config.shard_idx
    ]
    logger.info("Shard %d/%d: %d images", config.shard_idx, config.num_shards, len(sharded))
    return sharded


def _filter_already_scored(
    *,
    tasks: list[ScoringTask],
    output_csv: Path,
    resume: bool,
    formula: ProductScoreFormula | None = None,
    primary_score: str | None = None,
    manifest_path: str | None = None,
) -> tuple[list[ScoringTask], bool]:
    write_header = not output_csv.exists() or not resume
    if not resume or not output_csv.exists():
        return tasks, write_header

    expected_tasks = {(task.sample_id, task.version): task for task in tasks}
    scored_keys: set[tuple[str, int]] = set()
    with output_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != CANONICAL_SCORE_COLUMNS:
            raise ValueError(f"resume score header does not match canonical schema: {output_csv}")
        for row_index, row in enumerate(reader, start=2):
            try:
                key = (row["id"], int(row["version"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"resume score row {row_index} has an invalid id/version: {output_csv}"
                ) from exc
            if key in scored_keys:
                raise ValueError(f"resume score file contains duplicate row: {key}")
            task = expected_tasks.get(key)
            if task is None:
                raise ValueError(
                    f"resume score file contains rows outside the current shard: {[key]}"
                )
            if formula is not None and primary_score is not None and manifest_path is not None:
                _validate_resume_row(
                    row,
                    task=task,
                    formula=formula,
                    primary_score=primary_score,
                    manifest_path=manifest_path,
                    row_index=row_index,
                    output_csv=output_csv,
                )
            scored_keys.add(key)
    logger.info("Resuming: %d already scored", len(scored_keys))
    remaining = [task for task in tasks if (task.sample_id, task.version) not in scored_keys]
    logger.info("Remaining: %d to score", len(remaining))
    return remaining, write_header


def _parse_optional_finite_float(value: Any, *, field: str, row_index: int) -> float | None:
    if value == "" or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"resume score row {row_index} has invalid {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"resume score row {row_index} has non-finite {field}")
    return number


def _load_row_json(row: dict[str, str], field: str, *, row_index: int) -> Any:
    try:
        return json.loads(row[field])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"resume score row {row_index} has invalid {field} JSON") from exc


def _validate_resume_row(
    row: dict[str, str],
    *,
    task: ScoringTask,
    formula: ProductScoreFormula,
    primary_score: Literal["vlm", "ocr", "product"],
    manifest_path: str,
    row_index: int,
    output_csv: Path,
) -> None:
    """Validate persisted evidence before trusting a row during resume."""

    if row.get("sample_id") != task.sample_id or row.get("id") != task.sample_id:
        raise ValueError(f"resume score row {row_index} has inconsistent sample identity")
    if row.get("target_text") != task.target_text:
        raise ValueError(
            f"resume score row {row_index} target_text does not match current metadata"
        )
    if row.get("manifest_path") != manifest_path:
        raise ValueError(f"resume score row {row_index} manifest_path does not match current run")

    evidence = {
        field: _parse_optional_finite_float(row.get(field), field=field, row_index=row_index)
        for field in (
            "score_vlm",
            "score_ocr",
            "cer",
            "entropy",
            "official_conf",
            "min_p",
            "frac_unc",
        )
    }
    detected_text = row.get("ocr_detected") or ""
    text_metrics = _character_metrics(detected_text, task.target_text)
    evidence["exact_text_match"] = text_metrics["exact_text_match"]
    product = compute_product_score(evidence, formula=formula)

    product_score = _parse_optional_finite_float(
        row.get("product_score"), field="product_score", row_index=row_index
    )
    score = _parse_optional_finite_float(row.get("score"), field="score", row_index=row_index)
    if product_score is None or score is None:
        raise ValueError(f"resume score row {row_index} lacks a numeric score")
    if not math.isclose(product_score, product.score, rel_tol=0.0, abs_tol=5e-6):
        raise ValueError(f"resume score row {row_index} product_score is inconsistent")
    expected_primary = {
        "vlm": evidence["score_vlm"] or 0.0,
        "ocr": evidence["score_ocr"] or 0.0,
        "product": product.score,
    }.get(primary_score)
    if expected_primary is None or not math.isclose(
        score, expected_primary, rel_tol=0.0, abs_tol=5e-6
    ):
        raise ValueError(f"resume score row {row_index} primary score is inconsistent")

    expected_text_metrics = {
        **text_metrics,
        "char_accuracy": float(text_metrics["char_accuracy"]),
    }
    stored_text_metrics = _load_row_json(row, "text_metrics", row_index=row_index)
    if stored_text_metrics != expected_text_metrics:
        raise ValueError(f"resume score row {row_index} text_metrics are inconsistent")
    if row.get("detection_status") != text_metrics["detection_status"]:
        raise ValueError(f"resume score row {row_index} detection_status is inconsistent")
    if row.get("exact_text_match") != ("true" if text_metrics["exact_text_match"] else "false"):
        raise ValueError(f"resume score row {row_index} exact_text_match is inconsistent")
    if row.get("char_matches") != str(text_metrics["char_matches"]):
        raise ValueError(f"resume score row {row_index} char_matches is inconsistent")
    if row.get("char_total") != str(text_metrics["char_total"]):
        raise ValueError(f"resume score row {row_index} char_total is inconsistent")
    char_accuracy = _parse_optional_finite_float(
        row.get("char_accuracy"), field="char_accuracy", row_index=row_index
    )
    if char_accuracy is None or not math.isclose(
        char_accuracy, float(text_metrics["char_accuracy"]), rel_tol=0.0, abs_tol=5e-6
    ):
        raise ValueError(f"resume score row {row_index} char_accuracy is inconsistent")

    expected_metadata = {
        "formula_name": formula.name,
        "scorer_versions": dict(formula.scorer_versions),
        "primary_score": primary_score,
    }
    if _load_row_json(row, "scorer_metadata", row_index=row_index) != expected_metadata:
        raise ValueError(f"resume score row {row_index} scorer_metadata are inconsistent")
    if _load_row_json(row, "thresholds", row_index=row_index) != dict(product.threshold_flags):
        raise ValueError(f"resume score row {row_index} thresholds are inconsistent")
    if row.get("missing_components") != ",".join(product.missing_components):
        raise ValueError(f"resume score row {row_index} missing_components are inconsistent")
    if row.get("formula_complete") != ("true" if product.formula_complete else "false"):
        raise ValueError(f"resume score row {row_index} formula_complete is inconsistent")

    logger.debug("Validated resume score row %d in %s", row_index, output_csv)


def _formula_for_config(config: ScoringConfig) -> tuple[ProductScoreFormula, str]:
    scorer_versions: dict[str, str] = {}
    if config.scorer in {"vlm", "both"}:
        scorer_versions["vlm"] = (
            f"{config.vlm_model_id}@{config.vlm_model_revision}"
            if config.vlm_model_revision
            else config.vlm_model_id
        )
    if config.scorer in {"ocr", "both"}:
        scorer_versions["ocr"] = f"paddleocr-ru-entropy-lambda-{config.entropy_lambda}"
    formula = (
        thesis_product_formula(scorer_versions=scorer_versions)
        if config.product_formula == "thesis"
        else ProductScoreFormula(scorer_versions=scorer_versions)
    )
    primary_score = {"vlm": "vlm", "ocr": "ocr", "both": "product"}[config.scorer]
    return formula, primary_score


def _validate_resume_sidecar(
    output_csv: Path,
    *,
    formula: ProductScoreFormula,
    primary_score: str,
    source_manifest_paths: tuple[str, ...],
    shard_idx: int,
    num_shards: int,
    discovered_task_count: int | None = None,
    expected_shard_count: int | None = None,
) -> dict[str, Any]:
    sidecar = output_csv.with_suffix(".schema.json")
    if not sidecar.is_file():
        raise ValueError(f"resume requires an existing score sidecar: {sidecar}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"resume score sidecar is malformed: {sidecar}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"resume score sidecar must be an object: {sidecar}")
    expected_contract = {
        "score_file_schema_version": PHASE6_SCORE_FILE_SCHEMA_VERSION,
        "formula": formula.to_metadata(),
        "primary_score": primary_score,
        "source_manifest_paths": list(source_manifest_paths),
        "source_manifest_sha256": _source_manifest_hashes(source_manifest_paths),
        "shard_idx": shard_idx,
        "num_shards": num_shards,
    }
    actual_contract = {
        "score_file_schema_version": payload.get("score_file_schema_version"),
        "formula": payload.get("formula"),
        "primary_score": payload.get("primary_score"),
        "source_manifest_paths": payload.get("source_manifest_paths"),
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "shard_idx": payload.get("execution", {}).get("shard_idx"),
        "num_shards": payload.get("execution", {}).get("num_shards"),
    }
    if actual_contract != expected_contract:
        raise ValueError("resume score sidecar does not match the current scoring contract")

    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("resume score sidecar lacks execution metadata")
    status = execution.get("status")
    if status not in {"in-progress", "complete"}:
        raise ValueError(f"resume score sidecar has invalid execution status: {status!r}")
    if (
        discovered_task_count is not None
        and execution.get("discovered_task_count") != discovered_task_count
    ):
        raise ValueError("resume score sidecar discovered_task_count does not match current tasks")
    if (
        expected_shard_count is not None
        and execution.get("expected_shard_count") != expected_shard_count
    ):
        raise ValueError("resume score sidecar expected_shard_count does not match current shard")

    actual_row_count = _count_csv_rows(output_csv)
    recorded_row_count = execution.get("scored_row_count")
    if not isinstance(recorded_row_count, int) or recorded_row_count < 0:
        raise ValueError("resume score sidecar has invalid scored_row_count")
    if recorded_row_count != actual_row_count:
        raise ValueError("resume score sidecar row count does not match the score CSV")

    recorded_hash = execution.get("scores_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ValueError("resume score sidecar lacks a valid checkpoint scores_sha256")
    if recorded_hash != _sha256(output_csv):
        raise ValueError("resume score sidecar hash does not match the score CSV")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_hashes(paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            hashes[str(raw_path)] = _sha256(path)
    return hashes


def _source_manifests_for_sidecar(config: ScoringConfig) -> tuple[str, ...]:
    if config.source_manifests:
        return config.source_manifests
    if config.manifest_path:
        return (config.manifest_path,)
    return ()


def _require_source_manifests(paths: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
    from src.runtime.manifests import ManifestError, validate_source_manifest

    if not paths:
        raise ValueError("scoring requires at least one source manifest")
    validated: list[tuple[str, dict[str, Any]]] = []
    for raw_path in paths:
        try:
            payload = validate_source_manifest(raw_path)
        except ManifestError as exc:
            raise ValueError(f"invalid source manifest {raw_path}: {exc}") from exc
        validated.append((raw_path, payload))
    return validated


def _validate_generation_task_coverage(
    config: ScoringConfig,
    tasks: list[ScoringTask],
    source_manifests: list[tuple[str, dict[str, Any]]],
) -> None:
    generation_sources = [
        (raw_path, payload)
        for raw_path, payload in source_manifests
        if payload.get("schema_version") == "generation-manifest/v4"
    ]
    if not generation_sources:
        raise ValueError(
            "scoring requires at least one complete generation-manifest/v4 to bind image coverage"
        )

    images_dir = config.images_dir.resolve()
    text_embeds_dir = config.text_embeds_dir.resolve()
    expected_keys: set[tuple[str, int]] = set()
    for raw_path, payload in generation_sources:
        contract = payload["contract"]
        output_dir = Path(str(contract["output_dir"])).resolve()
        if images_dir != output_dir / "images" or text_embeds_dir != output_dir / "text_embeds":
            raise ValueError(
                f"generation source {raw_path} does not own the configured scoring directories"
            )
        if not contract.get("save_png"):
            raise ValueError(f"generation source {raw_path} does not declare PNG outputs")
        source_keys = {
            (f"{prompt_index:06d}", version)
            for prompt_index in range(int(contract["start_idx"]), int(contract["end_idx"]))
            for version in range(int(contract["versions_per_prompt"]))
        }
        overlap = expected_keys & source_keys
        if overlap:
            raise ValueError(
                "generation source manifests contain overlapping task coverage: "
                f"{sorted(overlap)[:3]}"
            )
        expected_keys.update(source_keys)

    actual_keys = {(task.sample_id, task.version) for task in tasks}
    if len(actual_keys) != len(tasks):
        raise ValueError("discovered scoring tasks contain duplicate prompt/version keys")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "scoring task coverage does not match generation source manifests: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def _assert_source_manifest_hashes(paths: tuple[str, ...], expected_hashes: dict[str, str]) -> None:
    if _source_manifest_hashes(paths) != expected_hashes:
        raise ValueError("source manifest bytes changed during scoring")


def run_scoring(config: ScoringConfig) -> None:
    """Run scorer selection, CSV writing, and score sidecar creation."""

    all_tasks = collect_scoring_tasks(
        images_dir=config.images_dir,
        text_embeds_dir=config.text_embeds_dir,
    )
    logger.info("Found %d images to score", len(all_tasks))

    tasks = _apply_sharding(all_tasks, config)
    expected_shard_count = len(tasks)
    output_csv = _sharded_output_csv(config)
    formula, primary_score = _formula_for_config(config)
    source_manifests = _source_manifests_for_sidecar(config)
    validated_source_manifests = _require_source_manifests(source_manifests)
    _validate_generation_task_coverage(config, all_tasks, validated_source_manifests)
    source_manifest_hashes = _source_manifest_hashes(source_manifests)
    resume_sidecar: dict[str, Any] | None = None
    if config.resume and output_csv.exists():
        resume_sidecar = _validate_resume_sidecar(
            output_csv,
            formula=formula,
            primary_score=primary_score,
            source_manifest_paths=source_manifests,
            shard_idx=config.shard_idx,
            num_shards=config.num_shards,
            discovered_task_count=len(all_tasks),
            expected_shard_count=expected_shard_count,
        )
    tasks, write_header = _filter_already_scored(
        tasks=tasks,
        output_csv=output_csv,
        resume=config.resume,
        formula=formula,
        primary_score=primary_score,
        manifest_path=config.manifest_path,
    )

    if (
        not tasks
        and resume_sidecar is not None
        and resume_sidecar["execution"]["status"] == "complete"
    ):
        logger.info("Nothing to score; existing shard is already complete.")
        return

    if not tasks:
        if write_header:
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS).writeheader()
        existing_count = _count_csv_rows(output_csv) if output_csv.exists() else 0
        if existing_count != expected_shard_count:
            raise ValueError(
                "score output is incomplete: "
                f"expected {expected_shard_count}, found {existing_count}"
            )
        if resume_sidecar is None or resume_sidecar["execution"]["status"] != "complete":
            _assert_source_manifest_hashes(source_manifests, source_manifest_hashes)
            write_score_schema_sidecar(
                output_csv,
                formula=formula,
                source_manifest_paths=source_manifests,
                primary_score=primary_score,
                execution_metadata=_complete_execution_metadata(
                    output_csv,
                    discovered_task_count=len(all_tasks),
                    expected_shard_count=expected_shard_count,
                    shard_idx=config.shard_idx,
                    num_shards=config.num_shards,
                ),
            )
            logger.info(
                "Repaired complete score sidecar → %s", output_csv.with_suffix(".schema.json")
            )
        logger.info("Nothing to score; existing shard is complete.")
        return

    from src.runtime.capabilities import check_stage_support

    support = check_stage_support(
        "score",
        scorer=config.scorer,
        ocr_device=config.ocr_device,
    )
    if not support.ok:
        raise RuntimeError("; ".join(support.errors))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    vlm_model = None
    ocr_model = None

    if config.scorer in ("vlm", "both"):
        from src.training.rewards import QwenYesProbReward

        vlm_model = QwenYesProbReward(
            model_id=config.vlm_model_id,
            device=config.vlm_device,
            revision=config.vlm_model_revision,
        )

    if config.scorer in ("ocr", "both"):
        from src.training.rewards import OcrCerEntropyReward

        ocr_model = OcrCerEntropyReward(
            lang="ru",
            device=config.ocr_device,
            entropy_lambda=config.entropy_lambda,
        )

    mode = "w" if write_header else "a"

    with output_csv.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS)
        if write_header:
            writer.writeheader()
            _persist_score_checkpoint(
                handle,
                output_csv=output_csv,
                formula=formula,
                source_manifests=source_manifests,
                primary_score=primary_score,
                discovered_task_count=len(all_tasks),
                expected_shard_count=expected_shard_count,
                shard_idx=config.shard_idx,
                num_shards=config.num_shards,
                source_manifest_hashes=source_manifest_hashes,
            )

        for task in tqdm(tasks, desc="Scoring"):
            evidence: dict[str, Any] = {}
            score_vlm = None
            score_ocr = None

            if vlm_model:
                import torchvision.transforms.functional as TF
                from PIL import Image

                pil_image = Image.open(task.image_path).convert("RGB")
                img_tensor = TF.to_tensor(pil_image).to(config.vlm_device)
                with torch.no_grad():
                    score_vlm = vlm_model.score_single(img_tensor, task.target_text).item()
                    evidence["score_vlm"] = score_vlm

            if ocr_model:
                ocr_result = ocr_model.score(str(task.image_path), task.target_text)
                score_ocr = ocr_result["reward_ocr"]
                evidence.update(
                    {
                        "score_ocr": score_ocr,
                        "official_conf": ocr_result.get("official_conf"),
                        "cer": ocr_result.get("cer"),
                        "entropy": ocr_result.get("entropy"),
                        "min_p": ocr_result.get("min_p"),
                        "frac_unc": ocr_result.get("frac_unc"),
                        "ocr_detected": ocr_result.get("ocr_detected", ""),
                    }
                )

            if score_vlm is not None and "score_vlm" not in evidence:
                evidence["score_vlm"] = score_vlm
            if score_ocr is not None and "score_ocr" not in evidence:
                evidence["score_ocr"] = score_ocr

            row = build_canonical_score_row(
                sample_id=task.sample_id,
                version=task.version,
                target_text=task.target_text,
                evidence=evidence,
                formula=formula,
                manifest_path=config.manifest_path,
                primary_score=primary_score,
            )
            writer.writerow(row)
            _persist_score_checkpoint(
                handle,
                output_csv=output_csv,
                formula=formula,
                source_manifests=source_manifests,
                primary_score=primary_score,
                discovered_task_count=len(all_tasks),
                expected_shard_count=expected_shard_count,
                shard_idx=config.shard_idx,
                num_shards=config.num_shards,
                source_manifest_hashes=source_manifest_hashes,
            )

    logger.info("Scoring complete. Results → %s", output_csv)
    scored_row_count = _count_csv_rows(output_csv)
    if scored_row_count != expected_shard_count:
        raise ValueError(
            f"score output is incomplete: expected {expected_shard_count}, found {scored_row_count}"
        )
    _assert_source_manifest_hashes(source_manifests, source_manifest_hashes)
    sidecar = write_score_schema_sidecar(
        output_csv,
        formula=formula,
        source_manifest_paths=source_manifests,
        primary_score=primary_score,
        execution_metadata=_complete_execution_metadata(
            output_csv,
            discovered_task_count=len(all_tasks),
            expected_shard_count=expected_shard_count,
            shard_idx=config.shard_idx,
            num_shards=config.num_shards,
        ),
    )
    logger.info("Score schema metadata → %s", sidecar)

    scores = []
    with output_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores.append(float(row["score"]))

    if scores:
        logger.info(
            "Score stats: n=%d, mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
            len(scores),
            statistics.mean(scores),
            statistics.median(scores),
            min(scores),
            max(scores),
        )


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _complete_execution_metadata(
    output_csv: Path,
    *,
    discovered_task_count: int,
    expected_shard_count: int,
    shard_idx: int,
    num_shards: int,
) -> dict[str, Any]:
    return {
        "discovered_task_count": discovered_task_count,
        "expected_shard_count": expected_shard_count,
        "scored_row_count": _count_csv_rows(output_csv),
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "scores_sha256": _sha256(output_csv),
        "status": "complete",
    }


def _persist_score_checkpoint(
    handle: Any,
    *,
    output_csv: Path,
    formula: ProductScoreFormula,
    source_manifests: tuple[str, ...],
    primary_score: Literal["vlm", "ocr", "product"],
    discovered_task_count: int,
    expected_shard_count: int,
    shard_idx: int,
    num_shards: int,
    source_manifest_hashes: dict[str, str],
) -> None:
    """Persist CSV bytes before atomically checkpointing their exact identity."""

    handle.flush()
    os.fsync(handle.fileno())
    _assert_source_manifest_hashes(source_manifests, source_manifest_hashes)
    write_score_schema_sidecar(
        output_csv,
        formula=formula,
        source_manifest_paths=source_manifests,
        primary_score=primary_score,
        execution_metadata={
            "discovered_task_count": discovered_task_count,
            "expected_shard_count": expected_shard_count,
            "scored_row_count": _count_csv_rows(output_csv),
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "scores_sha256": _sha256(output_csv),
            "status": "in-progress",
        },
    )
