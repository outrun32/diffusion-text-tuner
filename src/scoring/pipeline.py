"""Importable reward scoring pipeline implementation.

This module owns task discovery, canonical Phase 6 score row conversion,
schema sidecar writing, resume/shard behavior, scorer selection, and CSV
writing for ``python -m scripts.score_images`` while keeping optional model
and OCR stacks inside ``run_scoring`` execution branches.
"""

from __future__ import annotations

import csv
import json
import logging
import os
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
    entropy_lambda: float = 1.0
    batch_size: int = 1
    resume: bool = False
    shard_idx: int = 0
    num_shards: int = 1
    manifest_path: str = ""
    source_manifests: tuple[str, ...] = ()


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
) -> dict[str, Any]:
    """Convert scorer evidence into a canonical Phase 6 CSV row."""

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
    }

    row: dict[str, Any] = {
        "id": sample_id,
        "sample_id": sample_id,
        "version": version,
        "score": f"{product.score:.6f}",
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
) -> Path:
    """Write canonical score-file metadata next to a score CSV."""

    output_path = Path(output_csv)
    sidecar_path = output_path.with_suffix(".schema.json")
    metadata = build_score_metadata(
        formula=formula,
        source_manifest_paths=source_manifest_paths,
        generated_at=generated_at,
    )
    metadata.update(
        {
            "score_file_schema_version": PHASE6_SCORE_FILE_SCHEMA_VERSION,
            "required_fields": ["id", "version", "score", "target_text"],
            "required_phase6_fields": list(CANONICAL_SCORE_COLUMNS),
        }
    )
    sidecar_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
            logger.warning("No text embedding for %s, skipping", prompt_id)
            continue
        embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
        target_text = embed_data.get("target_text", "")
        if not target_text:
            logger.warning("No target_text in %s, skipping", prompt_id)
            continue

        for img_file in sorted(prompt_dir.glob("v*.png")):
            version = int(img_file.stem[1:])
            tasks.append(
                ScoringTask(
                    sample_id=prompt_id,
                    version=version,
                    image_path=img_file,
                    target_text=target_text,
                )
            )
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
) -> tuple[list[ScoringTask], bool]:
    write_header = not output_csv.exists() or not resume
    if not resume or not output_csv.exists():
        return tasks, write_header

    scored_keys: set[tuple[str, int]] = set()
    with output_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scored_keys.add((row["id"], int(row["version"])))
    logger.info("Resuming: %d already scored", len(scored_keys))
    remaining = [
        task for task in tasks if (task.sample_id, task.version) not in scored_keys
    ]
    logger.info("Remaining: %d to score", len(remaining))
    return remaining, write_header


def _source_manifests_for_sidecar(config: ScoringConfig) -> tuple[str, ...]:
    if config.source_manifests:
        return config.source_manifests
    if config.manifest_path:
        return (config.manifest_path,)
    return ()


def run_scoring(config: ScoringConfig) -> None:
    """Run scorer selection, CSV writing, and score sidecar creation."""

    tasks = collect_scoring_tasks(
        images_dir=config.images_dir,
        text_embeds_dir=config.text_embeds_dir,
    )
    logger.info("Found %d images to score", len(tasks))

    tasks = _apply_sharding(tasks, config)
    output_csv = _sharded_output_csv(config)
    tasks, write_header = _filter_already_scored(
        tasks=tasks,
        output_csv=output_csv,
        resume=config.resume,
    )

    if not tasks:
        logger.info("Nothing to score, exiting.")
        return

    vlm_model = None
    ocr_model = None

    if config.scorer in ("vlm", "both"):
        from src.training.rewards import QwenYesProbReward

        vlm_model = QwenYesProbReward(model_id=config.vlm_model_id, device="cuda")

    if config.scorer in ("ocr", "both"):
        from src.training.rewards import OcrCerEntropyReward

        ocr_model = OcrCerEntropyReward(
            lang="ru",
            device="gpu",
            entropy_lambda=config.entropy_lambda,
        )

    scorer_versions = {}
    if vlm_model:
        scorer_versions["vlm"] = config.vlm_model_id
    if ocr_model:
        scorer_versions["ocr"] = f"paddleocr-ru-entropy-lambda-{config.entropy_lambda}"
    formula = ProductScoreFormula(scorer_versions=scorer_versions)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"

    with output_csv.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS)
        if write_header:
            writer.writeheader()

        for task in tqdm(tasks, desc="Scoring"):
            evidence: dict[str, Any] = {}
            score_vlm = None
            score_ocr = None

            if vlm_model:
                import torchvision.transforms.functional as TF
                from PIL import Image

                pil_image = Image.open(task.image_path).convert("RGB")
                img_tensor = TF.to_tensor(pil_image).to("cuda")
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
            )
            writer.writerow(row)
            handle.flush()

    logger.info("Scoring complete. Results → %s", output_csv)
    sidecar = write_score_schema_sidecar(
        output_csv,
        formula=formula,
        source_manifest_paths=_source_manifests_for_sidecar(config),
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
