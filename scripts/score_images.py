"""
Score generated images using VLM or OCR reward model.

Reads PNGs from the image generation output, computes scores,
and writes a scores CSV for the SFT/DPO pipeline.

Scorers:
  vlm  — Qwen3.5-9B P(yes) (default, ~2.5GB VRAM, ~500ms/img)
  ocr  — PaddleOCR v5 CER+Entropy (~50MB VRAM, ~10ms/img, needs paddle)
  both — runs both and writes all columns

Usage:
  python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv \
    --scorer vlm

  python -m scripts.score_images \
    --scorer ocr --entropy_lambda 1.5 ...

  python -m scripts.score_images \
    --scorer both ...
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
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
    sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


def main():
    parser = argparse.ArgumentParser(description="Score generated images with VLM")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Dir with {prompt_id}/v{ver}.png images")
    parser.add_argument("--text_embeds_dir", type=str, required=True,
                        help="Dir with {prompt_id}.pt (contains target_text)")
    parser.add_argument("--output_csv", type=str, default="outputs/generated/scores.csv")
    parser.add_argument("--scorer", type=str, default="vlm", choices=["vlm", "ocr", "both"],
                        help="Scoring method: vlm (Qwen yes-prob), ocr (CER+entropy), both")
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--entropy_lambda", type=float, default=1.0,
                        help="Lambda for OCR entropy scaling (R = (1-CER)*exp(-λ*H))")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="VLM scoring batch size (1 is safest for memory)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scored entries if CSV exists")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Shard index for parallel scoring (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel scoring")
    parser.add_argument("--manifest_path", type=str, default="",
                        help="Run/evaluation manifest path to link in each score row")
    parser.add_argument("--source_manifest", action="append", default=[],
                        help="Source manifest path to include in the score schema sidecar")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    images_dir = Path(args.images_dir)
    text_embeds_dir = Path(args.text_embeds_dir)

    # Collect all (prompt_id, version, image_path, target_text) tuples
    tasks = []
    for prompt_dir in sorted(images_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_id = prompt_dir.name

        # Load target_text from text embedding file
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
            version = int(img_file.stem[1:])  # "v3" -> 3
            tasks.append({
                "id": prompt_id,
                "version": version,
                "image_path": str(img_file),
                "target_text": target_text,
            })

    logger.info("Found %d images to score", len(tasks))

    # Shard for parallel scoring (each GPU gets a disjoint slice)
    if args.num_shards > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_idx]
        logger.info("Shard %d/%d: %d images", args.shard_idx, args.num_shards, len(tasks))
        # Per-shard output CSV to avoid write conflicts
        base, ext = os.path.splitext(args.output_csv)
        output_csv = f"{base}_shard{args.shard_idx:03d}{ext}"
    else:
        output_csv = args.output_csv

    # Resume: load already scored
    scored_keys = set()
    if args.resume and os.path.exists(output_csv):
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scored_keys.add((row["id"], int(row["version"])))
        logger.info("Resuming: %d already scored", len(scored_keys))
        tasks = [t for t in tasks if (t["id"], t["version"]) not in scored_keys]
        logger.info("Remaining: %d to score", len(tasks))

    if not tasks:
        logger.info("Nothing to score, exiting.")
        return

    # Load scorer(s)
    vlm_model = None
    ocr_model = None

    if args.scorer in ("vlm", "both"):
        from src.training.rewards import QwenYesProbReward
        vlm_model = QwenYesProbReward(model_id=args.vlm_model_id, device="cuda")

    if args.scorer in ("ocr", "both"):
        from src.training.rewards import OcrCerEntropyReward
        ocr_model = OcrCerEntropyReward(
            lang="ru", device="gpu", entropy_lambda=args.entropy_lambda,
        )

    scorer_versions = {}
    if vlm_model:
        scorer_versions["vlm"] = args.vlm_model_id
    if ocr_model:
        scorer_versions["ocr"] = f"paddleocr-ru-entropy-lambda-{args.entropy_lambda}"
    formula = ProductScoreFormula(scorer_versions=scorer_versions)
    all_fields = CANONICAL_SCORE_COLUMNS

    # Score
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv) or not args.resume
    mode = "w" if write_header else "a"

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        if write_header:
            writer.writeheader()

        for task in tqdm(tasks, desc="Scoring"):
            evidence = {}
            score_vlm = None
            score_ocr = None

            if vlm_model:
                pil_image = Image.open(task["image_path"]).convert("RGB")
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(pil_image).to("cuda")
                with torch.no_grad():
                    score_vlm = vlm_model.score_single(
                        img_tensor, task["target_text"]).item()
                    evidence["score_vlm"] = score_vlm

            if ocr_model:
                ocr_result = ocr_model.score(task["image_path"], task["target_text"])
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
                sample_id=task["id"],
                version=task["version"],
                target_text=task["target_text"],
                evidence=evidence,
                formula=formula,
                manifest_path=args.manifest_path,
            )

            writer.writerow(row)
            f.flush()

    logger.info("Scoring complete. Results → %s", output_csv)
    sidecar = write_score_schema_sidecar(
        output_csv,
        formula=formula,
        source_manifest_paths=tuple(args.source_manifest or ([args.manifest_path] if args.manifest_path else [])),
    )
    logger.info("Score schema metadata → %s", sidecar)

    # Print summary stats
    scores = []
    with open(output_csv) as f:
        for row in csv.DictReader(f):
            scores.append(float(row["score"]))

    if scores:
        import statistics
        logger.info(
            "Score stats: n=%d, mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
            len(scores), statistics.mean(scores), statistics.median(scores),
            min(scores), max(scores),
        )


if __name__ == "__main__":
    main()
