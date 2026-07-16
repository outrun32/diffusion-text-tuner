"""
Reward evaluation for generated images.

Runs two reward signals on baseline images:
    1. Qwen3.5 VLM yes-token probability (CUDA, via transformers)
  2. PaddleOCR v3 character-level accuracy

Reads metadata.jsonl from generate_baseline and appends reward scores.

Usage:
    python -m src.evaluation.evaluate_rewards \
        --metadata outputs/baseline/metadata.jsonl \
        --output outputs/baseline/scores.jsonl \
        --reward qwen_yes_prob \
        --vlm-model Qwen/Qwen3.5-4B \
        --batch-size 4

    python -m src.evaluation.evaluate_rewards \
        --metadata outputs/baseline/metadata.jsonl \
        --output outputs/baseline/scores.jsonl \
        --reward paddleocr
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.reward_interface import (
    ProductScoreFormula,
    build_score_metadata,
    compute_product_score,
)
from src.runtime.artifacts import PHASE6_REQUIRED_SCORE_FIELDS
from src.runtime.capabilities import check_stage_support
from src.training.rewards import (
    EvaluationQwenYesProbReward as QwenYesProbReward,
)
from src.training.rewards import (
    PaddleOCRAccuracyReward as PaddleOCRReward,
)

PHASE6_JSONL_SCHEMA_VERSION = "phase6-score-jsonl/v1"


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


def _sample_id_from_record(record: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "prompt_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    image_path = str(record.get("image") or record.get("image_path") or "")
    return Path(image_path).stem if image_path else "unknown"


def build_canonical_evaluation_record(
    *,
    source_record: dict[str, Any],
    reward_outputs: dict[str, Any],
    version: int = 0,
    formula: ProductScoreFormula | None = None,
    manifest_path: str = "",
    primary_score: str = "product",
) -> dict[str, Any]:
    """Convert raw evaluation reward outputs into canonical score JSONL fields."""

    active_formula = formula or ProductScoreFormula()
    target_text = str(source_record.get("target_text") or "")
    detected_text = str(reward_outputs.get("ocr_detected") or "")
    text_metrics = _character_metrics(detected_text, target_text)
    evidence = {
        "score_vlm": reward_outputs.get("score_vlm", reward_outputs.get("reward_qwen_yes_prob")),
        "score_ocr": reward_outputs.get("score_ocr", reward_outputs.get("reward_paddleocr")),
        "cer": reward_outputs.get("cer"),
        "entropy": reward_outputs.get("entropy"),
        "exact_text_match": text_metrics["exact_text_match"],
    }
    product = compute_product_score(evidence, formula=active_formula)
    sample_id = _sample_id_from_record(source_record)
    primary_values = {
        "vlm": evidence["score_vlm"],
        "ocr": evidence["score_ocr"],
        "product": product.score,
    }
    if primary_score not in primary_values:
        raise ValueError(f"unsupported primary_score: {primary_score}")
    score = primary_values[primary_score]
    if score is None:
        score = 0.0

    return {
        **source_record,
        "schema_version": "reward-result/v1",
        "score_file_schema_version": PHASE6_JSONL_SCHEMA_VERSION,
        "sample_id": sample_id,
        "version": version,
        "target_text": target_text,
        "score": score,
        "product_score": product.score,
        "score_vlm": evidence["score_vlm"],
        "score_ocr": evidence["score_ocr"],
        "cer": evidence["cer"],
        "entropy": evidence["entropy"],
        "ocr_detected": detected_text,
        "detection_status": text_metrics["detection_status"],
        "exact_text_match": text_metrics["exact_text_match"],
        "char_accuracy": text_metrics["char_accuracy"],
        "char_matches": text_metrics["char_matches"],
        "char_total": text_metrics["char_total"],
        "missing_components": list(product.missing_components),
        "formula_complete": product.formula_complete,
        "manifest_path": manifest_path,
        "text_metrics": text_metrics,
        "scorer_metadata": {
            "formula_name": active_formula.name,
            "scorer_versions": dict(active_formula.scorer_versions),
            "primary_score": primary_score,
        },
        "thresholds": dict(product.threshold_flags),
    }


def write_evaluation_score_metadata(
    output_path: str | Path,
    *,
    formula: ProductScoreFormula | None = None,
    source_manifest_paths: list[str] | tuple[str, ...] = (),
    generated_at: str | None = None,
    primary_score: str = "product",
    expected_source_manifest_sha256: dict[str, str] | None = None,
) -> Path:
    """Write canonical JSONL score sidecar metadata."""

    path = Path(output_path)
    if not path.is_file():
        raise FileNotFoundError(f"score output does not exist: {path}")
    if not source_manifest_paths:
        raise ValueError("at least one source manifest is required for evaluation scores")
    _validate_source_manifests(tuple(source_manifest_paths))
    sidecar = path.with_suffix(".schema.json")
    metadata = build_score_metadata(
        formula=formula,
        source_manifest_paths=source_manifest_paths,
        generated_at=generated_at,
    )
    source_hashes = _source_manifest_hashes(tuple(source_manifest_paths))
    if (
        expected_source_manifest_sha256 is not None
        and source_hashes != expected_source_manifest_sha256
    ):
        raise ValueError("source manifest bytes changed during evaluation")
    metadata.update(
        {
            "score_file_schema_version": PHASE6_JSONL_SCHEMA_VERSION,
            "primary_score": primary_score,
            "source_manifest_sha256": source_hashes,
            "required_phase6_fields": sorted(PHASE6_REQUIRED_SCORE_FIELDS),
            "execution": {
                "status": "complete",
                "scored_row_count": _count_jsonl(path),
                "scores_sha256": _sha256(path),
            },
        }
    )
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_sidecar.replace(sidecar)
    return sidecar


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Evaluate rewards on generated images")
    p.add_argument(
        "--metadata", type=str, required=True, help="Path to metadata.jsonl from generate_baseline"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output scores JSONL (default: <metadata_dir>/scores.jsonl)",
    )
    p.add_argument(
        "--reward",
        type=str,
        nargs="+",
        choices=["qwen_yes_prob", "paddleocr", "all"],
        default=["all"],
        help="Which reward(s) to compute",
    )
    p.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="HuggingFace VLM model ID for yes-prob reward",
    )
    p.add_argument(
        "--vlm-revision",
        default=None,
        help="Optional immutable Hugging Face commit hash for the VLM scorer",
    )
    p.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Deprecated compatibility option; partial evaluation is rejected.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing complete score file instead of appending to it.",
    )
    p.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="Run/evaluation manifest path to link in each canonical record",
    )
    p.add_argument(
        "--source-manifest",
        action="append",
        default=[],
        help="Source manifest path to include in the JSONL schema sidecar",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.start_idx != 0:
        raise ValueError(
            "partial --start-idx evaluation is not supported; rerun the complete input "
            "with --overwrite"
        )

    # Resolve rewards
    rewards_to_run = set(args.reward)
    if "all" in rewards_to_run:
        rewards_to_run = {"qwen_yes_prob", "paddleocr"}

    # Load and validate the complete input before model loading or output mutation.
    print(f"Loading metadata from {args.metadata} ...")
    metadata_path = Path(args.metadata)
    records: list[dict[str, Any]] = []
    with metadata_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"metadata line {line_number} is not valid JSON: {metadata_path}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"metadata line {line_number} must be a JSON object")
                records.append(record)
    _validate_evaluation_records(records, metadata_path=metadata_path)
    print(f"  Total records: {len(records)}")

    # Output path
    out_path = Path(args.output) if args.output else metadata_path.parent / "scores.jsonl"
    sidecar_path = out_path.with_suffix(".schema.json")
    if (out_path.exists() or sidecar_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"evaluation output already exists: {out_path}; pass --overwrite to replace it"
        )

    source_manifests = _source_manifests_from_args(
        source_manifest_paths=args.source_manifest,
        manifest_path=args.manifest_path,
    )
    _validate_source_manifests(source_manifests)
    source_manifest_hashes = _source_manifest_hashes(source_manifests)
    record_manifest_path = args.manifest_path or source_manifests[0]

    scorer_kind = (
        "both"
        if rewards_to_run == {"qwen_yes_prob", "paddleocr"}
        else ("vlm" if "qwen_yes_prob" in rewards_to_run else "ocr")
    )
    support = check_stage_support("score", scorer=scorer_kind, ocr_device="cpu")
    if not support.ok:
        raise RuntimeError("; ".join(support.errors))

    # Initialize reward models
    scorers = {}
    if "qwen_yes_prob" in rewards_to_run:
        scorers["qwen_yes_prob"] = QwenYesProbReward(
            model_id=args.vlm_model,
            device="cuda",
            revision=args.vlm_revision,
        )
    if "paddleocr" in rewards_to_run:
        scorers["paddleocr"] = PaddleOCRReward()

    scorer_versions = {}
    if "qwen_yes_prob" in rewards_to_run:
        scorer_versions["vlm"] = (
            f"{args.vlm_model}@{args.vlm_revision}" if args.vlm_revision else args.vlm_model
        )
    if "paddleocr" in rewards_to_run:
        scorer_versions["ocr"] = "paddleocr-PP-OCRv3-cyrillic"
    formula = ProductScoreFormula(
        name="legacy_vlm_char_accuracy_product_v1",
        weights={"score_vlm": 1.0, "score_ocr": 1.0},
        scorer_versions=scorer_versions,
        aggregation="weighted_product",
        require_all=True,
    )
    primary_score = (
        "product"
        if rewards_to_run == {"qwen_yes_prob", "paddleocr"}
        else ("vlm" if "qwen_yes_prob" in rewards_to_run else "ocr")
    )

    # Score into a temporary file so failed runs never publish partial JSONL.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        with temporary_output.open("x", encoding="utf-8") as out_file:
            for record in tqdm(records, desc="Scoring", unit="img"):
                image_path = str(record["image"])
                target_text = str(record["target_text"])
                reward_outputs: dict[str, Any] = {}
                for scorer in scorers.values():
                    result = scorer.score(image_path, target_text)
                    if not isinstance(result, dict):
                        raise TypeError("reward scorer must return a dictionary")
                    reward_outputs.update(result)

                scored = build_canonical_evaluation_record(
                    source_record=record,
                    reward_outputs=reward_outputs,
                    version=int(record.get("version", 0) or 0),
                    formula=formula,
                    manifest_path=record_manifest_path,
                    primary_score=primary_score,
                )
                out_file.write(json.dumps(scored, ensure_ascii=False) + "\n")
            out_file.flush()
        if _count_jsonl(temporary_output) != len(records):
            raise RuntimeError("evaluation output row count does not match metadata input")
        _validate_source_manifests(source_manifests)
        if _source_manifest_hashes(source_manifests) != source_manifest_hashes:
            raise ValueError("source manifest bytes changed during evaluation")
        temporary_output.replace(out_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    # Print summary statistics
    sidecar = write_evaluation_score_metadata(
        out_path,
        formula=formula,
        source_manifest_paths=source_manifests,
        primary_score=primary_score,
        expected_source_manifest_sha256=source_manifest_hashes,
    )
    print(f"\nScores saved to: {out_path}")
    print(f"Score schema metadata saved to: {sidecar}")
    _print_summary(str(out_path), rewards_to_run)
    return 0


def _source_manifests_from_args(
    *, source_manifest_paths: list[str], manifest_path: str
) -> tuple[str, ...]:
    paths = [*source_manifest_paths]
    if manifest_path:
        paths.append(manifest_path)
    unique_paths = tuple(dict.fromkeys(paths))
    if not unique_paths:
        raise ValueError("evaluation requires --source-manifest or --manifest-path provenance")
    return unique_paths


def _validate_source_manifests(paths: tuple[str, ...]) -> None:
    from src.runtime.manifests import ManifestError, validate_source_manifest

    for raw_path in paths:
        try:
            validate_source_manifest(raw_path)
        except ManifestError as exc:
            raise ValueError(f"invalid source manifest {raw_path}: {exc}") from exc


def _source_manifest_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    return {str(raw_path): _sha256(Path(raw_path)) for raw_path in paths}


def _validate_evaluation_records(records: list[dict[str, Any]], *, metadata_path: Path) -> None:
    if not records:
        raise ValueError(f"metadata contains no scoreable records: {metadata_path}")
    seen_keys: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"metadata record {index} lacks a non-empty image path")
        if not Path(image_value).is_file():
            raise FileNotFoundError(f"metadata image does not exist: {image_value}")
        target_text = record.get("target_text")
        if not isinstance(target_text, str) or not target_text:
            raise ValueError(f"metadata record {index} lacks non-empty target_text")
        raw_version = record.get("version", 0)
        if isinstance(raw_version, bool) or not isinstance(raw_version, (int, str)):
            raise ValueError(f"metadata record {index} has invalid version")
        try:
            version = int(raw_version or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata record {index} has invalid version") from exc
        if version < 0 or (isinstance(raw_version, str) and str(version) != raw_version.strip()):
            raise ValueError(f"metadata record {index} has invalid version")
        record["version"] = version
        sample_id = _sample_id_from_record(record)
        if not sample_id or sample_id == "unknown":
            raise ValueError(f"metadata record {index} lacks a stable sample identity")
        key = (sample_id, version)
        if key in seen_keys:
            raise ValueError(f"metadata contains duplicate sample/version: {key}")
        seen_keys.add(key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _print_summary(scores_path: str, reward_names: set):
    """Print basic statistics for each reward."""
    import numpy as np

    records = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    print(f"\n{'=' * 60}")
    print(f"REWARD SUMMARY ({len(records)} images)")
    print(f"{'=' * 60}")

    if "qwen_yes_prob" in reward_names:
        vals = [r.get("reward_qwen_yes_prob", 0.0) for r in records if "reward_qwen_yes_prob" in r]
        if vals:
            arr = np.array(vals)
            print("\n  Qwen Yes-Prob:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"    median={np.median(arr):.4f}")
            # Distribution bins
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
            hist, _ = np.histogram(arr, bins=bins)
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist, strict=False):
                pct = 100 * cnt / len(arr)
                bar = "█" * int(pct / 2)
                print(f"    [{lo:.1f}-{hi:.1f}): {cnt:4d} ({pct:5.1f}%) {bar}")

    if "paddleocr" in reward_names:
        vals = [r.get("reward_paddleocr", 0.0) for r in records if "reward_paddleocr" in r]
        if vals:
            arr = np.array(vals)
            print("\n  PaddleOCR Accuracy:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"    median={np.median(arr):.4f}")
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
            hist, _ = np.histogram(arr, bins=bins)
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist, strict=False):
                pct = 100 * cnt / len(arr)
                bar = "█" * int(pct / 2)
                print(f"    [{lo:.1f}-{hi:.1f}): {cnt:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    raise SystemExit(main())
