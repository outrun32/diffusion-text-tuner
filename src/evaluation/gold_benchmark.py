"""CPU-safe gold diagnostic benchmark contracts.

Gold diagnostic records are small, human-authored JSONL metadata rows used to
check reward/evaluation signals before treating them as thesis evidence. This
module validates schema and compares prediction dictionaries only; it never
loads images, OCR engines, model weights, CUDA, or generated artifacts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.slices import classify_text_slices, summarize_slices

SCHEMA_VERSION = "gold-diagnostic-benchmark/v1"
REQUIRED_FIELDS = (
    "sample_id",
    "target_text",
    "image_path",
    "expected_exact_match",
    "expected_ocr_detected",
    "human_label",
)


class GoldBenchmarkError(ValueError):
    """Raised when a gold diagnostic benchmark cannot be trusted."""


def load_gold_benchmark(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSONL gold diagnostic benchmark.

    Validation aggregates all visible line errors so users can fix the benchmark
    before using it as reward-validity evidence.
    """
    benchmark_path = Path(path)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    try:
        lines = benchmark_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GoldBenchmarkError(f"{benchmark_path}: could not read benchmark") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: malformed JSON")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number}: record must be a JSON object")
            continue

        record_errors = _validate_record(payload, line_number, seen_sample_ids)
        errors.extend(record_errors)
        if not record_errors:
            seen_sample_ids.add(payload["sample_id"])
            records.append(_normalize_record(payload))

    if not records and not errors:
        errors.append("benchmark must contain at least one JSONL record")
    if errors:
        raise GoldBenchmarkError(f"{benchmark_path}: " + "; ".join(errors))

    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(benchmark_path),
        "records": records,
        "slice_summary": summarize_slices(records),
    }


def evaluate_gold_predictions(
    benchmark: Mapping[str, Any] | str | Path,
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare prediction records to a validated gold diagnostic benchmark."""
    loaded = load_gold_benchmark(benchmark) if isinstance(benchmark, (str, Path)) else benchmark
    gold_records = list(loaded["records"])
    predictions_by_id = _index_predictions(predictions)

    exact_agreement = {"agree": 0, "disagree": 0}
    ocr_detection_agreement = {"agree": 0, "disagree": 0}
    ocr_text_agreement = {"agree": 0, "disagree": 0}
    missing_prediction_sample_ids: list[str] = []
    disagreement_records: list[dict[str, Any]] = []
    per_slice: dict[str, dict[str, int]] = defaultdict(_empty_slice_summary)

    for gold_record in gold_records:
        sample_id = gold_record["sample_id"]
        prediction = predictions_by_id.get(sample_id)
        labels = classify_text_slices(gold_record)
        if not labels:
            labels = {"unsliced"}
        for label in labels:
            per_slice[label]["records"] += 1

        if prediction is None:
            missing_prediction_sample_ids.append(sample_id)
            for label in labels:
                per_slice[label]["missing_predictions"] += 1
            continue

        exact_matches = (
            _coerce_bool(prediction.get("exact_text_match")) == gold_record["expected_exact_match"]
        )
        ocr_matches = (
            _coerce_bool(prediction.get("ocr_detected")) == gold_record["expected_ocr_detected"]
        )
        text_matches = _normalize_text(prediction.get("detected_text")) == _normalize_text(
            gold_record["target_text"]
        )

        _count_agreement(exact_agreement, exact_matches)
        _count_agreement(ocr_detection_agreement, ocr_matches)
        _count_agreement(ocr_text_agreement, text_matches)

        if not exact_matches:
            disagreement_records.append(
                {
                    "sample_id": sample_id,
                    "field": "expected_exact_match",
                    "expected": gold_record["expected_exact_match"],
                    "observed": _coerce_bool(prediction.get("exact_text_match")),
                }
            )
            for label in labels:
                per_slice[label]["exact_disagreements"] += 1
        if not ocr_matches:
            disagreement_records.append(
                {
                    "sample_id": sample_id,
                    "field": "expected_ocr_detected",
                    "expected": gold_record["expected_ocr_detected"],
                    "observed": _coerce_bool(prediction.get("ocr_detected")),
                }
            )
            for label in labels:
                per_slice[label]["ocr_detection_disagreements"] += 1
        if not text_matches:
            for label in labels:
                per_slice[label]["ocr_text_disagreements"] += 1

    findings = _build_findings(
        missing_prediction_sample_ids,
        exact_agreement,
        ocr_detection_agreement,
        ocr_text_agreement,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(loaded["source_path"]),
        "total_gold_records": len(gold_records),
        "matched_prediction_count": len(gold_records) - len(missing_prediction_sample_ids),
        "missing_prediction_count": len(missing_prediction_sample_ids),
        "missing_prediction_sample_ids": missing_prediction_sample_ids,
        "exact_agreement": exact_agreement,
        "ocr_detection_agreement": ocr_detection_agreement,
        "ocr_text_agreement": ocr_text_agreement,
        "disagreement_records": disagreement_records,
        "per_slice": {key: dict(value) for key, value in sorted(per_slice.items())},
        "slice_summary": loaded["slice_summary"],
        "findings": findings,
    }


def format_gold_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a Markdown report suitable for diagnostics docs and summaries."""
    per_slice_lines = [
        "| Slice | Records | Missing predictions | Exact disagreements | "
        "OCR disagreements | OCR text disagreements |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in report["per_slice"].items():
        per_slice_lines.append(
            "| {label} | {records} | {missing} | {exact} | {ocr} | {text} |".format(
                label=label,
                records=summary.get("records", 0),
                missing=summary.get("missing_predictions", 0),
                exact=summary.get("exact_disagreements", 0),
                ocr=summary.get("ocr_detection_disagreements", 0),
                text=summary.get("ocr_text_disagreements", 0),
            )
        )

    finding_lines = [f"- {finding}" for finding in report["findings"]] or [
        "- No missing predictions or disagreements reported."
    ]
    missing_ids = ", ".join(report["missing_prediction_sample_ids"]) or "none"
    return "\n".join(
        [
            "# Gold diagnostic benchmark report",
            "",
            f"- schema_version: `{report['schema_version']}`",
            f"- source_path: `{report['source_path']}`",
            f"- total_gold_records: `{report['total_gold_records']}`",
            f"- matched_prediction_count: `{report['matched_prediction_count']}`",
            f"- missing_prediction_count: `{report['missing_prediction_count']}`",
            f"- missing_prediction_sample_ids: `{missing_ids}`",
            f"- exact_agreement: `{json.dumps(report['exact_agreement'], sort_keys=True)}`",
            "- ocr_detection_agreement: "
            f"`{json.dumps(report['ocr_detection_agreement'], sort_keys=True)}`",
            f"- ocr_text_agreement: `{json.dumps(report['ocr_text_agreement'], sort_keys=True)}`",
            "",
            "Missing predictions are diagnostic evidence, not hidden pass conditions.",
            "",
            "## Findings",
            "",
            *finding_lines,
            "",
            "## Per-slice summary",
            "",
            *per_slice_lines,
            "",
        ]
    )


def _validate_record(
    payload: Mapping[str, Any], line_number: int, seen_sample_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"line {line_number}: missing required field {field}")
    if not isinstance(payload.get("sample_id"), str) or not payload.get("sample_id", "").strip():
        errors.append(f"line {line_number}: sample_id must be a non-empty string")
    elif payload["sample_id"] in seen_sample_ids:
        errors.append(f"line {line_number}: duplicate sample_id {payload['sample_id']}")
    for field in ("target_text", "image_path", "human_label"):
        if field in payload and (not isinstance(payload[field], str) or not payload[field].strip()):
            errors.append(f"line {line_number}: {field} must be a non-empty string")
    for field in ("expected_exact_match", "expected_ocr_detected"):
        if field in payload and not isinstance(payload[field], bool):
            errors.append(f"line {line_number}: {field} must be a boolean")
    if "notes" in payload and not isinstance(payload["notes"], str):
        errors.append(f"line {line_number}: notes must be a string when provided")
    return errors


def _normalize_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = {field: payload[field] for field in REQUIRED_FIELDS}
    if "notes" in payload:
        record["notes"] = payload["notes"]
    return record


def _index_predictions(predictions: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        sample_id = prediction.get("sample_id")
        if isinstance(sample_id, str) and sample_id.strip() and sample_id not in indexed:
            indexed[sample_id] = prediction
    return indexed


def _empty_slice_summary() -> dict[str, int]:
    return {
        "records": 0,
        "missing_predictions": 0,
        "exact_disagreements": 0,
        "ocr_detection_disagreements": 0,
        "ocr_text_disagreements": 0,
    }


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _count_agreement(counter: dict[str, int], agrees: bool) -> None:
    counter["agree" if agrees else "disagree"] += 1


def _normalize_text(value: Any) -> str:
    cleaned = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in str(value or "")
    )
    return " ".join(cleaned.split()).casefold()


def _build_findings(
    missing_prediction_sample_ids: list[str],
    exact_agreement: Mapping[str, int],
    ocr_detection_agreement: Mapping[str, int],
    ocr_text_agreement: Mapping[str, int],
) -> list[str]:
    findings: list[str] = []
    if missing_prediction_sample_ids:
        findings.append(
            "Missing predictions: {count} sample(s): {ids}".format(
                count=len(missing_prediction_sample_ids),
                ids=", ".join(missing_prediction_sample_ids),
            )
        )
    if exact_agreement["disagree"]:
        findings.append(f"Exact-match expectation disagreements: {exact_agreement['disagree']}")
    if ocr_detection_agreement["disagree"]:
        findings.append(
            f"OCR-detection expectation disagreements: {ocr_detection_agreement['disagree']}"
        )
    if ocr_text_agreement["disagree"]:
        findings.append(f"OCR text disagreements: {ocr_text_agreement['disagree']}")
    return findings
