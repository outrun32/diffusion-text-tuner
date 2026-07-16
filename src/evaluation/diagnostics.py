"""CPU-safe reward disagreement diagnostics.

This module analyzes recorded score rows and optional gold benchmark metadata. It
does not run OCR, VLM, CUDA, diffusion models, or external model weights. PIL is
imported only inside the optional contact-sheet path.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from src.evaluation.gold_benchmark import load_gold_benchmark
from src.evaluation.slices import classify_text_slices

SCHEMA_VERSION = "reward-diagnostics/v1"
REQUIRED_SCORE_FIELDS = ("sample_id", "target_text")
SCORE_COMPONENT_FIELDS = ("score_vlm", "score_ocr", "product_score")
TEXT_METRIC_FIELDS = ("cer", "exact_match", "exact_text_match", "char_accuracy")
MISSING_MARKERS = (None, "", "none", "null", "nan", "NaN")


class DiagnosticsInputError(ValueError):
    """Raised when diagnostics inputs are malformed."""


def analyze_reward_disagreement(
    records: Iterable[Mapping[str, Any]],
    *,
    gold_records: Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | Path | None = None,
    positive_threshold: float = 0.8,
    negative_threshold: float = 0.5,
    contact_sheet_path: str | Path | None = None,
    contact_sheet_limit: int = 12,
) -> dict[str, Any]:
    """Analyze VLM/OCR/product-score disagreements from recorded rows.

    ``records`` are local score dictionaries using canonical fields such as
    ``score_vlm``, ``score_ocr``, ``cer``, ``exact_match``, ``char_accuracy``,
    ``product_score``, and ``missing_components``. Optional ``gold_records`` add
    expected pass/fail labels without invoking any reward model.
    """

    normalized_records = [
        _normalize_score_record(record, index) for index, record in enumerate(records)
    ]
    gold_by_id = _load_gold_by_id(gold_records)
    input_errors = _validate_records(normalized_records)

    missing_evidence = _summarize_missing_evidence(normalized_records)
    correlation = _summarize_correlation(normalized_records)
    false_positives, false_negatives = _classify_false_rows(
        normalized_records,
        gold_by_id,
        positive_threshold,
        negative_threshold,
    )
    confusion_summary = _summarize_character_confusions(normalized_records)
    per_slice = _summarize_slice_disagreements(
        normalized_records,
        false_positives,
        false_negatives,
        missing_evidence["rows"],
    )
    contact_sheet = _maybe_write_contact_sheet(
        false_positives,
        false_negatives,
        contact_sheet_path,
        contact_sheet_limit,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "record_counts": {
            "total": len(normalized_records),
            "with_vlm_and_ocr": correlation["n"],
            "missing_evidence": missing_evidence["count"],
        },
        "thresholds": {
            "positive_threshold": float(positive_threshold),
            "negative_threshold": float(negative_threshold),
        },
        "input_errors": input_errors,
        "missing_evidence": missing_evidence,
        "vlm_ocr_correlation": {
            "n": correlation["n"],
            "pearson": correlation["pearson"],
        },
        "scatter_summary": {
            "x": "score_vlm",
            "y": "score_ocr",
            "points": correlation["points"],
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "character_confusions": confusion_summary,
        "per_slice_disagreement_counts": per_slice,
        "contact_sheet": contact_sheet,
    }


def load_score_records(path: str | Path) -> list[dict[str, Any]]:
    """Load score records from CSV, JSONL, or JSON without model/OCR work."""
    score_path = Path(path)
    suffix = score_path.suffix.lower()
    try:
        if suffix == ".csv":
            with score_path.open("r", encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        if suffix == ".jsonl":
            records: list[dict[str, Any]] = []
            lines = score_path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DiagnosticsInputError(
                        f"{score_path}: line {line_number}: malformed JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise DiagnosticsInputError(
                        f"{score_path}: line {line_number}: record must be a JSON object"
                    )
                records.append(payload)
            return records
        if suffix == ".json":
            payload = json.loads(score_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
                records = payload["records"]
            else:
                raise DiagnosticsInputError(
                    f"{score_path}: JSON scores must be a list or contain records list"
                )
            if not all(isinstance(record, dict) for record in records):
                raise DiagnosticsInputError(f"{score_path}: all score records must be objects")
            return list(records)
    except OSError as exc:
        raise DiagnosticsInputError(f"{score_path}: could not read scores") from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticsInputError(f"{score_path}: malformed JSON") from exc
    raise DiagnosticsInputError(f"{score_path}: unsupported score format {suffix or '<none>'}")


def write_diagnostics_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write deterministic diagnostics JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_diagnostics_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise Markdown reward disagreement report."""
    correlation = report["vlm_ocr_correlation"]
    missing = report["missing_evidence"]
    lines = [
        "# Reward disagreement diagnostics",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- record_counts: `{json.dumps(report['record_counts'], sort_keys=True)}`",
        f"- thresholds: `{json.dumps(report['thresholds'], sort_keys=True)}`",
        "",
        "## VLM-vs-OCR correlation",
        "",
        f"- n: `{correlation['n']}`",
        f"- pearson: `{correlation['pearson']}`",
        "",
        "## Missing evidence",
        "",
        f"- count: `{missing['count']}`",
        "- by_component: "
        f"`{json.dumps(missing['by_component'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## False positives",
        "",
        *_format_false_row_lines(report["false_positives"]),
        "",
        "## False negatives",
        "",
        *_format_false_row_lines(report["false_negatives"]),
        "",
        "## Per-character confusion",
        "",
        f"- total_confusions: `{report['character_confusions']['total_confusions']}`",
        *_format_confusion_lines(report["character_confusions"]["pairs"]),
        "",
        "## Per-slice disagreement",
        "",
        "| Slice | Records | False positives | False negatives | Missing evidence |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in report["per_slice_disagreement_counts"].items():
        lines.append(
            "| {label} | {records} | {fp} | {fn} | {missing} |".format(
                label=label,
                records=summary.get("records", 0),
                fp=summary.get("false_positives", 0),
                fn=summary.get("false_negatives", 0),
                missing=summary.get("missing_evidence", 0),
            )
        )
    lines.extend(_format_contact_sheet_lines(report.get("contact_sheet", {})))
    return "\n".join(lines) + "\n"


def write_diagnostics_markdown(report: Mapping[str, Any], path: str | Path) -> None:
    """Write deterministic Markdown diagnostics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_diagnostics_markdown(report), encoding="utf-8")


def _normalize_score_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("sample_id", normalized.get("id") or str(index))
    normalized["sample_id"] = str(normalized["sample_id"])
    normalized["target_text"] = str(normalized.get("target_text") or "")
    normalized["score_vlm"] = _coerce_float(normalized.get("score_vlm"))
    normalized["score_ocr"] = _coerce_float(normalized.get("score_ocr"))
    normalized["cer"] = _coerce_float(normalized.get("cer"))
    normalized["char_accuracy"] = _coerce_float(normalized.get("char_accuracy"))
    normalized["product_score"] = _coerce_float(
        normalized.get("product_score", normalized.get("score"))
    )
    normalized["exact_match"] = _coerce_bool(
        normalized.get("exact_match", normalized.get("exact_text_match"))
    )
    normalized["missing_components"] = _parse_missing_components(
        normalized.get("missing_components")
    )
    return normalized


def _validate_records(records: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        for field in REQUIRED_SCORE_FIELDS:
            if not str(record.get(field) or "").strip():
                errors.append(f"record {index}: missing required field {field}")
        sample_id = str(record.get("sample_id") or "")
        if sample_id in seen:
            errors.append(f"record {index}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
    return errors


def _load_gold_by_id(
    gold_records: Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | Path | None,
) -> dict[str, Mapping[str, Any]]:
    if gold_records is None:
        return {}
    if isinstance(gold_records, (str, Path)):
        loaded = load_gold_benchmark(gold_records)
        records = loaded["records"]
    elif isinstance(gold_records, Mapping) and "records" in gold_records:
        records = gold_records["records"]
    else:
        records = gold_records
    return {
        str(record["sample_id"]): record
        for record in records  # type: ignore[union-attr]
        if isinstance(record, Mapping) and str(record.get("sample_id") or "").strip()
    }


def _summarize_missing_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_component: Counter[str] = Counter()
    for record in records:
        missing_components = set(record.get("missing_components") or [])
        for field in (*SCORE_COMPONENT_FIELDS, "cer"):
            if record.get(field) is None:
                missing_components.add(field)
        if not missing_components:
            continue
        components = sorted(missing_components)
        by_component.update(components)
        rows.append({"sample_id": record["sample_id"], "missing_components": components})
    return {
        "count": len(rows),
        "by_component": dict(sorted(by_component.items())),
        "rows": rows,
    }


def _summarize_correlation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    points = [
        {
            "sample_id": record["sample_id"],
            "score_vlm": record["score_vlm"],
            "score_ocr": record["score_ocr"],
            "product_score": record["product_score"],
        }
        for record in records
        if record.get("score_vlm") is not None and record.get("score_ocr") is not None
    ]
    points.sort(key=lambda point: (point["score_vlm"], point["sample_id"]))
    return {
        "n": len(points),
        "pearson": _pearson(
            [float(point["score_vlm"]) for point in points],
            [float(point["score_ocr"]) for point in points],
        ),
        "points": points,
    }


def _classify_false_rows(
    records: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for record in records:
        score = _diagnostic_score(record)
        if score is None:
            continue
        expected_positive, gold_label = _expected_positive(
            record, gold_by_id.get(record["sample_id"])
        )
        if expected_positive is None:
            continue
        row = _false_row(record, gold_label, score)
        if score >= positive_threshold and not expected_positive:
            false_positives.append(row)
        elif score <= negative_threshold and expected_positive:
            false_negatives.append(row)
    false_positives.sort(key=lambda row: row["sample_id"])
    false_negatives.sort(key=lambda row: row["sample_id"])
    return false_positives, false_negatives


def _false_row(record: Mapping[str, Any], gold_label: str, score: float) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "target_text": record.get("target_text", ""),
        "detected_text": record.get("detected_text", ""),
        "image_path": str(record.get("image_path") or ""),
        "product_score": score,
        "score_vlm": record.get("score_vlm"),
        "score_ocr": record.get("score_ocr"),
        "exact_match": record.get("exact_match"),
        "gold_label": gold_label,
        "slices": sorted(classify_text_slices(record)),
    }


def _summarize_character_confusions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counter: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        target_text = str(record.get("target_text") or "")
        detected_text = str(record.get("detected_text") or record.get("ocr_text") or "")
        if not target_text or not detected_text:
            continue
        for expected, observed in zip_longest(target_text, detected_text, fillvalue="∅"):
            if expected == observed:
                continue
            counter[(expected, observed)].add(str(record["sample_id"]))
    pairs = [
        {
            "expected": expected,
            "observed": observed,
            "count": len(sample_ids),
            "sample_ids": sorted(sample_ids),
        }
        for (expected, observed), sample_ids in counter.items()
    ]
    pairs.sort(key=lambda row: (-row["count"], row["expected"], row["observed"]))
    return {"total_confusions": sum(row["count"] for row in pairs), "pairs": pairs}


def _summarize_slice_disagreements(
    records: Sequence[Mapping[str, Any]],
    false_positives: Sequence[Mapping[str, Any]],
    false_negatives: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(_empty_slice_summary)
    false_positive_ids = {row["sample_id"] for row in false_positives}
    false_negative_ids = {row["sample_id"] for row in false_negatives}
    missing_ids = {row["sample_id"] for row in missing_rows}
    for record in records:
        labels = classify_text_slices(record) or {"unsliced"}
        for label in labels:
            summary[label]["records"] += 1
            if record["sample_id"] in false_positive_ids:
                summary[label]["false_positives"] += 1
            if record["sample_id"] in false_negative_ids:
                summary[label]["false_negatives"] += 1
            if record["sample_id"] in missing_ids:
                summary[label]["missing_evidence"] += 1
    return {key: dict(value) for key, value in sorted(summary.items())}


def _maybe_write_contact_sheet(
    false_positives: Sequence[Mapping[str, Any]],
    false_negatives: Sequence[Mapping[str, Any]],
    contact_sheet_path: str | Path | None,
    contact_sheet_limit: int,
) -> dict[str, Any]:
    if contact_sheet_path is None:
        return {
            "requested": False,
            "path": "",
            "limit": contact_sheet_limit,
            "entry_count": 0,
            "entries": [],
        }

    selected_rows = [("false_positive", row) for row in false_positives] + [
        ("false_negative", row) for row in false_negatives
    ]
    bounded_rows = selected_rows[: max(0, int(contact_sheet_limit))]
    entries = [
        {
            "sample_id": str(row["sample_id"]),
            "caption": f"{kind} {row['sample_id']} product={float(row['product_score']):.3f}",
            "source_path": str(row.get("image_path") or ""),
            "kind": kind,
        }
        for kind, row in bounded_rows
    ]
    _write_contact_sheet_image(entries, Path(contact_sheet_path))
    return {
        "requested": True,
        "path": str(contact_sheet_path),
        "limit": int(contact_sheet_limit),
        "entry_count": len(entries),
        "entries": entries,
    }


def _write_contact_sheet_image(entries: Sequence[Mapping[str, str]], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_width = 96
    thumb_height = 72
    caption_height = 24
    if not entries:
        Image.new("RGB", (thumb_width, thumb_height + caption_height), color="white").save(
            output_path
        )
        return
    sheet = Image.new("RGB", (thumb_width * len(entries), thumb_height + caption_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        x = index * thumb_width
        source_path = Path(entry["source_path"])
        try:
            with Image.open(source_path) as source:
                image = source.convert("RGB")
                image.thumbnail((thumb_width, thumb_height))
                sheet.paste(image, (x, 0))
        except OSError:
            draw.rectangle((x, 0, x + thumb_width - 1, thumb_height - 1), outline="black")
            draw.text((x + 4, 20), "missing", fill="black")
        draw.text((x + 2, thumb_height + 2), entry["sample_id"][:18], fill="black")
    sheet.save(output_path)


def _format_false_row_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["- None."]
    return [
        "- `{sample_id}` product={score:.3f} gold={gold} image=`{image}`".format(
            sample_id=row["sample_id"],
            score=float(row["product_score"]),
            gold=row["gold_label"],
            image=row.get("image_path") or "",
        )
        for row in rows
    ]


def _format_confusion_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["- None."]
    return [
        "- `{expected}` -> `{observed}`: {count} ({sample_ids})".format(
            expected=row["expected"],
            observed=row["observed"],
            count=row["count"],
            sample_ids=", ".join(row["sample_ids"]),
        )
        for row in rows[:10]
    ]


def _format_contact_sheet_lines(contact_sheet: Mapping[str, Any]) -> list[str]:
    if not contact_sheet.get("requested"):
        return []
    lines = [
        "",
        "## Contact sheet",
        "",
        f"- path: `{contact_sheet.get('path', '')}`",
        f"- entry_count: `{contact_sheet.get('entry_count', 0)}`",
    ]
    for entry in contact_sheet.get("entries", []):
        lines.append("- `{sample_id}` {kind}: `{source_path}` — {caption}".format(**entry))
    return lines


def _empty_slice_summary() -> dict[str, int]:
    return {"records": 0, "false_positives": 0, "false_negatives": 0, "missing_evidence": 0}


def _diagnostic_score(record: Mapping[str, Any]) -> float | None:
    return _coerce_float(record.get("product_score", record.get("score")))


def _expected_positive(
    record: Mapping[str, Any], gold_record: Mapping[str, Any] | None
) -> tuple[bool | None, str]:
    if gold_record is not None:
        label = str(gold_record.get("human_label") or "").casefold()
        if label in {"pass", "positive", "ok", "true"}:
            return True, str(gold_record.get("human_label"))
        if label in {"fail", "negative", "bad", "false"}:
            return False, str(gold_record.get("human_label"))
        expected = _coerce_bool(gold_record.get("expected_exact_match"))
        if expected is not None:
            return expected, str(gold_record.get("human_label") or expected)
    exact_match = _coerce_bool(record.get("exact_match", record.get("exact_text_match")))
    return exact_match, str(exact_match)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def _parse_missing_components(value: Any) -> tuple[str, ...]:
    if value in MISSING_MARKERS:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(component) for component in value if str(component).strip()))
    if isinstance(value, str):
        return tuple(
            sorted(component.strip() for component in value.split(",") if component.strip())
        )
    return (str(value),)


def _coerce_float(value: Any) -> float | None:
    if value in MISSING_MARKERS or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None
