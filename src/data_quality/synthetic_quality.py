"""CPU-safe synthetic masked-SFT quality inspection helpers.

The inspector reads local CSV/JSONL metadata plus tiny PIL image/mask files. It
does not instantiate OCR engines, load tensor/model artifacts, or import heavy
diffusion/training packages. OCR verification is represented by optional CSV or
JSONL result files produced by a separate opt-in diagnostic pipeline.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageStat

SYNTHETIC_QUALITY_SCHEMA_VERSION = "synthetic-quality/v1"
DEFAULT_IMAGE_DIR = "raw_imgs"
DEFAULT_MASK_DIR = "raw_masks"
TEXT_COLUMNS = ("text", "target_text", "label", "caption")


@dataclass(frozen=True)
class SyntheticQualityReport:
    """Aggregate quality report for a masked-SFT synthetic dataset."""

    data_dir: str
    raw_dir: str | None
    sample_count: int
    accepted_count: int
    rejected_count: int
    missing_files: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    mask_area_fraction: dict[str, Any] = field(default_factory=dict)
    bbox_height_fraction: dict[str, Any] = field(default_factory=dict)
    bbox_area_fraction: dict[str, Any] = field(default_factory=dict)
    contrast: dict[str, Any] = field(default_factory=dict)
    character_coverage: dict[str, Any] = field(default_factory=dict)
    font_coverage: dict[str, int] = field(default_factory=dict)
    resolution_distribution: dict[str, int] = field(default_factory=dict)
    ocr_summary: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SYNTHETIC_QUALITY_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return not self.missing_files and self.rejected_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_dir": self.data_dir,
            "raw_dir": self.raw_dir,
            "sample_count": self.sample_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "missing_files": self.missing_files,
            "rejection_reasons": self.rejection_reasons,
            "mask_area_fraction": self.mask_area_fraction,
            "bbox_height_fraction": self.bbox_height_fraction,
            "bbox_area_fraction": self.bbox_area_fraction,
            "contrast": self.contrast,
            "character_coverage": self.character_coverage,
            "font_coverage": self.font_coverage,
            "resolution_distribution": self.resolution_distribution,
            "ocr_summary": self.ocr_summary,
            "samples": self.samples,
            "warnings": self.warnings,
            "ok": self.ok,
        }


def inspect_synthetic_dataset(
    data_dir: str | Path,
    *,
    raw_dir: str | Path | None = None,
    ocr_results: str | Path | None = None,
    thresholds: Mapping[str, Any] | None = None,
    max_samples: int | None = None,
) -> SyntheticQualityReport:
    """Inspect a synthetic masked-SFT dataset without loading heavy ML/OCR modules."""

    dataset_dir = Path(data_dir)
    raw_path = Path(raw_dir) if raw_dir is not None else dataset_dir.parent / "raw"
    threshold_values = dict(thresholds or {})
    rows = _load_index_rows(dataset_dir / "index.csv")
    if max_samples is not None:
        rows = rows[: max(max_samples, 0)]
    prompt_lookup = _load_jsonl_by_id(dataset_dir / "prompts.jsonl")
    shape_lookup = _load_csv_by_id(dataset_dir / "shapes.csv")

    missing_files: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    font_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    mask_fractions: list[float] = []
    bbox_height_fractions: list[float] = []
    bbox_area_fractions: list[float] = []
    contrast_values: list[float] = []
    sample_payloads: list[dict[str, Any]] = []

    for row in rows:
        sample = _inspect_sample(dataset_dir, raw_path, row, prompt_lookup, shape_lookup)
        for key in sample.missing:
            missing_files[key] += 1
        _collect_text_coverage(character_counts, sample.text)
        for font_name in sample.fonts:
            font_counts[font_name] += 1
        if sample.resolution_key:
            resolution_counts[sample.resolution_key] += 1
        if sample.mask_area_fraction is not None:
            mask_fractions.append(sample.mask_area_fraction)
        if sample.bbox_height_fraction is not None:
            bbox_height_fractions.append(sample.bbox_height_fraction)
        if sample.bbox_area_fraction is not None:
            bbox_area_fractions.append(sample.bbox_area_fraction)
        if sample.contrast is not None:
            contrast_values.append(sample.contrast)

        reasons = _threshold_reasons(sample, threshold_values)
        for reason in reasons:
            rejection_reasons[reason] += 1
        sample_payloads.append(
            sample.to_payload(accepted=not sample.missing and not reasons, reasons=reasons)
        )

    ocr_summary = _load_ocr_summary(ocr_results) if ocr_results is not None else None
    return SyntheticQualityReport(
        data_dir=str(dataset_dir),
        raw_dir=str(raw_path) if raw_dir is not None else str(raw_path),
        sample_count=len(rows),
        accepted_count=sum(1 for sample in sample_payloads if sample["accepted"]),
        rejected_count=sum(1 for sample in sample_payloads if not sample["accepted"]),
        missing_files=dict(sorted(missing_files.items())),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        mask_area_fraction=_summary(mask_fractions),
        bbox_height_fraction=_summary(bbox_height_fractions),
        bbox_area_fraction=_summary(bbox_area_fractions),
        contrast=_summary(contrast_values),
        character_coverage=_character_coverage_payload(character_counts),
        font_coverage=dict(sorted(font_counts.items())),
        resolution_distribution=dict(sorted(resolution_counts.items())),
        ocr_summary=ocr_summary,
        samples=sample_payloads,
        warnings=[],
    )


@dataclass(frozen=True)
class _SampleInspection:
    sample_id: str
    text: str
    prompt: str | None
    image_path: Path
    mask_path: Path
    meta_path: Path | None
    missing: tuple[str, ...]
    mask_area_fraction: float | None
    bbox_height_fraction: float | None
    bbox_area_fraction: float | None
    contrast: float | None
    resolution_key: str | None
    fonts: tuple[str, ...]

    def to_payload(self, *, accepted: bool, reasons: Sequence[str]) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "text": self.text,
            "prompt": self.prompt,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
            "meta_path": str(self.meta_path) if self.meta_path is not None else None,
            "missing": list(self.missing),
            "mask_area_fraction": self.mask_area_fraction,
            "bbox_height_fraction": self.bbox_height_fraction,
            "bbox_area_fraction": self.bbox_area_fraction,
            "contrast": self.contrast,
            "resolution": self.resolution_key,
            "fonts": list(self.fonts),
            "accepted": accepted,
            "rejection_reasons": list(reasons),
        }


def _inspect_sample(
    data_dir: Path,
    raw_dir: Path,
    row: Mapping[str, str],
    prompt_lookup: Mapping[str, Mapping[str, Any]],
    shape_lookup: Mapping[str, Mapping[str, str]],
) -> _SampleInspection:
    sample_id = str(row.get("id", ""))
    image_path = data_dir / DEFAULT_IMAGE_DIR / f"{sample_id}.png"
    mask_path = data_dir / DEFAULT_MASK_DIR / f"{sample_id}.png"
    meta_path = raw_dir / "meta" / f"{sample_id}.json"
    missing = []
    if not sample_id:
        missing.append("id")
    if not image_path.is_file():
        missing.append("raw_img")
    if not mask_path.is_file():
        missing.append("raw_mask")
    if not (data_dir / "index.csv").is_file():
        missing.append("index_csv")
    if not (data_dir / "prompts.jsonl").is_file():
        missing.append("prompts_jsonl")
    if not (data_dir / "shapes.csv").is_file():
        missing.append("shapes_csv")
    if not meta_path.is_file():
        missing.append("raw_meta")

    meta = _load_json_object(meta_path) if meta_path.is_file() else {}
    mask_area_fraction = None
    bbox_height_fraction = None
    bbox_area_fraction = None
    contrast = None
    resolution_key = _resolution_key(row, shape_lookup.get(sample_id), meta)
    if image_path.is_file() and mask_path.is_file():
        image_metrics = _image_mask_metrics(image_path, mask_path, meta)
        mask_area_fraction = image_metrics["mask_area_fraction"]
        bbox_height_fraction = image_metrics["bbox_height_fraction"]
        bbox_area_fraction = image_metrics["bbox_area_fraction"]
        contrast = image_metrics["contrast"]
        resolution_key = resolution_key or image_metrics["resolution_key"]

    prompt = prompt_lookup.get(sample_id, {}).get("prompt")
    return _SampleInspection(
        sample_id=sample_id,
        text=_row_text(row),
        prompt=str(prompt) if isinstance(prompt, str) else None,
        image_path=image_path,
        mask_path=mask_path,
        meta_path=meta_path,
        missing=tuple(sorted(set(missing))),
        mask_area_fraction=mask_area_fraction,
        bbox_height_fraction=bbox_height_fraction,
        bbox_area_fraction=bbox_area_fraction,
        contrast=contrast,
        resolution_key=resolution_key,
        fonts=_extract_fonts(meta),
    )


def _load_index_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("id", "")): dict(row) for row in csv.DictReader(handle) if row.get("id")
        }


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                records[payload["id"]] = payload
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _image_mask_metrics(
    image_path: Path, mask_path: Path, meta: Mapping[str, Any]
) -> dict[str, Any]:
    with Image.open(image_path) as image_raw, Image.open(mask_path) as mask_raw:
        image = image_raw.convert("L")
        mask = mask_raw.convert("L")
        width, height = image.size
        histogram = mask.histogram()
        positive_pixels = sum(histogram[1:])
        total_pixels = max(width * height, 1)
        bbox = _metadata_bbox(meta, width, height) or mask.getbbox()
        bbox_width = 0
        bbox_height = 0
        if bbox is not None:
            left, top, right, bottom = bbox
            bbox_width = max(right - left, 0)
            bbox_height = max(bottom - top, 0)
        contrast = _contrast(image, mask)
        return {
            "mask_area_fraction": round(positive_pixels / total_pixels, 6),
            "bbox_height_fraction": round(bbox_height / max(height, 1), 6),
            "bbox_area_fraction": round((bbox_width * bbox_height) / total_pixels, 6),
            "contrast": contrast,
            "resolution_key": f"{width}x{height}",
        }


def _metadata_bbox(
    meta: Mapping[str, Any], width: int, height: int
) -> tuple[int, int, int, int] | None:
    annotations = meta.get("annotations")
    if not isinstance(annotations, Sequence) or isinstance(annotations, str):
        return None
    boxes = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        raw_bbox = annotation.get("bbox")
        if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, str) or len(raw_bbox) < 4:
            continue
        x, y, box_width, box_height = (int(float(raw_bbox[index])) for index in range(4))
        boxes.append((x, y, x + box_width, y + box_height))
    if not boxes:
        return None
    left = max(min(box[0] for box in boxes), 0)
    top = max(min(box[1] for box in boxes), 0)
    right = min(max(box[2] for box in boxes), width)
    bottom = min(max(box[3] for box in boxes), height)
    return left, top, right, bottom


def _contrast(image: Image.Image, mask: Image.Image) -> float:
    foreground_mask = mask.point(lambda value: 255 if value > 0 else 0)
    background_mask = mask.point(lambda value: 0 if value > 0 else 255)
    if foreground_mask.getbbox() is None or background_mask.getbbox() is None:
        return 0.0
    fg_mean = ImageStat.Stat(image, foreground_mask).mean[0]
    bg_mean = ImageStat.Stat(image, background_mask).mean[0]
    return round(abs(fg_mean - bg_mean), 6)


def _row_text(row: Mapping[str, str]) -> str:
    for column in TEXT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value:
            return value
    return ""


def _collect_text_coverage(counter: Counter[str], text: str) -> None:
    for character in text.lower():
        if not character.isspace() and character != "|":
            counter[character] += 1


def _extract_fonts(meta: Mapping[str, Any]) -> tuple[str, ...]:
    annotations = meta.get("annotations")
    if not isinstance(annotations, Sequence) or isinstance(annotations, str):
        return ()
    fonts = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        font = annotation.get("font") or annotation.get("font_name")
        if isinstance(font, str) and font:
            fonts.append(font)
    return tuple(dict.fromkeys(fonts))


def _resolution_key(
    row: Mapping[str, str],
    shape: Mapping[str, str] | None,
    meta: Mapping[str, Any],
) -> str | None:
    resolution = row.get("resolution") or meta.get("resolution")
    if isinstance(resolution, int | float) or (
        isinstance(resolution, str) and resolution.isdigit()
    ):
        value = int(resolution)
        return f"{value}x{value}"
    if shape and shape.get("H") and shape.get("W"):
        return f"latent-{shape['W']}x{shape['H']}"
    return None


def _threshold_reasons(
    sample: _SampleInspection,
    thresholds: Mapping[str, Any],
) -> tuple[str, ...]:
    if sample.missing:
        return ("missing_files",)
    reasons = []
    _append_min_reason(reasons, sample.mask_area_fraction, thresholds, "mask_area_fraction")
    _append_max_reason(reasons, sample.mask_area_fraction, thresholds, "mask_area_fraction")
    _append_min_reason(reasons, sample.bbox_height_fraction, thresholds, "bbox_height_fraction")
    _append_min_reason(reasons, sample.bbox_area_fraction, thresholds, "bbox_area_fraction")
    _append_min_reason(reasons, sample.contrast, thresholds, "contrast")
    max_text_length = thresholds.get("max_text_length")
    if isinstance(max_text_length, int | float) and len(sample.text) > int(max_text_length):
        reasons.append("text_length_above_max")
    return tuple(reasons)


def _append_min_reason(
    reasons: list[str],
    value: float | None,
    thresholds: Mapping[str, Any],
    metric: str,
) -> None:
    minimum = thresholds.get(f"min_{metric}")
    if isinstance(minimum, int | float) and (value is None or value < float(minimum)):
        reasons.append(f"{metric}_below_min")


def _append_max_reason(
    reasons: list[str],
    value: float | None,
    thresholds: Mapping[str, Any],
    metric: str,
) -> None:
    maximum = thresholds.get(f"max_{metric}")
    if isinstance(maximum, int | float) and (value is None or value > float(maximum)):
        reasons.append(f"{metric}_above_max")


def _load_ocr_summary(path: str | Path) -> dict[str, Any]:
    rows = _load_ocr_rows(Path(path))
    if not rows:
        return {"count": 0, "exact_matches": 0, "exact_match_rate": 0.0, "mean_cer": 0.0}
    exact_matches = 0
    cer_values = []
    for row in rows:
        target = str(row.get("target_text") or row.get("text") or row.get("label") or "")
        observed = str(
            row.get("ocr_text") or row.get("prediction") or row.get("recognized_text") or ""
        )
        if _normalize_for_match(target) == _normalize_for_match(observed):
            exact_matches += 1
        cer_values.append(_cer(target, observed))
    return {
        "count": len(rows),
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / len(rows), 6),
        "mean_cer": round(mean(cer_values), 6),
    }


def _load_ocr_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _cer(target: str, observed: str) -> float:
    if not target:
        return 0.0 if not observed else 1.0
    return _levenshtein(_normalize_for_match(target), _normalize_for_match(observed)) / len(
        _normalize_for_match(target)
    )


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _normalize_for_match(value: str) -> str:
    return " ".join(value.casefold().split())


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
    }


def _character_coverage_payload(counter: Counter[str]) -> dict[str, Any]:
    return {
        "unique_count": len(counter),
        "counts": dict(sorted(counter.items())),
    }
