"""CPU-safe prompt JSONL quality validation.

The validator intentionally uses only deterministic Python parsing and aggregate
heuristics. It reports line-numbered contract errors without importing model,
OCR, CUDA, or diffusion libraries.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.prompt_pipeline.config import RARE_CHARS

PROMPT_QUALITY_SCHEMA_VERSION = "prompt-quality/v1"
DEFAULT_REQUIRED_FIELDS = ("id", "prompt", "target_text", "content_type", "style", "lang")
DEFAULT_ALLOWED_SCRIPTS = frozenset({"cyrillic", "latin", "digits", "punctuation"})
MAX_DUPLICATE_EXAMPLES = 5
INSTRUCTION_MARKERS = (
    "ответь",
    "придумай",
    "сгенерируй",
    "напиши",
    "answer",
    "generate",
    "write",
    "return only",
)
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PromptQualityReport:
    """Aggregate prompt dataset quality report."""

    dataset_path: str
    schema_version: str = PROMPT_QUALITY_SCHEMA_VERSION
    total_lines: int = 0
    valid_records: int = 0
    malformed_records: int = 0
    missing_required_records: int = 0
    length_buckets: dict[str, int] = field(default_factory=dict)
    script_coverage: dict[str, int] = field(default_factory=dict)
    rare_character_coverage: dict[str, Any] = field(default_factory=dict)
    duplicate_rate: float = 0.0
    duplicate_examples: list[str] = field(default_factory=list)
    content_type_distribution: dict[str, int] = field(default_factory=dict)
    style_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_path": self.dataset_path,
            "total_lines": self.total_lines,
            "valid_records": self.valid_records,
            "malformed_records": self.malformed_records,
            "missing_required_records": self.missing_required_records,
            "length_buckets": self.length_buckets,
            "script_coverage": self.script_coverage,
            "rare_character_coverage": self.rare_character_coverage,
            "duplicate_rate": self.duplicate_rate,
            "duplicate_examples": self.duplicate_examples,
            "content_type_distribution": self.content_type_distribution,
            "style_distribution": self.style_distribution,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
            "ok": self.ok,
        }


def validate_prompt_dataset(
    path: str | Path,
    thresholds: Mapping[str, Any] | None = None,
) -> PromptQualityReport:
    """Validate prompt JSONL rows and return aggregate quality metrics."""

    dataset_path = Path(path)
    threshold_values = dict(thresholds or {})
    context = _PromptValidationContext(path=dataset_path, thresholds=threshold_values)
    if not dataset_path.is_file():
        context.errors.append(f"{dataset_path}: prompt dataset file is missing")
        return context.to_report()

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            context.total_lines = line_number
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                context.malformed_records += 1
                context.errors.append(f"{dataset_path}: line {line_number}: malformed JSON")
                continue
            if not isinstance(payload, dict):
                context.malformed_records += 1
                context.errors.append(f"{dataset_path}: line {line_number}: prompt record must be an object")
                continue
            _validate_record(context, payload, line_number)

    _finalize_duplicate_metrics(context)
    _finalize_rare_character_metrics(context)
    _apply_distribution_thresholds(context)
    return context.to_report()


@dataclass
class _PromptValidationContext:
    path: Path
    thresholds: Mapping[str, Any]
    total_lines: int = 0
    valid_records: int = 0
    malformed_records: int = 0
    missing_required_records: int = 0
    length_buckets: Counter[str] = field(
        default_factory=lambda: Counter({"1-4": 0, "5-12": 0, "13-24": 0, "25+": 0})
    )
    script_coverage: Counter[str] = field(default_factory=Counter)
    rare_counts: Counter[str] = field(default_factory=Counter)
    target_text_counts: Counter[str] = field(default_factory=Counter)
    content_types: Counter[str] = field(default_factory=Counter)
    style_values: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_report(self) -> PromptQualityReport:
        duplicate_examples = [
            text for text, count in sorted(self.target_text_counts.items()) if count > 1
        ][:MAX_DUPLICATE_EXAMPLES]
        duplicate_records = sum(count - 1 for count in self.target_text_counts.values() if count > 1)
        duplicate_rate = duplicate_records / self.valid_records if self.valid_records else 0.0
        style_distribution = {
            key: dict(sorted(values.items())) for key, values in sorted(self.style_values.items())
        }
        metadata = {
            "thresholds": dict(self.thresholds),
            "required_fields": list(_required_fields(self.thresholds)),
        }
        return PromptQualityReport(
            dataset_path=str(self.path),
            total_lines=self.total_lines,
            valid_records=self.valid_records,
            malformed_records=self.malformed_records,
            missing_required_records=self.missing_required_records,
            length_buckets=dict(self.length_buckets),
            script_coverage={key: self.script_coverage.get(key, 0) for key in _script_keys()},
            rare_character_coverage=_rare_character_coverage_payload(self),
            duplicate_rate=round(duplicate_rate, 6),
            duplicate_examples=duplicate_examples,
            content_type_distribution=dict(sorted(self.content_types.items())),
            style_distribution=style_distribution,
            warnings=list(dict.fromkeys(self.warnings)),
            errors=list(dict.fromkeys(self.errors)),
            metadata=metadata,
        )


def _validate_record(
    context: _PromptValidationContext,
    payload: Mapping[str, Any],
    line_number: int,
) -> None:
    missing = [
        field_name
        for field_name in _required_fields(context.thresholds)
        if field_name not in payload or payload[field_name] is None
    ]
    if missing:
        context.missing_required_records += 1
        for field_name in missing:
            context.errors.append(
                f"{context.path}: line {line_number}: missing required field: {field_name}"
            )
        return

    target_text = payload.get("target_text")
    if not isinstance(target_text, str):
        context.malformed_records += 1
        context.errors.append(f"{context.path}: line {line_number}: target_text must be a string")
        return
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        context.malformed_records += 1
        context.errors.append(f"{context.path}: line {line_number}: prompt must be a string")
        return

    context.valid_records += 1
    normalized_text = _normalize_text(target_text)
    context.target_text_counts[normalized_text] += 1
    context.length_buckets[_length_bucket(len(target_text))] += 1
    _collect_scripts(context, target_text, line_number)
    _collect_rare_characters(context, target_text)
    _collect_content_type(context, payload)
    _collect_style(context, payload)
    _apply_length_thresholds(context, target_text, line_number)
    _apply_naturalness_heuristics(context, target_text, line_number)


def _required_fields(thresholds: Mapping[str, Any]) -> tuple[str, ...]:
    value = thresholds.get("required_fields", DEFAULT_REQUIRED_FIELDS)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(str(item) for item in value)
    return DEFAULT_REQUIRED_FIELDS


def _script_keys() -> tuple[str, ...]:
    return ("cyrillic", "latin", "digits", "punctuation")


def _collect_scripts(context: _PromptValidationContext, target_text: str, line_number: int) -> None:
    allowed = set(context.thresholds.get("allowed_scripts", DEFAULT_ALLOWED_SCRIPTS))
    scripts = set()
    if CYRILLIC_RE.search(target_text):
        scripts.add("cyrillic")
    if LATIN_RE.search(target_text):
        scripts.add("latin")
    if DIGIT_RE.search(target_text):
        scripts.add("digits")
    if any(ch in string.punctuation or ch in "—–«»…№" for ch in target_text):
        scripts.add("punctuation")
    for script in sorted(scripts):
        context.script_coverage[script] += 1
        if script not in allowed:
            context.errors.append(
                f"{context.path}: line {line_number}: disallowed script {script} in target_text"
            )
    for ch in target_text:
        if ch.isspace() or CYRILLIC_RE.match(ch) or LATIN_RE.match(ch) or ch.isdigit():
            continue
        if ch in string.punctuation or ch in "—–«»…№":
            continue
        context.errors.append(f"{context.path}: line {line_number}: illegal character in target_text")
        break


def _collect_rare_characters(context: _PromptValidationContext, target_text: str) -> None:
    configured_rare = _required_rare_characters(context.thresholds)
    for ch in target_text.lower():
        if ch in {rare.lower() for rare in RARE_CHARS} or ch in configured_rare:
            context.rare_counts[ch] += 1


def _collect_content_type(context: _PromptValidationContext, payload: Mapping[str, Any]) -> None:
    content_type = payload.get("content_type")
    if isinstance(content_type, str) and content_type:
        context.content_types[content_type] += 1


def _collect_style(context: _PromptValidationContext, payload: Mapping[str, Any]) -> None:
    style = payload.get("style")
    if isinstance(style, Mapping):
        for key, value in style.items():
            if isinstance(value, str) and value:
                context.style_values[str(key)][value] += 1
    elif isinstance(style, str) and style:
        context.style_values["style"][style] += 1


def _apply_length_thresholds(
    context: _PromptValidationContext,
    target_text: str,
    line_number: int,
) -> None:
    min_length = context.thresholds.get("min_target_length")
    max_length = context.thresholds.get("max_target_length")
    if isinstance(min_length, int) and len(target_text) < min_length:
        context.warnings.append(
            f"{context.path}: line {line_number}: target_text shorter than min_target_length"
        )
    if isinstance(max_length, int) and len(target_text) > max_length:
        context.warnings.append(
            f"{context.path}: line {line_number}: target_text longer than max_target_length"
        )


def _apply_naturalness_heuristics(
    context: _PromptValidationContext,
    target_text: str,
    line_number: int,
) -> None:
    stripped = target_text.strip()
    if not stripped:
        context.errors.append(f"{context.path}: line {line_number}: empty target_text")
        return
    lowered = stripped.lower()
    if any(marker in lowered for marker in INSTRUCTION_MARKERS):
        context.warnings.append(f"{context.path}: line {line_number}: instruction-like output")
    tokens = [token for token in WHITESPACE_RE.split(lowered) if token]
    if tokens and max(Counter(tokens).values()) >= 4:
        context.warnings.append(f"{context.path}: line {line_number}: repeated token heuristic")
    if stripped.count('"') % 2 != 0 or stripped.count("'") % 2 != 0:
        context.warnings.append(f"{context.path}: line {line_number}: unmatched quote heuristic")
    if stripped in {'""', "''", "«»"}:
        context.errors.append(f"{context.path}: line {line_number}: quotes-only target_text")


def _finalize_duplicate_metrics(context: _PromptValidationContext) -> None:
    max_duplicate_rate = context.thresholds.get("max_duplicate_rate")
    duplicate_records = sum(count - 1 for count in context.target_text_counts.values() if count > 1)
    duplicate_rate = duplicate_records / context.valid_records if context.valid_records else 0.0
    if isinstance(max_duplicate_rate, int | float) and duplicate_rate > float(max_duplicate_rate):
        context.warnings.append(
            f"duplicate_rate {duplicate_rate:.3f} exceeds max_duplicate_rate {float(max_duplicate_rate):.3f}"
        )


def _finalize_rare_character_metrics(context: _PromptValidationContext) -> None:
    required = _required_rare_characters(context.thresholds)
    if not required:
        return
    coverage = sum(1 for ch in required if context.rare_counts.get(ch, 0) > 0) / len(required)
    minimum = context.thresholds.get("min_rare_character_coverage")
    if isinstance(minimum, int | float) and coverage < float(minimum):
        context.warnings.append(
            f"rare_character_coverage {coverage:.3f} below min_rare_character_coverage {float(minimum):.3f}"
        )


def _apply_distribution_thresholds(context: _PromptValidationContext) -> None:
    _apply_expected_distribution(
        context,
        label="content_type",
        counts=context.content_types,
        expected=context.thresholds.get("expected_content_distribution"),
    )
    expected_style = context.thresholds.get("expected_style_distribution")
    if isinstance(expected_style, Mapping):
        for style_key, expected in expected_style.items():
            _apply_expected_distribution(
                context,
                label=f"style {style_key}",
                counts=context.style_values.get(str(style_key), Counter()),
                expected=expected,
            )


def _apply_expected_distribution(
    context: _PromptValidationContext,
    *,
    label: str,
    counts: Counter[str],
    expected: Any,
) -> None:
    if not isinstance(expected, Mapping):
        return
    total = sum(counts.values())
    if total == 0:
        return
    for value, bounds in expected.items():
        if not isinstance(bounds, Sequence) or isinstance(bounds, str) or len(bounds) != 2:
            continue
        lower, upper = float(bounds[0]), float(bounds[1])
        ratio = counts.get(str(value), 0) / total
        if ratio < lower or ratio > upper:
            context.warnings.append(
                f"{label} {value} ratio {ratio:.3f} outside expected range [{lower:.3f}, {upper:.3f}]"
            )


def _rare_character_coverage_payload(context: _PromptValidationContext) -> dict[str, Any]:
    required = _required_rare_characters(context.thresholds)
    required_ordered = _required_rare_characters_ordered(context.thresholds)
    covered = [ch for ch in required_ordered if context.rare_counts.get(ch, 0) > 0]
    missing = [ch for ch in required_ordered if context.rare_counts.get(ch, 0) == 0]
    coverage_ratio = len(covered) / len(required) if required else 0.0
    return {
        "required": sorted(required),
        "covered": covered,
        "missing": missing,
        "coverage_ratio": round(coverage_ratio, 6),
        "counts": dict(sorted(context.rare_counts.items())),
    }


def _required_rare_characters(thresholds: Mapping[str, Any]) -> set[str]:
    return set(_required_rare_characters_ordered(thresholds))


def _required_rare_characters_ordered(thresholds: Mapping[str, Any]) -> list[str]:
    raw = thresholds.get("required_rare_characters", sorted({ch.lower() for ch in RARE_CHARS}))
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return list(dict.fromkeys(str(ch).lower() for ch in raw if str(ch)))
    return []


def _length_bucket(length: int) -> str:
    if length <= 4:
        return "1-4"
    if length <= 12:
        return "5-12"
    if length <= 24:
        return "13-24"
    return "25+"


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())
