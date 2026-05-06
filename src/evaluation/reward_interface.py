"""Canonical, import-safe reward and product-score contracts.

This module intentionally uses only the Python standard library. It is shared by
scoring, training, evaluation, diagnostics, and thesis reporting code without
importing optional model/OCR stacks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "reward-result/v1"
METADATA_SCHEMA_VERSION = "reward-score-metadata/v1"
DEFAULT_PRODUCT_FORMULA_NAME = "vlm_ocr_cer_entropy_exact_product_v1"

DEFAULT_PRODUCT_WEIGHTS: dict[str, float] = {
    "score_vlm": 0.35,
    "score_ocr": 0.25,
    "cer_quality": 0.20,
    "entropy_quality": 0.10,
    "exact_text_match": 0.10,
}


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(sorted(value.items())), ensure_ascii=False, sort_keys=True)


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping.items()))


def _coerce_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_unit_interval(value: Any) -> float | None:
    number = _coerce_finite_float(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def _component_source_name(component_name: str) -> str:
    if component_name == "cer_quality":
        return "cer"
    if component_name == "entropy_quality":
        return "entropy"
    return component_name


def _normalize_component(
    component_name: str,
    evidence: Mapping[str, Any],
    entropy_scale: float,
) -> float | None:
    if component_name == "cer_quality":
        cer = _coerce_unit_interval(evidence.get("cer"))
        return None if cer is None else 1.0 - cer
    if component_name == "entropy_quality":
        entropy = _coerce_finite_float(evidence.get("entropy"))
        if entropy is None or entropy < 0:
            return None
        return math.exp(-entropy_scale * entropy)
    if component_name == "exact_text_match":
        exact = evidence.get("exact_text_match")
        if isinstance(exact, bool):
            return 1.0 if exact else 0.0
        return _coerce_unit_interval(exact)
    return _coerce_unit_interval(evidence.get(component_name))


def _threshold_component_name(threshold_name: str) -> tuple[str, str] | None:
    if threshold_name.endswith("_min"):
        return threshold_name.removesuffix("_min"), "min"
    if threshold_name.endswith("_max"):
        return threshold_name.removesuffix("_max"), "max"
    return None


@dataclass(frozen=True)
class ProductScoreFormula:
    """Product-score formula metadata and reproducibility controls."""

    name: str = DEFAULT_PRODUCT_FORMULA_NAME
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PRODUCT_WEIGHTS))
    thresholds: Mapping[str, float] = field(default_factory=dict)
    scorer_versions: Mapping[str, str] = field(default_factory=dict)
    entropy_scale: float = 1.0

    def __post_init__(self) -> None:
        weights = {key: _coerce_finite_float(value) for key, value in self.weights.items()}
        invalid_weights = [key for key, value in weights.items() if value is None or value < 0]
        if invalid_weights:
            joined = ", ".join(sorted(invalid_weights))
            raise ValueError(f"Invalid product formula weights: {joined}")
        if sum(value for value in weights.values() if value is not None) <= 0:
            raise ValueError("Product formula weights must include a positive total weight")

        thresholds = {key: _coerce_finite_float(value) for key, value in self.thresholds.items()}
        invalid_thresholds = [key for key, value in thresholds.items() if value is None]
        if invalid_thresholds:
            joined = ", ".join(sorted(invalid_thresholds))
            raise ValueError(f"Invalid product formula thresholds: {joined}")

        entropy_scale = _coerce_finite_float(self.entropy_scale)
        if entropy_scale is None or entropy_scale < 0:
            raise ValueError("entropy_scale must be a finite non-negative number")

        object.__setattr__(self, "weights", _freeze_mapping(weights))
        object.__setattr__(self, "thresholds", _freeze_mapping(thresholds))
        object.__setattr__(self, "scorer_versions", _freeze_mapping(self.scorer_versions))
        object.__setattr__(self, "entropy_scale", entropy_scale)

    def to_metadata(self) -> dict[str, Any]:
        """Return deterministic JSON-safe formula metadata."""
        return {
            "name": self.name,
            "weights": dict(sorted(self.weights.items())),
            "thresholds": dict(sorted(self.thresholds.items())),
            "scorer_versions": dict(sorted(self.scorer_versions.items())),
            "entropy_scale": self.entropy_scale,
        }


@dataclass(frozen=True)
class ProductScoreResult:
    """Computed product score plus evidence accounting."""

    score: float
    components: Mapping[str, float]
    missing_components: tuple[str, ...]
    threshold_flags: Mapping[str, bool]
    formula: ProductScoreFormula
    formula_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _freeze_mapping(self.components))
        object.__setattr__(self, "threshold_flags", _freeze_mapping(self.threshold_flags))

    def to_metadata(self) -> dict[str, Any]:
        """Return deterministic JSON-safe result metadata for sidecars/reports."""
        return {
            "score": self.score,
            "components": dict(self.components),
            "missing_components": list(self.missing_components),
            "threshold_flags": dict(self.threshold_flags),
            "formula_complete": self.formula_complete,
            "formula": self.formula.to_metadata(),
        }


@dataclass(frozen=True)
class RewardResult:
    """Canonical reward record for score CSV/JSON sidecars and reports."""

    sample_id: str
    version: int
    target_text: str
    score: float
    components: Mapping[str, Any] = field(default_factory=dict)
    text_metrics: Mapping[str, Any] = field(default_factory=dict)
    scorer_metadata: Mapping[str, Any] = field(default_factory=dict)
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: str = ""
    missing_components: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        score = _coerce_finite_float(self.score)
        if score is None:
            raise ValueError("RewardResult.score must be finite")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "components", _freeze_mapping(self.components))
        object.__setattr__(self, "text_metrics", _freeze_mapping(self.text_metrics))
        object.__setattr__(self, "scorer_metadata", _freeze_mapping(self.scorer_metadata))
        object.__setattr__(self, "thresholds", _freeze_mapping(self.thresholds))
        object.__setattr__(self, "missing_components", tuple(self.missing_components))

    def to_row(self) -> dict[str, Any]:
        """Return deterministic CSV/JSON-safe fields for the canonical reward row."""
        row: dict[str, Any] = {
            "sample_id": self.sample_id,
            "version": self.version,
            "target_text": self.target_text,
            "score": self.score,
        }
        row.update(dict(self.components))
        row.update(
            {
                "text_metrics": _json_dumps(self.text_metrics),
                "scorer_metadata": _json_dumps(self.scorer_metadata),
                "thresholds": _json_dumps(self.thresholds),
                "missing_components": ",".join(self.missing_components),
                "manifest_path": self.manifest_path,
            }
        )
        return row


def compute_product_score(
    evidence: Mapping[str, Any] | RewardResult,
    *,
    formula: ProductScoreFormula | None = None,
) -> ProductScoreResult:
    """Compute a missing-aware weighted product score from local evidence.

    The score is a weighted geometric product over available normalized
    components. Missing or non-finite components are excluded from the numeric
    product and reported in ``missing_components`` so callers do not treat those
    rows as equally comparable.
    """
    active_formula = formula or ProductScoreFormula()
    raw_evidence: Mapping[str, Any]
    if isinstance(evidence, RewardResult):
        raw_evidence = evidence.components
    else:
        raw_evidence = evidence

    components: dict[str, float] = {}
    missing_components: list[str] = []
    weighted_log_sum = 0.0
    available_weight = 0.0

    for component_name, raw_weight in active_formula.weights.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        value = _normalize_component(component_name, raw_evidence, active_formula.entropy_scale)
        if value is None:
            missing_components.append(_component_source_name(component_name))
            continue
        components[component_name] = value
        available_weight += weight
        if value <= 0:
            weighted_log_sum = float("-inf")
        elif not math.isinf(weighted_log_sum):
            weighted_log_sum += weight * math.log(value)

    if available_weight <= 0:
        score = 0.0
    elif math.isinf(weighted_log_sum):
        score = 0.0
    else:
        score = math.exp(weighted_log_sum / available_weight)

    threshold_flags = _compute_threshold_flags(
        raw_evidence,
        active_formula.thresholds,
        active_formula.entropy_scale,
    )
    return ProductScoreResult(
        score=score,
        components=components,
        missing_components=tuple(missing_components),
        threshold_flags=threshold_flags,
        formula=active_formula,
        formula_complete=not missing_components,
    )


def _compute_threshold_flags(
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, float],
    entropy_scale: float,
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for threshold_name, threshold_value in thresholds.items():
        parsed = _threshold_component_name(threshold_name)
        if parsed is None:
            flags[threshold_name] = False
            continue
        component_name, direction = parsed
        if component_name in {"cer_quality", "entropy_quality", "exact_text_match"}:
            value = _normalize_component(component_name, evidence, entropy_scale)
        else:
            value = _coerce_finite_float(evidence.get(component_name))
        if value is None:
            flags[threshold_name] = False
        elif direction == "min":
            flags[threshold_name] = value >= threshold_value
        else:
            flags[threshold_name] = value <= threshold_value
    return flags


def build_score_metadata(
    *,
    formula: ProductScoreFormula | None = None,
    source_manifest_paths: Sequence[str] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build deterministic score-file sidecar metadata.

    Manifest paths are stored exactly as caller-supplied provenance references;
    this helper never inspects local files, cache locations, environment
    variables, or secrets.
    """
    active_formula = formula or ProductScoreFormula()
    timestamp = generated_at or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "generated_at": timestamp,
        "formula": active_formula.to_metadata(),
        "source_manifest_paths": list(source_manifest_paths),
    }
