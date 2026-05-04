"""CPU-safe generated-vs-synthetic dataset source comparison.

The comparison layer consumes existing Phase 3 JSON/JSONL metadata artifacts. It
does not inspect generated images, load tensors, import OCR/model stacks, or run
training code. Missing optional inputs are represented explicitly so reports do
not fabricate unavailable evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

DATA_SOURCE_COMPARISON_SCHEMA_VERSION = "data-source-comparison/v1"

_EVIDENCE_INPUTS = {
    "generated_prompt_quality_report": "generated_prompt_quality_report",
    "selected_samples": "selected_samples",
    "preference_pairs": "preference_pairs",
    "generated_dataset_manifest": "generated_dataset_manifest",
    "synthetic_quality_report": "synthetic_quality_report",
    "synthetic_manifest": "synthetic_manifest",
}


@dataclass(frozen=True)
class DataSourceComparison:
    """Serializable generated reward-filtered vs synthetic masked-SFT report."""

    evidence_available: list[str]
    evidence_missing: list[str]
    counts: dict[str, int | None]
    rare_character_coverage: dict[str, Any]
    distribution_differences: dict[str, Any]
    generated_score_summary: dict[str, dict[str, float | int | None]]
    synthetic_mask_contrast_health: dict[str, Any]
    expected_help: dict[str, list[str]]
    expected_failure: dict[str, list[str]]
    provenance: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = DATA_SOURCE_COMPARISON_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_available": self.evidence_available,
            "evidence_missing": self.evidence_missing,
            "counts": self.counts,
            "rare_character_coverage": self.rare_character_coverage,
            "distribution_differences": self.distribution_differences,
            "generated_score_summary": self.generated_score_summary,
            "synthetic_mask_contrast_health": self.synthetic_mask_contrast_health,
            "expected_help": self.expected_help,
            "expected_failure": self.expected_failure,
            "provenance": self.provenance,
            "warnings": self.warnings,
        }


def compare_data_sources(
    *,
    generated_prompt_quality_report: str | Path | None = None,
    selected_samples: str | Path | None = None,
    preference_pairs: str | Path | None = None,
    generated_dataset_manifest: str | Path | None = None,
    synthetic_quality_report: str | Path | None = None,
    synthetic_manifest: str | Path | None = None,
) -> DataSourceComparison:
    """Compare generated reward-filtered and synthetic masked-SFT metadata.

    All inputs are optional. Provided JSON/JSONL files are minimally parsed and
    recorded with path/hash provenance; absent inputs stay visible in
    ``evidence_missing`` and their dependent metrics are ``None`` or empty.
    """

    context = _ComparisonContext()
    generated_prompt = context.load_json(
        "generated_prompt_quality_report", generated_prompt_quality_report
    )
    selected_rows = context.load_jsonl("selected_samples", selected_samples)
    pair_rows = context.load_jsonl("preference_pairs", preference_pairs)
    context.load_json("generated_dataset_manifest", generated_dataset_manifest)
    synthetic_report = context.load_json("synthetic_quality_report", synthetic_quality_report)
    context.load_json("synthetic_manifest", synthetic_manifest)

    generated_rare = _nonzero_counts(
        _nested_counts(generated_prompt, "rare_character_coverage")
    )
    synthetic_rare = _nonzero_counts(_nested_counts(synthetic_report, "character_coverage"))
    rare_coverage = _coverage_overlap(generated_rare, synthetic_rare)

    return DataSourceComparison(
        evidence_available=context.evidence_available(),
        evidence_missing=context.evidence_missing(),
        counts=_counts(generated_prompt, selected_rows, pair_rows, synthetic_report),
        rare_character_coverage=rare_coverage,
        distribution_differences=_distribution_differences(generated_prompt, synthetic_report),
        generated_score_summary=_generated_score_summary(selected_rows, pair_rows),
        synthetic_mask_contrast_health=_synthetic_health(synthetic_report),
        expected_help=_expected_help(),
        expected_failure=_expected_failure(),
        provenance=context.provenance,
        warnings=context.warnings,
    )


@dataclass
class _ComparisonContext:
    available: set[str] = field(default_factory=set)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def load_json(self, evidence_name: str, path: str | Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        evidence_path = Path(path)
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.warnings.append(f"{evidence_path}: malformed JSON evidence")
            self._record_provenance(evidence_name, evidence_path, record_count=None)
            return None
        except OSError:
            self.warnings.append(f"{evidence_path}: could not read evidence")
            self._record_provenance(evidence_name, evidence_path, record_count=None)
            return None
        if not isinstance(payload, dict):
            self.warnings.append(f"{evidence_path}: JSON evidence must be an object")
            self._record_provenance(evidence_name, evidence_path, record_count=None)
            return None
        self.available.add(evidence_name)
        self._record_provenance(evidence_name, evidence_path, payload=payload, record_count=1)
        return payload

    def load_jsonl(
        self, evidence_name: str, path: str | Path | None
    ) -> list[dict[str, Any]] | None:
        if path is None:
            return None
        evidence_path = Path(path)
        rows: list[dict[str, Any]] = []
        try:
            with evidence_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        self.warnings.append(
                            f"{evidence_path}: line {line_number}: malformed JSONL evidence"
                        )
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
                    else:
                        self.warnings.append(
                            f"{evidence_path}: line {line_number}: JSONL row must be an object"
                        )
        except OSError:
            self.warnings.append(f"{evidence_path}: could not read evidence")
            self._record_provenance(evidence_name, evidence_path, record_count=None)
            return None
        self.available.add(evidence_name)
        self._record_provenance(
            evidence_name,
            evidence_path,
            payload=rows[0] if rows else {},
            record_count=len(rows),
        )
        return rows

    def evidence_available(self) -> list[str]:
        return [name for name in _EVIDENCE_INPUTS if name in self.available]

    def evidence_missing(self) -> list[str]:
        return [name for name in _EVIDENCE_INPUTS if name not in self.available]

    def _record_provenance(
        self,
        evidence_name: str,
        path: Path,
        *,
        payload: Mapping[str, Any] | None = None,
        record_count: int | None,
    ) -> None:
        path_key = str(path)
        provenance: dict[str, Any] = {
            "evidence": evidence_name,
            "path": path_key,
            "sha256": _sha256_file(path),
            "record_count": record_count,
        }
        if payload is not None:
            if isinstance(payload.get("schema_version"), str):
                provenance["schema_version"] = payload["schema_version"]
            if isinstance(payload.get("dataset_kind"), str):
                provenance["dataset_kind"] = payload["dataset_kind"]
        self.provenance[path_key] = provenance


def _counts(
    generated_prompt: Mapping[str, Any] | None,
    selected_rows: list[dict[str, Any]] | None,
    pair_rows: list[dict[str, Any]] | None,
    synthetic_report: Mapping[str, Any] | None,
) -> dict[str, int | None]:
    return {
        "generated_prompt_records": _int_field(generated_prompt, "valid_records"),
        "generated_selected_samples": len(selected_rows) if selected_rows is not None else None,
        "generated_preference_pairs": len(pair_rows) if pair_rows is not None else None,
        "synthetic_samples": _int_field(synthetic_report, "sample_count"),
        "synthetic_accepted": _int_field(synthetic_report, "accepted_count"),
        "synthetic_rejected": _int_field(synthetic_report, "rejected_count"),
    }


def _distribution_differences(
    generated_prompt: Mapping[str, Any] | None,
    synthetic_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "content_type": {
            "generated": _dict_field(generated_prompt, "content_type_distribution"),
            "synthetic": {},
        },
        "style": {
            "generated": _dict_field(generated_prompt, "style_distribution"),
            "synthetic": {"font": _dict_field(synthetic_report, "font_coverage")},
        },
        "resolution": {
            "generated": {},
            "synthetic": _dict_field(synthetic_report, "resolution_distribution"),
        },
    }


def _generated_score_summary(
    selected_rows: list[dict[str, Any]] | None,
    pair_rows: list[dict[str, Any]] | None,
) -> dict[str, dict[str, float | int | None]]:
    return {
        "selected_score": _summary(
            _numeric_values(selected_rows or [], "selected_score")
        ),
        "preference_margin": _summary(_numeric_values(pair_rows or [], "margin")),
        "winner_score": _summary(_numeric_values(pair_rows or [], "winner_score")),
        "loser_score": _summary(_numeric_values(pair_rows or [], "loser_score")),
    }


def _synthetic_health(synthetic_report: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "mask_area_fraction": _dict_field(synthetic_report, "mask_area_fraction"),
        "contrast": _dict_field(synthetic_report, "contrast"),
        "rejection_reasons": _dict_field(synthetic_report, "rejection_reasons"),
        "ocr_summary": _optional_dict_field(synthetic_report, "ocr_summary"),
    }


def _coverage_overlap(
    generated_counts: Mapping[str, int],
    synthetic_counts: Mapping[str, int],
) -> dict[str, Any]:
    generated_keys = set(generated_counts)
    synthetic_keys = set(synthetic_counts)
    return {
        "generated": dict(sorted(generated_counts.items())),
        "synthetic": dict(sorted(synthetic_counts.items())),
        "overlap": sorted(generated_keys & synthetic_keys),
        "generated_only": sorted(generated_keys - synthetic_keys),
        "synthetic_only": sorted(synthetic_keys - generated_keys),
    }


def _expected_help() -> dict[str, list[str]]:
    return {
        "generated_reward_filtered": [
            "Aligns SFT/DPO training evidence to actual generated FLUX outputs.",
            "Captures reward-model preferences, score margins, and prompt-image failure patterns.",
            "Supports comparison against reward-filtered artifacts used by generated-image runs.",
        ],
        "synthetic_masked_sft": [
            "Provides controlled text-region reconstruction examples with mask evidence.",
            "Improves coverage of rare characters, fonts, and resolutions from curated inputs.",
            "Supports masked-SFT where local text reconstruction is the supervision signal.",
        ],
    }


def _expected_failure() -> dict[str, list[str]]:
    return {
        "generated_reward_filtered": [
            "Can inherit reward or OCR false positives from the scorer used for filtering.",
            "May preserve prompt gaps, rare-character gaps, and style imbalance.",
            "Score thresholds and DPO margins are internal selection evidence only.",
        ],
        "synthetic_masked_sft": [
            "Can miss natural scene complexity, domain realism, and FLUX artifacts.",
            "Renderer masks, contrast, and fonts may create shortcuts that do not transfer.",
            "Synthetic reconstruction loss is internal until Phase 6 validates text quality.",
        ],
    }


def _nested_counts(payload: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    nested = _dict_field(payload, field_name)
    return _dict_field(nested, "counts")


def _nonzero_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    result = {}
    for key, value in counts.items():
        if isinstance(value, int | float) and value > 0:
            result[str(key)] = int(value)
    return result


def _numeric_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
            continue
        if isinstance(value, str):
            try:
                values.append(float(value))
            except ValueError:
                continue
    return values


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
    }


def _int_field(payload: Mapping[str, Any] | None, key: str) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _dict_field(payload: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    if payload is None:
        return {}
    value = payload.get(key)
    return dict(sorted(value.items())) if isinstance(value, Mapping) else {}


def _optional_dict_field(payload: Mapping[str, Any] | None, key: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    value = payload.get(key)
    return dict(sorted(value.items())) if isinstance(value, Mapping) else None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
