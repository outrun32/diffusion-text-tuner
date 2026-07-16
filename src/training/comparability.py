"""CPU-safe controlled-field comparison helpers for training runs and configs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

COMPARABILITY_SCHEMA_VERSION = "training-comparability/v1"

CONTROLLED_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "training": ("num_training_steps",),
    "optimization": (
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "lr",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "mixed_precision",
        "lr_schedule",
        "lr_min",
        "resolution",
    ),
    "objective": (
        "beta",
        "shift",
        "num_train_timesteps",
        "masked_lambda",
        "score_diff_min",
        "ambiguity_margin",
        "hard_negative_threshold",
        "sft_lora_path",
        "resume_lora_path",
        "resume_step",
        "selection_mode",
        "selected_samples_path",
        "sample_weighting",
        "pair_construction_mode",
        "preference_pairs_path",
        "pair_weighting",
    ),
    "lora": ("lora",),
    "inference": ("num_inference_steps", "guidance_scale", "prompt_embedding_padding"),
    "prompt": ("seed", "sample_prompt", "sample_target_text"),
    "model": ("model_id", "model_revision"),
    "data_source": (
        "latents_dir",
        "text_embeds_dir",
        "scores_csv",
        "data_dir",
    ),
    "reward": ("score_column", "score_threshold", "reward_model", "scorer"),
    "metrics": ("metric_columns",),
    "artifacts": ("samples_dir",),
}

BLOCKING_GROUPS = frozenset(
    {
        "training",
        "optimization",
        "lora",
        "inference",
        "prompt",
        "model",
        "data_source",
        "reward",
    }
)


def compare_training_configs(
    left: Any,
    right: Any,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, Any]:
    """Compare controlled training fields from dictionaries or dataclasses.

    The comparison is intentionally pure and import-safe: callers may pass raw dictionaries,
    runtime config snapshots, or dataclass config instances. Missing-vs-present fields are reported
    explicitly instead of being treated as comparable.
    """

    left_payload = _normalize_config(left)
    right_payload = _normalize_config(right)

    blocking_mismatches: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for group, fields in CONTROLLED_FIELD_GROUPS.items():
        for field in fields:
            mismatch = _compare_field(left_payload, right_payload, group, field)
            if mismatch is None:
                continue
            if mismatch["severity"] == "blocking":
                blocking_mismatches.append(mismatch)
            else:
                warnings.append(mismatch)

    return {
        "schema_version": COMPARABILITY_SCHEMA_VERSION,
        "left_label": left_label,
        "right_label": right_label,
        "blocking_mismatches": blocking_mismatches,
        "warnings": warnings,
        "controlled_fields": {
            group: list(fields) for group, fields in CONTROLLED_FIELD_GROUPS.items()
        },
        "summary": {
            "blocking_count": len(blocking_mismatches),
            "warning_count": len(warnings),
            "is_comparable": not blocking_mismatches,
        },
    }


def compare_training_manifests(
    left_manifest: str | Path,
    right_manifest: str | Path,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> dict[str, Any]:
    """Load two run manifests and compare their training comparability fields."""

    from src.runtime.manifests import load_run_manifest

    left = load_run_manifest(left_manifest)
    right = load_run_manifest(right_manifest)
    return compare_training_configs(
        _manifest_comparison_payload(left),
        _manifest_comparison_payload(right),
        left_label=left_label or left.run_id,
        right_label=right_label or right.run_id,
    )


def format_comparability_report(report: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown report for humans and CI logs."""

    lines = [
        "# Training comparability report",
        "",
        f"Schema: {report.get('schema_version', COMPARABILITY_SCHEMA_VERSION)}",
        f"Left: {report.get('left_label', 'left')}",
        f"Right: {report.get('right_label', 'right')}",
        "",
    ]
    summary = report.get("summary", {})
    if isinstance(summary, Mapping):
        lines.extend(
            [
                "## Summary",
                "",
                f"- Comparable: {str(bool(summary.get('is_comparable', False))).lower()}",
                f"- Blocking mismatches: {summary.get('blocking_count', 0)}",
                f"- Warnings: {summary.get('warning_count', 0)}",
                "",
            ]
        )

    lines.extend(_format_mismatch_section("Blocking mismatches", report.get("blocking_mismatches")))
    lines.extend(_format_mismatch_section("Warnings", report.get("warnings")))
    lines.extend(_format_controlled_fields(report.get("controlled_fields")))
    return "\n".join(lines).rstrip() + "\n"


def _normalize_config(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return _normalize_config(asdict(value))
    if isinstance(value, Mapping):
        payload = {str(key): _normalize_value(item) for key, item in value.items()}
        batch_size = payload.get("batch_size")
        accumulation = payload.get("gradient_accumulation_steps")
        if isinstance(batch_size, int) and isinstance(accumulation, int):
            payload["effective_batch_size"] = batch_size * accumulation
        return payload
    raise TypeError("training comparison inputs must be mappings or dataclass instances")


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_config(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _compare_field(
    left: Mapping[str, Any], right: Mapping[str, Any], group: str, field: str
) -> dict[str, Any] | None:
    left_has = field in left
    right_has = field in right
    if not left_has and not right_has:
        return None
    left_value = left.get(field)
    right_value = right.get(field)
    if left_has and right_has and left_value == right_value:
        return None

    if not left_has:
        reason = "missing_left"
    elif not right_has:
        reason = "missing_right"
    else:
        reason = "value_mismatch"
    severity = "blocking" if group in BLOCKING_GROUPS else "warning"
    return {
        "field": field,
        "group": group,
        "left": left_value if left_has else None,
        "right": right_value if right_has else None,
        "reason": reason,
        "severity": severity,
    }


def _manifest_comparison_payload(manifest: Any) -> dict[str, Any]:
    payload = _normalize_config(manifest.config_snapshot)
    payload.update(_extract_known_fields(manifest.inputs))
    payload.update(_extract_known_fields(manifest.outputs))
    if manifest.metrics:
        payload.setdefault("metric_columns", sorted(str(key) for key in manifest.metrics))
    return payload


def _extract_known_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    known = {field for fields in CONTROLLED_FIELD_GROUPS.values() for field in fields}
    return {key: _normalize_value(value) for key, value in values.items() if key in known}


def _format_mismatch_section(title: str, mismatches: Any) -> list[str]:
    rows = mismatches if isinstance(mismatches, list) else []
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["None.", ""])
        return lines
    lines.extend(
        ["| Field | Group | Left | Right | Reason |", "|-------|-------|------|-------|--------|"]
    )
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| {field} | {group} | {left} | {right} | {reason} |".format(
                field=item.get("field", ""),
                group=item.get("group", ""),
                left=_markdown_value(item.get("left")),
                right=_markdown_value(item.get("right")),
                reason=item.get("reason", ""),
            )
        )
    lines.append("")
    return lines


def _format_controlled_fields(controlled_fields: Any) -> list[str]:
    if not isinstance(controlled_fields, Mapping):
        return []
    lines = ["## Controlled fields", ""]
    for group in CONTROLLED_FIELD_GROUPS:
        fields = controlled_fields.get(group, [])
        if isinstance(fields, list):
            field_list = ", ".join(str(field) for field in fields)
            lines.append(f"- **{group}:** {field_list}")
    lines.append("")
    return lines


def _markdown_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value.replace("|", "\\|")
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
