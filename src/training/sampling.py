"""Import-safe sampling helper contracts for trainer variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def should_sample_step(step: int, interval: int) -> bool:
    """Return whether a positive training step should emit samples."""

    return interval > 0 and step > 0 and step % interval == 0


def normalize_eval_suite_items(
    items: Sequence[Mapping[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    """Return deterministic eval-suite prompt dictionaries without mutating inputs.

    The helper accepts the existing masked-SFT ``target`` alias and normalizes it to
    ``target_text`` so future trainer variants can consume one small contract.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    selected_items = items if limit is None else items[:limit]
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(selected_items):
        prompt = item.get("prompt")
        target_text = item.get("target_text", item.get("target"))
        if prompt is None:
            raise ValueError(f"eval suite item {index} is missing prompt")
        if target_text is None:
            raise ValueError(f"eval suite item {index} is missing target_text")

        normalized_item = {
            key: value for key, value in item.items() if key not in {"target", "target_text"}
        }
        normalized_item["name"] = str(item.get("name", f"item_{index:02d}"))
        normalized_item["prompt"] = prompt
        normalized_item["target_text"] = target_text
        normalized.append({key: normalized_item[key] for key in sorted(normalized_item)})

    return normalized
