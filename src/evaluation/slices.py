"""CPU-safe Russian text difficulty slice classification.

The helpers in this module are pure Python and deterministic. They inspect
local record dictionaries only; they never load images, OCR engines, model
weights, CUDA, or generated artifacts.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

RARE_CYRILLIC_LETTERS = frozenset("ёЁжЖцЦщЩъЪ")
PUNCTUATION_CHARS = frozenset(string.punctuation + "—–…«»№")
STYLE_FIELDS = ("font", "font_family", "font_style", "style", "text_style")
SCENE_FIELDS = ("scene", "background", "background_type", "environment")


def classify_text_slices(record: Mapping[str, Any]) -> set[str]:
    """Return deterministic difficulty slice labels for one evaluation record.

    The primary input is ``target_text``. Optional metadata fields such as
    ``font``, ``style``, ``scene``, and ``background`` add visual-context slices
    without requiring image inspection.
    """
    target_text = str(record.get("target_text") or "")
    stripped = target_text.strip()
    if not stripped:
        return set()

    labels: set[str] = set()
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]

    if any(character in RARE_CYRILLIC_LETTERS for character in stripped):
        labels.add("rare_cyrillic")
    if any(character.isdigit() for character in stripped):
        labels.add("has_digits")
    if any(character in PUNCTUATION_CHARS for character in stripped):
        labels.add("has_punctuation")
    if "\n" in target_text or "\r" in target_text:
        labels.add("multiline")
    if _has_mixed_case(stripped):
        labels.add("mixed_case")
    if len(words) >= 2:
        labels.add("multi_word_phrase")
    elif words:
        labels.add("short_word")
    if any(len(_strip_punctuation(word)) >= 10 for word in words):
        labels.add("long_word")
    if _has_metadata(record, STYLE_FIELDS):
        labels.add("font_or_style")
    if _has_metadata(record, SCENE_FIELDS):
        labels.add("scene_or_background")

    return labels


def summarize_slices(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count difficulty slices and records missing target text."""
    total_records = 0
    classified_records = 0
    slice_counts: Counter[str] = Counter()
    missing_target_text_records: list[str] = []

    for index, record in enumerate(records):
        total_records += 1
        target_text = str(record.get("target_text") or "").strip()
        sample_id = str(record.get("sample_id") or record.get("id") or index)
        if not target_text:
            missing_target_text_records.append(sample_id)
            continue

        labels = classify_text_slices(record)
        if labels:
            classified_records += 1
        slice_counts.update(labels)

    return {
        "total_records": total_records,
        "classified_records": classified_records,
        "missing_target_text_records": missing_target_text_records,
        "slice_counts": dict(sorted(slice_counts.items())),
    }


def _has_mixed_case(text: str) -> bool:
    words = [word for word in text.replace("\n", " ").split(" ") if word]
    has_lowercase = any(character.islower() for character in text if character.isalpha())
    has_uppercase_word = any(
        len(_strip_punctuation(word)) > 1
        and _strip_punctuation(word).isupper()
        and any(character.isalpha() for character in word)
        for word in words
    )
    has_internal_uppercase = any(
        any(character.isupper() for character in _strip_punctuation(word)[1:])
        for word in words
    )
    return has_lowercase and (has_uppercase_word or has_internal_uppercase)


def _strip_punctuation(word: str) -> str:
    return "".join(character for character in word if character not in PUNCTUATION_CHARS)


def _has_metadata(record: Mapping[str, Any], fields: tuple[str, ...]) -> bool:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return True
        if value not in (None, "", [], {}):
            return True
    return False
