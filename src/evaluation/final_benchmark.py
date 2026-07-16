"""Final-review benchmark and artifact-report helpers.

This module is intentionally CPU-safe: it builds prompt JSONL files and analyzes
recorded CSV artifacts without importing CUDA, FLUX, OCR, or VLM stacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HARD_CYRILLIC = set("ЖЩЪЫЁЮЯЭЦЧШ")
CYRILLIC_RANGES = (("А", "я"), ("Ё", "Ё"), ("ё", "ё"))
LATIN_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
TARGET_HASH_INDEX_SCHEMA_VERSION = "normalized-target-hash-index/v1"
TARGET_HASH_ALGORITHM = "sha256"
TARGET_NORMALIZATION = "strip-collapse-whitespace-unicode-uppercase/v1"

_LATIN_TO_CYRILLIC: dict[str, str] = {
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "a": "а",
    "c": "с",
    "e": "е",
    "o": "о",
    "p": "р",
    "x": "х",
}
_HOMOGLYPH_TABLE = str.maketrans(_LATIN_TO_CYRILLIC)

PROMPT_SLICES: dict[str, list[str]] = {
    "easy_short": [
        "ДОМ",
        "ЛЕС",
        "КОТ",
        "СНЕГ",
        "РЕКА",
        "ГОРОД",
        "КНИГА",
        "ВЕТЕР",
        "СОЛНЦЕ",
        "МОРЕ",
        "ОКНО",
        "МОСТ",
        "ПОЛЕ",
        "ЛАМПА",
        "ШКОЛА",
        "ПОЕЗД",
        "ЗВЕЗДА",
        "РАДИО",
        "ТЕАТР",
        "МУЗЕЙ",
    ],
    "hard_letters": [
        "ЖУК",
        "ЩИТ",
        "ЧАЙ",
        "ЦЕНА",
        "ШАР",
        "ЖИЗНЬ",
        "ЩЕКА",
        "ЧАСЫ",
        "ЦИРК",
        "ШКОЛА",
        "ЖЕЛЕЗО",
        "ЩЕДРОСТЬ",
        "ЧЕРТЕЖ",
        "ЦВЕТОК",
        "ШИРОТА",
        "ЖУРНАЛ",
        "ЩЕКОТКА",
        "ЧЕМОДАН",
        "ЦИФРА",
        "ШЕПОТ",
        "ШЛЮЗ",
        "ЖЁЛУДЬ",
        "ГИЛЬЗА",
    ],
    "signs_yo_yeru": [
        "ОБЪЕМ",
        "ПОДЪЕМ",
        "СЪЕЗД",
        "ВЪЕЗД",
        "ОБЪЯВЛЕНИЕ",
        "СЕМЬЯ",
        "ПИСЬМО",
        "МЕДВЕДЬ",
        "ДЕНЬ",
        "ПЫЛЬ",
        "ЁЛКА",
        "СЪЕМКА",
        "БЕЛЫЙ",
        "СЫР",
        "МЫС",
        "НОЧЬЮ",
        "ПОДЪЕЗД",
        "ВЬЮГА",
        "ОГОНЁК",
        "РЫНОК",
        "РАЗЪЁМ",
        "АДЪЮТАНТ",
        "КОНЪЮНКТУРА",
        "ПЬЕСА",
        "РУЖЬЁ",
    ],
    "medium_long": [
        "БИБЛИОТЕКА",
        "ТЕЛЕВИЗОР",
        "АРХИТЕКТУРА",
        "КОСМОНАВТИКА",
        "ПУТЕШЕСТВИЕ",
        "ЭЛЕКТРИЧЕСТВО",
        "ФОТОГРАФИЯ",
        "УНИВЕРСИТЕТ",
        "ЛАБОРАТОРИЯ",
        "ПРОГРАММИСТ",
        "ВЕЛОСИПЕД",
        "ПРАЗДНИЧНЫЙ",
        "РАСПИСАНИЕ",
        "ВОСПОМИНАНИЕ",
        "СООБЩЕНИЕ",
        "НАБЛЮДЕНИЕ",
        "СОТРУДНИЧЕСТВО",
        "ИССЛЕДОВАНИЕ",
        "ПРЕДУПРЕЖДЕНИЕ",
        "ВОССТАНОВЛЕНИЕ",
    ],
    "phrases": [
        "С НОВЫМ ГОДОМ",
        "ВХОД СВОБОДНЫЙ",
        "ТИХИЙ ЧАС",
        "ДОБРОЕ УТРО",
        "СКОРАЯ ПОМОЩЬ",
        "ОСТОРОЖНО ДВЕРИ",
        "МУЗЫКА ВЕЧЕРА",
        "КРАСНЫЙ МОСТ",
        "ЧАЙ И КОФЕ",
        "СЕВЕРНЫЙ ВЕТЕР",
        "НОЧНОЙ РЕЙС",
        "СВЕТЛЫЙ ДОМ",
        "ЗИМНИЙ САД",
        "НОВЫЙ МАРШРУТ",
        "ПЕРВЫЙ ЭТАЖ",
        "БИЛЕТ В КИНО",
        "ГОРОДСКОЙ ПАРК",
        "КРАСИВАЯ ВЫВЕСКА",
        "СИЛЬНЫЙ ДОЖДЬ",
        "ДОБРЫЙ ВЕЧЕР",
    ],
    "complex_style": [
        "ЖЁЛТЫЙ ПЛАКАТ",
        "ЩУКА НА РЫНКЕ",
        "СЪЕМКА ФИЛЬМА",
        "ПОДЪЕЗД НОМЕР ПЯТЬ",
        "ЦЕНТР ГОРОДА",
        "ЧЕРНАЯ КНИГА",
        "ШИРОКИЙ ПРОСПЕКТ",
        "ЮЖНЫЙ ВОКЗАЛ",
        "ЯРКИЙ НЕОН",
        "ЭХО МУЗЫКИ",
        "ЗИМНЯЯ ЯРМАРКА",
        "БОЛЬШОЙ ОБЪЕМ",
        "ВЪЕЗД ЗАКРЫТ",
        "НОЧЬЮ СНЕГ",
        "СЫР И ХЛЕБ",
        "ЁЖИК В ТУМАНЕ",
        "ЩЕДРЫЙ ДЕНЬ",
        "ЖИВОЙ ОГОНЁК",
        "ЧИСТЫЙ ВОЗДУХ",
        "ЦВЕТНОЙ ФОНАРЬ",
    ],
}

SCENE_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "poster",
        "A clean typography poster with the exact Russian text '{text}', centered, high contrast.",
    ),
    (
        "street_sign",
        (
            "A realistic street sign on a city wall with the exact Russian text '{text}', "
            "readable letters."
        ),
    ),
    (
        "neon",
        (
            "A dark cafe wall with a neon sign showing the exact Russian text '{text}', "
            "glowing letters."
        ),
    ),
    (
        "packaging",
        "A product package label with the exact Russian text '{text}', crisp printed typography.",
    ),
    (
        "graffiti",
        "A brick wall with graffiti text '{text}' in bright paint, urban style, readable letters.",
    ),
    (
        "book_cover",
        "A book cover design with the exact Russian title '{text}', bold serif typography.",
    ),
)

STYLE_TAGS = ("plain", "bold", "serif", "sans", "neon", "printed")


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class DpoRewardSpec:
    name: str
    scores_csv: Path
    winner_threshold: float
    margin: float


def normalize_homoglyphs(text: str) -> str:
    """Replace Latin homoglyphs with visually equivalent Cyrillic letters."""
    return text.translate(_HOMOGLYPH_TABLE)


def normalize_text(text: str, *, homoglyphs: bool = False) -> str:
    """Normalize text for exact/CER comparisons while preserving script choice."""
    value = normalize_homoglyphs(text) if homoglyphs else text
    return " ".join(value.casefold().strip().split())


def character_error_rate(hypothesis: str, reference: str, *, homoglyphs: bool = False) -> float:
    """Compute Levenshtein character error rate."""
    hyp = normalize_text(hypothesis, homoglyphs=homoglyphs)
    ref = normalize_text(reference, homoglyphs=homoglyphs)
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        next_dp = [i] + [0] * m
        ref_ch = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_ch == hyp[j - 1] else 1
            next_dp[j] = min(next_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = next_dp
    return dp[m] / max(n, 1)


def is_cyrillic_letter(char: str) -> bool:
    return any(start <= char <= end for start, end in CYRILLIC_RANGES)


def script_mixing_stats(text: str) -> dict[str, float | bool | int]:
    """Return simple Latin/Cyrillic script-mixing diagnostics for OCR output."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return {
            "latin_letters": 0,
            "cyrillic_letters": 0,
            "letter_count": 0,
            "latin_letter_rate": 0.0,
            "script_mixing_flag": False,
        }
    latin = sum(1 for char in letters if char in LATIN_LETTERS)
    cyrillic = sum(1 for char in letters if is_cyrillic_letter(char))
    return {
        "latin_letters": latin,
        "cyrillic_letters": cyrillic,
        "letter_count": len(letters),
        "latin_letter_rate": latin / len(letters),
        "script_mixing_flag": latin > 0 and cyrillic > 0,
    }


def target_length(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def hard_glyphs(text: str) -> list[str]:
    return sorted({char for char in text.upper() if char in HARD_CYRILLIC})


def make_benchmark_prompts(
    output_path: str | Path,
    *,
    count_per_slice: int = 20,
    manifest_path: str | Path | None = None,
    excluded_targets: Iterable[str] = (),
    exclusion_manifest: str | Path | None = None,
    exclusion_target_hash_index: str | Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic held-out prompt JSONL benchmark."""
    if count_per_slice < 1:
        raise ValueError("count_per_slice must be >= 1")
    output = Path(output_path)
    excluded = {_normalize_target(target) for target in excluded_targets if str(target).strip()}
    used_targets: set[str] = set()
    excluded_matches: set[str] = set()
    records: list[dict[str, Any]] = []
    for slice_name, targets in PROMPT_SLICES.items():
        selected: list[str] = []
        for text in targets:
            normalized = _normalize_target(text)
            if normalized in excluded:
                excluded_matches.add(text)
                continue
            if normalized in used_targets:
                continue
            selected.append(text)
            used_targets.add(normalized)
            if len(selected) == count_per_slice:
                break
        if len(selected) != count_per_slice:
            raise ValueError(
                f"slice {slice_name!r} has only {len(selected)} unique, non-excluded targets; "
                f"need {count_per_slice}"
            )
        for text in selected:
            global_index = len(records)
            scene_tag, template = SCENE_TEMPLATES[global_index % len(SCENE_TEMPLATES)]
            style_tag = STYLE_TAGS[global_index % len(STYLE_TAGS)]
            records.append(
                {
                    "id": f"heldout_{global_index:04d}",
                    "prompt": template.format(text=text),
                    "target_text": text,
                    "slice": slice_name,
                    "scene_tag": scene_tag,
                    "style_tag": style_tag,
                    "contains_hard_glyph": bool(hard_glyphs(text)),
                    "hard_glyphs": hard_glyphs(text),
                    "target_length": target_length(text),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    training_overlap = used_targets & excluded
    summary = {
        "schema_version": "final-benchmark-prompts/v1",
        "output_path": str(output),
        "count_per_slice": count_per_slice,
        "total_prompts": len(records),
        "slice_counts": dict(Counter(record["slice"] for record in records)),
        "hard_glyph_prompt_count": sum(1 for record in records if record["contains_hard_glyph"]),
        "mean_target_length": _mean(record["target_length"] for record in records),
        "unique_target_count": len(used_targets),
        "excluded_target_count": len(excluded),
        "excluded_benchmark_matches": sorted(excluded_matches),
        "training_target_overlap_count": len(training_overlap),
        "training_target_overlap_hashes": sorted(
            normalized_target_sha256(target) for target in training_overlap
        ),
        "target_disjoint": not training_overlap,
        "benchmark_sha256": _sha256_file(output),
    }
    if exclusion_manifest is not None:
        source_path = Path(exclusion_manifest)
        source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
        summary["exclusion_source_manifest"] = {
            "path": str(source_path),
            "schema_version": source_manifest.get("schema_version"),
            "repository": source_manifest.get("repository"),
            "resolved_revision": source_manifest.get("resolved_revision"),
            "source_sha256": source_manifest.get("source_sha256"),
            "output_sha256": source_manifest.get("output_sha256"),
        }
    if exclusion_target_hash_index is not None:
        index_path = Path(exclusion_target_hash_index)
        target_hash_index = load_target_hash_index(index_path)
        indexed_hashes = set(target_hash_index["target_hashes"])
        excluded_hashes = {normalized_target_sha256(target) for target in excluded}
        if indexed_hashes != excluded_hashes:
            raise ValueError(
                "exclusion target hash index does not match the supplied excluded target set"
            )
        benchmark_hashes = {normalized_target_sha256(target) for target in used_targets}
        hash_overlap = benchmark_hashes & indexed_hashes
        if bool(hash_overlap) != bool(training_overlap):
            raise AssertionError("plain-text and hashed target overlap checks disagree")
        summary["training_target_hash_index"] = {
            "path": str(index_path),
            "sha256": _sha256_file(index_path),
            "schema_version": target_hash_index["schema_version"],
            "hash_algorithm": target_hash_index["hash_algorithm"],
            "normalization": target_hash_index["normalization"],
            "unique_normalized_target_count": target_hash_index["unique_normalized_target_count"],
            "source_manifest_path": target_hash_index["source"]["manifest_path"],
            "source_manifest_sha256": target_hash_index["source"]["manifest_sha256"],
            "source_output_sha256": target_hash_index["source"]["output_sha256"],
        }
    if manifest_path:
        _write_json(Path(manifest_path), summary)
    return summary


def load_excluded_targets(path: str | Path) -> set[str]:
    """Load target strings from a JSONL dataset used to enforce split disjointness."""

    targets: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            target = row.get("target_text") if isinstance(row, dict) else None
            if isinstance(target, str) and target.strip():
                targets.add(_normalize_target(target))
    return targets


def build_target_hash_index(
    source_jsonl: str | Path,
    output_path: str | Path,
    *,
    source_manifest: str | Path,
) -> dict[str, Any]:
    """Write an exact, compact hash index for offline target-overlap checks."""

    source_path = Path(source_jsonl)
    source_manifest_path = Path(source_manifest)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be a JSON object")
    source_sha256 = _sha256_file(source_path)
    if manifest.get("output_sha256") != source_sha256:
        raise ValueError("source dataset SHA-256 does not match the pinned source manifest")

    normalized_targets: set[str] = set()
    row_count = 0
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source_path}:{line_number}: malformed JSON") from exc
            target = row.get("target_text") if isinstance(row, dict) else None
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"{source_path}:{line_number}: missing target_text")
            normalized_targets.add(_normalize_target(target))

    if manifest.get("row_count") != row_count:
        raise ValueError("source dataset row count does not match the pinned source manifest")

    payload = {
        "schema_version": TARGET_HASH_INDEX_SCHEMA_VERSION,
        "hash_algorithm": TARGET_HASH_ALGORITHM,
        "normalization": TARGET_NORMALIZATION,
        "source": {
            "dataset_path": str(source_path),
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": _sha256_file(source_manifest_path),
            "repository": manifest.get("repository"),
            "resolved_revision": manifest.get("resolved_revision"),
            "output_sha256": source_sha256,
            "row_count": row_count,
        },
        "unique_normalized_target_count": len(normalized_targets),
        "target_hashes": sorted(normalized_target_sha256(target) for target in normalized_targets),
    }
    _write_json(Path(output_path), payload)
    return payload


def load_target_hash_index(path: str | Path) -> dict[str, Any]:
    """Load and validate a normalized-target hash index."""

    index_path = Path(path)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed target hash index: {index_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("target hash index must be a JSON object")
    if payload.get("schema_version") != TARGET_HASH_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported target hash index schema")
    if payload.get("hash_algorithm") != TARGET_HASH_ALGORITHM:
        raise ValueError("unsupported target hash algorithm")
    if payload.get("normalization") != TARGET_NORMALIZATION:
        raise ValueError("unsupported target normalization contract")
    hashes = payload.get("target_hashes")
    if not isinstance(hashes, list) or not all(_is_sha256(value) for value in hashes):
        raise ValueError("target_hashes must contain SHA-256 hex digests")
    if hashes != sorted(set(hashes)):
        raise ValueError("target_hashes must be sorted and unique")
    if payload.get("unique_normalized_target_count") != len(hashes):
        raise ValueError("target hash count does not match target_hashes")
    source = payload.get("source")
    if not isinstance(source, dict) or not _is_sha256(source.get("output_sha256")):
        raise ValueError("target hash index source provenance is incomplete")
    return payload


def normalized_target_sha256(value: object) -> str:
    """Hash one target using the split-disjointness normalization contract."""

    return hashlib.sha256(_normalize_target(value).encode("utf-8")).hexdigest()


def _normalize_target(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_bias_report(
    *,
    prompts_jsonl: str | Path,
    scores_csv: str | Path,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Analyze which prompt/data strata survive product-score filtering."""
    prompts = load_prompt_metadata(prompts_jsonl)
    rows = load_score_rows(scores_csv)
    annotated = [annotate_score_row(row, prompts) for row in rows]
    selected = [
        row
        for row in annotated
        if _float_or_none(row.get("score")) is not None and float(row["score"]) >= threshold
    ]
    return {
        "schema_version": "product-selection-bias/v1",
        "scores_csv": str(scores_csv),
        "prompts_jsonl": str(prompts_jsonl),
        "threshold": threshold,
        "total_rows": len(annotated),
        "selected_rows": len(selected),
        "selected_fraction": _safe_div(len(selected), len(annotated)),
        "total_prompts": len({row["id"] for row in annotated}),
        "selected_prompts": len({row["id"] for row in selected}),
        "all": distribution_summary(annotated),
        "selected": distribution_summary(selected),
        "bias_delta": bias_delta(distribution_summary(annotated), distribution_summary(selected)),
    }


def dpo_provenance_report(
    *,
    prompts_jsonl: str | Path,
    reward: DpoRewardSpec,
    pair_source_model: str = "base-generated candidates",
    policy_initialization: str = "matching SFT final checkpoint",
    ablation_margins: Sequence[float] = (0.1, 0.2, 0.3),
) -> dict[str, Any]:
    """Summarize current DPO pair construction and prompt provenance."""
    prompts = load_prompt_metadata(prompts_jsonl)
    rows = load_score_rows(reward.scores_csv)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)

    pairs: list[dict[str, Any]] = []
    stats = Counter()
    for prompt_id, prompt_rows in grouped.items():
        if len(prompt_rows) < 2:
            stats["insufficient_versions"] += 1
            continue
        ordered = sorted(prompt_rows, key=lambda row: float(row["score"]))
        loser = ordered[0]
        winner = ordered[-1]
        winner_score = float(winner["score"])
        loser_score = float(loser["score"])
        margin = winner_score - loser_score
        if winner_score < reward.winner_threshold:
            stats["winner_below_threshold"] += 1
            continue
        if margin < reward.margin:
            stats["below_margin"] += 1
            continue
        prompt_meta = prompts.get(prompt_id, {})
        text = str(winner.get("target_text") or prompt_meta.get("target_text") or "")
        pairs.append(
            {
                "prompt_id": prompt_id,
                "target_text": text,
                "winner_version": int(winner["version"]),
                "loser_version": int(loser["version"]),
                "winner_score": winner_score,
                "loser_score": loser_score,
                "score_gap": margin,
                "target_length": target_length(text),
                "hard_glyphs": hard_glyphs(text),
                "contains_hard_glyph": bool(hard_glyphs(text)),
                "tier": prompt_meta.get("tier"),
                "content_type": prompt_meta.get("content_type"),
                "slice": prompt_meta.get("slice"),
                "scene_tag": prompt_meta.get("scene_tag"),
                "style_tag": _style_value(prompt_meta, "font"),
                "pair_source_model": pair_source_model,
                "policy_initialization": policy_initialization,
            }
        )
        stats["selected"] += 1

    return {
        "schema_version": "dpo-pair-provenance/v1",
        "reward_name": reward.name,
        "scores_csv": str(reward.scores_csv),
        "prompts_jsonl": str(prompts_jsonl),
        "winner_threshold": reward.winner_threshold,
        "margin": reward.margin,
        "pair_source_model": pair_source_model,
        "policy_initialization": policy_initialization,
        "input_prompts": len(grouped),
        "pair_count": len(pairs),
        "filtering_stats": dict(stats),
        "margin_ablation_pair_counts": {
            str(margin): count_dpo_pairs(grouped, reward.winner_threshold, margin)
            for margin in ablation_margins
        },
        "pair_summary": pair_distribution_summary(pairs),
        "pairs": pairs,
    }


def benchmark_score_report(
    *,
    prompts_jsonl: str | Path,
    score_specs: Sequence[ScoreSpec],
    output_rows_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize held-out benchmark score CSVs with strict/normalized OCR metrics."""
    prompts = load_prompt_metadata(prompts_jsonl)
    aliases, expected_prompt_ids, expected_targets = _benchmark_prompt_aliases(prompts_jsonl)
    report_runs = []
    augmented_rows: list[dict[str, Any]] = []
    expected_seed_set: set[str] | None = None
    expected_score_contract: dict[str, Any] | None = None
    for spec in score_specs:
        rows = load_score_rows(spec.path)
        score_contract = _load_benchmark_score_contract(spec.path, expected_rows=len(rows))
        coverage = validate_benchmark_score_coverage(
            rows,
            aliases=aliases,
            expected_prompt_ids=expected_prompt_ids,
            expected_targets=expected_targets,
            run_name=spec.name,
        )
        seed_set = set(coverage["groups"])
        if expected_seed_set is None:
            expected_seed_set = seed_set
        elif seed_set != expected_seed_set:
            raise ValueError(
                f"{spec.name}: seed set {sorted(seed_set)} does not match "
                f"{sorted(expected_seed_set)}"
            )
        comparable_contract = {
            "formula": score_contract["formula"],
            "primary_score": score_contract["primary_score"],
        }
        if expected_score_contract is None:
            expected_score_contract = comparable_contract
        elif comparable_contract != expected_score_contract:
            raise ValueError(f"{spec.name}: score formula/primary score differs across runs")
        per_row = [augment_benchmark_row(row, prompts, run_name=spec.name) for row in rows]
        augmented_rows.extend(per_row)
        report_runs.append(
            {
                "run_name": spec.name,
                "scores_csv": str(spec.path),
                "coverage": coverage,
                "score_contract": score_contract,
                "overall": benchmark_summary(per_row),
                "by_slice": {
                    slice_name: benchmark_summary(slice_rows)
                    for slice_name, slice_rows in sorted(group_by(per_row, "slice").items())
                },
            }
        )
    if output_rows_csv:
        write_augmented_rows_csv(output_rows_csv, augmented_rows)
    return {
        "schema_version": "heldout-benchmark-summary/v1",
        "prompts_jsonl": str(prompts_jsonl),
        "runs": report_runs,
    }


def validate_benchmark_score_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    aliases: Mapping[str, str],
    expected_prompt_ids: set[str],
    expected_targets: Mapping[str, str],
    run_name: str,
) -> dict[str, Any]:
    """Require complete, duplicate-free prompt coverage for every recorded seed."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("seed") or "single")].append(row)
    if not groups:
        raise ValueError(f"{run_name}: score file is empty")

    group_summaries: dict[str, Any] = {}
    for seed, seed_rows in sorted(groups.items()):
        seen: set[str] = set()
        duplicates: set[str] = set()
        unknown: set[str] = set()
        for row in seed_rows:
            raw_id = str(row.get("id") or row.get("sample_id") or "")
            canonical_id = aliases.get(raw_id)
            if canonical_id is None:
                unknown.add(raw_id or "<missing>")
                continue
            row_target = normalize_text(str(row.get("target_text") or ""), homoglyphs=False)
            expected_target = normalize_text(expected_targets[canonical_id], homoglyphs=False)
            if row_target != expected_target:
                raise ValueError(
                    f"{run_name} seed={seed}: target_text mismatch for {canonical_id}: "
                    f"{row.get('target_text')!r} != {expected_targets[canonical_id]!r}"
                )
            if canonical_id in seen:
                duplicates.add(canonical_id)
            seen.add(canonical_id)
        missing = expected_prompt_ids - seen
        if duplicates or unknown or missing:
            parts = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if duplicates:
                parts.append(f"duplicates={sorted(duplicates)}")
            if unknown:
                parts.append(f"unknown={sorted(unknown)}")
            raise ValueError(
                f"{run_name} seed={seed}: invalid benchmark coverage: " + "; ".join(parts)
            )
        group_summaries[seed] = {
            "row_count": len(seed_rows),
            "unique_prompt_count": len(seen),
        }
    return {
        "expected_prompt_count": len(expected_prompt_ids),
        "seed_count": len(groups),
        "groups": group_summaries,
    }


def _benchmark_prompt_aliases(
    path: str | Path,
) -> tuple[dict[str, str], set[str], dict[str, str]]:
    aliases: dict[str, str] = {}
    expected: set[str] = set()
    targets: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        index = 0
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            canonical = str(row.get("id") or f"{index:06d}")
            if canonical in expected:
                raise ValueError(f"benchmark prompt file has duplicate id: {canonical}")
            expected.add(canonical)
            target_text = str(row.get("target_text") or "")
            if not target_text:
                raise ValueError(f"benchmark prompt {canonical} has no target_text")
            targets[canonical] = target_text
            aliases[canonical] = canonical
            aliases[f"{index:06d}"] = canonical
            index += 1
    return aliases, expected, targets


def _load_benchmark_score_contract(path: Path, *, expected_rows: int) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".schema.json")
    if not sidecar_path.is_file():
        raise ValueError(f"benchmark score sidecar is missing: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed benchmark score sidecar: {sidecar_path}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError(f"benchmark score sidecar must be an object: {sidecar_path}")
    if sidecar.get("schema_version") != "reward-score-metadata/v1":
        raise ValueError(f"invalid benchmark score sidecar schema: {sidecar_path}")
    execution = sidecar.get("execution")
    if not isinstance(execution, dict) or execution.get("status") != "complete":
        raise ValueError(f"benchmark score sidecar is not complete: {sidecar_path}")
    row_count = execution.get("scored_row_count", execution.get("row_count"))
    if row_count != expected_rows:
        raise ValueError(f"benchmark score sidecar row count mismatch: {sidecar_path}")
    if execution.get("scores_sha256") != _sha256_file(path):
        raise ValueError(f"benchmark score CSV hash mismatch: {path}")
    formula = sidecar.get("formula")
    primary_score = sidecar.get("primary_score")
    if not isinstance(formula, dict) or primary_score not in {"vlm", "ocr", "product"}:
        raise ValueError(f"benchmark score formula/primary score is invalid: {sidecar_path}")
    source_paths = sidecar.get("source_manifest_paths")
    source_hashes = sidecar.get("source_manifest_sha256")
    if (
        not isinstance(source_paths, list)
        or not source_paths
        or not isinstance(source_hashes, dict)
    ):
        raise ValueError(f"benchmark score provenance is incomplete: {sidecar_path}")
    for raw_source_path in source_paths:
        source_path = Path(str(raw_source_path))
        if not source_path.is_file():
            raise ValueError(f"benchmark source manifest is missing: {source_path}")
        if source_hashes.get(str(raw_source_path)) != _sha256_file(source_path):
            raise ValueError(f"benchmark source manifest hash mismatch: {source_path}")
    return {
        "score_file_schema_version": sidecar.get("score_file_schema_version"),
        "formula": formula,
        "primary_score": primary_score,
        "source_manifest_paths": source_paths,
        "source_manifest_sha256": source_hashes,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256_file(sidecar_path),
    }


def augment_benchmark_row(
    row: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Any]],
    *,
    run_name: str,
) -> dict[str, Any]:
    prompt_id = str(row.get("id") or row.get("sample_id") or "")
    prompt_meta = prompts.get(prompt_id, {})
    target = str(row.get("target_text") or prompt_meta.get("target_text") or "")
    detected = str(row.get("ocr_detected") or "")
    strict_cer = character_error_rate(detected, target, homoglyphs=False) if detected else 1.0
    normalized_cer = character_error_rate(detected, target, homoglyphs=True) if detected else 1.0
    strict_exact = normalize_text(detected, homoglyphs=False) == normalize_text(
        target, homoglyphs=False
    )
    normalized_exact = normalize_text(detected, homoglyphs=True) == normalize_text(
        target, homoglyphs=True
    )
    script_stats = script_mixing_stats(detected)
    score_vlm = _float_or_none(row.get("score_vlm"))
    score_ocr = _float_or_none(row.get("score_ocr"))
    simple_product = (
        score_vlm * score_ocr if score_vlm is not None and score_ocr is not None else None
    )
    return {
        "run_name": run_name,
        "id": prompt_id,
        "version": int(row.get("version") or 0),
        "target_text": target,
        "slice": prompt_meta.get("slice", "unknown"),
        "scene_tag": prompt_meta.get("scene_tag", prompt_meta.get("content_type", "unknown")),
        "style_tag": prompt_meta.get("style_tag", _style_value(prompt_meta, "font") or "unknown"),
        "target_length": target_length(target),
        "hard_glyphs": ",".join(hard_glyphs(target)),
        "contains_hard_glyph": bool(hard_glyphs(target)),
        "ocr_detected": detected,
        "ocr_detected_normalized": normalize_homoglyphs(detected),
        "detected": bool(detected.strip()),
        "strict_cer": strict_cer,
        "normalized_cer": normalized_cer,
        "strict_exact": strict_exact,
        "normalized_exact": normalized_exact,
        "score_vlm": score_vlm,
        "score_ocr": score_ocr,
        "recorded_score": _float_or_none(row.get("score")),
        "recorded_product_score": _float_or_none(row.get("product_score")),
        "simple_product_score": simple_product,
        **script_stats,
    }


def benchmark_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "detection_rate": _mean_bool(row.get("detected") for row in rows),
        "strict_exact_rate": _mean_bool(row.get("strict_exact") for row in rows),
        "normalized_exact_rate": _mean_bool(row.get("normalized_exact") for row in rows),
        "strict_cer_mean": _mean_numeric(row.get("strict_cer") for row in rows),
        "strict_cer_median": _median_numeric(row.get("strict_cer") for row in rows),
        "normalized_cer_mean": _mean_numeric(row.get("normalized_cer") for row in rows),
        "normalized_cer_median": _median_numeric(row.get("normalized_cer") for row in rows),
        "script_mixing_rate": _mean_bool(row.get("script_mixing_flag") for row in rows),
        "latin_letter_rate_mean": _mean_numeric(row.get("latin_letter_rate") for row in rows),
        "score_vlm_mean": _mean_numeric(row.get("score_vlm") for row in rows),
        "score_ocr_mean": _mean_numeric(row.get("score_ocr") for row in rows),
        "simple_product_score_mean": _mean_numeric(row.get("simple_product_score") for row in rows),
        "recorded_product_score_mean": _mean_numeric(
            row.get("recorded_product_score") for row in rows
        ),
    }


def load_prompt_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load prompt JSONL and key records by generated prompt id plus original id."""
    prompts: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            generated_id = f"{index:06d}"
            prompts[generated_id] = record
            original_id = str(record.get("id") or "")
            if original_id:
                prompts[original_id] = record
    return prompts


def load_score_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def annotate_score_row(
    row: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prompt_id = str(row.get("id") or row.get("sample_id") or "")
    prompt_meta = prompts.get(prompt_id, {})
    target = str(row.get("target_text") or prompt_meta.get("target_text") or "")
    annotated = dict(row)
    annotated.update(
        {
            "target_text": target,
            "target_length": target_length(target),
            "contains_hard_glyph": bool(hard_glyphs(target)),
            "hard_glyphs": hard_glyphs(target),
            "tier": prompt_meta.get("tier"),
            "content_type": prompt_meta.get("content_type"),
            "slice": prompt_meta.get("slice"),
            "font": _style_value(prompt_meta, "font"),
            "effect": _style_value(prompt_meta, "effect"),
            "size": _style_value(prompt_meta, "size"),
            "color": _style_value(prompt_meta, "color"),
        }
    )
    return annotated


def distribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "mean_score": _mean_numeric(row.get("score") for row in rows),
        "median_score": _median_numeric(row.get("score") for row in rows),
        "mean_target_length": _mean_numeric(row.get("target_length") for row in rows),
        "median_target_length": _median_numeric(row.get("target_length") for row in rows),
        "hard_glyph_share": _mean_bool(row.get("contains_hard_glyph") for row in rows),
        "tier_distribution": _counter_fraction(rows, "tier"),
        "content_type_distribution": _counter_fraction(rows, "content_type"),
        "font_distribution": _counter_fraction(rows, "font"),
        "effect_distribution": _counter_fraction(rows, "effect"),
        "size_distribution": _counter_fraction(rows, "size"),
    }


def pair_distribution_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(pairs),
        "winner_score_mean": _mean_numeric(pair.get("winner_score") for pair in pairs),
        "winner_score_median": _median_numeric(pair.get("winner_score") for pair in pairs),
        "loser_score_mean": _mean_numeric(pair.get("loser_score") for pair in pairs),
        "loser_score_median": _median_numeric(pair.get("loser_score") for pair in pairs),
        "score_gap_mean": _mean_numeric(pair.get("score_gap") for pair in pairs),
        "score_gap_median": _median_numeric(pair.get("score_gap") for pair in pairs),
        "target_length_mean": _mean_numeric(pair.get("target_length") for pair in pairs),
        "target_length_median": _median_numeric(pair.get("target_length") for pair in pairs),
        "hard_glyph_share": _mean_bool(pair.get("contains_hard_glyph") for pair in pairs),
        "tier_distribution": _counter_fraction(pairs, "tier"),
        "content_type_distribution": _counter_fraction(pairs, "content_type"),
        "slice_distribution": _counter_fraction(pairs, "slice"),
    }


def bias_delta(
    all_summary: Mapping[str, Any],
    selected_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mean_target_length_delta": _none_safe_sub(
            selected_summary.get("mean_target_length"), all_summary.get("mean_target_length")
        ),
        "median_target_length_delta": _none_safe_sub(
            selected_summary.get("median_target_length"), all_summary.get("median_target_length")
        ),
        "hard_glyph_share_delta": _none_safe_sub(
            selected_summary.get("hard_glyph_share"), all_summary.get("hard_glyph_share")
        ),
        "tier_distribution_delta": _distribution_delta(
            all_summary.get("tier_distribution", {}), selected_summary.get("tier_distribution", {})
        ),
        "content_type_distribution_delta": _distribution_delta(
            all_summary.get("content_type_distribution", {}),
            selected_summary.get("content_type_distribution", {}),
        ),
    }


def count_dpo_pairs(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: float,
    margin: float,
) -> int:
    count = 0
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        scores = [float(row["score"]) for row in rows]
        if max(scores) >= threshold and max(scores) - min(scores) >= margin:
            count += 1
    return count


def group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return grouped


def write_markdown_report(path: str | Path, title: str, payload: Mapping[str, Any]) -> None:
    """Write a compact Markdown report for thesis review."""
    lines = [
        f"# {title}",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_augmented_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pairs_jsonl(path: str | Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")


def _style_value(record: Mapping[str, Any], key: str) -> Any:
    style = record.get("style")
    if isinstance(style, Mapping):
        return style.get(key)
    return record.get(f"style_{key}")


def _counter_fraction(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, float | int]]:
    counter = Counter(str(row.get(key) if row.get(key) is not None else "unknown") for row in rows)
    total = len(rows)
    return {
        value: {"count": count, "fraction": _safe_div(count, total)}
        for value, count in sorted(counter.items())
    }


def _distribution_delta(
    all_dist: Mapping[str, Mapping[str, Any]],
    selected_dist: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    keys = set(all_dist) | set(selected_dist)
    return {
        key: float(selected_dist.get(key, {}).get("fraction", 0.0))
        - float(all_dist.get(key, {}).get("fraction", 0.0))
        for key in sorted(keys)
    }


def _none_safe_sub(left: Any, right: Any) -> float | None:
    left_num = _float_or_none(left)
    right_num = _float_or_none(right)
    if left_num is None or right_num is None:
        return None
    return left_num - right_num


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_values(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _float_or_none(value)) is not None]


def _mean(values: Iterable[Any]) -> float | None:
    return _mean_numeric(values)


def _mean_numeric(values: Iterable[Any]) -> float | None:
    nums = _numeric_values(values)
    return statistics.mean(nums) if nums else None


def _median_numeric(values: Iterable[Any]) -> float | None:
    nums = _numeric_values(values)
    return statistics.median(nums) if nums else None


def _mean_bool(values: Iterable[Any]) -> float | None:
    bools = [bool(value) for value in values if value is not None]
    return statistics.mean(1.0 if value else 0.0 for value in bools) if bools else None


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_score_spec(raw: str) -> ScoreSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("score spec must be NAME=PATH")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("score spec must be NAME=PATH")
    return ScoreSpec(name=name.strip(), path=Path(path.strip()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final review benchmark/report tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompts = sub.add_parser("make-prompts", help="Create deterministic held-out prompts")
    p_prompts.add_argument("--output", required=True)
    p_prompts.add_argument("--count-per-slice", type=int, default=20)
    p_prompts.add_argument("--manifest", default=None)
    p_prompts.add_argument(
        "--exclude-targets",
        default=None,
        help=(
            "Optional training/source JSONL; exact target strings are excluded from the benchmark."
        ),
    )
    p_prompts.add_argument(
        "--exclusion-manifest",
        default=None,
        help="Optional download/source manifest recorded in the benchmark manifest.",
    )
    p_prompts.add_argument(
        "--exclusion-target-hash-index",
        default=None,
        help="Optional exact target-hash index recorded in the benchmark manifest.",
    )

    p_hashes = sub.add_parser(
        "make-target-hash-index",
        help="Create a compact normalized-target hash index from a pinned JSONL dataset",
    )
    p_hashes.add_argument("--source", required=True)
    p_hashes.add_argument("--source-manifest", required=True)
    p_hashes.add_argument("--output", required=True)

    p_bias = sub.add_parser("product-bias", help="Analyze product-selected subset bias")
    p_bias.add_argument("--prompts", required=True)
    p_bias.add_argument("--scores", required=True)
    p_bias.add_argument("--threshold", type=float, default=0.3)
    p_bias.add_argument("--output-json", required=True)
    p_bias.add_argument("--markdown", default=None)

    p_dpo = sub.add_parser("dpo-provenance", help="Analyze DPO pair provenance")
    p_dpo.add_argument("--prompts", required=True)
    p_dpo.add_argument("--scores", required=True)
    p_dpo.add_argument("--reward-name", required=True)
    p_dpo.add_argument("--winner-threshold", type=float, required=True)
    p_dpo.add_argument("--margin", type=float, default=0.1)
    p_dpo.add_argument("--pair-source-model", default="base-generated candidates")
    p_dpo.add_argument("--policy-initialization", default="matching SFT final checkpoint")
    p_dpo.add_argument("--output-json", required=True)
    p_dpo.add_argument("--pairs-jsonl", default=None)
    p_dpo.add_argument("--markdown", default=None)

    p_bench = sub.add_parser("benchmark-report", help="Summarize held-out benchmark scores")
    p_bench.add_argument("--prompts", required=True)
    p_bench.add_argument("--score", action="append", type=_parse_score_spec, required=True)
    p_bench.add_argument("--output-json", required=True)
    p_bench.add_argument("--rows-csv", default=None)
    p_bench.add_argument("--markdown", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-prompts":
        summary = make_benchmark_prompts(
            args.output,
            count_per_slice=args.count_per_slice,
            manifest_path=args.manifest,
            excluded_targets=(
                load_excluded_targets(args.exclude_targets) if args.exclude_targets else ()
            ),
            exclusion_manifest=args.exclusion_manifest,
            exclusion_target_hash_index=args.exclusion_target_hash_index,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "make-target-hash-index":
        payload = build_target_hash_index(
            args.source,
            args.output,
            source_manifest=args.source_manifest,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "product-bias":
        report = product_bias_report(
            prompts_jsonl=args.prompts,
            scores_csv=args.scores,
            threshold=args.threshold,
        )
        _write_json(Path(args.output_json), report)
        if args.markdown:
            write_markdown_report(args.markdown, "Product selection bias", report)
        return 0
    if args.command == "dpo-provenance":
        report = dpo_provenance_report(
            prompts_jsonl=args.prompts,
            reward=DpoRewardSpec(
                name=args.reward_name,
                scores_csv=Path(args.scores),
                winner_threshold=args.winner_threshold,
                margin=args.margin,
            ),
            pair_source_model=args.pair_source_model,
            policy_initialization=args.policy_initialization,
        )
        pairs = report.get("pairs", [])
        if args.pairs_jsonl:
            write_pairs_jsonl(args.pairs_jsonl, pairs)  # type: ignore[arg-type]
        report_without_pairs = {key: value for key, value in report.items() if key != "pairs"}
        _write_json(Path(args.output_json), report_without_pairs)
        if args.markdown:
            write_markdown_report(
                args.markdown,
                f"DPO provenance: {args.reward_name}",
                report_without_pairs,
            )
        return 0
    if args.command == "benchmark-report":
        report = benchmark_score_report(
            prompts_jsonl=args.prompts,
            score_specs=args.score,
            output_rows_csv=args.rows_csv,
        )
        _write_json(Path(args.output_json), report)
        if args.markdown:
            write_markdown_report(args.markdown, "Held-out benchmark summary", report)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
