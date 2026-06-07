"""Final-review benchmark and artifact-report helpers.

This module is intentionally CPU-safe: it builds prompt JSONL files and analyzes
recorded CSV artifacts without importing CUDA, FLUX, OCR, or VLM stacks.
"""

from __future__ import annotations

import argparse
import csv
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
        ("A realistic street sign on a city wall with the exact Russian text '{text}', "
         "readable letters."),
    ),
    (
        "neon",
        ("A dark cafe wall with a neon sign showing the exact Russian text '{text}', "
         "glowing letters."),
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
) -> dict[str, Any]:
    """Write a deterministic held-out prompt JSONL benchmark."""
    if count_per_slice < 1:
        raise ValueError("count_per_slice must be >= 1")
    output = Path(output_path)
    records: list[dict[str, Any]] = []
    for slice_name, targets in PROMPT_SLICES.items():
        selected = targets[:count_per_slice]
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

    summary = {
        "schema_version": "final-benchmark-prompts/v1",
        "output_path": str(output),
        "count_per_slice": count_per_slice,
        "total_prompts": len(records),
        "slice_counts": dict(Counter(record["slice"] for record in records)),
        "hard_glyph_prompt_count": sum(1 for record in records if record["contains_hard_glyph"]),
        "mean_target_length": _mean(record["target_length"] for record in records),
    }
    if manifest_path:
        _write_json(Path(manifest_path), summary)
    return summary


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
    report_runs = []
    augmented_rows: list[dict[str, Any]] = []
    for spec in score_specs:
        rows = load_score_rows(spec.path)
        per_row = [augment_benchmark_row(row, prompts, run_name=spec.name) for row in rows]
        augmented_rows.extend(per_row)
        report_runs.append(
            {
                "run_name": spec.name,
                "scores_csv": str(spec.path),
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
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
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
