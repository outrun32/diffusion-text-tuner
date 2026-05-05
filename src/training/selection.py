"""Materialize SFT sample and DPO pair selections from score CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SELECTED_SAMPLES_SCHEMA_VERSION = "selected-samples/v1"
PREFERENCE_PAIRS_SCHEMA_VERSION = "preference-pairs/v1"

SFT_SELECTION_MODES = {"hard_positive", "score_weighted", "threshold", "top_k_per_prompt"}
DPO_PAIR_MODES = {
    "all_separated_pairs",
    "ambiguity_filtered",
    "best_vs_worst",
    "margin_weighted",
}


def materialize_sft_samples(
    scores_csv: str | Path,
    output_path: str | Path,
    mode: str = "threshold",
    score_column: str = "score",
    threshold: float = 0.3,
    top_k_per_prompt: int = 1,
    hard_negative_threshold: float = 0.2,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write selected SFT samples as deterministic JSONL rows.

    The default ``threshold`` mode preserves the existing ``SFTDataset`` constructor
    behavior: every row with ``score >= score_threshold`` is selected.
    """

    if mode not in SFT_SELECTION_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(SFT_SELECTION_MODES))}")
    if top_k_per_prompt < 1:
        raise ValueError("top_k_per_prompt must be >= 1")

    source_path = Path(scores_csv)
    rows = _read_score_rows(source_path, score_column=score_column)
    source_hash = _sha256_file(source_path)
    selected_rows = sorted(
        _select_sft_rows(
            rows,
            mode=mode,
            threshold=threshold,
            score_column=score_column,
            top_k_per_prompt=top_k_per_prompt,
            hard_negative_threshold=hard_negative_threshold,
        ),
        key=lambda row: (row.prompt_id, row.version),
    )
    sample_weights = _sft_sample_weights(selected_rows) if mode == "score_weighted" else {}

    output_rows = [
        _sft_output_row(
            row,
            mode=mode,
            score_column=score_column,
            source_path=source_path,
            source_hash=source_hash,
            manifest_path=manifest_path,
            sample_weight=sample_weights.get((row.prompt_id, row.version)),
        )
        for row in selected_rows
    ]
    _write_jsonl(output_path, output_rows)

    filtering_stats = _sft_filtering_stats(
        rows,
        selected_rows,
        threshold=threshold,
        score_column=score_column,
        mode=mode,
        hard_negative_threshold=hard_negative_threshold,
    )
    summary = {
        "schema_version": SELECTED_SAMPLES_SCHEMA_VERSION,
        "selection_mode": mode,
        "score_column": score_column,
        "threshold": threshold,
        "top_k_per_prompt": top_k_per_prompt if mode == "top_k_per_prompt" else None,
        "hard_negative_threshold": hard_negative_threshold if mode == "hard_positive" else None,
        "source_scores_path": str(source_path),
        "source_scores_sha256": source_hash,
        "output_path": str(Path(output_path)),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "input_rows": len(rows),
        "selected_count": len(selected_rows),
        "filtered_count": len(rows) - len(selected_rows),
        "filtering_stats": filtering_stats,
    }
    if manifest_path is not None:
        _write_json(manifest_path, summary)
    return summary


def materialize_dpo_pairs(
    scores_csv: str | Path,
    output_path: str | Path,
    mode: str = "best_vs_worst",
    score_column: str = "score",
    threshold: float = 0.5,
    margin: float = 0.1,
    ambiguity_margin: float = 0.0,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write DPO preference pairs as deterministic JSONL rows.

    The default ``best_vs_worst`` mode preserves the existing ``DPODataset``
    constructor behavior while making winner/loser semantics explicit.
    """

    if mode not in DPO_PAIR_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(DPO_PAIR_MODES))}")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be >= 0")

    source_path = Path(scores_csv)
    rows = _read_score_rows(source_path, score_column=score_column)
    source_hash = _sha256_file(source_path)
    pairs, filtering_stats, prompt_count = _select_dpo_pairs(
        rows,
        mode=mode,
        threshold=threshold,
        margin=margin,
        ambiguity_margin=ambiguity_margin,
    )
    pairs = sorted(
        pairs,
        key=lambda pair: (pair.prompt_id, pair.winner.version, pair.loser.version),
    )
    pair_weights = _dpo_pair_weights(pairs) if mode == "margin_weighted" else {}
    output_rows = [
        _dpo_output_row(
            pair,
            mode=mode,
            score_column=score_column,
            source_path=source_path,
            source_hash=source_hash,
            manifest_path=manifest_path,
            pair_weight=pair_weights.get((pair.prompt_id, pair.winner.version, pair.loser.version)),
        )
        for pair in pairs
    ]
    _write_jsonl(output_path, output_rows)

    summary = {
        "schema_version": PREFERENCE_PAIRS_SCHEMA_VERSION,
        "pair_construction_mode": mode,
        "score_column": score_column,
        "threshold": threshold,
        "margin": margin,
        "ambiguity_margin": ambiguity_margin,
        "source_scores_path": str(source_path),
        "source_scores_sha256": source_hash,
        "output_path": str(Path(output_path)),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "input_rows": len(rows),
        "prompt_count": prompt_count,
        "pair_count": len(pairs),
        "filtering_stats": filtering_stats,
    }
    if manifest_path is not None:
        _write_json(manifest_path, summary)
    return summary


@dataclass(frozen=True)
class _ScoreRow:
    prompt_id: str
    version: int
    score: float
    target_text: str


@dataclass(frozen=True)
class _DpoPair:
    prompt_id: str
    winner: _ScoreRow
    loser: _ScoreRow

    @property
    def margin(self) -> float:
        return round(self.winner.score - self.loser.score, 12)


def _read_score_rows(path: Path, *, score_column: str) -> list[_ScoreRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        for column in ["id", "version", "target_text", score_column]:
            if column not in fieldnames:
                raise ValueError(f"{path}: missing required column: {column}")

        rows: list[_ScoreRow] = []
        for line_number, raw_row in enumerate(reader, start=2):
            prompt_id = str(raw_row.get("id") or "")
            target_text = str(raw_row.get("target_text") or "")
            if not prompt_id:
                raise ValueError(f"{path}: line {line_number}: missing prompt id")
            try:
                version = int(str(raw_row.get("version") or ""))
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid integer version") from exc
            try:
                score = float(str(raw_row.get(score_column) or ""))
            except ValueError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid numeric score in {score_column}"
                ) from exc
            rows.append(
                _ScoreRow(
                    prompt_id=prompt_id,
                    version=version,
                    score=score,
                    target_text=target_text,
                )
            )
    return rows


def _select_sft_rows(
    rows: Iterable[_ScoreRow],
    *,
    mode: str,
    threshold: float,
    score_column: str,
    top_k_per_prompt: int,
    hard_negative_threshold: float,
) -> list[_ScoreRow]:
    if mode in {"threshold", "score_weighted"}:
        return _select_sft_threshold(rows, threshold=threshold, score_column=score_column)
    if mode == "top_k_per_prompt":
        return _select_sft_top_k(
            rows,
            threshold=threshold,
            score_column=score_column,
            top_k_per_prompt=top_k_per_prompt,
        )
    return _select_sft_hard_positive(
        rows,
        threshold=threshold,
        hard_negative_threshold=hard_negative_threshold,
    )


def _select_sft_threshold(
    rows: Iterable[_ScoreRow],
    *,
    threshold: float,
    score_column: str,
) -> list[_ScoreRow]:
    del score_column
    return [row for row in rows if row.score >= threshold]


def _select_sft_top_k(
    rows: Iterable[_ScoreRow],
    *,
    threshold: float,
    score_column: str,
    top_k_per_prompt: int,
) -> list[_ScoreRow]:
    del score_column
    grouped = _group_rows_by_prompt(rows)
    selected: list[_ScoreRow] = []
    for prompt_rows in grouped.values():
        eligible = [row for row in prompt_rows if row.score >= threshold]
        eligible.sort(key=lambda row: (-row.score, row.version))
        selected.extend(eligible[:top_k_per_prompt])
    return selected


def _select_sft_hard_positive(
    rows: Iterable[_ScoreRow],
    *,
    threshold: float,
    hard_negative_threshold: float,
) -> list[_ScoreRow]:
    grouped = _group_rows_by_prompt(rows)
    selected: list[_ScoreRow] = []
    for prompt_id in sorted(grouped):
        prompt_rows = grouped[prompt_id]
        has_hard_negative = any(row.score < hard_negative_threshold for row in prompt_rows)
        if has_hard_negative:
            selected.extend(row for row in prompt_rows if row.score >= threshold)
    return selected


def _sft_sample_weights(rows: list[_ScoreRow]) -> dict[tuple[str, int], float]:
    if not rows:
        return {}
    max_score = max(row.score for row in rows)
    if max_score <= 0:
        return {(row.prompt_id, row.version): 0.0 for row in rows}
    return {
        (row.prompt_id, row.version): round(row.score / max_score, 12)
        for row in rows
    }


def _sft_output_row(
    row: _ScoreRow,
    *,
    mode: str,
    score_column: str,
    source_path: Path,
    source_hash: str,
    manifest_path: str | Path | None,
    sample_weight: float | None = None,
) -> dict[str, Any]:
    output = {
        "schema_version": SELECTED_SAMPLES_SCHEMA_VERSION,
        "sample_id": f"sft:{row.prompt_id}:v{row.version}:{score_column}",
        "prompt_id": row.prompt_id,
        "version": row.version,
        "target_text": row.target_text,
        "selected_score": row.score,
        "score_column": score_column,
        "selection_mode": mode,
        "source_scores_path": str(source_path),
        "source_scores_sha256": source_hash,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }
    if sample_weight is not None:
        output["sample_weight"] = sample_weight
    return output


def _sft_filtering_stats(
    rows: list[_ScoreRow],
    selected_rows: list[_ScoreRow],
    *,
    threshold: float,
    score_column: str,
    mode: str,
    hard_negative_threshold: float,
) -> dict[str, int]:
    del score_column
    selected_keys = {(row.prompt_id, row.version) for row in selected_rows}
    below_threshold = sum(1 for row in rows if row.score < threshold)
    if mode in {"threshold", "score_weighted"}:
        return {"below_threshold": below_threshold, "selected": len(selected_rows)}
    if mode == "top_k_per_prompt":
        return {
            "below_threshold": below_threshold,
            "selected": len(selected_rows),
            "unselected_by_top_k": sum(
                1
                for row in rows
                if row.score >= threshold and (row.prompt_id, row.version) not in selected_keys
            ),
        }

    grouped = _group_rows_by_prompt(rows)
    prompts_without_hard_negative = sum(
        1
        for prompt_rows in grouped.values()
        if any(row.score >= threshold for row in prompt_rows)
        and not any(row.score < hard_negative_threshold for row in prompt_rows)
    )
    return {
        "below_threshold": below_threshold,
        "prompts_without_hard_negative": prompts_without_hard_negative,
        "selected": len(selected_rows),
    }


def _select_dpo_pairs(
    rows: Iterable[_ScoreRow],
    *,
    mode: str,
    threshold: float,
    margin: float,
    ambiguity_margin: float,
) -> tuple[list[_DpoPair], dict[str, int], int]:
    if mode == "all_separated_pairs":
        return _select_dpo_all_separated_pairs(rows, threshold=threshold, margin=margin)
    return _select_dpo_best_vs_worst(
        rows,
        threshold=threshold,
        margin=margin,
        ambiguity_margin=ambiguity_margin,
        reject_best_second_ambiguity=mode == "ambiguity_filtered",
    )


def _select_dpo_best_vs_worst(
    rows: Iterable[_ScoreRow],
    *,
    threshold: float,
    margin: float,
    ambiguity_margin: float,
    reject_best_second_ambiguity: bool,
) -> tuple[list[_DpoPair], dict[str, int], int]:
    grouped = _group_rows_by_prompt(rows)

    pairs: list[_DpoPair] = []
    stats = {
        "ambiguous_below_margin": 0,
        "insufficient_versions": 0,
        "selected": 0,
        "winner_below_threshold": 0,
    }
    if reject_best_second_ambiguity:
        stats["ambiguous_best_second_margin"] = 0
    for prompt_id in sorted(grouped):
        prompt_rows = grouped[prompt_id]
        if len(prompt_rows) < 2:
            stats["insufficient_versions"] += 1
            continue
        ordered = sorted(prompt_rows, key=lambda row: (row.score, row.version))
        loser = ordered[0]
        winner = ordered[-1]
        if winner.score < threshold:
            stats["winner_below_threshold"] += 1
            continue
        if reject_best_second_ambiguity:
            second_best = ordered[-2]
            best_second_margin = winner.score - second_best.score
            if best_second_margin <= ambiguity_margin:
                stats["ambiguous_best_second_margin"] += 1
                continue
        pair_margin = winner.score - loser.score
        if pair_margin <= ambiguity_margin or pair_margin < margin:
            stats["ambiguous_below_margin"] += 1
            continue
        pairs.append(_DpoPair(prompt_id=prompt_id, winner=winner, loser=loser))
        stats["selected"] += 1
    return pairs, stats, len(grouped)


def _select_dpo_all_separated_pairs(
    rows: Iterable[_ScoreRow],
    *,
    threshold: float,
    margin: float,
) -> tuple[list[_DpoPair], dict[str, int], int]:
    grouped = _group_rows_by_prompt(rows)

    pairs: list[_DpoPair] = []
    stats = {
        "equal_score_pairs_rejected": 0,
        "insufficient_versions": 0,
        "pairs_below_margin": 0,
        "selected": 0,
        "winners_below_threshold": 0,
    }
    for prompt_id in sorted(grouped):
        prompt_rows = sorted(grouped[prompt_id], key=lambda row: row.version)
        if len(prompt_rows) < 2:
            stats["insufficient_versions"] += 1
            continue
        for left_index, left in enumerate(prompt_rows):
            for right in prompt_rows[left_index + 1 :]:
                if left.score == right.score:
                    stats["equal_score_pairs_rejected"] += 1
                    continue
                winner, loser = (left, right) if left.score > right.score else (right, left)
                if winner.score < threshold:
                    stats["winners_below_threshold"] += 1
                    continue
                if winner.score - loser.score < margin:
                    stats["pairs_below_margin"] += 1
                    continue
                pairs.append(_DpoPair(prompt_id=prompt_id, winner=winner, loser=loser))
                stats["selected"] += 1
    return pairs, stats, len(grouped)


def _dpo_pair_weights(pairs: list[_DpoPair]) -> dict[tuple[str, int, int], float]:
    if not pairs:
        return {}
    max_margin = max(pair.margin for pair in pairs)
    if max_margin <= 0:
        return {
            (pair.prompt_id, pair.winner.version, pair.loser.version): 0.0
            for pair in pairs
        }
    return {
        (pair.prompt_id, pair.winner.version, pair.loser.version): round(
            pair.margin / max_margin,
            12,
        )
        for pair in pairs
    }


def _dpo_output_row(
    pair: _DpoPair,
    *,
    mode: str,
    score_column: str,
    source_path: Path,
    source_hash: str,
    manifest_path: str | Path | None,
    pair_weight: float | None = None,
) -> dict[str, Any]:
    output = {
        "schema_version": PREFERENCE_PAIRS_SCHEMA_VERSION,
        "pair_id": (
            f"dpo:{pair.prompt_id}:w{pair.winner.version}:l{pair.loser.version}:{score_column}"
        ),
        "prompt_id": pair.prompt_id,
        "target_text": pair.winner.target_text,
        "winner_version": pair.winner.version,
        "loser_version": pair.loser.version,
        "winner_score": pair.winner.score,
        "loser_score": pair.loser.score,
        "margin": pair.margin,
        "score_column": score_column,
        "pair_construction_mode": mode,
        "source_scores_path": str(source_path),
        "source_scores_sha256": source_hash,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }
    if pair_weight is not None:
        output["pair_weight"] = pair_weight
    return output


def _group_rows_by_prompt(rows: Iterable[_ScoreRow]) -> dict[str, list[_ScoreRow]]:
    grouped: dict[str, list[_ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt_id].append(row)
    return grouped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
