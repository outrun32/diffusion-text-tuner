"""Materialize SFT sample and DPO pair selections from score CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SELECTED_SAMPLES_SCHEMA_VERSION = "selected-samples/v1"


def materialize_sft_samples(
    scores_csv: str | Path,
    output_path: str | Path,
    mode: str = "threshold",
    score_column: str = "score",
    threshold: float = 0.3,
    top_k_per_prompt: int = 1,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write selected SFT samples as deterministic JSONL rows.

    The default ``threshold`` mode preserves the existing ``SFTDataset`` constructor
    behavior: every row with ``score >= score_threshold`` is selected.
    """

    if mode not in {"threshold", "top_k_per_prompt"}:
        raise ValueError("mode must be one of: threshold, top_k_per_prompt")
    if top_k_per_prompt < 1:
        raise ValueError("top_k_per_prompt must be >= 1")

    source_path = Path(scores_csv)
    rows = _read_score_rows(source_path, score_column=score_column)
    source_hash = _sha256_file(source_path)
    selected_rows = (
        _select_sft_threshold(rows, threshold=threshold, score_column=score_column)
        if mode == "threshold"
        else _select_sft_top_k(
            rows,
            threshold=threshold,
            score_column=score_column,
            top_k_per_prompt=top_k_per_prompt,
        )
    )
    selected_rows = sorted(selected_rows, key=lambda row: (row.prompt_id, row.version))

    output_rows = [
        _sft_output_row(
            row,
            mode=mode,
            score_column=score_column,
            source_path=source_path,
            source_hash=source_hash,
            manifest_path=manifest_path,
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
    )
    summary = {
        "schema_version": SELECTED_SAMPLES_SCHEMA_VERSION,
        "selection_mode": mode,
        "score_column": score_column,
        "threshold": threshold,
        "top_k_per_prompt": top_k_per_prompt if mode == "top_k_per_prompt" else None,
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


class _ScoreRow:
    def __init__(self, *, prompt_id: str, version: int, score: float, target_text: str) -> None:
        self.prompt_id = prompt_id
        self.version = version
        self.score = score
        self.target_text = target_text


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
    grouped: dict[str, list[_ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt_id].append(row)
    selected: list[_ScoreRow] = []
    for prompt_rows in grouped.values():
        eligible = [row for row in prompt_rows if row.score >= threshold]
        eligible.sort(key=lambda row: (-row.score, row.version))
        selected.extend(eligible[:top_k_per_prompt])
    return selected


def _sft_output_row(
    row: _ScoreRow,
    *,
    mode: str,
    score_column: str,
    source_path: Path,
    source_hash: str,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    return {
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


def _sft_filtering_stats(
    rows: list[_ScoreRow],
    selected_rows: list[_ScoreRow],
    *,
    threshold: float,
    score_column: str,
    mode: str,
) -> dict[str, int]:
    del score_column
    selected_keys = {(row.prompt_id, row.version) for row in selected_rows}
    below_threshold = sum(1 for row in rows if row.score < threshold)
    if mode == "threshold":
        return {"below_threshold": below_threshold, "selected": len(selected_rows)}
    above_threshold = sum(1 for row in rows if row.score >= threshold)
    return {
        "below_threshold": below_threshold,
        "selected": len(selected_rows),
        "unselected_by_top_k": sum(
            1
            for row in rows
            if row.score >= threshold and (row.prompt_id, row.version) not in selected_keys
        ),
    } | ({"_above_threshold": above_threshold} if False else {})


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
