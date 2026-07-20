"""Tests for complete, provenance-safe score shard merging."""

from __future__ import annotations

import csv
import hashlib

import pytest

from src.evaluation.reward_interface import vlm_ocr_product_formula
from src.scoring.pipeline import CANONICAL_SCORE_COLUMNS, write_score_schema_sidecar


def _write_shard(
    root,
    index,
    num_shards,
    sample_id,
    *,
    discovered_task_count=None,
    expected_shard_count=1,
):
    path = root / f"scores_shard{index:03d}.csv"
    row = {field: "" for field in CANONICAL_SCORE_COLUMNS}
    row.update(
        {
            "id": sample_id,
            "sample_id": sample_id,
            "version": "0",
            "score": "0.5",
            "product_score": "0.5",
            "target_text": "ТЕСТ",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    write_score_schema_sidecar(
        path,
        formula=vlm_ocr_product_formula(),
        source_manifest_paths=("runs/generation/manifest.json",),
        primary_score="product",
        execution_metadata={
            "status": "complete",
            "shard_idx": index,
            "num_shards": num_shards,
            "discovered_task_count": (
                num_shards if discovered_task_count is None else discovered_task_count
            ),
            "expected_shard_count": expected_shard_count,
            "scored_row_count": 1,
            "scores_sha256": digest,
        },
    )
    return path


def test_merge_requires_all_shards_and_writes_aggregate_sidecar(tmp_path):
    from scripts.merge_score_shards import merge_shards

    _write_shard(tmp_path, 0, 2, "a")
    _write_shard(tmp_path, 1, 2, "b")
    output = tmp_path / "scores.csv"

    execution = merge_shards(tmp_path, output=output, expected_shards=2)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["a", "b"]
    assert execution["scored_row_count"] == 2
    assert execution["discovered_task_count"] == 2
    assert execution["expected_shard_count"] == 2
    assert len(execution["merged_shards"]) == 2


def test_merge_rejects_missing_shard(tmp_path):
    from scripts.merge_score_shards import merge_shards

    _write_shard(tmp_path, 0, 2, "a")

    with pytest.raises(ValueError, match="shard indices mismatch"):
        merge_shards(tmp_path, output=tmp_path / "scores.csv", expected_shards=2)


def test_merge_rejects_inconsistent_discovery_snapshots(tmp_path):
    from scripts.merge_score_shards import merge_shards

    _write_shard(tmp_path, 0, 2, "a", discovered_task_count=2)
    _write_shard(tmp_path, 1, 2, "b", discovered_task_count=3)

    with pytest.raises(ValueError, match="discovered_task_count mismatch"):
        merge_shards(tmp_path, output=tmp_path / "scores.csv", expected_shards=2)


def test_merge_rejects_expected_count_without_full_snapshot_coverage(tmp_path):
    from scripts.merge_score_shards import merge_shards

    _write_shard(tmp_path, 0, 2, "a", discovered_task_count=3)
    _write_shard(tmp_path, 1, 2, "b", discovered_task_count=3)

    with pytest.raises(ValueError, match="do not cover the discovered task snapshot"):
        merge_shards(tmp_path, output=tmp_path / "scores.csv", expected_shards=2)


def test_merge_rejects_shard_rows_different_from_expected_count(tmp_path):
    from scripts.merge_score_shards import merge_shards

    _write_shard(tmp_path, 0, 1, "a", expected_shard_count=2)

    with pytest.raises(ValueError, match="expected_shard_count mismatch"):
        merge_shards(tmp_path, output=tmp_path / "scores.csv", expected_shards=1)
