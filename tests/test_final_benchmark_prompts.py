"""Tests for deterministic, target-disjoint benchmark prompt construction."""

from __future__ import annotations

import json
from hashlib import sha256


def test_benchmark_targets_are_unique_by_default(tmp_path):
    from src.evaluation.final_benchmark import make_benchmark_prompts

    output = tmp_path / "benchmark.jsonl"
    summary = make_benchmark_prompts(output, count_per_slice=20)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 120
    assert summary["unique_target_count"] == 120
    assert len({row["target_text"] for row in rows}) == 120


def test_benchmark_can_exclude_training_targets(tmp_path):
    from src.evaluation.final_benchmark import make_benchmark_prompts

    excluded = {"ОБЪЕМ", "ПОДЪЕМ", "СЪЕЗД", "ВЪЕЗД", "СЪЕМКА"}
    output = tmp_path / "benchmark.jsonl"
    summary = make_benchmark_prompts(
        output,
        count_per_slice=20,
        excluded_targets=excluded,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    targets = {row["target_text"] for row in rows}

    assert len(rows) == 120
    assert targets.isdisjoint(excluded)
    assert summary["training_target_overlap_count"] == 0
    assert summary["training_target_overlap_hashes"] == []
    assert summary["target_disjoint"] is True
    assert summary["excluded_benchmark_matches"] == sorted(excluded)
    assert {"РАЗЪЁМ", "АДЪЮТАНТ", "КОНЪЮНКТУРА", "ПЬЕСА", "РУЖЬЁ"} <= targets


def test_target_hash_index_is_exact_deterministic_and_linked_from_benchmark(tmp_path):
    from src.evaluation.final_benchmark import (
        build_target_hash_index,
        load_excluded_targets,
        make_benchmark_prompts,
        normalized_target_sha256,
    )

    source = tmp_path / "training.jsonl"
    source.write_text(
        "".join(
            json.dumps({"target_text": target}, ensure_ascii=False) + "\n"
            for target in [" ОБЪЕМ ", "объем", "ПОДЪЕМ"]
        ),
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "repository": "example/prompts",
                "resolved_revision": "abc123",
                "output_sha256": sha256(source.read_bytes()).hexdigest(),
                "row_count": 3,
            }
        ),
        encoding="utf-8",
    )
    first_index = tmp_path / "targets-a.json"
    second_index = tmp_path / "targets-b.json"

    first = build_target_hash_index(source, first_index, source_manifest=source_manifest)
    second = build_target_hash_index(source, second_index, source_manifest=source_manifest)

    assert first == second
    assert first_index.read_bytes() == second_index.read_bytes()
    assert first["unique_normalized_target_count"] == 2
    assert first["target_hashes"] == sorted(
        [normalized_target_sha256("ОБЪЕМ"), normalized_target_sha256("ПОДЪЕМ")]
    )

    benchmark = tmp_path / "benchmark.jsonl"
    summary = make_benchmark_prompts(
        benchmark,
        count_per_slice=20,
        excluded_targets=load_excluded_targets(source),
        exclusion_manifest=source_manifest,
        exclusion_target_hash_index=first_index,
    )

    assert summary["target_disjoint"] is True
    assert summary["training_target_overlap_count"] == 0
    assert (
        summary["training_target_hash_index"]["sha256"]
        == sha256(first_index.read_bytes()).hexdigest()
    )
