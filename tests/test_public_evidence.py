"""Checks for the committed public evidence bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def test_public_evidence_manifest_hashes_match_checkout():
    from scripts.build_evidence_manifest import verify_manifest

    root = Path.cwd()
    manifest = root / "reports" / "final" / "evidence_manifest.json"

    assert manifest.is_file()
    assert verify_manifest(root, manifest) == []


def test_historical_results_are_explicitly_aggregate_only():
    path = Path("reports/final/historical_benchmark_summary.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["run"] for row in rows] == ["Base", "Product SFT", "Product DPO"]
    assert {row["evidence_status"] for row in rows} == {"historical-aggregate-only"}


def test_historical_selection_bias_is_machine_readable_and_bounded():
    payload = json.loads(
        Path("reports/final/historical_selection_bias.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "historical-selection-bias/v1"
    assert payload["evidence_status"] == "historical-aggregate-only"
    assert payload["metric"] == {
        "name": "median_target_length",
        "reported_unit": "characters",
        "before_product_selection": 15,
        "after_product_selection": 8,
        "absolute_change": -7,
    }
    assert payload["provenance"]["raw_selection_rows_available"] is False
    assert payload["provenance"]["recomputable_from_checkout"] is False


def test_prompt_dataset_quality_report_matches_pinned_source():
    quality = json.loads(
        Path("reports/final/prompt_dataset_quality_v1.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        Path("reports/final/prompt_dataset_source.manifest.json").read_text(encoding="utf-8")
    )

    assert quality["valid_records"] == 15000
    assert quality["malformed_records"] == 0
    assert quality["errors"] == []
    assert quality["metadata"]["source"]["sha256"] == source["output_sha256"]


def test_benchmark_manifest_is_unique_and_target_disjoint():
    benchmark_manifest = json.loads(
        Path("reports/final/benchmark_prompts_v2.manifest.json").read_text(encoding="utf-8")
    )
    target_index_path = Path(benchmark_manifest["training_target_hash_index"]["path"])
    target_index = json.loads(target_index_path.read_text(encoding="utf-8"))
    benchmark_rows = [
        json.loads(line)
        for line in Path("reports/final/benchmark_prompts_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    from src.evaluation.final_benchmark import normalized_target_sha256

    benchmark_hashes = {normalized_target_sha256(row["target_text"]) for row in benchmark_rows}
    training_hashes = set(target_index["target_hashes"])
    actual_overlap = benchmark_hashes & training_hashes

    assert benchmark_manifest["total_prompts"] == 120
    assert benchmark_manifest["unique_target_count"] == 120
    assert actual_overlap == set()
    assert benchmark_manifest["training_target_overlap_count"] == len(actual_overlap)
    assert benchmark_manifest["target_disjoint"] is (not actual_overlap)
    assert (
        benchmark_manifest["training_target_hash_index"]["sha256"]
        == hashlib.sha256(target_index_path.read_bytes()).hexdigest()
    )
    assert target_index["target_hashes"] == sorted(training_hashes)
    assert target_index["unique_normalized_target_count"] == len(training_hashes)
    assert (
        target_index["source"]["output_sha256"]
        == benchmark_manifest["exclusion_source_manifest"]["output_sha256"]
    )
    assert benchmark_manifest["exclusion_source_manifest"]["resolved_revision"] == (
        "ecd8b2da9820b35afc65e2d56eaf37a662c37976"
    )


def test_empty_evidence_manifest_cannot_pass(tmp_path):
    from scripts.build_evidence_manifest import verify_manifest

    manifest = tmp_path / "empty.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "public-evidence-manifest/v1",
                "historical_results_status": "unknown",
                "checkpoint_status": "unknown",
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    errors = verify_manifest(Path.cwd(), manifest)
    assert any("missing required evidence artifacts" in error for error in errors)
