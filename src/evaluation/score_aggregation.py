"""CPU-safe aggregation for multi-seed held-out score CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from src.runtime.manifests import ManifestError, validate_source_manifest

AGGREGATE_SCHEMA_VERSION = "heldout-score-aggregate/v1"


class ScoreAggregationError(ValueError):
    """Raised when score inputs cannot be compared safely."""


def aggregate_score_files(
    inputs: Sequence[tuple[int, str | Path]],
    *,
    output_path: str | Path,
) -> dict[str, object]:
    """Combine canonical per-seed score CSVs and attach explicit seed provenance."""

    if not inputs:
        raise ScoreAggregationError("at least one seed=score input is required")

    rows: list[dict[str, str]] = []
    expected_fields: list[str] | None = None
    expected_contract: dict[str, object] | None = None
    expected_sample_keys: set[tuple[str, str]] | None = None
    common_manifest_paths: set[str] | None = None
    manifest_hashes: dict[str, str] = {}
    source_files: list[dict[str, object]] = []
    seen_keys: set[tuple[int, str, str]] = set()
    seen_seeds: set[int] = set()

    for seed, raw_path in inputs:
        if seed in seen_seeds:
            raise ScoreAggregationError(f"duplicate seed input: {seed}")
        seen_seeds.add(seed)
        path = Path(raw_path)
        if not path.is_file():
            raise ScoreAggregationError(f"score input does not exist: {path}")
        sidecar_path = path.with_suffix(".schema.json")
        sidecar = _load_score_sidecar(path, sidecar_path)
        contract = {
            "score_file_schema_version": sidecar.get("score_file_schema_version"),
            "formula": sidecar.get("formula"),
            "primary_score": sidecar.get("primary_score"),
            "required_phase6_fields": sidecar.get("required_phase6_fields"),
        }
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ScoreAggregationError(f"score formula/schema mismatch: {sidecar_path}")

        source_manifest_paths = sidecar.get("source_manifest_paths")
        source_manifest_sha256 = sidecar.get("source_manifest_sha256")
        if not isinstance(source_manifest_paths, list) or not source_manifest_paths:
            raise ScoreAggregationError(f"sidecar lacks source manifests: {sidecar_path}")
        if not isinstance(source_manifest_sha256, dict):
            raise ScoreAggregationError(f"sidecar lacks source manifest hashes: {sidecar_path}")
        current_paths = {str(value) for value in source_manifest_paths}
        common_manifest_paths = (
            current_paths
            if common_manifest_paths is None
            else common_manifest_paths & current_paths
        )
        for manifest_path in current_paths:
            manifest_hash = source_manifest_sha256.get(manifest_path)
            if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
                raise ScoreAggregationError(
                    f"source manifest hash is missing for {manifest_path}: {sidecar_path}"
                )
            actual_manifest = Path(manifest_path)
            if not actual_manifest.is_file():
                raise ScoreAggregationError(f"source manifest does not exist: {actual_manifest}")
            try:
                validate_source_manifest(actual_manifest)
            except ManifestError as exc:
                raise ScoreAggregationError(
                    f"invalid source manifest {actual_manifest}: {exc}"
                ) from exc
            actual_hash = _sha256(actual_manifest)
            if actual_hash != manifest_hash:
                raise ScoreAggregationError(f"source manifest hash mismatch: {actual_manifest}")
            previous = manifest_hashes.setdefault(manifest_path, manifest_hash)
            if previous != manifest_hash:
                raise ScoreAggregationError(
                    f"source manifest hash changed across seeds: {manifest_path}"
                )

        seed_keys: set[tuple[str, str]] = set()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if not fields:
                raise ScoreAggregationError(f"score input has no header: {path}")
            if expected_fields is None:
                expected_fields = fields
            elif fields != expected_fields:
                raise ScoreAggregationError(
                    f"score input schema mismatch: {path} has {fields}, expected {expected_fields}"
                )
            for row in reader:
                sample_id = row.get("sample_id") or row.get("id") or ""
                version = row.get("version") or ""
                key = (seed, sample_id, version)
                if key in seen_keys:
                    raise ScoreAggregationError(
                        "duplicate score row for "
                        f"seed={seed}, sample={sample_id}, version={version}"
                    )
                seen_keys.add(key)
                seed_key = (sample_id, version)
                if seed_key in seed_keys:
                    raise ScoreAggregationError(
                        f"duplicate sample/version within seed {seed}: {seed_key}"
                    )
                seed_keys.add(seed_key)
                rows.append({"seed": str(seed), **row})
        if expected_sample_keys is None:
            expected_sample_keys = seed_keys
        elif seed_keys != expected_sample_keys:
            missing = sorted(expected_sample_keys - seed_keys)
            extra = sorted(seed_keys - expected_sample_keys)
            raise ScoreAggregationError(
                f"sample coverage mismatch for seed {seed}: missing={missing}, extra={extra}"
            )
        source_files.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": _sha256(path),
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": _sha256(sidecar_path),
            }
        )

    if not common_manifest_paths:
        raise ScoreAggregationError("seed score files share no common source manifest")

    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            row.get("sample_id") or row.get("id") or "",
            int(row.get("version") or 0),
        )
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", *(expected_fields or [])]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if expected_contract is None:
        raise AssertionError("score contract must exist after non-empty inputs")
    common_hashes = {path: manifest_hashes[path] for path in sorted(common_manifest_paths)}
    execution = {
        "status": "complete",
        "row_count": len(rows),
        "seed_count": len(seen_seeds),
        "seeds": sorted(seen_seeds),
        "sample_count_per_seed": len(expected_sample_keys or ()),
        "scores_sha256": _sha256(output),
        "source_files": source_files,
    }
    metadata = {
        "schema_version": "reward-score-metadata/v1",
        "score_file_schema_version": AGGREGATE_SCHEMA_VERSION,
        "formula": expected_contract["formula"],
        "primary_score": expected_contract["primary_score"],
        "required_phase6_fields": expected_contract["required_phase6_fields"],
        "source_manifest_paths": sorted(common_manifest_paths),
        "source_manifest_sha256": common_hashes,
        "execution": execution,
    }
    for sidecar in (output.with_suffix(".schema.json"), output.with_suffix(".aggregate.json")):
        _write_json_atomic(sidecar, metadata)
    return metadata


def _load_score_sidecar(path: Path, sidecar_path: Path) -> dict[str, object]:
    if not sidecar_path.is_file():
        raise ScoreAggregationError(f"score sidecar does not exist: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScoreAggregationError(f"malformed score sidecar: {sidecar_path}") from exc
    if not isinstance(sidecar, dict):
        raise ScoreAggregationError(f"score sidecar must be an object: {sidecar_path}")
    execution = sidecar.get("execution")
    if not isinstance(execution, dict) or execution.get("status") != "complete":
        raise ScoreAggregationError(f"score sidecar is not complete: {sidecar_path}")
    if execution.get("scores_sha256") != _sha256(path):
        raise ScoreAggregationError(f"score CSV hash mismatch: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _row in csv.DictReader(handle))
    if execution.get("scored_row_count") != row_count:
        raise ScoreAggregationError(f"score row count mismatch: {path}")
    return sidecar


def parse_seed_input(value: str) -> tuple[int, Path]:
    """Parse a CLI ``SEED=PATH`` input."""

    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text or not path_text:
        raise ScoreAggregationError("score input must use SEED=PATH syntax")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ScoreAggregationError(f"invalid score seed: {seed_text!r}") from exc
    return seed, Path(path_text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
