"""Merge a complete set of canonical score shards after contract validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

SHARD_PATTERN = re.compile(r"scores_shard(\d{3})\.csv$")


def merge_shards(
    input_dir: Path,
    *,
    output: Path,
    expected_shards: int,
    delete: bool = False,
) -> dict[str, object]:
    if expected_shards < 1:
        raise ValueError("expected_shards must be >= 1")
    indexed: dict[int, Path] = {}
    for path in input_dir.glob("scores_shard*.csv"):
        match = SHARD_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"unexpected shard filename: {path.name}")
        indexed[int(match.group(1))] = path
    expected_indices = set(range(expected_shards))
    if set(indexed) != expected_indices:
        raise ValueError(
            f"shard indices mismatch: expected {sorted(expected_indices)}, got {sorted(indexed)}"
        )

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    expected_header: list[str] | None = None
    expected_contract: dict[str, object] | None = None
    shard_records: list[dict[str, object]] = []
    first_sidecar: dict[str, object] | None = None
    discovered_task_count: int | None = None
    expected_row_count = 0
    for index in sorted(indexed):
        path = indexed[index]
        sidecar_path = path.with_suffix(".schema.json")
        if not sidecar_path.is_file():
            raise ValueError(f"missing shard sidecar: {sidecar_path}")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed shard sidecar: {sidecar_path}") from exc
        if not isinstance(sidecar, dict):
            raise ValueError(f"shard sidecar must be an object: {sidecar_path}")
        execution = sidecar.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"shard sidecar lacks execution metadata: {sidecar_path}")
        contract = {
            "score_file_schema_version": sidecar.get("score_file_schema_version"),
            "formula": sidecar.get("formula"),
            "primary_score": sidecar.get("primary_score"),
            "source_manifest_paths": sidecar.get("source_manifest_paths"),
            "source_manifest_sha256": sidecar.get("source_manifest_sha256"),
            "num_shards": execution.get("num_shards"),
        }
        if expected_contract is None:
            expected_contract = contract
            first_sidecar = sidecar
        elif contract != expected_contract:
            raise ValueError(f"shard contract mismatch: {sidecar_path}")
        if execution.get("shard_idx") != index:
            raise ValueError(f"sidecar shard_idx mismatch: {sidecar_path}")
        if execution.get("num_shards") != expected_shards:
            raise ValueError(f"sidecar num_shards mismatch: {sidecar_path}")
        if execution.get("status") != "complete":
            raise ValueError(f"shard is not marked complete: {sidecar_path}")
        shard_discovered_count = _non_negative_int(
            execution.get("discovered_task_count"),
            field="discovered_task_count",
            sidecar_path=sidecar_path,
        )
        shard_expected_count = _non_negative_int(
            execution.get("expected_shard_count"),
            field="expected_shard_count",
            sidecar_path=sidecar_path,
        )
        if discovered_task_count is None:
            discovered_task_count = shard_discovered_count
        elif shard_discovered_count != discovered_task_count:
            raise ValueError(f"discovered_task_count mismatch: {sidecar_path}")
        csv_sha256 = _sha256(path)
        if execution.get("scores_sha256") != csv_sha256:
            raise ValueError(f"shard CSV hash mismatch: {path}")

        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise ValueError(f"shard header mismatch: {path}")
            shard_rows = list(reader)
        scored_row_count = _non_negative_int(
            execution.get("scored_row_count"),
            field="scored_row_count",
            sidecar_path=sidecar_path,
        )
        if scored_row_count != len(shard_rows):
            raise ValueError(f"sidecar row count mismatch: {path}")
        if shard_expected_count != len(shard_rows):
            raise ValueError(f"expected_shard_count mismatch: {path}")
        expected_row_count += shard_expected_count
        for row in shard_rows:
            key = (str(row.get("id") or ""), str(row.get("version") or ""))
            if key in seen:
                raise ValueError(f"duplicate score key across shards: {key}")
            seen.add(key)
            rows.append(row)
        shard_records.append(
            {
                "shard_idx": index,
                "path": str(path),
                "rows": len(shard_rows),
                "expected_rows": shard_expected_count,
                "sha256": csv_sha256,
                "sidecar_sha256": _sha256(sidecar_path),
            }
        )

    if discovered_task_count is None:
        raise AssertionError("expected at least one shard sidecar")
    if expected_row_count != discovered_task_count:
        raise ValueError(
            "shard expected counts do not cover the discovered task snapshot: "
            f"expected sum {expected_row_count}, discovered {discovered_task_count}"
        )
    if len(rows) != discovered_task_count:
        raise ValueError(
            "merged shard rows do not cover the discovered task snapshot: "
            f"rows {len(rows)}, discovered {discovered_task_count}"
        )

    rows.sort(key=lambda row: (str(row.get("id") or ""), int(row.get("version") or 0)))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_header or [])
        writer.writeheader()
        writer.writerows(rows)
    temporary_output.replace(output)

    if first_sidecar is None:
        raise AssertionError("expected at least one shard sidecar")
    merged_sidecar = dict(first_sidecar)
    merged_sidecar["execution"] = {
        "status": "complete",
        "num_shards": expected_shards,
        "discovered_task_count": discovered_task_count,
        "expected_shard_count": expected_row_count,
        "scored_row_count": len(rows),
        "scores_sha256": _sha256(output),
        "merged_shards": shard_records,
    }
    sidecar_output = output.with_suffix(".schema.json")
    temporary_sidecar = sidecar_output.with_suffix(sidecar_output.suffix + ".tmp")
    temporary_sidecar.write_text(
        json.dumps(merged_sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_sidecar.replace(sidecar_output)

    if delete:
        for path in indexed.values():
            path.unlink()
            path.with_suffix(".schema.json").unlink()
    return merged_sidecar["execution"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--output", type=Path, default=Path("outputs/generated/scores.csv"))
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = merge_shards(
            args.input_dir,
            output=args.output,
            expected_shards=args.expected_shards,
            delete=args.delete,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_negative_int(value: object, *, field: str, sidecar_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"sidecar {field} must be a non-negative integer: {sidecar_path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
