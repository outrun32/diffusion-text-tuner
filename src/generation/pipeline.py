"""Import-safe FLUX image generation pipeline implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GenerationResumeError(ValueError):
    """Raised when existing generation artifacts cannot be resumed safely."""


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for the generated-image pipeline."""

    prompts: Path
    output_dir: Path = Path("outputs/generated")
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    model_revision: str | None = None
    lora_path: str | None = None
    versions_per_prompt: int = 5
    batch_size: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    resolution: int = 512
    seed: int = 42
    device: str = "cuda"
    start_idx: int = 0
    end_idx: int | None = None
    shard_index: int = 0
    shard_count: int = 1
    save_latents: bool = True
    save_png: bool = True
    manifest_path: Path | None = None
    run_manifest_path: str = ""

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("batch_size currently supports only 1; batching is not implemented")
        if self.versions_per_prompt < 1:
            raise ValueError("versions_per_prompt must be >= 1")
        if self.resolution < 64:
            raise ValueError("resolution must be >= 64")
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0")
        if self.end_idx is not None and self.end_idx < self.start_idx:
            raise ValueError("end_idx must be >= start_idx")
        if self.shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")


@dataclass(frozen=True)
class GenerationPaths:
    """Filesystem paths used by the generated-image pipeline."""

    output_dir: Path
    latents_dir: Path
    text_embeds_dir: Path
    images_dir: Path
    manifest_path: Path


def load_prompt_records(
    prompts_path: Path | str,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[dict[str, Any]]:
    """Load prompt JSONL records and apply the CLI-compatible index slice."""
    records, _ = _load_prompt_snapshot(
        prompts_path,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    return records


def _load_prompt_snapshot(
    prompts_path: Path | str,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    raw = Path(prompts_path).read_bytes()
    records: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        _ = record["prompt"]
        records.append(record)
    return records[start_idx:end_idx], hashlib.sha256(raw).hexdigest()


def resolve_generation_paths(output_dir: Path | str) -> GenerationPaths:
    """Resolve deterministic generated-image output paths without side effects."""
    root = Path(output_dir)
    return GenerationPaths(
        output_dir=root,
        latents_dir=root / "latents",
        text_embeds_dir=root / "text_embeds",
        images_dir=root / "images",
        manifest_path=root / "manifest.json",
    )


def plan_generation_seed(
    *,
    seed: int,
    prompt_index: int,
    versions_per_prompt: int,
    version: int,
) -> int:
    """Return the deterministic per-prompt/version seed used by the original script."""
    return seed + prompt_index * versions_per_prompt + version


GENERATION_MANIFEST_SCHEMA_VERSION = "generation-manifest/v4"
_TOPOLOGY_FIELDS = (
    "output_dir",
    "model_id",
    "model_revision",
    "lora",
    "versions_per_prompt",
    "batch_size",
    "num_inference_steps",
    "guidance_scale",
    "resolution",
    "device",
    "seed",
    "save_latents",
    "save_png",
    "prompts_path",
    "prompts_sha256",
    "total_prompt_count",
    "slice_count",
)


def build_generation_manifest(
    config: GenerationConfig,
    paths: GenerationPaths,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a hashed immutable contract plus a separately hashed completion record."""

    all_records, prompts_sha256 = _load_prompt_snapshot(config.prompts)
    start_idx, end_idx = _resolved_slice(config, len(all_records))
    if config.start_idx != start_idx or (config.end_idx or len(all_records)) != end_idx:
        raise GenerationResumeError(
            "generation slice does not match deterministic shard topology: "
            f"expected [{start_idx}, {end_idx}), got [{config.start_idx}, {config.end_idx})"
        )
    if all_records[start_idx:end_idx] != records:
        raise GenerationResumeError(
            "prompt input changed after it was loaded; restart generation with a stable input file"
        )
    expected = _expected_counts(len(records), config)
    contract = {
        "output_dir": str(paths.output_dir.resolve()),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "lora": _artifact_identity(config.lora_path),
        "versions_per_prompt": config.versions_per_prompt,
        "batch_size": config.batch_size,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "resolution": config.resolution,
        "device": config.device,
        "seed": config.seed,
        "save_latents": config.save_latents,
        "save_png": config.save_png,
        "prompts_path": str(config.prompts.resolve()),
        "prompts_sha256": prompts_sha256,
        "selected_records_sha256": _records_sha256(records),
        "total_prompt_count": len(all_records),
        "slice_index": config.shard_index,
        "slice_count": config.shard_count,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "expected": expected,
        "expected_paths_sha256": _expected_paths_hashes(paths, start_idx, end_idx, config),
        "artifact_layout": {
            "text_embeddings": "text_embeds/{prompt_id}.pt",
            "images": "images/{prompt_id}/v{version}.png" if config.save_png else None,
            "latents": "latents/{prompt_id}/v{version}.pt" if config.save_latents else None,
            "prompt_id": "{global_index:06d}",
            "version_start": 0,
            "version_stop_exclusive": config.versions_per_prompt,
        },
    }
    completion = _completion_record(
        contract,
        status="planned",
        run_manifest_path=config.run_manifest_path,
        attempt=0,
        generated=_zero_counts(),
        skipped=_zero_counts(),
    )
    return {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": _canonical_json_sha256(contract),
        "completion": completion,
    }


def ensure_generation_resume_contract(
    config: GenerationConfig,
    paths: GenerationPaths,
    records: list[dict[str, Any]],
) -> Path:
    """Create or verify one shard contract while allowing compatible sibling shards."""

    expected = build_generation_manifest(config, paths, records)
    manifest_path = _manifest_path(config, paths)
    if paths.output_dir.is_symlink():
        raise GenerationResumeError(
            f"generation output root {paths.output_dir} must not be a symlink"
        )
    if paths.output_dir.exists() and not paths.output_dir.is_dir():
        raise GenerationResumeError(
            f"generation output root {paths.output_dir} exists but is not a directory"
        )

    siblings = _related_manifests(manifest_path, paths.output_dir, exclude=manifest_path)
    _validate_topology(expected["contract"], [item[1]["contract"] for item in siblings])
    sibling_paths = [item[0] for item in siblings]
    sibling_contracts = [item[1]["contract"] for item in siblings]

    if manifest_path.exists():
        actual = validate_generation_manifest(manifest_path)
        differences = _manifest_differences(expected["contract"], actual["contract"])
        if differences:
            raise GenerationResumeError(
                "generation resume contract mismatch; use a new output directory: "
                + "; ".join(differences[:8])
            )
        _validate_shared_layout(
            paths.output_dir,
            [actual["contract"], *sibling_contracts],
            [manifest_path, *sibling_paths],
        )
        return manifest_path

    current_paths = _contract_artifact_paths(expected["contract"])
    existing_owned = [
        path for paths_by_kind in current_paths.values() for path in paths_by_kind if path.exists()
    ]
    if existing_owned:
        raise GenerationResumeError(
            "generation shard contains artifacts but its immutable manifest is missing at "
            f"{manifest_path}: {existing_owned[0]}"
        )
    _validate_shared_layout(paths.output_dir, sibling_contracts, sibling_paths)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        raise GenerationResumeError(
            f"generation manifest appeared concurrently at {manifest_path}; rerun to verify it"
        ) from None
    return manifest_path


def validate_generation_manifest(
    path: str | Path,
    *,
    require_complete: bool = False,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate immutable and mutable manifest hashes, optionally including artifact coverage."""

    manifest_path = Path(path)
    payload = _load_manifest(manifest_path)
    if payload.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise GenerationResumeError(
            f"{manifest_path}: schema_version must be {GENERATION_MANIFEST_SCHEMA_VERSION}"
        )
    contract = payload.get("contract")
    completion = payload.get("completion")
    if not isinstance(contract, dict) or not isinstance(completion, dict):
        raise GenerationResumeError(f"{manifest_path}: contract and completion must be objects")
    if payload.get("contract_sha256") != _canonical_json_sha256(contract):
        raise GenerationResumeError(f"{manifest_path}: contract SHA-256 mismatch")
    _validate_prompt_source(manifest_path, contract)
    completion_payload = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    if completion.get("completion_sha256") != _canonical_json_sha256(completion_payload):
        raise GenerationResumeError(f"{manifest_path}: completion SHA-256 mismatch")
    if completion.get("expected") != contract.get("expected"):
        raise GenerationResumeError(
            f"{manifest_path}: completion expected counts do not match contract"
        )
    status = completion.get("status")
    if status not in {"planned", "in-progress", "failed", "complete"}:
        raise GenerationResumeError(f"{manifest_path}: invalid completion status {status!r}")
    if require_complete and status != "complete":
        raise GenerationResumeError(
            f"{manifest_path}: generation is not complete (status={status!r})"
        )
    if require_complete or verify_artifacts:
        coverage = _artifact_coverage(contract)
        if completion.get("coverage") != coverage:
            raise GenerationResumeError(f"{manifest_path}: artifact coverage/hash mismatch")
        discovered = {kind: int(item["count"]) for kind, item in coverage.items()}
        if completion.get("discovered") != discovered:
            raise GenerationResumeError(
                f"{manifest_path}: discovered counts do not match artifacts"
            )
        if require_complete and discovered != contract.get("expected"):
            raise GenerationResumeError(
                f"{manifest_path}: generation artifact coverage is incomplete"
            )
    return payload


def _validate_prompt_source(manifest_path: Path, contract: dict[str, Any]) -> None:
    raw_path = contract.get("prompts_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise GenerationResumeError(f"{manifest_path}: prompts_path is missing")
    prompts_path = Path(raw_path)
    try:
        all_records, prompts_sha256 = _load_prompt_snapshot(prompts_path)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        raise GenerationResumeError(
            f"{manifest_path}: could not validate prompt source {prompts_path}: {exc}"
        ) from exc
    if prompts_sha256 != contract.get("prompts_sha256"):
        raise GenerationResumeError(f"{manifest_path}: prompt source SHA-256 mismatch")
    if len(all_records) != contract.get("total_prompt_count"):
        raise GenerationResumeError(f"{manifest_path}: prompt source row count mismatch")
    try:
        start_idx = int(contract["start_idx"])
        end_idx = int(contract["end_idx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GenerationResumeError(f"{manifest_path}: invalid prompt slice") from exc
    if not 0 <= start_idx <= end_idx <= len(all_records):
        raise GenerationResumeError(f"{manifest_path}: prompt slice is outside the source dataset")
    if _records_sha256(all_records[start_idx:end_idx]) != contract.get("selected_records_sha256"):
        raise GenerationResumeError(f"{manifest_path}: selected prompt slice SHA-256 mismatch")


def begin_generation_attempt(path: str | Path, *, run_manifest_path: str) -> dict[str, Any]:
    payload = validate_generation_manifest(path)
    completion = payload["completion"]
    return _replace_completion(
        Path(path),
        payload,
        status="in-progress",
        run_manifest_path=run_manifest_path,
        attempt=int(completion.get("attempt", 0)) + 1,
        generated=_zero_counts(),
        skipped=_zero_counts(),
    )


def complete_generation_attempt(
    path: str | Path,
    *,
    generated: dict[str, int],
    skipped: dict[str, int],
) -> dict[str, Any]:
    payload = validate_generation_manifest(path)
    contract = payload["contract"]
    expected = _dict_counts(contract.get("expected"))
    generated_counts = _dict_counts(generated)
    skipped_counts = _dict_counts(skipped)
    processed = {kind: generated_counts[kind] + skipped_counts[kind] for kind in expected}
    if processed != expected:
        raise GenerationResumeError(
            f"generation processed counts are incomplete: expected {expected}, found {processed}"
        )
    coverage = _artifact_coverage(contract)
    discovered = {kind: int(item["count"]) for kind, item in coverage.items()}
    if discovered != expected:
        raise GenerationResumeError(
            f"generation artifact coverage is incomplete: expected {expected}, found {discovered}"
        )
    _replace_completion(
        Path(path),
        payload,
        status="complete",
        run_manifest_path=str(payload["completion"].get("run_manifest_path") or ""),
        attempt=int(payload["completion"].get("attempt", 0)),
        generated=generated_counts,
        skipped=skipped_counts,
    )
    return validate_generation_manifest(path, require_complete=True, verify_artifacts=True)


def fail_generation_attempt(
    path: str | Path,
    *,
    error: Exception,
    generated: dict[str, int] | None = None,
    skipped: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload = validate_generation_manifest(path)
    return _replace_completion(
        Path(path),
        payload,
        status="failed",
        run_manifest_path=str(payload["completion"].get("run_manifest_path") or ""),
        attempt=int(payload["completion"].get("attempt", 0)),
        generated=_dict_counts(generated),
        skipped=_dict_counts(skipped),
        error_type=type(error).__name__,
    )


def _replace_completion(
    path: Path,
    payload: dict[str, Any],
    *,
    status: str,
    run_manifest_path: str,
    attempt: int,
    generated: dict[str, int],
    skipped: dict[str, int],
    error_type: str | None = None,
) -> dict[str, Any]:
    contract = payload["contract"]
    updated = dict(payload)
    updated["completion"] = _completion_record(
        contract,
        status=status,
        run_manifest_path=run_manifest_path,
        attempt=attempt,
        generated=generated,
        skipped=skipped,
        error_type=error_type,
    )
    _atomic_write_json(path, updated)
    return updated


def _completion_record(
    contract: dict[str, Any],
    *,
    status: str,
    run_manifest_path: str,
    attempt: int,
    generated: dict[str, int],
    skipped: dict[str, int],
    error_type: str | None = None,
) -> dict[str, Any]:
    coverage = _artifact_coverage(contract)
    completion = {
        "status": status,
        "attempt": attempt,
        "run_manifest_path": run_manifest_path,
        "expected": dict(contract["expected"]),
        "discovered": {kind: int(item["count"]) for kind, item in coverage.items()},
        "generated": _dict_counts(generated),
        "skipped": _dict_counts(skipped),
        "coverage": coverage,
        "error_type": error_type,
    }
    completion["completion_sha256"] = _canonical_json_sha256(completion)
    return completion


def _resolved_slice(config: GenerationConfig, total: int) -> tuple[int, int]:
    if config.shard_count == 1:
        return config.start_idx, min(total, total if config.end_idx is None else config.end_idx)
    chunk_size = (total + config.shard_count - 1) // config.shard_count
    start = min(total, config.shard_index * chunk_size)
    return start, min(total, start + chunk_size)


def _manifest_path(config: GenerationConfig, paths: GenerationPaths) -> Path:
    if config.manifest_path is not None:
        return config.manifest_path
    if config.shard_count == 1:
        return paths.manifest_path
    return (
        paths.output_dir
        / "manifests"
        / (f"generation-shard-{config.shard_index:05d}-of-{config.shard_count:05d}.json")
    )


def _expected_counts(num_prompts: int, config: GenerationConfig) -> dict[str, int]:
    variants = num_prompts * config.versions_per_prompt
    return {
        "text_embeddings": num_prompts,
        "images": variants if config.save_png else 0,
        "latents": variants if config.save_latents else 0,
    }


def _zero_counts() -> dict[str, int]:
    return {"text_embeddings": 0, "images": 0, "latents": 0}


def _dict_counts(value: Any) -> dict[str, int]:
    source = value if isinstance(value, dict) else {}
    return {kind: int(source.get(kind, 0)) for kind in _zero_counts()}


def _contract_artifact_paths(contract: dict[str, Any]) -> dict[str, list[Path]]:
    root = Path(str(contract["output_dir"]))
    prompt_ids = [f"{index:06d}" for index in range(contract["start_idx"], contract["end_idx"])]
    versions = range(int(contract["versions_per_prompt"]))
    return {
        "text_embeddings": [root / "text_embeds" / f"{prompt_id}.pt" for prompt_id in prompt_ids],
        "images": (
            [
                root / "images" / prompt_id / f"v{version}.png"
                for prompt_id in prompt_ids
                for version in versions
            ]
            if contract["save_png"]
            else []
        ),
        "latents": (
            [
                root / "latents" / prompt_id / f"v{version}.pt"
                for prompt_id in prompt_ids
                for version in versions
            ]
            if contract["save_latents"]
            else []
        ),
    }


def _expected_paths_hashes(
    paths: GenerationPaths,
    start_idx: int,
    end_idx: int,
    config: GenerationConfig,
) -> dict[str, str]:
    contract = {
        "output_dir": str(paths.output_dir.resolve()),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "versions_per_prompt": config.versions_per_prompt,
        "save_png": config.save_png,
        "save_latents": config.save_latents,
    }
    return {
        kind: _relative_paths_sha256(items, paths.output_dir.resolve())
        for kind, items in _contract_artifact_paths(contract).items()
    }


def _artifact_coverage(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(str(contract["output_dir"]))
    coverage: dict[str, dict[str, Any]] = {}
    for kind, expected_paths in _contract_artifact_paths(contract).items():
        discovered = [path for path in expected_paths if path.is_file() and not path.is_symlink()]
        coverage[kind] = {
            "count": len(discovered),
            "paths_sha256": _relative_paths_sha256(discovered, root),
            "contents_sha256": _artifact_contents_sha256(discovered, root),
        }
    return coverage


def _relative_paths_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
    return digest.hexdigest()


def _artifact_contents_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _related_manifests(
    manifest_path: Path,
    output_dir: Path,
    *,
    exclude: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    related = []
    if not manifest_path.parent.is_dir():
        return related
    for candidate in sorted(manifest_path.parent.glob("*.json")):
        if candidate.resolve() == exclude.resolve():
            continue
        try:
            payload = _load_manifest(candidate)
        except GenerationResumeError:
            continue
        if payload.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
            continue
        validated = validate_generation_manifest(candidate)
        if validated["contract"].get("output_dir") == str(output_dir.resolve()):
            related.append((candidate, validated))
    return related


def _validate_topology(current: dict[str, Any], siblings: list[dict[str, Any]]) -> None:
    for sibling in siblings:
        mismatched = [
            field for field in _TOPOLOGY_FIELDS if sibling.get(field) != current.get(field)
        ]
        if mismatched:
            raise GenerationResumeError(
                "incompatible generation shard topology/config: " + ", ".join(mismatched)
            )
        if sibling.get("slice_index") == current.get("slice_index"):
            raise GenerationResumeError(
                f"duplicate generation shard index {current.get('slice_index')} for one output root"
            )
        if max(current["start_idx"], sibling["start_idx"]) < min(
            current["end_idx"], sibling["end_idx"]
        ):
            raise GenerationResumeError("generation shard ranges overlap")


def _validate_shared_layout(
    output_dir: Path,
    contracts: list[dict[str, Any]],
    manifest_paths: list[Path],
) -> None:
    if not output_dir.exists():
        return
    root = output_dir.resolve()
    allowed_files = {
        path.resolve()
        for contract in contracts
        for items in _contract_artifact_paths(contract).values()
        for path in items
    }
    allowed_files.update(
        path.resolve() for path in manifest_paths if path.resolve().is_relative_to(root)
    )
    allowed_directories = {
        root,
        *(root / name for name in ("images", "latents", "text_embeds", "manifests")),
    }
    allowed_directories = {path.resolve() for path in allowed_directories}
    for contract in contracts:
        for kind, items in _contract_artifact_paths(contract).items():
            if kind in {"images", "latents"}:
                allowed_directories.update(path.parent.resolve() for path in items)
    for entry in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix()):
        resolved = entry.resolve()
        if entry.is_symlink():
            raise GenerationResumeError(f"generation output contains unsupported symlink {entry}")
        if entry.is_dir():
            if resolved not in allowed_directories:
                raise GenerationResumeError(
                    f"generation output contains undeclared directory {entry}"
                )
        elif resolved not in allowed_files:
            raise GenerationResumeError(f"generation output contains undeclared artifact {entry}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationResumeError(f"could not read generation manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerationResumeError(f"generation manifest {path} must contain a JSON object")
    return payload


def _manifest_differences(expected: Any, actual: Any, *, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(expected.keys() | actual.keys()):
            location = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                differences.append(f"{location} is missing")
            elif key not in expected:
                differences.append(f"{location} is unexpected")
            else:
                differences.extend(
                    _manifest_differences(expected[key], actual[key], prefix=location)
                )
        return differences
    return [] if expected == actual else [f"{prefix}: expected {expected!r}, found {actual!r}"]


def _artifact_identity(locator: str | None) -> dict[str, str | None]:
    if locator is None:
        return {"locator": None, "kind": "none", "sha256": None}
    path = Path(locator).expanduser()
    if path.is_file():
        return {"locator": locator, "kind": "file", "sha256": _sha256(path)}
    if path.is_dir():
        return {"locator": locator, "kind": "directory", "sha256": _directory_sha256(path)}
    return {"locator": locator, "kind": "unresolved", "sha256": None}


def _directory_sha256(path: Path) -> str:
    return _artifact_contents_sha256(
        sorted(candidate for candidate in path.rglob("*") if candidate.is_file()), path
    )


def _records_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        serialized = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(serialized).to_bytes(8, "big"))
        digest.update(serialized)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_generation(config: GenerationConfig) -> None:
    """Generate prompt variants, latents, text embeddings, images, and a manifest."""
    from src.runtime.capabilities import check_stage_support

    if config.device != "cuda":
        raise ValueError(
            "FLUX generation currently supports only device='cuda'; MLX is used only for "
            "text-prompt generation"
        )
    support = check_stage_support("generate")
    if not support.ok:
        raise RuntimeError("; ".join(support.errors))

    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(
        config.prompts,
        start_idx=config.start_idx,
        end_idx=config.end_idx,
    )
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    current = validate_generation_manifest(manifest_path)
    if current["completion"]["status"] == "complete":
        validate_generation_manifest(manifest_path, require_complete=True, verify_artifacts=True)
        logger.info("Generation shard is already complete: %s", manifest_path)
        return

    begin_generation_attempt(manifest_path, run_manifest_path=config.run_manifest_path)
    generated = _zero_counts()
    skipped = _zero_counts()
    try:
        paths.latents_dir.mkdir(parents=True, exist_ok=True)
        paths.text_embeds_dir.mkdir(parents=True, exist_ok=True)
        paths.images_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import torch
        from diffusers import Flux2KleinPipeline
        from tqdm import tqdm

        from src.training.flux2_utils import encode_image

        logger.info("Loading pipeline: %s", config.model_id)
        pipe = Flux2KleinPipeline.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            torch_dtype=torch.bfloat16,
        ).to(config.device)
        if config.lora_path:
            try:
                pipe.load_lora_weights(config.lora_path)
            except ValueError:
                from peft import PeftModel

                pipe.transformer = PeftModel.from_pretrained(pipe.transformer, config.lora_path).to(
                    config.device
                )
        vae = pipe.vae

        for rec_idx, record in enumerate(tqdm(records, desc="Generating")):
            global_idx = config.start_idx + rec_idx
            prompt_id = f"{global_idx:06d}"
            prompt = record["prompt"]
            target_text = record.get("target_text", "")
            prompt_latent_dir = paths.latents_dir / prompt_id
            prompt_image_dir = paths.images_dir / prompt_id
            prompt_latent_dir.mkdir(parents=True, exist_ok=True)
            prompt_image_dir.mkdir(parents=True, exist_ok=True)

            embed_path = paths.text_embeds_dir / f"{prompt_id}.pt"
            if embed_path.exists():
                skipped["text_embeddings"] += 1
            else:
                with torch.no_grad():
                    prompt_embeds = pipe._get_qwen3_prompt_embeds(
                        text_encoder=pipe.text_encoder,
                        tokenizer=pipe.tokenizer,
                        prompt=[prompt],
                        device=config.device,
                        max_sequence_length=512,
                        hidden_states_layers=(9, 18, 27),
                    )
                torch.save(
                    {
                        "prompt_embeds": prompt_embeds[0].cpu(),
                        "target_text": target_text,
                        "prompt": prompt,
                    },
                    embed_path,
                )
                generated["text_embeddings"] += 1

            for version in range(config.versions_per_prompt):
                latent_path = prompt_latent_dir / f"v{version}.pt"
                image_path = prompt_image_dir / f"v{version}.png"
                expected_outputs = {
                    "latents": latent_path if config.save_latents else None,
                    "images": image_path if config.save_png else None,
                }
                active_outputs = {kind: path for kind, path in expected_outputs.items() if path}
                if active_outputs and all(path.exists() for path in active_outputs.values()):
                    for kind in active_outputs:
                        skipped[kind] += 1
                    continue

                generation_seed = plan_generation_seed(
                    seed=config.seed,
                    prompt_index=global_idx,
                    versions_per_prompt=config.versions_per_prompt,
                    version=version,
                )
                generator = torch.Generator(device=config.device).manual_seed(generation_seed)
                with torch.no_grad():
                    pil_image = pipe(
                        prompt=prompt,
                        height=config.resolution,
                        width=config.resolution,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        generator=generator,
                        output_type="pil",
                    ).images[0]
                if config.save_png:
                    pil_image.save(image_path)
                    generated["images"] += 1
                if config.save_latents:
                    img_array = np.asarray(pil_image.convert("RGB"), dtype="uint8")
                    img_tensor = (
                        torch.from_numpy(img_array)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        .div(255.0)
                        .to(config.device, dtype=torch.bfloat16)
                    )
                    latent = encode_image(img_tensor, vae)
                    torch.save({"latent": latent[0].cpu()}, latent_path)
                    generated["latents"] += 1

        complete_generation_attempt(manifest_path, generated=generated, skipped=skipped)
    except Exception as exc:
        try:
            fail_generation_attempt(manifest_path, error=exc, generated=generated, skipped=skipped)
        except Exception:
            logger.exception("Could not record failed generation attempt")
        raise

    logger.info("Generation complete → %s (manifest: %s)", config.output_dir, manifest_path)
