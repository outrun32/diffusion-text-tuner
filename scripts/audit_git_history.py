"""Audit reachable Git history for forbidden handoff/editor archives and large blobs."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_BLOB_BYTES = 10 * 1024 * 1024
FORBIDDEN_PATH_MARKERS = (
    "outputs/handoff/",
    "chatsessions/",
    ".vscdb",
    "handoff_bundle",
)
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")
SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?:gh[pousr]|github_pat)_[A-Za-z0-9_-]{20,}"),
)


@dataclass(frozen=True)
class HistoryFinding:
    object_id: str
    path: str
    size: int
    reasons: tuple[str, ...]
    archive_secret_matches: int = 0
    archive_sensitive_members: int = 0


def audit_history(repo: Path) -> list[HistoryFinding]:
    objects = _reachable_objects(repo)
    sizes = _blob_sizes(repo, [object_id for object_id, _path in objects])
    findings: list[HistoryFinding] = []
    for object_id, path in objects:
        size = sizes.get(object_id)
        if size is None:
            continue
        lowered = path.casefold()
        reasons: list[str] = []
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            reasons.append("forbidden handoff/editor path")
        if size > MAX_BLOB_BYTES:
            reasons.append(f"blob exceeds {MAX_BLOB_BYTES} bytes")

        secret_matches = 0
        sensitive_members = 0
        if path.casefold().endswith(ARCHIVE_SUFFIXES) and (reasons or size > 0):
            blob = _git_blob(repo, object_id)
            secret_matches, sensitive_members = inspect_archive_bytes(blob)
            if secret_matches:
                reasons.append("archive contains credential-shaped strings")
            if sensitive_members:
                reasons.append("archive contains chat/editor state members")

        if reasons:
            findings.append(
                HistoryFinding(
                    object_id=object_id[:12],
                    path=path,
                    size=size,
                    reasons=tuple(dict.fromkeys(reasons)),
                    archive_secret_matches=secret_matches,
                    archive_sensitive_members=sensitive_members,
                )
            )
    return findings


def inspect_archive_bytes(payload: bytes) -> tuple[int, int]:
    """Return secret-pattern and sensitive-member counts without exposing contents."""

    return _inspect_archive_bytes(payload, depth=0)


def _inspect_archive_bytes(payload: bytes, *, depth: int) -> tuple[int, int]:
    if depth > 2:
        return 0, 0

    secret_matches = 0
    sensitive_members = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except tarfile.TarError:
        return 0, 0
    with archive:
        for member in archive.getmembers():
            lowered_name = member.name.casefold()
            if any(
                marker in lowered_name
                for marker in ("chatsessions/", ".vscdb", "copilot-chat-session")
            ):
                sensitive_members += 1
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            if lowered_name.endswith(ARCHIVE_SUFFIXES) and member.size <= 50 * 1024 * 1024:
                nested_matches, nested_members = _inspect_archive_bytes(
                    extracted.read(),
                    depth=depth + 1,
                )
                secret_matches += nested_matches
                sensitive_members += nested_members
                continue
            if member.size > 64 * 1024 * 1024:
                continue
            content = extracted.read()
            secret_matches += sum(len(pattern.findall(content)) for pattern in SECRET_PATTERNS)
    return secret_matches, sensitive_members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        findings = audit_history(args.repo.resolve())
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"history audit failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": "git-history-audit/v1",
                    "finding_count": len(findings),
                    "findings": [asdict(finding) for finding in findings],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for finding in findings:
            print(
                f"{finding.object_id} {finding.path} ({finding.size} bytes): "
                + "; ".join(finding.reasons),
                file=sys.stderr,
            )
    return 1 if findings else 0


def _reachable_objects(repo: Path) -> list[tuple[str, str]]:
    result = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    objects = []
    for line in result.stdout.splitlines():
        object_id, separator, path = line.partition(" ")
        if separator and path:
            objects.append((object_id, path))
    return objects


def _blob_sizes(repo: Path, object_ids: list[str]) -> dict[str, int]:
    if not object_ids:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=repo,
        input="\n".join(object_ids) + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    sizes: dict[str, int] = {}
    for line in result.stdout.splitlines():
        object_id, object_type, size_text = line.split()
        if object_type == "blob":
            sizes[object_id] = int(size_text)
    return sizes


def _git_blob(repo: Path, object_id: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
