"""Prepare a filtered mirror clone that removes private handoff artifacts.

The command never pushes. It creates a pre-rewrite bundle, rewrites a disposable
mirror clone with git-filter-repo, verifies every reachable ref, and emits the
two explicit force-push commands a repository owner may run after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

FILTER_REPO_VERSION = "2.47.0"
FORBIDDEN_PATHS = (
    "outputs/handoff/",
    "scripts/export_handoff_bundle.sh",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(Path.cwd()),
        help="Local checkout or remote Git URL to mirror-clone.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="New empty directory that will contain the filtered bare mirror.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: sibling of work-dir).",
    )
    args = parser.parse_args(argv)

    try:
        report = prepare_cleanup(
            source=args.source,
            work_dir=args.work_dir.resolve(),
            report_path=args.report.resolve() if args.report else None,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        safe_error = redact_sensitive_text(str(exc), sensitive_values=(args.source,))
        print(f"history cleanup preparation failed: {safe_error}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def prepare_cleanup(
    *,
    source: str,
    work_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Clone, back up, rewrite, and verify without touching the source repo."""

    if work_dir.exists():
        raise ValueError(f"work-dir must not exist: {work_dir}")
    if work_dir == Path("/") or work_dir == Path.home().resolve():
        raise ValueError("work-dir must be a disposable subdirectory, not root or home")
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    _clone_mirror(source, work_dir)
    source_url = _git(work_dir, "config", "--get", "remote.origin.url")
    safe_source_url = redact_remote_url(source_url)
    if safe_source_url != source_url:
        _run(["git", "remote", "set-url", "origin", safe_source_url], cwd=work_dir)
    refs_before = _refs(work_dir)
    forbidden_before = find_forbidden_paths(work_dir)
    if not forbidden_before:
        raise ValueError("source history contains none of the configured forbidden paths")

    bundle_path = work_dir.parent / f"{work_dir.name}-before-filter.bundle"
    _run(["git", "bundle", "create", str(bundle_path), "--all"], cwd=work_dir)
    _run(_filter_repo_command(), cwd=work_dir)

    forbidden_after = find_forbidden_paths(work_dir)
    if forbidden_after:
        raise ValueError("forbidden paths remain after filtering: " + ", ".join(forbidden_after))
    refs_after = _refs(work_dir)
    if set(refs_before) != set(refs_after):
        raise ValueError("head/tag ref names changed unexpectedly during filtering")
    _run(["git", "fsck", "--full", "--no-dangling"], cwd=work_dir)

    output_report = report_path or work_dir.parent / f"{work_dir.name}-cleanup-report.json"
    report = {
        "schema_version": "history-cleanup-report/v1",
        "source": redact_remote_url(source),
        "source_url": safe_source_url,
        "filtered_mirror": str(work_dir),
        "backup_bundle": str(bundle_path),
        "backup_bundle_sha256": _sha256(bundle_path),
        "forbidden_paths": list(FORBIDDEN_PATHS),
        "forbidden_objects_before": forbidden_before,
        "forbidden_objects_after": forbidden_after,
        "refs_before": refs_before,
        "refs_after": refs_after,
        "push_commands": build_push_commands(work_dir, safe_source_url),
        "pushed": False,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_report.with_suffix(output_report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_report)
    report["report_path"] = str(output_report)
    return report


def find_forbidden_paths(repo: Path) -> list[str]:
    """Return forbidden paths still reachable from any head or tag."""

    output = _git(repo, "rev-list", "--objects", "--branches", "--tags")
    matches: set[str] = set()
    for line in output.splitlines():
        _object_id, separator, path = line.partition(" ")
        if not separator:
            continue
        normalized = path.casefold()
        if any(
            normalized == forbidden.rstrip("/").casefold()
            or normalized.startswith(forbidden.casefold())
            for forbidden in FORBIDDEN_PATHS
        ):
            matches.add(path)
    return sorted(matches)


def build_push_commands(repo: Path, source_url: str) -> list[str]:
    """Return reviewable commands; do not execute them automatically."""

    quoted_repo = _shell_quote(str(repo))
    quoted_url = _shell_quote(redact_remote_url(source_url))
    return [
        f"git -C {quoted_repo} push --force --all {quoted_url}",
        f"git -C {quoted_repo} push --force --tags {quoted_url}",
    ]


def redact_remote_url(value: str) -> str:
    """Remove URL userinfo and query credentials from a reportable remote."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1", value)
    if not parsed.scheme or not parsed.netloc:
        return value

    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    username = parsed.username if parsed.scheme == "ssh" and parsed.password is None else None
    userinfo = f"{username}@" if username else ""
    safe_netloc = f"{userinfo}{hostname}{port}"
    return urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))


def redact_sensitive_text(text: str, *, sensitive_values: tuple[str, ...] = ()) -> str:
    """Redact credential-bearing remotes in diagnostics before printing."""

    safe = text
    for value in sensitive_values:
        if value:
            safe = safe.replace(value, redact_remote_url(value))
    url_pattern = re.compile(r"(?i)(?:https?|ssh|git)://[^\s'\"\]]+")
    return url_pattern.sub(lambda match: redact_remote_url(match.group(0)), safe)


def _clone_mirror(source: str, work_dir: Path) -> None:
    result = subprocess.run(
        ["git", "clone", "--mirror", source, str(work_dir)],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        details = redact_sensitive_text(
            (result.stderr or result.stdout or "git clone failed").strip(),
            sensitive_values=(source,),
        )
        raise ValueError(details)


def _filter_repo_command() -> list[str]:
    arguments = [
        "git-filter-repo",
        "--force",
        "--sensitive-data-removal",
        "--invert-paths",
        "--path",
        "outputs/handoff/",
        "--path",
        "scripts/export_handoff_bundle.sh",
    ]
    executable = shutil.which("git-filter-repo")
    if executable:
        return [executable, *arguments[1:]]
    uvx = shutil.which("uvx")
    if uvx:
        return [
            uvx,
            "--from",
            f"git-filter-repo=={FILTER_REPO_VERSION}",
            *arguments,
        ]
    raise ValueError(
        "git-filter-repo is unavailable; install uv and run "
        f"uvx --from git-filter-repo=={FILTER_REPO_VERSION} git-filter-repo --help"
    )


def _refs(repo: Path) -> dict[str, str]:
    output = _git(
        repo,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/heads",
        "refs/tags",
    )
    return {
        ref: object_id for line in output.splitlines() for ref, object_id in [line.split(" ", 1)]
    }


def _git(repo: Path, *arguments: str) -> str:
    result = _run(["git", *arguments], cwd=repo, capture=True)
    return result.stdout.strip()


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
