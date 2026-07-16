"""Tests for archive-aware Git history safety checks."""

from __future__ import annotations

import io
import tarfile


def test_archive_inspection_counts_secrets_without_returning_values():
    from scripts.audit_git_history import inspect_archive_bytes

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = b"credential=AKIAABCDEFGHIJKLMNOP"  # gitleaks:allow
        info = tarfile.TarInfo("chatSessions/session.jsonl")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    secret_matches, sensitive_members = inspect_archive_bytes(buffer.getvalue())

    assert secret_matches == 1
    assert sensitive_members == 1
