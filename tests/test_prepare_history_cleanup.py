"""Unit tests for the non-pushing history cleanup preparer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_push_commands_update_only_heads_and_tags(tmp_path):
    from scripts.prepare_history_cleanup import build_push_commands

    commands = build_push_commands(tmp_path / "filtered.git", "git@example/repo.git")

    assert commands == [
        f"git -C '{tmp_path / 'filtered.git'}' push --force --all 'git@example/repo.git'",
        f"git -C '{tmp_path / 'filtered.git'}' push --force --tags 'git@example/repo.git'",
    ]
    assert all("--mirror" not in command for command in commands)


def test_cleanup_reports_and_push_commands_strip_remote_credentials(tmp_path):
    from scripts.prepare_history_cleanup import (
        build_push_commands,
        redact_remote_url,
        redact_sensitive_text,
    )

    remote = "https://alice:ghp_secret@example.com/org/repo.git?access_token=also-secret"
    safe = "https://example.com/org/repo.git"

    assert redact_remote_url(remote) == safe
    assert build_push_commands(tmp_path / "filtered.git", remote) == [
        f"git -C '{tmp_path / 'filtered.git'}' push --force --all '{safe}'",
        f"git -C '{tmp_path / 'filtered.git'}' push --force --tags '{safe}'",
    ]
    diagnostic = redact_sensitive_text(f"clone failed for {remote}", sensitive_values=(remote,))
    assert diagnostic == f"clone failed for {safe}"
    assert "secret" not in diagnostic


def test_cleanup_script_help_works_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_history_cleanup.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--source" in completed.stdout
    assert "--work-dir" in completed.stdout
