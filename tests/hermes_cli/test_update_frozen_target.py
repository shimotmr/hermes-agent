from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli import update_cmd


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_capture_fetched_target_is_immutable_when_tracking_ref_advances(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Updater Test")
    git(repo, "config", "user.email", "updater@example.invalid")
    (repo / "file.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", "file.txt")
    git(repo, "commit", "-m", "a")
    reviewed = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "upstream", reviewed)

    frozen = update_cmd._capture_fetched_target_sha(["git"], repo, "upstream")
    (repo / "file.txt").write_text("b\n", encoding="utf-8")
    git(repo, "commit", "-am", "b")
    git(repo, "branch", "-f", "upstream", "HEAD")

    assert frozen == reviewed
    assert frozen != git(repo, "rev-parse", "upstream")


def test_capture_fetched_target_rejects_non_sha(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "not-a-sha\n", ""),
    )

    assert update_cmd._capture_fetched_target_sha(["git"], tmp_path, "origin/main") is None
