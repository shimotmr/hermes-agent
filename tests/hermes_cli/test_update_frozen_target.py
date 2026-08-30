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


def test_parked_branch_probe_uses_frozen_target_after_tracking_ref_advances(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Updater Test")
    git(repo, "config", "user.email", "updater@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "base.txt")
    git(repo, "commit", "-m", "base")
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-b", "parked")
    (repo / "feature.txt").write_text("same patch\n", encoding="utf-8")
    git(repo, "add", "feature.txt")
    git(repo, "commit", "-m", "local feature")
    git(repo, "checkout", "main")
    (repo / "feature.txt").write_text("same patch\n", encoding="utf-8")
    git(repo, "add", "feature.txt")
    git(repo, "commit", "-m", "later equivalent upstream patch")
    git(repo, "branch", "-f", "origin/main", "HEAD")
    git(repo, "checkout", "parked")

    safe, reason = update_cmd._assess_parked_branch_switch(
        ["git"], repo, "parked", "main", frozen_target_sha=frozen
    )

    assert safe is True
    assert reason == "unmerged:1"


def test_target_checkout_anchors_frozen_sha_without_discarding_local_commits(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Updater Test")
    git(repo, "config", "user.email", "updater@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "base.txt")
    git(repo, "commit", "-m", "base")
    git(repo, "branch", "parked")
    (repo / "local.txt").write_text("local\n", encoding="utf-8")
    git(repo, "add", "local.txt")
    git(repo, "commit", "-m", "local-only main commit")
    local_tip = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--detach", "HEAD~1")
    (repo / "remote.txt").write_text("remote\n", encoding="utf-8")
    git(repo, "add", "remote.txt")
    git(repo, "commit", "-m", "frozen remote commit")
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "parked")

    result = update_cmd._checkout_target_at_frozen(["git"], repo, "main", frozen)

    assert result.returncode == 0
    assert git(repo, "branch", "--show-current") == "main"
    assert git(repo, "rev-parse", "main") == local_tip
    assert git(repo, "merge-base", "--is-ancestor", local_tip, "main") == ""
    assert git(repo, "rev-list", "--count", f"{frozen}..main") == "1"
