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


def test_target_checkout_does_not_overwrite_branch_advanced_after_probe(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Updater Test")
    git(repo, "config", "user.email", "updater@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "base.txt")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--detach", base)
    (repo / "remote.txt").write_text("remote\n", encoding="utf-8")
    git(repo, "add", "remote.txt")
    git(repo, "commit", "-m", "frozen remote commit")
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-b", "parked", base)

    real_run = subprocess.run
    raced_tip: list[str] = []

    def run_with_branch_race(cmd, *args, **kwargs):
        result = real_run(cmd, *args, **kwargs)
        if cmd[-3:] == ["rev-list", "--count", f"{frozen}..refs/heads/main"]:
            tree = git(repo, "rev-parse", "main^{tree}")
            local_tip = real_run(
                ["git", "commit-tree", tree, "-p", "main", "-m", "raced local commit"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            git(repo, "update-ref", "refs/heads/main", local_tip, base)
            raced_tip.append(local_tip)
        return result

    monkeypatch.setattr(update_cmd.subprocess, "run", run_with_branch_race)

    result = update_cmd._checkout_target_at_frozen(["git"], repo, "main", frozen)

    assert result.returncode == 0
    assert raced_tip
    assert git(repo, "rev-parse", "main") == raced_tip[0]
    assert git(repo, "branch", "--show-current") == "main"


def test_destructive_reset_refuses_when_head_advanced_after_observation(
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
    observed = git(repo, "rev-parse", "HEAD")
    (repo / "remote.txt").write_text("remote\n", encoding="utf-8")
    git(repo, "add", "remote.txt")
    git(repo, "commit", "-m", "remote")
    target = git(repo, "rev-parse", "HEAD")
    git(repo, "reset", "--hard", observed)
    (repo / "local.txt").write_text("local\n", encoding="utf-8")
    git(repo, "add", "local.txt")
    git(repo, "commit", "-m", "raced local")
    raced = git(repo, "rev-parse", "HEAD")

    result = update_cmd._reset_hard_if_head_matches(
        ["git"], repo, target, expected_head=observed
    )

    assert result.returncode != 0
    assert git(repo, "rev-parse", "HEAD") == raced
    assert (repo / "local.txt").read_text() == "local\n"


def _reset_race_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Updater Test")
    git(repo, "config", "user.email", "updater@example.invalid")
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-m", "base")
    observed = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--detach", observed)
    tracked.write_text("target\n", encoding="utf-8")
    git(repo, "commit", "-am", "target")
    target = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "main")
    return repo, tracked, observed, target


def test_destructive_reset_preserves_edit_created_when_ref_cas_fails(
    monkeypatch, tmp_path: Path
) -> None:
    repo, tracked, observed, target = _reset_race_repo(tmp_path)
    real_run = subprocess.run

    def fail_ref_cas(cmd, *args, **kwargs):
        if "update-ref" in cmd:
            tracked.write_text("raced edit\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 1, "", "ref raced")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(update_cmd.subprocess, "run", fail_ref_cas)

    result = update_cmd._reset_hard_if_head_matches(
        ["git"], repo, target, expected_head=observed
    )

    assert result.returncode != 0
    assert tracked.read_text(encoding="utf-8") == "raced edit\n"
    assert git(repo, "rev-parse", "main") == observed


def test_destructive_reset_preserves_edit_racing_successful_ref_cas(
    monkeypatch, tmp_path: Path
) -> None:
    repo, tracked, observed, target = _reset_race_repo(tmp_path)
    real_run = subprocess.run

    def race_after_ref_cas(cmd, *args, **kwargs):
        result = real_run(cmd, *args, **kwargs)
        if "update-ref" in cmd and result.returncode == 0:
            tracked.write_text("raced edit\n", encoding="utf-8")
        return result

    monkeypatch.setattr(update_cmd.subprocess, "run", race_after_ref_cas)

    result = update_cmd._reset_hard_if_head_matches(
        ["git"], repo, target, expected_head=observed
    )

    assert result.returncode != 0
    assert tracked.read_text(encoding="utf-8") == "raced edit\n"
