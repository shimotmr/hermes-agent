from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.release_gate import (
    EvidenceCorrupt,
    append_evidence,
    apply_overlap_to_matrix,
    classify_overlap,
    freeze_snapshot,
    load_acceptance_matrix,
    load_critical_paths,
    validate_evidence,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def commit(repo: Path, path: str, text: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    git(repo, "add", path)
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Release Gate Test")
    git(root, "config", "user.email", "release-gate@example.invalid")
    commit(root, "hermes_cli/update_cmd.py", "base\n", "base")
    return root


def test_freeze_snapshot_is_immutable_after_ref_advances(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    reviewed = commit(repo, "docs/notes.md", "reviewed\n", "reviewed")
    git(repo, "branch", "-f", "upstream", reviewed)

    snapshot = freeze_snapshot(repo, "upstream", local_base_sha=base)
    later = commit(repo, "docs/later.md", "later\n", "later")
    git(repo, "branch", "-f", "upstream", later)

    assert snapshot.target_sha == reviewed
    assert snapshot.target_sha != git(repo, "rev-parse", "upstream")
    assert snapshot.changed_paths == ("docs/notes.md",)
    assert len(snapshot.snapshot_id) == 64


def test_docs_only_drift_is_disjoint(repo: Path) -> None:
    reviewed = git(repo, "rev-parse", "HEAD")
    latest = commit(repo, "docs/readme.md", "docs\n", "docs")

    report = classify_overlap(
        repo,
        reviewed,
        latest,
        critical_paths=("hermes_cli/", "gateway/"),
    )

    assert report.classification == "disjoint"
    assert report.drift_paths == ("docs/readme.md",)
    assert report.overlapping_paths == ()


def test_critical_drift_requires_full_review(repo: Path) -> None:
    reviewed = git(repo, "rev-parse", "HEAD")
    latest = commit(repo, "hermes_cli/update_cmd.py", "changed\n", "critical")

    report = classify_overlap(repo, reviewed, latest, critical_paths=("hermes_cli/",))

    assert report.classification == "overlap"
    assert report.overlapping_paths == ("hermes_cli/update_cmd.py",)


def test_rename_across_critical_boundary_includes_old_and_new_paths(repo: Path) -> None:
    commit(repo, "docs/move-me.py", "x\n", "source")
    reviewed = git(repo, "rev-parse", "HEAD")
    (repo / "hermes_cli").mkdir(exist_ok=True)
    git(repo, "mv", "docs/move-me.py", "hermes_cli/moved.py")
    git(repo, "commit", "-m", "rename")
    latest = git(repo, "rev-parse", "HEAD")

    report = classify_overlap(repo, reviewed, latest, critical_paths=("hermes_cli/",))

    assert report.classification == "overlap"
    assert report.drift_paths == ("docs/move-me.py", "hermes_cli/moved.py")


def test_git_failure_is_indeterminate_not_disjoint(repo: Path) -> None:
    report = classify_overlap(
        repo,
        "0" * 40,
        "f" * 40,
        critical_paths=("hermes_cli/",),
    )

    assert report.classification == "indeterminate"
    assert report.reason.startswith("git-diff-failed:")


@pytest.mark.parametrize(
    "paths",
    [[], ["/absolute"], ["../escape"], ["a//b"], ["a/./b"], ["a\\b"]],
)
def test_critical_path_policy_rejects_unsafe_entries(tmp_path: Path, paths: list[str]) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"schema_version": "hermes.update.critical-paths.v1", "paths": paths}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_critical_paths(policy)


def test_acceptance_matrix_invalidates_only_overlapping_criteria(
    repo: Path, tmp_path: Path
) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "hermes.update.acceptance-matrix.v1",
                "criteria": [
                    {
                        "id": "gateway-restart",
                        "required": True,
                        "path_rules": ["gateway/", "hermes_cli/"],
                        "status": "passed",
                    },
                    {
                        "id": "docs",
                        "required": False,
                        "path_rules": ["docs/"],
                        "status": "passed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    reviewed = git(repo, "rev-parse", "HEAD")
    latest = commit(repo, "hermes_cli/update_cmd.py", "changed\n", "critical")
    report = classify_overlap(
        repo, reviewed, latest, critical_paths=("gateway/", "hermes_cli/")
    )

    result = apply_overlap_to_matrix(load_acceptance_matrix(matrix_path), report)

    assert [criterion["status"] for criterion in result["criteria"]] == [
        "invalidated",
        "passed",
    ]
    assert result["criteria"][0]["invalidation_reason"] == "path-overlap"


def test_indeterminate_overlap_invalidates_every_required_criterion(
    repo: Path, tmp_path: Path
) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "hermes.update.acceptance-matrix.v1",
                "criteria": [
                    {
                        "id": "required-check",
                        "required": True,
                        "path_rules": ["gateway/"],
                        "status": "passed",
                    },
                    {
                        "id": "optional-check",
                        "required": False,
                        "path_rules": ["docs/"],
                        "status": "passed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = classify_overlap(
        repo, "0" * 40, "f" * 40, critical_paths=("gateway/",)
    )

    result = apply_overlap_to_matrix(load_acceptance_matrix(matrix_path), report)

    assert [criterion["status"] for criterion in result["criteria"]] == [
        "invalidated",
        "passed",
    ]
    assert result["criteria"][0]["invalidation_reason"] == "overlap-indeterminate"


def test_evidence_append_is_prefix_preserving_and_hash_chained(tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.jsonl"
    first = append_evidence(
        manifest,
        frozen_sha="a" * 40,
        candidate_sha="b" * 40,
        command="pytest -q tests/unit",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    prefix = manifest.read_bytes()
    second = append_evidence(
        manifest,
        frozen_sha="a" * 40,
        candidate_sha="b" * 40,
        command="ruff check .",
        exit_code=0,
        recorded_at="2026-08-30T04:01:00Z",
    )

    assert manifest.read_bytes().startswith(prefix)
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_event_hash"] == first["event_hash"]
    assert validate_evidence(manifest) == [first, second]


def test_evidence_tampering_is_detected_without_rewriting(tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.jsonl"
    append_evidence(
        manifest,
        frozen_sha="a" * 40,
        candidate_sha="b" * 40,
        command="pytest -q",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    original = manifest.read_bytes()
    manifest.write_bytes(original.replace(b"pytest -q", b"pytest -x"))
    tampered = manifest.read_bytes()

    with pytest.raises(EvidenceCorrupt, match="event-hash-mismatch"):
        validate_evidence(manifest)

    assert manifest.read_bytes() == tampered


def test_truncated_tail_is_detected(tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.jsonl"
    append_evidence(
        manifest,
        frozen_sha="a" * 40,
        candidate_sha="b" * 40,
        command="pytest -q",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    with manifest.open("ab") as handle:
        handle.write(b'{"sequence":2')

    with pytest.raises(EvidenceCorrupt, match="invalid-json-line:2"):
        validate_evidence(manifest)
