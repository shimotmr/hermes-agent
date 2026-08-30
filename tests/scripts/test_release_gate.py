from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.release_gate import (
    EvidenceCorrupt,
    REQUIRED_RELEASE_GATE_IDS,
    append_evidence,
    apply_overlap_to_matrix,
    classify_overlap,
    freeze_snapshot,
    load_acceptance_matrix,
    load_critical_paths,
    main,
    probe_merge_tree,
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


def authority(repo: Path, frozen: str, candidate: str) -> dict[str, str]:
    return {
        "expected_candidate_sha": candidate,
        "expected_frozen_sha": frozen,
        "expected_candidate_tree_sha": git(repo, "rev-parse", f"{candidate}^{{tree}}"),
    }


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


def test_freeze_snapshot_resolves_symbolic_base_to_commit(repo: Path) -> None:
    expected = git(repo, "rev-parse", "HEAD")

    snapshot = freeze_snapshot(repo, "HEAD", local_base_sha="HEAD")

    assert snapshot.local_base_sha == expected
    assert snapshot.target_sha == expected


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


def test_critical_policy_covers_every_required_acceptance_path() -> None:
    matrix = load_acceptance_matrix(Path("docs/update-acceptance-matrix.json"))
    critical_paths = load_critical_paths(Path("docs/release-gate-critical-paths.json"))

    required_rules = {
        rule
        for criterion in matrix["criteria"]
        if criterion.get("required") is True
        for rule in criterion["path_rules"]
    }

    uncovered = {
        rule
        for rule in required_rules
        if not any(rule == critical or rule.startswith(critical) for critical in critical_paths)
    }
    assert uncovered == set()


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


def test_critical_overlap_invalidates_entire_acceptance_matrix(
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
        "invalidated",
    ]
    assert {criterion["invalidation_reason"] for criterion in result["criteria"]} == {
        "critical-path-overlap"
    }


def test_indeterminate_overlap_invalidates_entire_acceptance_matrix(
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
        "invalidated",
    ]
    assert result["criteria"][0]["invalidation_reason"] == "overlap-indeterminate"


@pytest.mark.parametrize("classification_sha", ["overlap", "indeterminate"])
def test_blocking_classification_returns_nonzero(
    repo: Path, tmp_path: Path, classification_sha: str
) -> None:
    reviewed = git(repo, "rev-parse", "HEAD")
    latest = (
        commit(repo, "hermes_cli/update_cmd.py", "changed\n", "critical")
        if classification_sha == "overlap"
        else "f" * 40
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "hermes.update.critical-paths.v1",
                "paths": ["hermes_cli/"],
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "classify",
            "--repo",
            str(repo),
            "--from-sha",
            reviewed,
            "--latest-sha",
            latest,
            "--policy",
            str(policy),
        ]
    ) != 0


def test_merge_tree_probe_detects_conflict_without_touching_checkout(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-b", "candidate")
    candidate = commit(repo, "shared.txt", "candidate\n", "candidate")
    git(repo, "checkout", "main")
    current = commit(repo, "shared.txt", "main\n", "main")
    status_before = git(repo, "status", "--porcelain=v1")
    head_before = git(repo, "rev-parse", "HEAD")

    result = probe_merge_tree(repo, current, candidate)

    assert result.classification == "conflict"
    assert git(repo, "status", "--porcelain=v1") == status_before
    assert git(repo, "rev-parse", "HEAD") == head_before


def test_evidence_append_binds_real_commits_snapshot_trees_and_ancestry(
    repo: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    first = append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        command="pytest -q tests/unit",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    prefix = manifest.read_bytes()
    second = append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        command="ruff check .",
        exit_code=0,
        recorded_at="2026-08-30T04:01:00Z",
    )

    assert manifest.read_bytes().startswith(prefix)
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_event_hash"] == first["event_hash"]
    assert first["snapshot_id"] == snapshot.snapshot_id
    assert first["frozen_tree_sha"] == git(repo, "rev-parse", f"{frozen}^{{tree}}")
    assert first["candidate_tree_sha"] == git(repo, "rev-parse", f"{candidate}^{{tree}}")
    assert validate_evidence(manifest, repo=repo) == [first, second]


@pytest.mark.require_symlinks
def test_evidence_append_rejects_symlink_manifest(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text("do not append\n", encoding="utf-8")
    manifest = tmp_path / "evidence.jsonl"
    manifest.symlink_to(outside)
    frozen = git(repo, "rev-parse", "HEAD")
    snapshot = freeze_snapshot(repo, "HEAD", local_base_sha=frozen)

    with pytest.raises(ValueError, match="evidence-target-symlink"):
        append_evidence(
            manifest,
            repo=repo,
            snapshot=snapshot,
            frozen_sha=frozen,
            candidate_sha=frozen,
            command="tests",
            exit_code=0,
            recorded_at="2026-08-30T04:00:00Z",
        )

    assert outside.read_text(encoding="utf-8") == "do not append\n"


def test_evidence_append_rejects_non_regular_manifest(repo: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.jsonl"
    manifest.mkdir()
    frozen = git(repo, "rev-parse", "HEAD")
    snapshot = freeze_snapshot(repo, "HEAD", local_base_sha=frozen)

    with pytest.raises(ValueError, match="evidence-target-not-regular"):
        append_evidence(
            manifest,
            repo=repo,
            snapshot=snapshot,
            frozen_sha=frozen,
            candidate_sha=frozen,
            command="tests",
            exit_code=0,
            recorded_at="2026-08-30T04:00:00Z",
        )


def test_release_evidence_rejects_a_failed_required_gate(repo: Path, tmp_path: Path) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        gate_id="tests",
        command="tests",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        gate_id="ruff",
        command="ruff",
        exit_code=1,
        recorded_at="2026-08-30T04:01:00Z",
    )

    with pytest.raises(EvidenceCorrupt, match="required-gate-failed:ruff"):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("tests", "ruff"),
            **authority(repo, frozen, candidate),
        )


def test_release_evidence_rejects_missing_gate_and_mixed_authority(
    repo: Path, tmp_path: Path
) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    first_candidate = commit(repo, "candidate-1.txt", "one\n", "candidate one")
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=first_candidate)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=first_candidate,
        gate_id="tests",
        command="tests",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )

    with pytest.raises(EvidenceCorrupt, match="required-gates-missing"):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("tests", "ruff"),
            **authority(repo, frozen, first_candidate),
        )

    second_candidate = commit(repo, "candidate-2.txt", "two\n", "candidate two")
    second_snapshot = freeze_snapshot(repo, "frozen", local_base_sha=second_candidate)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=second_snapshot,
        frozen_sha=frozen,
        candidate_sha=second_candidate,
        gate_id="ruff",
        command="ruff",
        exit_code=0,
        recorded_at="2026-08-30T04:01:00Z",
    )

    with pytest.raises(EvidenceCorrupt, match="release-authority-mixed"):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("tests", "ruff"),
            **authority(repo, frozen, first_candidate),
        )


def test_evidence_tampering_is_detected_without_rewriting(repo: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        command="pytest -q",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    original = manifest.read_bytes()
    manifest.write_bytes(original.replace(b"pytest -q", b"pytest -x"))
    tampered = manifest.read_bytes()

    with pytest.raises(EvidenceCorrupt, match="event-hash-mismatch"):
        validate_evidence(manifest, repo=repo)

    assert manifest.read_bytes() == tampered


def test_truncated_tail_is_detected(repo: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=frozen)
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        command="pytest -q",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )
    with manifest.open("ab") as handle:
        handle.write(b'{"sequence":2')

    with pytest.raises(EvidenceCorrupt, match="invalid-json-line:2"):
        validate_evidence(manifest, repo=repo)


def test_evidence_rejects_non_commit_and_non_descendant_candidate(
    repo: Path, tmp_path: Path
) -> None:
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=frozen)
    git(repo, "checkout", "--orphan", "unrelated")
    git(repo, "rm", "-rf", ".")
    unrelated = commit(repo, "other.txt", "other\n", "unrelated")

    with pytest.raises(ValueError, match="candidate-not-descendant"):
        append_evidence(
            tmp_path / "evidence.jsonl",
            repo=repo,
            snapshot=snapshot,
            frozen_sha=frozen,
            candidate_sha=unrelated,
            command="pytest -q",
            exit_code=0,
            recorded_at="2026-08-30T04:00:00Z",
        )


def test_evidence_rejects_snapshot_tree_not_bound_to_frozen_commit(
    repo: Path, tmp_path: Path
) -> None:
    frozen = git(repo, "rev-parse", "HEAD")
    later = commit(repo, "later.txt", "later\n", "later")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=frozen)
    forged = snapshot.__class__(
        **{
            **snapshot.__dict__,
            "target_tree_sha": git(repo, "rev-parse", f"{later}^{{tree}}"),
        }
    )

    with pytest.raises(ValueError, match="snapshot-tree-mismatch"):
        append_evidence(
            tmp_path / "evidence.jsonl",
            repo=repo,
            snapshot=forged,
            frozen_sha=frozen,
            candidate_sha=later,
            command="pytest -q",
            exit_code=0,
            recorded_at="2026-08-30T04:00:00Z",
        )


def test_release_gate_imports_when_fcntl_is_unavailable() -> None:
    result = subprocess.run(
        [
            "python",
            "-c",
            "import sys; sys.modules['fcntl']=None; import scripts.release_gate",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_windows_evidence_lock_uses_sidecar_and_never_sentinels_manifest(
    tmp_path: Path,
) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    lock_path = release_gate._evidence_lock_path(manifest, is_windows=True)

    assert lock_path != manifest
    assert lock_path == tmp_path / "evidence.jsonl.lock"
    assert not manifest.exists()


def test_windows_first_append_does_not_truncate_writer_that_won_lock(
) -> None:
    from scripts import release_gate

    class LockBuffer(io.BytesIO):
        def fileno(self) -> int:
            return 123

    handle = LockBuffer()
    lock_calls = 0

    def locking(_fd: int, mode: int, _length: int) -> None:
        nonlocal lock_calls
        if mode == 1:
            lock_calls += 1
            handle.seek(0)
            handle.write(b"first-writer-event\n")
            handle.truncate()

    windows_lock = SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=locking)

    with release_gate._exclusive_file_lock(
        handle, is_windows=True, windows_lock_module=windows_lock
    ):
        handle.seek(0)
        assert handle.read() == b"first-writer-event\n"

    assert lock_calls == 1


def test_overlap_classification_rejects_identical_nonexistent_commits(
    repo: Path,
) -> None:
    missing = "f" * 40

    report = classify_overlap(
        repo,
        missing,
        missing,
        critical_paths=("hermes_cli/update_cmd.py",),
    )

    assert report.classification == "indeterminate"
    assert "git-" in report.reason


def test_evidence_append_works_without_fchmod(monkeypatch, repo: Path, tmp_path: Path) -> None:
    from scripts import release_gate

    frozen = git(repo, "rev-parse", "HEAD")
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    monkeypatch.delattr(release_gate.os, "fchmod")

    event = append_evidence(
        tmp_path / "evidence.jsonl",
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        gate_id="freeze",
        command="freeze",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )

    assert event["sequence"] == 1


def test_release_validation_cannot_be_weakened_to_one_caller_gate(
    repo: Path, tmp_path: Path
) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        gate_id="freeze",
        command="freeze",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )

    with pytest.raises(EvidenceCorrupt, match="required-gates-missing"):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("freeze",),
            **authority(repo, frozen, candidate),
        )


def test_release_validation_recomputes_snapshot_from_candidate(
    repo: Path, tmp_path: Path
) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "frozen", frozen)
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    wrong_base_snapshot = freeze_snapshot(repo, "frozen", local_base_sha=frozen)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=wrong_base_snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        gate_id="freeze",
        command="freeze",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )

    with pytest.raises(EvidenceCorrupt, match="snapshot-local-base-mismatch"):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("freeze",),
            **authority(repo, frozen, candidate),
        )


@pytest.mark.parametrize(
    "recorded_at",
    ["not-a-timestamp", "2026-08-30T04:00:00", "2026-08-30", ""],
)
def test_evidence_rejects_non_timezone_aware_iso_timestamp(
    repo: Path, tmp_path: Path, recorded_at: str
) -> None:
    frozen = git(repo, "rev-parse", "HEAD")
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)

    with pytest.raises(ValueError, match="evidence-recorded-at-invalid"):
        append_evidence(
            tmp_path / "evidence.jsonl",
            repo=repo,
            snapshot=snapshot,
            frozen_sha=frozen,
            candidate_sha=candidate,
            gate_id="freeze",
            command="freeze",
            exit_code=0,
            recorded_at=recorded_at,
        )


def test_release_validation_is_bound_to_caller_expected_authority(
    repo: Path, tmp_path: Path
) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    append_evidence(
        manifest,
        repo=repo,
        snapshot=snapshot,
        frozen_sha=frozen,
        candidate_sha=candidate,
        gate_id="freeze",
        command="freeze",
        exit_code=0,
        recorded_at="2026-08-30T04:00:00Z",
    )

    with pytest.raises(EvidenceCorrupt, match="release-candidate-mismatch"):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("freeze",),
            expected_candidate_sha="f" * 40,
            expected_frozen_sha=frozen,
            expected_candidate_tree_sha=git(repo, "rev-parse", f"{candidate}^{{tree}}"),
        )


def test_overlap_authority_requires_overlap_compatibility_gate(
    repo: Path, tmp_path: Path
) -> None:
    from scripts import release_gate

    manifest = tmp_path / "evidence.jsonl"
    frozen = git(repo, "rev-parse", "HEAD")
    candidate = commit(repo, "candidate.txt", "candidate\n", "candidate")
    git(repo, "branch", "frozen", frozen)
    snapshot = freeze_snapshot(repo, "frozen", local_base_sha=candidate)
    overlap = classify_overlap(
        repo, frozen, candidate, critical_paths=("candidate.txt",)
    )
    for gate_id in REQUIRED_RELEASE_GATE_IDS:
        append_evidence(
            manifest,
            repo=repo,
            snapshot=snapshot,
            frozen_sha=frozen,
            candidate_sha=candidate,
            gate_id=gate_id,
            command=gate_id,
            exit_code=0,
            recorded_at="2026-08-30T04:00:00Z",
            overlap_report=overlap if gate_id == "freeze" else None,
            overlap_critical_paths=("candidate.txt",) if gate_id == "freeze" else (),
        )

    with pytest.raises(
        EvidenceCorrupt, match="required-gates-missing:overlap-compatibility"
    ):
        release_gate.validate_release_evidence(
            manifest,
            repo=repo,
            required_gate_ids=("freeze",),
            expected_candidate_sha=candidate,
            expected_frozen_sha=frozen,
            expected_candidate_tree_sha=git(repo, "rev-parse", f"{candidate}^{{tree}}"),
        )
