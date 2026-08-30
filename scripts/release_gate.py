#!/usr/bin/env python3
"""Freeze update review targets and record append-only evidence.

This tool is intentionally independent from updater/service orchestration. It reads
Git state, classifies path overlap, and writes only the explicitly supplied JSONL
manifest path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence

_SCHEMA_SNAPSHOT = "hermes.update.release-snapshot.v1"
_SCHEMA_OVERLAP = "hermes.update.overlap-report.v1"
_SCHEMA_EVIDENCE = "hermes.update.evidence-event.v2"
_SCHEMA_EVIDENCE_LEGACY = "hermes.update.evidence-event.v1"
_SHA_LEN = 40


class EvidenceCorrupt(ValueError):
    """Raised when an evidence manifest is malformed or hash-invalid."""


@dataclass(frozen=True)
class ReleaseSnapshot:
    schema_version: str
    snapshot_id: str
    remote_ref: str
    local_base_sha: str
    target_sha: str
    target_tree_sha: str
    merge_base_sha: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class OverlapReport:
    schema_version: str
    from_sha: str
    latest_sha: str
    classification: Literal["unchanged", "disjoint", "overlap", "indeterminate"]
    drift_paths: tuple[str, ...]
    overlapping_paths: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class MergeTreeReport:
    schema_version: str
    current_sha: str
    candidate_sha: str
    classification: Literal["clean", "conflict", "indeterminate"]
    tree_sha: str | None
    reason: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA_LEN
        and all(character in "0123456789abcdef" for character in value)
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
    )


def _git_text(repo: Path, *args: str) -> str:
    result = _git(repo, *args)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ValueError(f"git-{'-'.join(args[:2])}-failed:{detail or result.returncode}")
    return result.stdout.decode("utf-8", "replace").strip()


@contextmanager
def _exclusive_file_lock(handle):
    """Serialize appends with the host's native advisory file lock."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        was_empty = handle.tell() == 0
        if was_empty:
            handle.write(b"\n")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        if was_empty:
            handle.seek(0)
            handle.truncate(0)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _changed_paths(repo: Path, old_sha: str, new_sha: str) -> tuple[str, ...]:
    result = _git(repo, "diff", "--name-status", "-z", old_sha, new_sha)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ValueError(f"git-diff-failed:{detail or result.returncode}")
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii", "replace")
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        if index + path_count > len(fields):
            raise ValueError("git-diff-failed:malformed-name-status")
        for raw_path in fields[index : index + path_count]:
            paths.append(raw_path.decode("utf-8", "surrogateescape"))
        index += path_count
    return tuple(sorted(set(paths)))


def freeze_snapshot(
    repo: Path,
    remote_ref: str,
    *,
    local_base_sha: str | None = None,
) -> ReleaseSnapshot:
    """Resolve a mutable ref once and return a content-addressed snapshot."""
    repo = Path(repo)
    base_ref = local_base_sha or "HEAD"
    base = _git_text(repo, "rev-parse", f"{base_ref}^{{commit}}")
    target = _git_text(repo, "rev-parse", f"{remote_ref}^{{commit}}")
    if not _is_sha(base) or not _is_sha(target):
        raise ValueError("snapshot-sha-invalid")
    tree = _git_text(repo, "rev-parse", f"{target}^{{tree}}")
    merge_base = _git_text(repo, "merge-base", base, target)
    paths = _changed_paths(repo, base, target)
    body = {
        "schema_version": _SCHEMA_SNAPSHOT,
        "remote_ref": remote_ref,
        "local_base_sha": base,
        "target_sha": target,
        "target_tree_sha": tree,
        "merge_base_sha": merge_base,
        "changed_paths": list(paths),
    }
    snapshot_id = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return ReleaseSnapshot(
        schema_version=_SCHEMA_SNAPSHOT,
        snapshot_id=snapshot_id,
        remote_ref=remote_ref,
        local_base_sha=base,
        target_sha=target,
        target_tree_sha=tree,
        merge_base_sha=merge_base,
        changed_paths=paths,
    )


def _validate_critical_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("critical-path-invalid")
    if "\\" in value or "//" in value:
        raise ValueError("critical-path-invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError("critical-path-invalid")
    normalized = pure.as_posix()
    if value.endswith("/"):
        normalized += "/"
    if normalized != value:
        raise ValueError("critical-path-not-normalized")
    return value


def load_critical_paths(path: Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("critical-path-policy-invalid")
    if payload.get("schema_version") != "hermes.update.critical-paths.v1":
        raise ValueError("critical-path-schema-invalid")
    entries = payload.get("paths")
    if not isinstance(entries, list) or not entries:
        raise ValueError("critical-paths-empty")
    result = tuple(_validate_critical_path(entry) for entry in entries)
    if len(set(result)) != len(result):
        raise ValueError("critical-path-duplicate")
    return result


def _path_matches(path: str, critical: str) -> bool:
    return path.startswith(critical) if critical.endswith("/") else path == critical


def classify_overlap(
    repo: Path,
    from_sha: str,
    latest_sha: str,
    *,
    critical_paths: Sequence[str],
) -> OverlapReport:
    try:
        normalized = tuple(_validate_critical_path(item) for item in critical_paths)
        if not normalized:
            raise ValueError("critical-paths-empty")
        if not _is_sha(from_sha) or not _is_sha(latest_sha):
            raise ValueError("git-diff-failed:sha-invalid")
        if from_sha == latest_sha:
            return OverlapReport(
                _SCHEMA_OVERLAP,
                from_sha,
                latest_sha,
                "unchanged",
                (),
                (),
                "upstream-unchanged",
            )
        drift = _changed_paths(Path(repo), from_sha, latest_sha)
    except (OSError, ValueError) as exc:
        return OverlapReport(
            _SCHEMA_OVERLAP,
            from_sha,
            latest_sha,
            "indeterminate",
            (),
            (),
            str(exc),
        )
    overlapping = tuple(
        path for path in drift if any(_path_matches(path, rule) for rule in normalized)
    )
    classification: Literal["disjoint", "overlap"] = (
        "overlap" if overlapping else "disjoint"
    )
    return OverlapReport(
        _SCHEMA_OVERLAP,
        from_sha,
        latest_sha,
        classification,
        drift,
        overlapping,
        "critical-path-overlap" if overlapping else "critical-paths-disjoint",
    )


def load_acceptance_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("acceptance-matrix-invalid")
    if payload.get("schema_version") != "hermes.update.acceptance-matrix.v1":
        raise ValueError("acceptance-matrix-schema-invalid")
    criteria = payload.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("acceptance-matrix-criteria-empty")
    seen: set[str] = set()
    allowed_statuses = {"pending", "passed", "failed", "invalidated", "not_run"}
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise ValueError("acceptance-criterion-invalid")
        criterion_id = criterion.get("id")
        if not isinstance(criterion_id, str) or not criterion_id or criterion_id in seen:
            raise ValueError("acceptance-criterion-id-invalid")
        seen.add(criterion_id)
        if type(criterion.get("required")) is not bool:
            raise ValueError("acceptance-criterion-required-invalid")
        rules = criterion.get("path_rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("acceptance-criterion-path-rules-empty")
        criterion["path_rules"] = [_validate_critical_path(rule) for rule in rules]
        if criterion.get("status") not in allowed_statuses:
            raise ValueError("acceptance-criterion-status-invalid")
    return payload


def apply_overlap_to_matrix(
    matrix: dict[str, Any], report: OverlapReport
) -> dict[str, Any]:
    """Invalidate the complete matrix when review authority is blocked."""
    result = json.loads(json.dumps(matrix))
    criteria = result.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("acceptance-matrix-criteria-empty")
    if report.classification in {"overlap", "indeterminate"}:
        reason = (
            "critical-path-overlap"
            if report.classification == "overlap"
            else "overlap-indeterminate"
        )
        for criterion in criteria:
            criterion["status"] = "invalidated"
            criterion["invalidation_reason"] = reason
    result["overlap"] = asdict(report)
    return result


def probe_merge_tree(
    repo: Path, current_sha: str, candidate_sha: str
) -> MergeTreeReport:
    """Ask Git to synthesize the merge tree without touching refs or checkout."""
    if not _is_sha(current_sha) or not _is_sha(candidate_sha):
        return MergeTreeReport(
            "hermes.update.merge-tree-report.v1",
            current_sha,
            candidate_sha,
            "indeterminate",
            None,
            "sha-invalid",
        )
    result = _git(Path(repo), "merge-tree", "--write-tree", current_sha, candidate_sha)
    output = result.stdout.decode("utf-8", "replace").splitlines()
    tree = output[0].strip() if output and _is_sha(output[0].strip()) else None
    if result.returncode == 0 and tree is not None:
        classification: Literal["clean", "conflict", "indeterminate"] = "clean"
        reason = "merge-clean"
    elif result.returncode == 1:
        classification = "conflict"
        reason = "merge-conflict"
    else:
        classification = "indeterminate"
        detail = result.stderr.decode("utf-8", "replace").strip()
        reason = f"merge-tree-failed:{detail or result.returncode}"
    return MergeTreeReport(
        "hermes.update.merge-tree-report.v1",
        current_sha,
        candidate_sha,
        classification,
        tree,
        reason,
    )


def _validate_git_binding(
    repo: Path, event: dict[str, Any], expected_sequence: int
) -> None:
    frozen = event["frozen_sha"]
    candidate = event["candidate_sha"]
    try:
        if _git_text(repo, "rev-parse", f"{frozen}^{{commit}}") != frozen:
            raise ValueError("frozen-not-commit")
        if _git_text(repo, "rev-parse", f"{candidate}^{{commit}}") != candidate:
            raise ValueError("candidate-not-commit")
        frozen_tree = _git_text(repo, "rev-parse", f"{frozen}^{{tree}}")
        candidate_tree = _git_text(repo, "rev-parse", f"{candidate}^{{tree}}")
    except ValueError as exc:
        raise EvidenceCorrupt(f"git-object-invalid:{expected_sequence}:{exc}") from exc
    if event.get("frozen_tree_sha") != frozen_tree:
        raise EvidenceCorrupt(f"frozen-tree-mismatch:{expected_sequence}")
    if event.get("candidate_tree_sha") != candidate_tree:
        raise EvidenceCorrupt(f"candidate-tree-mismatch:{expected_sequence}")
    ancestry = _git(repo, "merge-base", "--is-ancestor", frozen, candidate)
    if ancestry.returncode != 0:
        raise EvidenceCorrupt(f"candidate-not-descendant:{expected_sequence}")
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        raise EvidenceCorrupt(f"snapshot-invalid:{expected_sequence}")
    snapshot_id = snapshot.get("snapshot_id")
    body = dict(snapshot)
    body.pop("snapshot_id", None)
    if snapshot_id != hashlib.sha256(_canonical_bytes(body)).hexdigest():
        raise EvidenceCorrupt(f"snapshot-id-mismatch:{expected_sequence}")
    if event.get("snapshot_id") != snapshot_id or snapshot.get("target_sha") != frozen:
        raise EvidenceCorrupt(f"snapshot-binding-mismatch:{expected_sequence}")
    if snapshot.get("target_tree_sha") != frozen_tree:
        raise EvidenceCorrupt(f"snapshot-tree-mismatch:{expected_sequence}")


def _validate_event_shape(
    event: object,
    expected_sequence: int,
    previous_hash: str | None,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise EvidenceCorrupt(f"event-not-object:{expected_sequence}")
    schema = event.get("schema_version")
    if schema not in {_SCHEMA_EVIDENCE, _SCHEMA_EVIDENCE_LEGACY}:
        raise EvidenceCorrupt(f"schema-invalid:{expected_sequence}")
    if event.get("sequence") != expected_sequence:
        raise EvidenceCorrupt(f"sequence-invalid:{expected_sequence}")
    if event.get("previous_event_hash") != previous_hash:
        raise EvidenceCorrupt(f"previous-hash-mismatch:{expected_sequence}")
    for field in ("frozen_sha", "candidate_sha"):
        if not _is_sha(event.get(field)):
            raise EvidenceCorrupt(f"{field}-invalid:{expected_sequence}")
    if not isinstance(event.get("command"), str) or not event["command"]:
        raise EvidenceCorrupt(f"command-invalid:{expected_sequence}")
    if type(event.get("exit_code")) is not int:
        raise EvidenceCorrupt(f"exit-code-invalid:{expected_sequence}")
    if not isinstance(event.get("recorded_at"), str) or not event["recorded_at"]:
        raise EvidenceCorrupt(f"recorded-at-invalid:{expected_sequence}")
    if schema == _SCHEMA_EVIDENCE:
        if repo is None:
            raise EvidenceCorrupt(f"evidence-repo-required:{expected_sequence}")
        _validate_git_binding(repo, event, expected_sequence)
    claimed = event.get("event_hash")
    unsigned = dict(event)
    unsigned.pop("event_hash", None)
    actual = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if claimed != actual:
        raise EvidenceCorrupt(f"event-hash-mismatch:{expected_sequence}")
    return event


def _read_evidence_bytes(raw: bytes, *, repo: Path | None = None) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        line_number = raw.count(b"\n") + 1
        raise EvidenceCorrupt(f"invalid-json-line:{line_number}")
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            parsed = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceCorrupt(f"invalid-json-line:{line_number}") from exc
        event = _validate_event_shape(parsed, line_number, previous_hash, repo=repo)
        events.append(event)
        previous_hash = event["event_hash"]
    return events


def validate_evidence(path: Path, *, repo: Path | None = None) -> list[dict[str, Any]]:
    return _read_evidence_bytes(Path(path).read_bytes(), repo=repo)


def append_evidence(
    path: Path,
    *,
    repo: Path,
    snapshot: ReleaseSnapshot,
    frozen_sha: str,
    candidate_sha: str,
    command: str,
    exit_code: int,
    recorded_at: str,
) -> dict[str, Any]:
    if not _is_sha(frozen_sha) or not _is_sha(candidate_sha):
        raise ValueError("evidence-sha-invalid")
    if not isinstance(command, str) or not command:
        raise ValueError("evidence-command-invalid")
    if type(exit_code) is not int:
        raise ValueError("evidence-exit-code-invalid")
    if not isinstance(recorded_at, str) or not recorded_at:
        raise ValueError("evidence-recorded-at-invalid")
    if snapshot.target_sha != frozen_sha:
        raise ValueError("evidence-snapshot-frozen-mismatch")
    try:
        frozen_commit = _git_text(Path(repo), "rev-parse", f"{frozen_sha}^{{commit}}")
        candidate_commit = _git_text(
            Path(repo), "rev-parse", f"{candidate_sha}^{{commit}}"
        )
    except ValueError as exc:
        raise ValueError("evidence-git-object-invalid") from exc
    if frozen_commit != frozen_sha or candidate_commit != candidate_sha:
        raise ValueError("evidence-git-object-invalid")
    if _git(Path(repo), "merge-base", "--is-ancestor", frozen_sha, candidate_sha).returncode != 0:
        raise ValueError("candidate-not-descendant")
    frozen_tree = _git_text(Path(repo), "rev-parse", f"{frozen_sha}^{{tree}}")
    candidate_tree = _git_text(Path(repo), "rev-parse", f"{candidate_sha}^{{tree}}")
    if snapshot.target_tree_sha != frozen_tree:
        raise ValueError("snapshot-tree-mismatch")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "r+b", closefd=False) as handle:
            with _exclusive_file_lock(handle):
                handle.seek(0)
                events = _read_evidence_bytes(handle.read(), repo=Path(repo))
                snapshot_payload = asdict(snapshot)
                event: dict[str, Any] = {
                    "schema_version": _SCHEMA_EVIDENCE,
                    "sequence": len(events) + 1,
                    "frozen_sha": frozen_sha,
                    "candidate_sha": candidate_sha,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot": snapshot_payload,
                    "frozen_tree_sha": frozen_tree,
                    "candidate_tree_sha": candidate_tree,
                    "command": command,
                    "exit_code": exit_code,
                    "recorded_at": recorded_at,
                    "previous_event_hash": events[-1]["event_hash"] if events else None,
                }
                event["event_hash"] = hashlib.sha256(_canonical_bytes(event)).hexdigest()
                event = json.loads(_canonical_bytes(event))
                encoded = _canonical_bytes(event) + b"\n"
                os.write(handle.fileno(), encoded)
                os.fsync(handle.fileno())
                return event
    finally:
        os.close(descriptor)


def _load_snapshot(path: Path) -> ReleaseSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot-invalid")
    try:
        payload["changed_paths"] = tuple(payload["changed_paths"])
        snapshot = ReleaseSnapshot(**payload)
    except (KeyError, TypeError) as exc:
        raise ValueError("snapshot-invalid") from exc
    body = asdict(snapshot)
    claimed = body.pop("snapshot_id")
    if claimed != hashlib.sha256(_canonical_bytes(body)).hexdigest():
        raise ValueError("snapshot-id-mismatch")
    return snapshot


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--repo", type=Path, required=True)
    freeze.add_argument("--ref", required=True)
    freeze.add_argument("--base")

    classify = subparsers.add_parser("classify")
    classify.add_argument("--repo", type=Path, required=True)
    classify.add_argument("--from-sha", required=True)
    classify.add_argument("--latest-sha", required=True)
    classify.add_argument("--policy", type=Path, required=True)

    append = subparsers.add_parser("append-evidence")
    append.add_argument("--path", type=Path, required=True)
    append.add_argument("--repo", type=Path, required=True)
    append.add_argument("--snapshot", type=Path, required=True)
    append.add_argument("--frozen-sha", required=True)
    append.add_argument("--candidate-sha", required=True)
    append.add_argument("--command-text", required=True)
    append.add_argument("--exit-code", type=int, required=True)
    append.add_argument("--recorded-at", required=True)

    validate = subparsers.add_parser("validate-evidence")
    validate.add_argument("--path", type=Path, required=True)
    validate.add_argument("--repo", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "freeze":
        _print_json(asdict(freeze_snapshot(args.repo, args.ref, local_base_sha=args.base)))
        return 0
    if args.command == "classify":
        report = classify_overlap(
            args.repo,
            args.from_sha,
            args.latest_sha,
            critical_paths=load_critical_paths(args.policy),
        )
        _print_json(asdict(report))
        return 1 if report.classification in {"overlap", "indeterminate"} else 0
    if args.command == "append-evidence":
        _print_json(
            append_evidence(
                args.path,
                repo=args.repo,
                snapshot=_load_snapshot(args.snapshot),
                frozen_sha=args.frozen_sha,
                candidate_sha=args.candidate_sha,
                command=args.command_text,
                exit_code=args.exit_code,
                recorded_at=args.recorded_at,
            )
        )
        return 0
    _print_json(validate_evidence(args.path, repo=args.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
