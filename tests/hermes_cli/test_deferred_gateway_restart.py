from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.deferred_gateway_restart import (
    deferred_restart_intent_path,
    deferred_restart_receipt_path,
    read_deferred_restart_receipt,
    run_deferred_restart,
    schedule_deferred_restart,
)
from hermes_cli.gateway_restart_contract import RestartProbe, ServingIdentity


class Clock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def identity(home: Path, *, pid: int = 100, start: int = 200, sha: str = "old") -> ServingIdentity:
    return ServingIdentity(pid, start, home, sha)


@pytest.mark.parametrize(
    "reason",
    [
        "active-work:0",
        "active-work:-1",
        "active-work:01",
        "active-work:1 ",
        "active-work:unknown",
        "active-work-unknown",
        "listener-owner-mismatch",
        "",
    ],
)
def test_schedule_rejects_everything_except_positive_exact_active_work(
    tmp_path: Path,
    reason: str,
) -> None:
    spawned: list[list[str]] = []

    result = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason=reason,
        spawn=lambda argv: spawned.append(argv) or 321,
        now=lambda: 1_000.0,
    )

    assert result.scheduled is False
    assert result.reason == "reason-not-schedulable"
    assert spawned == []
    assert not deferred_restart_intent_path(tmp_path).exists()


def test_schedule_writes_atomic_intent_and_spawns_worker(tmp_path: Path) -> None:
    spawned: list[list[str]] = []

    result = schedule_deferred_restart(
        home=tmp_path,
        port=8765,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:2",
        spawn=lambda argv: spawned.append(argv) or 321,
        now=lambda: 1_000.0,
        timeout=600.0,
    )

    assert result.scheduled is True
    assert result.reason == "scheduled"
    assert len(spawned) == 1
    assert spawned[0][-2:] == ["--intent", str(deferred_restart_intent_path(tmp_path))]
    payload = json.loads(deferred_restart_intent_path(tmp_path).read_text())
    assert payload == {
        "schema": 1,
        "state": "pending",
        "home": str(tmp_path.resolve()),
        "port": 8765,
        "health_url": "http://127.0.0.1:8765/health",
        "expected_sha": "a" * 40,
        "old_pid": 100,
        "old_start_time": 200,
        "created_at": 1_000.0,
        "deadline_at": 1_600.0,
        "reason": "active-work:2",
        "worker_pid": 321,
    }
    assert not list(deferred_restart_intent_path(tmp_path).parent.glob("*.tmp"))


def test_schedule_deduplicates_live_worker_for_exact_target(tmp_path: Path) -> None:
    first = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=lambda: 1_000.0,
    )
    assert first.scheduled

    spawned: list[list[str]] = []
    second = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:3",
        spawn=lambda argv: spawned.append(argv) or 999,
        process_alive=lambda pid: pid == 321,
        now=lambda: 1_001.0,
    )

    assert second.scheduled is True
    assert second.reason == "already-scheduled"
    assert spawned == []


def test_schedule_replaces_dead_worker_for_exact_target(tmp_path: Path) -> None:
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=lambda: 1_000.0,
    )

    result = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:1",
        spawn=lambda _argv: 999,
        process_alive=lambda _pid: False,
        now=lambda: 1_010.0,
    )

    assert result.reason == "worker-replaced"
    assert json.loads(deferred_restart_intent_path(tmp_path).read_text())["worker_pid"] == 999


def test_schedule_fails_closed_on_corrupt_existing_intent(tmp_path: Path) -> None:
    path = deferred_restart_intent_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    spawned: list[list[str]] = []

    result = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:1",
        spawn=lambda argv: spawned.append(argv) or 321,
    )

    assert result.scheduled is False
    assert result.reason == "existing-intent-invalid"
    assert spawned == []
    assert path.read_text(encoding="utf-8") == "not-json"


def test_worker_rejects_noncanonical_intent_path_before_probe(tmp_path: Path) -> None:
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=lambda: 1_000.0,
    )
    canonical = deferred_restart_intent_path(tmp_path)
    foreign = tmp_path / "copied-intent.json"
    foreign.write_bytes(canonical.read_bytes())
    probe_calls = []

    exit_code = run_deferred_restart(
        foreign,
        probe=lambda intent: probe_calls.append(intent) or RestartProbe(
            False, "active-work:1", identity(tmp_path)
        ),
        restart=lambda _intent: pytest.fail("restart must not run"),
    )

    assert exit_code == 1
    assert probe_calls == []
    assert foreign.is_file()
    assert not deferred_restart_receipt_path(tmp_path).exists()


def test_worker_waits_for_active_work_then_records_verified_replacement(tmp_path: Path) -> None:
    clock = Clock()
    old = identity(tmp_path)
    new = identity(tmp_path, pid=101, start=201, sha="a" * 40)
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=clock.now,
        timeout=60.0,
    )
    probes = iter(
        [
            RestartProbe(False, "active-work:1", old),
            RestartProbe(True, "gateway-healthy", old),
        ]
    )
    restart_calls: list[object] = []

    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=lambda _intent: next(probes),
        restart=lambda intent: restart_calls.append(intent) or RestartProbe(
            True, "replacement-healthy", new
        ),
        now=clock.now,
        sleep=clock.sleep,
        poll_interval=2.0,
    )

    assert exit_code == 0
    assert len(restart_calls) == 1
    receipt = read_deferred_restart_receipt(tmp_path)
    assert receipt == {
        "schema": 1,
        "state": "completed",
        "reason": "replacement-healthy",
        "home": str(tmp_path.resolve()),
        "port": 8642,
        "expected_sha": "a" * 40,
        "serving_sha": "a" * 40,
        "old_pid": 100,
        "old_start_time": 200,
        "new_pid": 101,
        "new_start_time": 201,
        "listener_owner_pid": 101,
        "control_health": "verified",
        "http_health": "verified",
        "finished_at": 1_002.0,
    }
    assert not deferred_restart_intent_path(tmp_path).exists()
    assert deferred_restart_receipt_path(tmp_path).is_file()


def test_worker_fails_closed_on_unknown_authority_without_restart(tmp_path: Path) -> None:
    clock = Clock()
    old = identity(tmp_path)
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=clock.now,
    )
    restart_calls: list[object] = []

    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=lambda _intent: RestartProbe(False, "active-work-unknown", old),
        restart=lambda intent: restart_calls.append(intent),
        now=clock.now,
        sleep=clock.sleep,
    )

    assert exit_code == 1
    assert restart_calls == []
    receipt = read_deferred_restart_receipt(tmp_path)
    assert receipt is not None
    assert receipt["state"] == "failed"
    assert receipt["reason"] == "active-work-unknown"
    assert receipt["new_pid"] is None


def test_worker_expires_without_restart(tmp_path: Path) -> None:
    clock = Clock()
    old = identity(tmp_path)
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=clock.now,
        timeout=1.0,
    )

    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=lambda _intent: RestartProbe(False, "active-work:1", old),
        restart=lambda _intent: pytest.fail("restart must not run"),
        now=clock.now,
        sleep=clock.sleep,
        poll_interval=2.0,
    )

    assert exit_code == 1
    receipt = read_deferred_restart_receipt(tmp_path)
    assert receipt is not None
    assert receipt["state"] == "expired"
    assert receipt["reason"] == "deadline-exceeded"
