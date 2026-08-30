from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli.deferred_gateway_restart import (
    deferred_restart_intent_path,
    deferred_restart_receipt_path,
    read_deferred_restart_receipt,
    run_deferred_restart as _run_deferred_restart,
    schedule_deferred_restart as _schedule_deferred_restart,
)
from hermes_cli.gateway_restart_contract import RestartProbe, ServingIdentity


class Clock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def schedule_deferred_restart(**kwargs):
    """Give fake worker PIDs stable identities unless a test overrides them."""
    kwargs.setdefault("process_start_time", lambda pid: pid * 10)
    return _schedule_deferred_restart(**kwargs)


def identity(home: Path, *, pid: int = 100, start: int = 200, sha: str = "old") -> ServingIdentity:
    return ServingIdentity(pid, start, home, sha)


def run_deferred_restart(intent_path: Path, **kwargs) -> int:
    """Run the scheduled worker as the PID published by the parent fixture."""
    payload = json.loads(intent_path.read_text(encoding="utf-8"))
    if payload.get("state") == "pending":
        payload["worker_pid"] = os.getpid()
        payload["worker_start_time"] = os.getpid() * 10
        intent_path.write_text(json.dumps(payload), encoding="utf-8")
    return _run_deferred_restart(
        intent_path,
        generation=payload["generation"],
        owner_token=payload["owner_token"],
        process_start_time=lambda pid: pid * 10,
        **kwargs,
    )


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
    payload = json.loads(deferred_restart_intent_path(tmp_path).read_text())
    assert spawned[0][-6:] == [
        "--intent",
        str(deferred_restart_intent_path(tmp_path)),
        "--generation",
        payload["generation"],
        "--owner-token",
        payload["owner_token"],
    ]
    assert isinstance(payload.pop("generation"), str)
    assert isinstance(payload.pop("owner_token"), str)
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
        "worker_start_time": 3210,
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


def test_scheduler_does_not_overwrite_worker_that_claims_intent_during_spawn(
    tmp_path: Path,
) -> None:
    path = deferred_restart_intent_path(tmp_path)

    def spawn(_argv):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["state"] = "running"
        payload["worker_pid"] = 777
        payload["worker_start_time"] = 7770
        path.write_text(json.dumps(payload), encoding="utf-8")
        return 321

    result = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=identity(tmp_path),
        reason="active-work:1",
        spawn=spawn,
        now=lambda: 1_000.0,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert result.scheduled is True
    assert payload["state"] == "running"
    assert payload["worker_pid"] == 777


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
    intent_payload = json.loads(deferred_restart_intent_path(tmp_path).read_text())
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
    receipt = read_deferred_restart_receipt(
        tmp_path, expected_generation=intent_payload["generation"]
    )
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
        "generation": intent_payload["generation"],
        "owner_token": intent_payload["owner_token"],
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
    generation = json.loads(deferred_restart_intent_path(tmp_path).read_text())["generation"]
    restart_calls: list[object] = []

    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=lambda _intent: RestartProbe(False, "active-work-unknown", old),
        restart=lambda intent: restart_calls.append(intent)
        or RestartProbe(False, "unexpected-restart", old),
        now=clock.now,
        sleep=clock.sleep,
    )

    assert exit_code == 1
    assert restart_calls == []
    receipt = read_deferred_restart_receipt(tmp_path, expected_generation=generation)
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

    generation = json.loads(deferred_restart_intent_path(tmp_path).read_text())["generation"]
    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=lambda _intent: RestartProbe(False, "active-work:1", old),
        restart=lambda _intent: pytest.fail("restart must not run"),
        now=clock.now,
        sleep=clock.sleep,
        poll_interval=2.0,
    )

    assert exit_code == 1
    receipt = read_deferred_restart_receipt(tmp_path, expected_generation=generation)
    assert receipt is not None
    assert receipt["state"] == "expired"
    assert receipt["reason"] == "deadline-exceeded"


def test_stale_worker_cannot_overwrite_receipt_or_delete_newer_intent(
    tmp_path: Path,
) -> None:
    clock = Clock()
    old = identity(tmp_path)
    first = schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: 321,
        now=clock.now,
    )
    assert first.scheduled
    replacement_generation: list[str] = []

    def replace_then_fail(_intent):
        replacement = schedule_deferred_restart(
            home=tmp_path,
            port=8642,
            expected_sha="a" * 40,
            old_identity=old,
            reason="active-work:2",
            spawn=lambda _argv: 999,
            process_alive=lambda _pid: False,
            now=lambda: clock.now() + 1,
        )
        assert replacement.scheduled
        replacement_generation.append(
            json.loads(deferred_restart_intent_path(tmp_path).read_text())["generation"]
        )
        return RestartProbe(False, "active-work-unknown", old)

    assert run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=replace_then_fail,
        restart=lambda _intent: pytest.fail("restart must not run"),
        now=clock.now,
        sleep=clock.sleep,
    ) == 1

    surviving = json.loads(deferred_restart_intent_path(tmp_path).read_text())
    assert surviving["generation"] == replacement_generation[0]
    assert surviving["worker_pid"] == 999
    assert not deferred_restart_receipt_path(tmp_path).exists()


def test_worker_losing_authority_during_ready_probe_cannot_restart(tmp_path: Path) -> None:
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

    def restart_attempt(intent):
        restart_calls.append(intent)
        return RestartProbe(False, "unexpected-restart", old)

    def replace_during_probe(_intent):
        intent_path = deferred_restart_intent_path(tmp_path)
        payload = json.loads(intent_path.read_text())
        payload["generation"] = "replacement-generation"
        payload["owner_token"] = "replacement-owner"
        payload["worker_pid"] = 999
        intent_path.write_text(json.dumps(payload))
        return RestartProbe(True, "idle", old)

    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=replace_during_probe,
        restart=restart_attempt,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert exit_code == 1
    assert restart_calls == []


def test_second_worker_cannot_regress_restarting_intent_to_running(
    tmp_path: Path,
) -> None:
    old = identity(tmp_path)
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: os.getpid(),
        now=lambda: 1_000.0,
    )
    path = deferred_restart_intent_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"] = "restarting"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restart_calls: list[object] = []

    exit_code = run_deferred_restart(
        path,
        probe=lambda _intent: RestartProbe(True, "idle", old),
        restart=lambda intent: restart_calls.append(intent)
        or RestartProbe(False, "unexpected-restart", old),
        now=lambda: 1_001.0,
    )

    assert exit_code == 1
    assert restart_calls == []
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "restarting"


def test_worker_rechecks_deadline_after_ready_probe_before_restart(
    tmp_path: Path,
) -> None:
    clock = Clock()
    old = identity(tmp_path)
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: os.getpid(),
        now=clock.now,
        timeout=1.0,
    )
    restart_calls: list[object] = []

    def probe_after_deadline(_intent):
        clock.value = 1_002.0
        return RestartProbe(True, "idle", old)

    exit_code = run_deferred_restart(
        deferred_restart_intent_path(tmp_path),
        probe=probe_after_deadline,
        restart=lambda intent: restart_calls.append(intent)
        or RestartProbe(False, "unexpected-restart", old),
        now=clock.now,
    )

    assert exit_code == 1
    assert restart_calls == []


def test_worker_waits_for_parent_to_publish_its_pid(tmp_path: Path) -> None:
    old = identity(tmp_path)
    new = identity(tmp_path, pid=101, start=201, sha="a" * 40)
    schedule_deferred_restart(
        home=tmp_path,
        port=8642,
        expected_sha="a" * 40,
        old_identity=old,
        reason="active-work:1",
        spawn=lambda _argv: os.getpid(),
        now=lambda: 1_000.0,
    )
    path = deferred_restart_intent_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["worker_pid"] = None
    payload["worker_start_time"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    published = False

    def publish_pid(_seconds: float) -> None:
        nonlocal published
        current = json.loads(path.read_text(encoding="utf-8"))
        current["worker_pid"] = os.getpid()
        current["worker_start_time"] = os.getpid() * 10
        path.write_text(json.dumps(current), encoding="utf-8")
        published = True

    exit_code = _run_deferred_restart(
        path,
        generation=payload["generation"],
        owner_token=payload["owner_token"],
        probe=lambda _intent: RestartProbe(True, "idle", old),
        restart=lambda _intent: RestartProbe(True, "replacement-healthy", new),
        now=lambda: 1_000.0,
        sleep=publish_pid,
        process_start_time=lambda pid: pid * 10,
    )

    assert exit_code == 0
    assert published is True


@pytest.mark.parametrize("restart_result", [None, object()])
def test_restart_adapter_invalid_result_writes_failed_terminal_receipt(
    tmp_path: Path, restart_result: object
) -> None:
    old = identity(tmp_path)
    result = schedule_deferred_restart(
        home=tmp_path, port=8642, expected_sha="a" * 40, old_identity=old,
        reason="active-work:1", spawn=lambda _argv: os.getpid(), now=lambda: 1_000.0,
    )
    generation = json.loads(result.path.read_text())["generation"]

    assert run_deferred_restart(
        result.path, probe=lambda _intent: RestartProbe(True, "idle", old),
        restart=lambda _intent: restart_result, now=lambda: 1_001.0,
    ) == 1
    receipt = read_deferred_restart_receipt(tmp_path, expected_generation=generation)
    assert receipt is not None
    assert receipt["state"] == "failed"
    assert receipt["reason"] == "restart-result-invalid"


def test_restart_adapter_exception_writes_failed_terminal_receipt(tmp_path: Path) -> None:
    old = identity(tmp_path)
    result = schedule_deferred_restart(
        home=tmp_path, port=8642, expected_sha="a" * 40, old_identity=old,
        reason="active-work:1", spawn=lambda _argv: os.getpid(), now=lambda: 1_000.0,
    )
    generation = json.loads(result.path.read_text())["generation"]

    def explode(_intent):
        raise RuntimeError("adapter failed")

    assert run_deferred_restart(
        result.path, probe=lambda _intent: RestartProbe(True, "idle", old),
        restart=explode, now=lambda: 1_001.0,
    ) == 1
    receipt = read_deferred_restart_receipt(tmp_path, expected_generation=generation)
    assert receipt is not None
    assert receipt["state"] == "failed"
    assert receipt["reason"] == "restart-failed:RuntimeError"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", True), ("old_pid", 0), ("old_pid", -1),
        ("old_start_time", 0), ("old_start_time", -1),
        ("control_health", True), ("control_health", "verified"),
        ("http_health", "unknown"),
    ],
)
def test_receipt_rejects_malformed_schema_identity_and_failure_health(
    tmp_path: Path, field: str, value: object
) -> None:
    generation = "generation-one"
    payload = {
        "schema": 1, "state": "failed", "reason": "restart-result-invalid",
        "home": str(tmp_path.resolve()), "port": 8642, "expected_sha": "a" * 40,
        "serving_sha": None, "old_pid": 100, "old_start_time": 200,
        "new_pid": None, "new_start_time": None, "listener_owner_pid": None,
        "control_health": "unverified", "http_health": "unverified",
        "finished_at": 1_001.0, "generation": generation, "owner_token": "owner-one",
    }
    payload[field] = value
    path = deferred_restart_receipt_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_deferred_restart_receipt(tmp_path, expected_generation=generation) is None


def test_new_generation_cannot_read_stale_receipt(tmp_path: Path) -> None:
    old = identity(tmp_path)
    first = schedule_deferred_restart(
        home=tmp_path, port=8642, expected_sha="a" * 40, old_identity=old,
        reason="active-work:1", spawn=lambda _argv: os.getpid(), now=lambda: 1_000.0,
    )
    first_generation = json.loads(first.path.read_text())["generation"]
    assert run_deferred_restart(
        first.path, probe=lambda _intent: RestartProbe(False, "active-work-unknown", old),
        restart=lambda _intent: pytest.fail("restart must not run"), now=lambda: 1_001.0,
    ) == 1
    assert read_deferred_restart_receipt(tmp_path, expected_generation=first_generation) is not None

    second = schedule_deferred_restart(
        home=tmp_path, port=8642, expected_sha="a" * 40, old_identity=old,
        reason="active-work:1", spawn=lambda _argv: 999, now=lambda: 1_002.0,
    )
    second_generation = json.loads(second.path.read_text())["generation"]
    assert second_generation != first_generation
    assert read_deferred_restart_receipt(tmp_path, expected_generation=second_generation) is None


def test_reused_worker_pid_does_not_count_as_same_live_worker(tmp_path: Path) -> None:
    old = identity(tmp_path)
    first = schedule_deferred_restart(
        home=tmp_path, port=8642, expected_sha="a" * 40, old_identity=old,
        reason="active-work:1", spawn=lambda _argv: 321,
        process_start_time=lambda _pid: 10, now=lambda: 1_000.0,
    )
    assert first.scheduled

    second = schedule_deferred_restart(
        home=tmp_path, port=8642, expected_sha="a" * 40, old_identity=old,
        reason="active-work:1", spawn=lambda _argv: 999,
        process_start_time=lambda pid: 20 if pid == 321 else 30, now=lambda: 1_001.0,
    )
    assert second.scheduled
    assert second.reason == "worker-replaced"
    payload = json.loads(second.path.read_text())
    assert payload["worker_pid"] == 999
    assert payload["worker_start_time"] == 30
