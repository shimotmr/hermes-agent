"""Hermetic control-protocol tests for atomic restart-if-idle."""
import json
import os
import threading
import asyncio

import pytest
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.control_socket import GatewayControlServer
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.fixture(autouse=True)
def _isolate_restart_control_from_relaunch_attestation(monkeypatch):
    """These tests exercise control semantics, not OS manager probing."""
    import gateway.restart_relaunch as relaunch

    monkeypatch.setattr(relaunch, "verify_relaunch_attestation", lambda *args, **kwargs: True)


def test_production_snapshot_keeps_answering_identity_and_complete_live_state(monkeypatch, tmp_path):
    """Exercise the real default snapshot builder, not the synthetic injection seam."""
    import gateway.control_socket as cs

    stale = {
        "kind": "hermes-gateway", "pid": 999999, "start_time": 1,
        "hermes_home": "/stale", "code_sha": "b" * 40,
        "gateway_state": "running", "session_store": {"status": "ok"},
        "active_agents": 7,
        "platforms": {"telegram": {"state": "connected", "writer_pid": 999999,
                                      "writer_start_time": 1}},
    }
    live = {"protocol": 1, "kind": "hermes-gateway", "pid": os.getpid(),
            "start_time": 2468, "hermes_home": str(tmp_path.resolve()),
            "code_sha": "a" * 40, "supervisor": "launchd"}
    monkeypatch.setattr(cs, "build_identify_payload", lambda: live.copy())
    monkeypatch.setattr(cs, "build_status_payload", lambda: stale.copy())
    server = GatewayControlServer(
        tmp_path, active_agents=lambda: 0,
        obligations=lambda: _zero_obligations())

    snapshot = server._build_restart_snapshot()

    assert {key: snapshot[key] for key in ("kind", "pid", "start_time", "hermes_home", "code_sha")} == {
        key: live[key] for key in ("kind", "pid", "start_time", "hermes_home", "code_sha")}
    assert snapshot["answering_pid"] == os.getpid() == snapshot["signal_target_pid"]
    assert snapshot["supervisor_pid"] == os.getppid()
    assert snapshot["active_agents"] == 0
    assert snapshot["session_store"] == {"status": "ok"}
    assert snapshot["obligations"] == _zero_obligations()
    assert snapshot["platforms"]["telegram"]["writer_pid"] == 999999


def test_production_snapshot_concurrent_success_signals_answering_pid_once(monkeypatch, tmp_path):
    import gateway.control_socket as cs
    live = {"protocol": 1, "kind": "hermes-gateway", "pid": os.getpid(),
            "start_time": 2468, "hermes_home": str(tmp_path.resolve()), "code_sha": "a" * 40,
            "supervisor": "launchd"}
    status = {"gateway_state": "running", "session_store": {"status": "ok"},
              "configured_platforms": [], "platforms": {}}
    monkeypatch.setattr(cs, "build_identify_payload", lambda: live.copy())
    monkeypatch.setattr(cs, "build_status_payload", lambda: status.copy())
    signals = []
    monkeypatch.setattr(cs.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    server = GatewayControlServer(
        tmp_path, active_agents=lambda: 0,
        obligations=lambda: _zero_obligations(),
        sigusr1_supported=True)
    barrier = threading.Barrier(3)
    results = []
    def invoke():
        barrier.wait()
        results.append(_request(server, {**live, "required_platforms": []})["result"])
    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads: thread.start()
    barrier.wait()
    for thread in threads: thread.join()
    assert sum(result["accepted"] is True for result in results) == 1
    assert signals == [(os.getpid(), cs.signal.SIGUSR1)]


def _expected(home: Path):
    return {
        "protocol": 1,
        "kind": "hermes-gateway",
        "pid": 42,
        "start_time": 1234,
        "hermes_home": str(home.resolve()),
        "code_sha": "a" * 40,
        "required_platforms": ["telegram"],
    }


def _snapshot(home: Path):
    return {
        **_expected(home),
        "gateway_state": "running",
        "session_store": {"status": "ok"},
        "active_agents": 0,
        "answering_pid": 42,
        "signal_target_pid": 42,
        "supervisor_pid": 24,
        "supervisor": "launchd",
        "configured_platforms": ["telegram"],
        "platforms": {
            "telegram": {
                "state": "connected",
                "writer_pid": 42,
                "writer_start_time": 1234,
            }
        },
        "obligations": _zero_obligations(),
    }


def _zero_obligations():
    return {
        "delivery_queue": 0,
        "delivery_ledger": 0,
        "pending_final": 0,
        "drain": 0,
        "delegation_workers": 0,
        "updater": 0,
        "update_lock": 0,
        "oauth_refresh": 0,
        "oauth_token_lock": 0,
    }


def _attestation(pid, start_time):
    return {"manager": "launchd", "service": "gui/501/test.gateway", "policy": "keepalive",
            "pid": pid, "start_time": start_time}


@pytest.fixture(autouse=True)
def manager_contract(monkeypatch):
    monkeypatch.setattr("gateway.restart_relaunch.probe_gateway_relaunch", _attestation)


def _request(server, expected=None):
    request = {"id": 7, "protocol": 1, "verb": "restart-if-idle"}
    if expected is not None:
        request["expected_identity"] = expected
        request["relaunch_attestation"] = _attestation(expected["pid"], expected["start_time"])
    return json.loads(server.handle_request_line(json.dumps(request).encode()))


def test_atomic_success_rechecks_and_signals_sigusr1_exactly_once(tmp_path):
    calls = []
    snapshots = [_snapshot(tmp_path)]
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: snapshots.pop(0),
        graceful_signal=lambda: calls.append("SIGUSR1"),
        sigusr1_supported=True,
    )
    reply = _request(server, _expected(tmp_path))
    assert reply["result"] == {"accepted": True, "reason": "accepted", "pid": 42}
    assert calls == ["SIGUSR1"]


def test_state_change_at_atomic_recheck_denies_without_signal(tmp_path):
    calls = []
    changed = _snapshot(tmp_path)
    changed["active_agents"] = 1
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: changed,
        graceful_signal=lambda: calls.append("SIGUSR1"),
        sigusr1_supported=True,
    )
    reply = _request(server, _expected(tmp_path))
    assert reply["result"]["accepted"] is False
    assert reply["result"]["reason"] == "active-agents"
    assert calls == []


def test_missing_expected_identity_denies_without_signal(tmp_path):
    calls = []
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: _snapshot(tmp_path),
        graceful_signal=lambda: calls.append("SIGUSR1"),
        sigusr1_supported=True,
    )
    reply = _request(server)
    assert reply["result"]["accepted"] is False
    assert reply["result"]["reason"] == "invalid-expected-identity"
    assert calls == []


def test_unsupported_sigusr1_denies_without_signal(tmp_path):
    calls = []
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: _snapshot(tmp_path),
        graceful_signal=lambda: calls.append("fallback"),
        sigusr1_supported=False,
    )
    reply = _request(server, _expected(tmp_path))
    assert reply["result"] == {"accepted": False, "reason": "unsupported"}
    assert calls == []


def test_wrong_request_protocol_denies_without_signal(tmp_path):
    calls = []
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: _snapshot(tmp_path),
        graceful_signal=lambda: calls.append("SIGUSR1"),
        sigusr1_supported=True,
    )
    request = {"id": 7, "protocol": 2, "verb": "restart-if-idle",
               "expected_identity": _expected(tmp_path)}
    reply = json.loads(server.handle_request_line(json.dumps(request).encode()))
    assert reply["result"] == {"accepted": False, "reason": "protocol-mismatch"}
    assert calls == []


def test_duplicate_restart_request_signals_at_most_once(tmp_path):
    calls = []
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: _snapshot(tmp_path),
        graceful_signal=lambda: calls.append("SIGUSR1"),
        sigusr1_supported=True,
    )
    assert _request(server, _expected(tmp_path))["result"]["accepted"] is True
    assert _request(server, _expected(tmp_path))["result"] == {
        "accepted": False, "reason": "restart-already-requested"}
    assert calls == ["SIGUSR1"]


def test_request_fields_are_transported(monkeypatch, tmp_path):
    import gateway.control_socket as cs

    captured = {}

    def query(_home, request, _timeout):
        captured.update(json.loads(request))
        return b'{"ok":true,"result":{"accepted":false,"reason":"busy"}}\n'

    monkeypatch.setattr(cs, "_query_unix_socket", query)
    monkeypatch.setattr(cs, "_IS_WINDOWS", False)
    result = cs.query_gateway_control(
        tmp_path,
        "restart-if-idle",
        request_fields={"expected_identity": _expected(tmp_path)},
    )
    assert captured["expected_identity"] == _expected(tmp_path)
    assert result == {"accepted": False, "reason": "busy"}


def test_empty_requested_platforms_still_enforces_configured_platforms(tmp_path):
    from hermes_cli.gateway_restart_contract import ExpectedGatewayIdentity, authorize_restart_if_idle

    snapshot = _snapshot(tmp_path)
    snapshot["platforms"]["telegram"]["writer_pid"] = 999999
    expected = ExpectedGatewayIdentity.from_mapping({**_expected(tmp_path), "required_platforms": []})

    accepted, reason = authorize_restart_if_idle(snapshot, expected)

    assert accepted is False
    assert reason == "platform-telegram-stale-writer"


def test_answering_and_signal_target_pid_are_authorization_bound(tmp_path):
    from hermes_cli.gateway_restart_contract import ExpectedGatewayIdentity, authorize_restart_if_idle

    for field, reason in (
        ("answering_pid", "identity-answering-pid-mismatch"),
        ("signal_target_pid", "identity-signal-target-pid-mismatch"),
    ):
        snapshot = _snapshot(tmp_path)
        snapshot[field] = 999999
        accepted, actual_reason = authorize_restart_if_idle(
            snapshot, ExpectedGatewayIdentity.from_mapping(_expected(tmp_path))
        )
        assert accepted is False
        assert actual_reason == reason


def test_unsupported_supervisor_fails_closed(tmp_path):
    from hermes_cli.gateway_restart_contract import ExpectedGatewayIdentity, authorize_restart_if_idle

    snapshot = _snapshot(tmp_path)
    snapshot["supervisor"] = "manual"

    accepted, reason = authorize_restart_if_idle(snapshot, ExpectedGatewayIdentity.from_mapping(_expected(tmp_path)))

    assert accepted is False
    assert reason == "unsupported-supervisor"


def test_gateway_runner_live_snapshot_overrides_stale_status(monkeypatch, tmp_path):
    import gateway.control_socket as cs
    import gateway.restart_runtime as rr
    from gateway.run import GatewayRunner

    live = {
        "protocol": 1,
        "kind": "hermes-gateway",
        "pid": os.getpid(),
        "start_time": 2468,
        "hermes_home": str(tmp_path.resolve()),
        "code_sha": "a" * 40,
        "supervisor": "launchd",
    }
    monkeypatch.setattr(cs, "build_identify_payload", lambda: live.copy())
    monkeypatch.setattr(rr, "count_owned_delivery_obligations", lambda: 0)
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")})
    runner = GatewayRunner(config)
    runner._running = True
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(
        restart_writer_health_probe=lambda: {
            "state": "connected",
            "writer_pid": os.getpid(),
            "writer_start_time": 2468,
        }
    )
    runner.session_store = SimpleNamespace(restart_health_probe=lambda: {"status": "ok"})
    snapshot = runner._build_restart_control_snapshot()

    assert snapshot["pid"] == os.getpid()
    assert snapshot["answering_pid"] == os.getpid() == snapshot["signal_target_pid"]
    assert snapshot["supervisor_pid"] == os.getppid()
    assert snapshot["supervisor"] == "launchd"
    assert snapshot["session_store"] == {"status": "ok"}
    assert snapshot["configured_platforms"] == ["telegram"]
    assert snapshot["platforms"]["telegram"] == {
        "state": "connected",
        "writer_pid": os.getpid(),
        "writer_start_time": 2468,
    }


def test_gateway_control_socket_wiring_uses_runner_owner_and_barrier(monkeypatch, tmp_path):
    import gateway.control_socket as cs
    import gateway.restart_runtime as rr
    import gateway.run as gw_run
    from gateway.run import GatewayRunner

    live = {
        "protocol": 1,
        "kind": "hermes-gateway",
        "pid": os.getpid(),
        "start_time": 2468,
        "hermes_home": str(tmp_path.resolve()),
        "code_sha": "a" * 40,
        "supervisor": "launchd",
    }
    monkeypatch.setattr(cs, "build_identify_payload", lambda: live.copy())
    monkeypatch.setattr(rr, "count_owned_delivery_obligations", lambda: 0)
    monkeypatch.setattr(cs.GatewayControlServer, "start", lambda self: _true_async())
    signals = []
    monkeypatch.setattr(cs.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    runner = GatewayRunner(GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    ))
    runner._running = True
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(
        restart_writer_health_probe=lambda: {
            "state": "connected",
            "writer_pid": os.getpid(),
            "writer_start_time": 2468,
        }
    )
    runner.session_store = SimpleNamespace(restart_health_probe=lambda: {"status": "ok"})
    server = asyncio.run(gw_run._start_gateway_start_control_socket(runner))
    result = _request(server, {**live, "required_platforms": []})["result"]

    assert server._admission_barrier is runner._restart_admission_barrier
    assert server._restart_snapshot == runner._build_restart_control_snapshot
    assert result == {"accepted": True, "reason": "accepted", "pid": os.getpid()}
    assert signals == [(os.getpid(), cs.signal.SIGUSR1)]


def test_admission_barrier_closes_snapshot_authorize_signal_toctou(tmp_path):
    from gateway.restart_runtime import GatewayAdmissionBarrier

    barrier = GatewayAdmissionBarrier()
    calls = []
    active = {"value": 0}
    snapshot_started = threading.Event()

    def snapshot():
        snapshot_started.set()
        snap = _snapshot(tmp_path)
        snap["active_agents"] = active["value"]
        return snap

    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=snapshot,
        admission_barrier=barrier,
        graceful_signal=lambda: calls.append("SIGUSR1"),
        sigusr1_supported=True,
    )
    with barrier.admission():
        thread = threading.Thread(target=lambda: calls.append(_request(server, _expected(tmp_path))["result"]))
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not snapshot_started.is_set()
        active["value"] = 1

    assert calls == [{"accepted": False, "reason": "admissions-active"}]


async def _true_async():
    return True
