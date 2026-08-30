import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.gateway_restart_contract import (
    RestartProbe,
    RuntimeAdapters,
    ServingIdentity,
    evaluate_current,
    evaluate_replacement,
    main,
    parse_serving_identity,
    perform_verified_graceful_restart,
    probe_current_gateway,
)


HOME = Path("/tmp/hermes-home")
OLD = ServingIdentity(pid=68821, start_time=12345, hermes_home=HOME, code_sha="old")
NEW = ServingIdentity(pid=94350, start_time=22345, hermes_home=HOME, code_sha="new")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol", True, "control-protocol-mismatch"),
        ("pid", True, "invalid-serving-pid"),
        ("start_time", True, "invalid-serving-start-time"),
    ],
)
def test_serving_identity_rejects_boolean_integers(
    field: str, value: object, message: str
) -> None:
    payload = {
        "protocol": 1,
        "kind": "hermes-gateway",
        "pid": 94350,
        "start_time": 22345,
        "hermes_home": str(HOME),
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        parse_serving_identity(payload, expected_home=HOME)


def _status(pid: int = NEW.pid, *, active_agents: object = 0) -> dict:
    return {
        "gateway_state": "running",
        "active_agents": active_agents,
        "answering_pid": pid,
        "answered_at": time.time(),
        "pid": pid,
        "start_time": NEW.start_time if pid == NEW.pid else OLD.start_time,
        "platforms": {
            name: {
                "state": "connected",
                "writer_pid": pid,
                "writer_start_time": (
                    NEW.start_time if pid == NEW.pid else OLD.start_time
                ),
            }
            for name in ("api_server", "webhook", "telegram")
        },
    }


def test_stale_control_status_fails_closed() -> None:
    status = _status()
    status["answered_at"] = time.time() - 60

    result = evaluate_current(
        current=NEW,
        status=status,
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "status-answer-stale", NEW)


@pytest.mark.parametrize("answered_at", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_control_status_timestamp_fails_closed(answered_at: float) -> None:
    status = _status()
    status["answered_at"] = answered_at

    result = evaluate_current(
        current=NEW,
        status=status,
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "status-answer-time-invalid", NEW)


def test_empty_dynamic_writer_inventory_fails_closed() -> None:
    status = _status()
    status["platforms"] = {}

    result = evaluate_current(
        current=NEW,
        status=status,
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "platform-inventory-empty", NEW)


def test_supervisor_rotation_cannot_hide_live_old_serving_process() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=_status(),
        old_process_alive=True,
        new_process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "old-serving-process-still-alive", NEW)


def test_health_200_from_old_listener_is_failure() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=_status(),
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={OLD.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "listener-owner-mismatch", NEW)


def test_ambiguous_listener_owners_fail_closed() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=_status(),
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={OLD.pid, NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "listener-owner-mismatch", NEW)


def test_platform_writer_must_match_control_socket_identity() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=_status(pid=OLD.pid),
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "status-answering-pid-mismatch", NEW)


def test_platform_writer_start_time_must_match_control_socket_identity() -> None:
    status = _status()
    status["platforms"]["telegram"]["writer_start_time"] = OLD.start_time

    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=status,
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(
        False, "platform-writer-start-time-mismatch:telegram", NEW
    )


def test_status_start_time_must_match_control_socket_identity() -> None:
    status = _status()
    status["start_time"] = OLD.start_time

    result = evaluate_current(
        current=NEW,
        status=status,
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "status-start-time-mismatch", NEW)


def test_dynamic_writer_inventory_cannot_disappear_before_signal() -> None:
    identities = [OLD, OLD, OLD]
    statuses = [
        _status(pid=OLD.pid),
        {**_status(pid=OLD.pid), "platforms": {}},
        {**_status(pid=OLD.pid), "platforms": {}},
    ]
    signals: list[ServingIdentity] = []

    def identify(home: Path) -> dict:
        current = identities.pop(0)
        return {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": current.pid,
            "start_time": current.start_time,
            "hermes_home": str(home),
            "code_sha": current.code_sha,
        }

    result = perform_verified_graceful_restart(
        HOME,
        adapters=RuntimeAdapters(
            identify=identify,
            status=lambda _home: statuses.pop(0),
            process_alive=lambda _pid: True,
            listener_owners=lambda _port: {OLD.pid},
            health_ok=lambda _url: True,
        ),
        signal_process=signals.append,
        timeout=0,
    )

    assert result.reason == "platform-not-connected:api_server"
    assert signals == []


def test_new_identity_listener_and_platform_readiness_are_success() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=_status(),
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(True, "replacement-healthy", NEW)


def test_current_verifies_only_platforms_published_by_runtime_inventory() -> None:
    status = _status()
    status["platforms"] = {
        "api_server": {
            "state": "connected",
            "writer_pid": NEW.pid,
            "writer_start_time": NEW.start_time,
        }
    }

    result = evaluate_current(
        current=NEW,
        status=status,
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result == RestartProbe(True, "gateway-healthy", NEW)


def test_replacement_must_run_expected_code_sha_when_supplied() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=NEW,
        status=_status(),
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
        expected_code_sha="expected-after-update",
    )

    assert result == RestartProbe(False, "replacement-code-sha-mismatch", NEW)


def test_same_pid_and_start_time_is_not_a_replacement() -> None:
    result = evaluate_replacement(
        old=OLD,
        current=OLD,
        status=_status(pid=OLD.pid),
        old_process_alive=False,
        new_process_alive=True,
        listener_pids={OLD.pid},
        health_ok=True,
    )

    assert result == RestartProbe(False, "serving-identity-did-not-change", OLD)


def test_current_runtime_requires_listener_and_zero_active_work() -> None:
    busy = evaluate_current(
        current=NEW,
        status=_status(active_agents=1),
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
        require_idle=True,
    )
    ready = evaluate_current(
        current=NEW,
        status=_status(),
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
        require_idle=True,
    )

    assert busy == RestartProbe(False, "active-work:1", NEW)
    assert ready == RestartProbe(True, "gateway-healthy", NEW)


@pytest.mark.parametrize("active_agents", [False, True, 0.0, "0", None])
def test_current_runtime_requires_exact_integer_active_work(active_agents) -> None:
    result = evaluate_current(
        current=NEW,
        status=_status(active_agents=active_agents),
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
        require_idle=True,
    )

    assert result == RestartProbe(False, "active-work-unknown", NEW)


@pytest.mark.parametrize(
    "field",
    ["answering_pid", "pid", "start_time", "writer_pid", "writer_start_time"],
)
def test_status_identity_fields_require_exact_integers(field) -> None:
    status = _status()
    if field.startswith("writer_"):
        status["platforms"]["telegram"][field] = float(
            NEW.pid if field == "writer_pid" else NEW.start_time
        )
    else:
        status[field] = float(NEW.pid if field != "start_time" else NEW.start_time)

    result = evaluate_current(
        current=NEW,
        status=status,
        process_alive=True,
        listener_pids={NEW.pid},
        health_ok=True,
    )

    assert result.ready is False


def test_control_socket_identity_is_home_scoped_and_typed() -> None:
    identity = parse_serving_identity(
        {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": 94350,
            "start_time": 22345,
            "hermes_home": str(HOME),
            "code_sha": "new",
        },
        expected_home=HOME,
    )

    assert identity == NEW


def test_probe_uses_control_socket_identity_and_independent_adapters() -> None:
    calls: list[str] = []

    result = probe_current_gateway(
        HOME,
        identify=lambda home: {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": NEW.pid,
            "start_time": NEW.start_time,
            "hermes_home": str(home),
            "code_sha": NEW.code_sha,
        },
        status=lambda home: _status(),
        process_alive=lambda pid: calls.append(f"pid:{pid}") or True,
        listener_owners=lambda port: calls.append(f"port:{port}") or {NEW.pid},
        health_ok=lambda url: calls.append(f"health:{url}") or True,
        port=8642,
        health_url="http://127.0.0.1:8642/health",
        require_idle=False,
    )

    assert result == RestartProbe(True, "gateway-healthy", NEW)
    assert calls == ["pid:94350", "port:8642", "health:http://127.0.0.1:8642/health"]


def test_verified_graceful_restart_refuses_active_gateway_without_signal() -> None:
    signals: list[int] = []
    adapters = RuntimeAdapters(
        identify=lambda home: {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": OLD.pid,
            "start_time": OLD.start_time,
            "hermes_home": str(home),
            "code_sha": OLD.code_sha,
        },
        status=lambda home: _status(pid=OLD.pid, active_agents=1),
        process_alive=lambda pid: True,
        listener_owners=lambda port: {OLD.pid},
        health_ok=lambda url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        signal_process=signals.append,
        port=8642,
        health_url="http://127.0.0.1:8642/health",
        timeout=1,
        poll_interval=0,
    )

    assert result == RestartProbe(False, "active-work:1", OLD)
    assert signals == []


def test_verified_graceful_restart_fails_closed_when_status_is_unavailable() -> None:
    signals: list[int] = []

    def missing_status(_home):
        raise OSError("socket unavailable")

    adapters = RuntimeAdapters(
        identify=lambda _home: {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": OLD.pid,
            "start_time": OLD.start_time,
            "hermes_home": str(OLD.hermes_home),
            "code_sha": OLD.code_sha,
        },
        status=missing_status,
        process_alive=lambda _pid: True,
        listener_owners=lambda _port: {OLD.pid},
        health_ok=lambda _url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        signal_process=signals.append,
    )

    assert result.ready is False
    assert "socket unavailable" in result.reason
    assert signals == []


@pytest.mark.parametrize("adapter_name", ["process_alive", "listener_owners", "health_ok"])
def test_verified_graceful_restart_fails_closed_on_probe_adapter_exception(
    adapter_name: str,
) -> None:
    signals: list[int] = []

    def explode(*_args):
        raise RuntimeError(f"{adapter_name} unavailable")

    adapters = RuntimeAdapters(
        identify=lambda home: {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": OLD.pid,
            "start_time": OLD.start_time,
            "hermes_home": str(home),
            "code_sha": OLD.code_sha,
        },
        status=lambda _home: _status(pid=OLD.pid),
        process_alive=explode if adapter_name == "process_alive" else lambda _pid: True,
        listener_owners=explode if adapter_name == "listener_owners" else lambda _port: {OLD.pid},
        health_ok=explode if adapter_name == "health_ok" else lambda _url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        signal_process=signals.append,
    )

    assert result.ready is False
    assert f"{adapter_name} unavailable" in result.reason
    assert signals == []


def test_verified_graceful_restart_signals_serving_pid_and_verifies_new_sha() -> None:
    signaled: list[ServingIdentity] = []

    def identity(home: Path) -> dict:
        current = NEW if signaled else OLD
        return {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": current.pid,
            "start_time": current.start_time,
            "hermes_home": str(home),
            "code_sha": current.code_sha,
        }

    def status(home: Path) -> dict:
        return _status(pid=NEW.pid if signaled else OLD.pid)

    adapters = RuntimeAdapters(
        identify=identity,
        status=status,
        process_alive=lambda pid: pid == NEW.pid if signaled else pid == OLD.pid,
        listener_owners=lambda port: {NEW.pid if signaled else OLD.pid},
        health_ok=lambda url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        signal_process=signaled.append,
        port=8642,
        health_url="http://127.0.0.1:8642/health",
        expected_code_sha=NEW.code_sha,
        timeout=1,
        poll_interval=0,
    )

    assert result == RestartProbe(True, "replacement-healthy", NEW)
    assert signaled == [OLD]


def test_default_restart_uses_atomic_control_verb_not_external_kill(
    monkeypatch,
) -> None:
    accepted = False
    requests: list[tuple[Path, str, dict]] = []

    def query(home, verb, *, request_fields, timeout):
        nonlocal accepted
        requests.append((home, verb, request_fields))
        accepted = True
        return {
            "accepted": True,
            "identity": {"pid": OLD.pid, "start_time": OLD.start_time},
            "signal": "SIGUSR1",
        }

    monkeypatch.setattr("gateway.control_socket.query_gateway_control", query)
    monkeypatch.setattr(
        "os.kill",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("external PID signal must not be used")
        ),
    )

    def identity(home: Path) -> dict:
        current = NEW if accepted else OLD
        return {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": current.pid,
            "start_time": current.start_time,
            "hermes_home": str(home),
            "code_sha": current.code_sha,
        }

    adapters = RuntimeAdapters(
        identify=identity,
        status=lambda _home: _status(pid=NEW.pid if accepted else OLD.pid),
        process_alive=lambda pid: pid == (NEW.pid if accepted else OLD.pid),
        listener_owners=lambda _port: {NEW.pid if accepted else OLD.pid},
        health_ok=lambda _url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        expected_code_sha=NEW.code_sha,
        timeout=1,
        poll_interval=0,
    )

    assert result == RestartProbe(True, "replacement-healthy", NEW)
    assert requests == [
        (
            HOME,
            "restart-if-idle",
            {"expected_pid": OLD.pid, "expected_start_time": OLD.start_time},
        )
    ]


@pytest.mark.parametrize("field", ["pid", "start_time"])
def test_atomic_restart_ack_identity_requires_exact_integers(monkeypatch, field) -> None:
    identity: dict[str, object] = {"pid": OLD.pid, "start_time": OLD.start_time}
    identity[field] = float(OLD.pid if field == "pid" else OLD.start_time)
    monkeypatch.setattr(
        "gateway.control_socket.query_gateway_control",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "identity": identity,
            "signal": "SIGUSR1",
        },
    )
    adapters = RuntimeAdapters(
        identify=lambda home: {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": OLD.pid,
            "start_time": OLD.start_time,
            "hermes_home": str(home),
            "code_sha": OLD.code_sha,
        },
        status=lambda _home: _status(pid=OLD.pid),
        process_alive=lambda pid: pid == OLD.pid,
        listener_owners=lambda _port: {OLD.pid},
        health_ok=lambda _url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        expected_code_sha=NEW.code_sha,
        timeout=0,
    )

    assert result == RestartProbe(False, "atomic-restart-request-rejected", OLD)


def test_signal_adapter_receives_identity_revalidated_immediately_before_signal() -> None:
    identities = [OLD, OLD, OLD._replace(start_time=OLD.start_time + 1)]
    signals: list[ServingIdentity] = []

    def identify(home: Path) -> dict:
        current = identities.pop(0)
        return {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": current.pid,
            "start_time": current.start_time,
            "hermes_home": str(home),
            "code_sha": current.code_sha,
        }

    adapters = RuntimeAdapters(
        identify=identify,
        status=lambda _home: _status(pid=OLD.pid),
        process_alive=lambda _pid: True,
        listener_owners=lambda _port: {OLD.pid},
        health_ok=lambda _url: True,
    )

    result = perform_verified_graceful_restart(
        HOME,
        adapters=adapters,
        signal_process=signals.append,
        expected_code_sha="new",
        timeout=0,
    )

    assert result.reason == "serving-identity-changed-before-signal"
    assert signals == []


def test_cli_ready_prints_only_verified_replacement_identity(capsys) -> None:
    adapters = RuntimeAdapters(
        identify=lambda home: {
            "protocol": 1,
            "kind": "hermes-gateway",
            "pid": NEW.pid,
            "start_time": NEW.start_time,
            "hermes_home": str(home),
            "code_sha": NEW.code_sha,
        },
        status=lambda home: _status(),
        process_alive=lambda pid: pid == NEW.pid,
        listener_owners=lambda port: {NEW.pid},
        health_ok=lambda url: True,
    )

    exit_code = main(
        [
            "ready",
            "--home",
            str(HOME),
            "--old-pid",
            str(OLD.pid),
            "--old-start",
            str(OLD.start_time),
        ],
        adapters=adapters,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "94350 22345\n"
    assert captured.err == ""


def test_direct_script_invocation_imports_top_level_gateway_package(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "hermes_cli" / "gateway_restart_contract.py"

    result = subprocess.run(
        [sys.executable, str(script), "identity", "--home", str(tmp_path)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 1
    assert "control-identity-missing" in result.stderr
    assert "'gateway' is not a package" not in result.stderr
