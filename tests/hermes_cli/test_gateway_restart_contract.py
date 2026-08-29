from pathlib import Path

from hermes_cli.gateway_restart_contract import (
    RestartProbe,
    RuntimeAdapters,
    ServingIdentity,
    evaluate_current,
    evaluate_replacement,
    main,
    parse_serving_identity,
    probe_current_gateway,
)


HOME = Path("/tmp/hermes-home")
OLD = ServingIdentity(pid=68821, start_time=12345, hermes_home=HOME, code_sha="old")
NEW = ServingIdentity(pid=94350, start_time=22345, hermes_home=HOME, code_sha="new")


def _status(pid: int = NEW.pid, *, active_agents: int = 0) -> dict:
    return {
        "gateway_state": "running",
        "active_agents": active_agents,
        "answering_pid": pid,
        "platforms": {
            name: {"state": "connected", "writer_pid": pid}
            for name in ("api_server", "webhook", "telegram")
        },
    }


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
