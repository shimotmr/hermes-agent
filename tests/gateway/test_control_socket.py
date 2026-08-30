"""Tests for the gateway control socket (#92091 migration step 1)."""

import asyncio
import json
import socket
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    identify_gateway,
    query_gateway_control,
    resolve_client_socket_path,
    resolve_server_socket_path,
    windows_pipe_name,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-socket transport; the named-pipe half is covered on the wine2e lane",
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def _serve(home: Path, handlers=None):
    """Context helper: start a server in a fresh loop, yield inside coro."""
    return GatewayControlServer(home, verb_handlers=handlers)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_short_home_binds_in_home(tmp_path: Path):
    # A home short enough for sun_path binds in-home with no pointer.
    # tmp_path can exceed the limit on CI runners, so build one in the
    # system temp root directly.
    import tempfile

    try:
        short_root = Path(tempfile.mkdtemp(prefix="hgw-", dir="/tmp"))
    except OSError:
        pytest.skip("/tmp not writable on this host")
    try:
        short_home = short_root / ".hermes"
        short_home.mkdir()
        assert len(str(short_home / "gateway.sock").encode()) <= 100
        bind, pointer = resolve_server_socket_path(short_home)
        assert bind == short_home / "gateway.sock"
        assert pointer is None
    finally:
        import shutil

        shutil.rmtree(short_root, ignore_errors=True)


def test_long_home_uses_pointer_fallback(tmp_path: Path):
    deep = tmp_path / ("x" * 120) / ".hermes"
    deep.mkdir(parents=True)
    bind, pointer = resolve_server_socket_path(deep)
    assert bind != deep / "gateway.sock"
    assert len(str(bind).encode()) <= 100
    assert pointer == deep / "gateway.sock.path"


def test_client_resolution_prefers_direct_then_pointer(home: Path, tmp_path: Path):
    assert resolve_client_socket_path(home) is None
    # pointer file to an existing socket-ish file
    target = tmp_path / "elsewhere.sock"
    target.touch()
    (home / "gateway.sock.path").write_text(str(target))
    assert resolve_client_socket_path(home) == target
    # direct file wins over pointer
    direct = home / "gateway.sock"
    direct.touch()
    assert resolve_client_socket_path(home) == direct


def test_windows_pipe_name_is_stable_and_home_scoped(tmp_path: Path):
    a = windows_pipe_name(tmp_path / "a")
    b = windows_pipe_name(tmp_path / "b")
    assert a.startswith(r"\\.\pipe\hermes-gateway-")
    assert a != b
    assert a == windows_pipe_name(tmp_path / "a")


# ---------------------------------------------------------------------------
# Server lifecycle + verbs (real sockets, real event loop)
# ---------------------------------------------------------------------------

def test_server_answers_identify_and_status(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home,
            verb_handlers={
                "identify": lambda: {"pid": 4242, "code_sha": "abc123", "protocol": 1},
                "status": lambda: {"gateway_state": "running"},
            },
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            ident = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "identify")
            )
            status = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "status")
            )
            return ident, status
        finally:
            await server.stop()

    ident, status = _run(scenario())
    assert ident == {"pid": 4242, "code_sha": "abc123", "protocol": 1}
    assert status == {"gateway_state": "running"}


@pytest.mark.parametrize(
    "response",
    [
        {"ok": True, "id": 999, "protocol": CONTROL_PROTOCOL_VERSION, "result": {}},
        {"ok": True, "id": 1, "protocol": 999, "result": {}},
        {"ok": True, "protocol": CONTROL_PROTOCOL_VERSION, "result": {}},
        {"ok": True, "id": 1, "result": {}},
        {"ok": True, "id": True, "protocol": True, "result": {}},
    ],
)
def test_client_rejects_response_with_wrong_or_missing_envelope(
    home: Path, monkeypatch, response
):
    import gateway.control_socket as control_socket

    monkeypatch.setattr(
        control_socket,
        "_query_unix_socket",
        lambda *_args, **_kwargs: json.dumps(response).encode(),
    )

    assert query_gateway_control(home, "identify") is None


def test_unix_reader_rejects_oversized_newline_terminated_response(
    home: Path, monkeypatch
):
    import gateway.control_socket as control_socket

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _timeout):
            return None

        def connect(self, _path):
            return None

        def sendall(self, _request):
            return None

        def recv(self, _size):
            return b"x" * (control_socket._MAX_RESPONSE_BYTES + 1) + b"\n"

    monkeypatch.setattr(
        control_socket, "resolve_client_socket_path", lambda _home: home / "sock"
    )
    monkeypatch.setattr(
        control_socket.socket, "socket", lambda *_args: FakeSocket()
    )

    assert control_socket._query_unix_socket(home, b"request\n", 1) is None


def test_windows_reader_rejects_oversized_newline_terminated_response(
    home: Path, monkeypatch
):
    import builtins
    import gateway.control_socket as control_socket

    class FakePipe:
        def write(self, _request):
            return None

        def read(self, _size):
            return b"x" * (control_socket._MAX_RESPONSE_BYTES + 1) + b"\n"

        def close(self):
            return None

    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: FakePipe())

    assert control_socket._query_windows_pipe(home, b"request\n", 1) is None


def test_unknown_verb_and_malformed_request(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            unknown = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "restart")
            )

            def raw_garbage():
                path = resolve_client_socket_path(home)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect(str(path))
                    s.sendall(b"this is not json\n")
                    return s.recv(65536)

            garbage_reply = await loop.run_in_executor(None, raw_garbage)
            return unknown, garbage_reply
        finally:
            await server.stop()

    unknown, garbage_reply = _run(scenario())
    # unknown verb → ok:false → client returns None (fallback signal)
    assert unknown is None
    payload = json.loads(garbage_reply.decode())
    assert payload["ok"] is False
    assert payload["protocol"] == CONTROL_PROTOCOL_VERSION


def test_stop_removes_socket_and_pointer(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        bind, _ = resolve_server_socket_path(home)
        assert bind.exists()
        await server.stop()
        return bind

    bind = _run(scenario())
    assert not bind.exists()
    assert resolve_client_socket_path(home) is None
    # queries after stop cleanly return None
    assert query_gateway_control(home, "identify") is None


def test_stale_socket_file_is_replaced_on_bind(home: Path):
    # Plant the stale file at wherever the server will actually bind
    # (in-home OR the temp-dir fallback, depending on path length).
    bind, _ = resolve_server_socket_path(home)
    bind.parent.mkdir(parents=True, exist_ok=True)
    bind.touch()  # crashed predecessor's leftover

    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 7}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 7}


def test_long_home_end_to_end_via_pointer(tmp_path: Path):
    deep = tmp_path / ("p" * 120) / ".hermes"
    deep.mkdir(parents=True)

    async def scenario():
        server = GatewayControlServer(
            deep, verb_handlers={"identify": lambda: {"pid": 9}}
        )
        assert await server.start()
        try:
            assert (deep / "gateway.sock.path").is_file()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(deep))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 9}
    assert not (deep / "gateway.sock.path").exists()


def test_no_socket_returns_none_fast(home: Path):
    assert identify_gateway(home) is None
    assert query_gateway_control(home, "status") is None


def test_query_rejects_unserializable_request_fields_without_raising(home: Path):
    assert (
        query_gateway_control(home, "status", request_fields={"bad": object()})
        is None
    )


def test_query_sends_fields_to_request_aware_handler(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home,
            request_handlers={
                "echo-request": lambda request: {
                    "expected_pid": request["expected_pid"],
                    "expected_start_time": request["expected_start_time"],
                }
            },
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: query_gateway_control(
                    home,
                    "echo-request",
                    request_fields={
                        "expected_pid": 123,
                        "expected_start_time": 456,
                    },
                ),
            )
        finally:
            await server.stop()

    assert _run(scenario()) == {
        "expected_pid": 123,
        "expected_start_time": 456,
    }


def test_restart_if_idle_accepts_live_reservation_and_defers_signal(
    home: Path, monkeypatch
):
    import os
    import signal

    import gateway.control_socket as control_socket
    import gateway.status as gateway_status

    pid = os.getpid()
    start_time = 987654
    monkeypatch.setattr(
        gateway_status, "get_process_start_time", lambda candidate: start_time
    )
    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: (_ for _ in ()).throw(AssertionError("persisted status must not authorize")),
    )
    kills = []
    cancellations = []
    monkeypatch.setattr(control_socket.os, "kill", lambda *args: kills.append(args))

    def handler(request):
        return control_socket.restart_if_idle(
            request,
            try_reserve=lambda: True,
            confirm_reservation=lambda: True,
            cancel_reservation=lambda: cancellations.append("cancel"),
        )

    server = GatewayControlServer(
        home,
        request_handlers={"restart-if-idle": handler},
    )
    prepared = server.prepare_request_line(
        json.dumps(
            {
                "verb": "restart-if-idle",
                "id": 1,
                "protocol": CONTROL_PROTOCOL_VERSION,
                "expected_pid": pid,
                "expected_start_time": start_time,
            }
        ).encode()
    )
    response = json.loads(prepared.payload)

    assert response == {
        "ok": True,
        "id": 1,
        "protocol": CONTROL_PROTOCOL_VERSION,
        "result": {
            "accepted": True,
            "identity": {"pid": pid, "start_time": start_time},
            "signal": "SIGUSR1",
        },
    }
    assert kills == []
    assert prepared.after_drain is not None
    prepared.after_drain()
    assert kills == [(pid, signal.SIGUSR1)]
    assert cancellations == []


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"protocol": CONTROL_PROTOCOL_VERSION, "unexpected": True},
        {"protocol": CONTROL_PROTOCOL_VERSION + 1},
    ],
)
def test_restart_if_idle_rejects_missing_wrong_or_extra_protocol_fields(
    home: Path, monkeypatch, extra
):
    import gateway.control_socket as control_socket
    import gateway.status as gateway_status

    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda _pid: 42)
    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "active_agents": 0,
            "pid": 123,
            "start_time": 42,
        },
    )
    kills = []
    monkeypatch.setattr(control_socket.os, "getpid", lambda: 123)
    monkeypatch.setattr(control_socket.os, "kill", lambda *args: kills.append(args))
    request = {
        "verb": "restart-if-idle",
        "expected_pid": 123,
        "expected_start_time": 42,
        **extra,
    }
    server = GatewayControlServer(
        home,
        request_handlers={"restart-if-idle": control_socket.restart_if_idle},
    )

    response = json.loads(server.handle_request_line(json.dumps(request).encode()))

    assert response["ok"] is False
    assert kills == []


@pytest.mark.parametrize("request_id", [None, True, 1.0, "1", 2])
def test_restart_if_idle_requires_exact_request_id(home: Path, monkeypatch, request_id):
    import gateway.control_socket as control_socket
    import gateway.status as gateway_status

    monkeypatch.setattr(control_socket.os, "getpid", lambda: 123)
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda _pid: 42)
    request = {
        "verb": "restart-if-idle",
        "protocol": CONTROL_PROTOCOL_VERSION,
        "expected_pid": 123,
        "expected_start_time": 42,
    }
    if request_id is not None:
        request["id"] = request_id
    with pytest.raises(ValueError, match="request id"):
        control_socket.restart_if_idle(
            request,
            try_reserve=lambda: True,
            confirm_reservation=lambda: True,
            cancel_reservation=lambda: None,
        )


def test_restart_signal_is_cancelled_when_reservation_is_superseded(monkeypatch):
    import gateway.control_socket as control_socket
    import gateway.status as gateway_status

    monkeypatch.setattr(control_socket.os, "getpid", lambda: 123)
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda _pid: 42)
    kills: list[tuple[int, int]] = []
    cancellations: list[str] = []
    monkeypatch.setattr(control_socket.os, "kill", lambda *args: kills.append(args))
    result = control_socket.restart_if_idle(
        {
            "verb": "restart-if-idle",
            "id": 1,
            "protocol": CONTROL_PROTOCOL_VERSION,
            "expected_pid": 123,
            "expected_start_time": 42,
        },
        try_reserve=lambda: True,
        confirm_reservation=lambda: False,
        cancel_reservation=lambda: cancellations.append("cancel"),
    )

    result.after_drain()

    assert kills == []
    assert cancellations == ["cancel"]


@pytest.mark.parametrize(
    ("request_fields", "reserve"),
    [
        ({"expected_pid": 999999, "expected_start_time": 42}, True),
        ({"expected_pid": True, "expected_start_time": 42}, True),
        ({"expected_pid": 123}, True),
        ({"expected_pid": 123, "expected_start_time": 42}, False),
    ],
)
def test_restart_if_idle_fails_closed_without_signalling(
    home: Path, monkeypatch, request_fields, reserve
):
    import gateway.control_socket as control_socket
    import gateway.status as gateway_status

    monkeypatch.setattr(control_socket.os, "getpid", lambda: 123)
    monkeypatch.setattr(
        gateway_status, "get_process_start_time", lambda candidate: 42
    )
    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: (_ for _ in ()).throw(AssertionError("persisted status must not be read")),
    )
    kills = []
    cancellations = []
    monkeypatch.setattr(control_socket.os, "kill", lambda *args: kills.append(args))

    def handler(request):
        return control_socket.restart_if_idle(
            request,
            try_reserve=lambda: reserve,
            confirm_reservation=lambda: True,
            cancel_reservation=lambda: cancellations.append("cancel"),
        )

    server = GatewayControlServer(
        home,
        request_handlers={"restart-if-idle": handler},
    )
    prepared = server.prepare_request_line(
        json.dumps(
            {
                "verb": "restart-if-idle",
                "id": 1,
                "protocol": CONTROL_PROTOCOL_VERSION,
                **request_fields,
            }
        ).encode()
    )
    response = json.loads(prepared.payload)

    assert response["ok"] is False
    assert prepared.after_drain is None
    assert kills == []
    assert cancellations == []


def test_restart_signal_occurs_only_after_ack_drain(home: Path):
    import gateway.control_socket as control_socket

    events: list[str] = []

    class Reader:
        async def readline(self):
            return b'{"verb":"restart-if-idle"}\n'

    class Writer:
        def write(self, _payload):
            events.append("write")

        async def drain(self):
            events.append("drain")

        def close(self):
            events.append("close")

    server = GatewayControlServer(
        home,
        request_handlers={
            "restart-if-idle": lambda _request: control_socket.DeferredControlResult(
                result={"accepted": True},
                after_drain=lambda: events.append("signal"),
            )
        },
    )

    _run(server._handle_connection(cast(Any, Reader()), cast(Any, Writer())))

    assert events[:3] == ["write", "drain", "signal"]


def test_restart_drain_failure_never_signals_and_cancels_reservation(home: Path):
    import gateway.control_socket as control_socket

    events: list[str] = []

    class Reader:
        async def readline(self):
            return b'{"verb":"restart-if-idle"}\n'

    class Writer:
        def write(self, _payload):
            events.append("write")

        async def drain(self):
            events.append("drain-failed")
            raise ConnectionError("peer closed")

        def close(self):
            events.append("close")

    server = GatewayControlServer(
        home,
        request_handlers={
            "restart-if-idle": lambda _request: control_socket.DeferredControlResult(
                result={"accepted": True},
                after_drain=lambda: events.append("signal"),
                on_abort=lambda: events.append("cancel"),
            )
        },
    )

    _run(server._handle_connection(cast(Any, Reader()), cast(Any, Writer())))

    assert "signal" not in events
    assert events.count("cancel") == 1
    assert events[:3] == ["write", "drain-failed", "cancel"]


def test_oversized_restart_ack_discards_signal_and_cancels_reservation(home: Path):
    import gateway.control_socket as control_socket

    events: list[str] = []
    server = GatewayControlServer(
        home,
        request_handlers={
            "restart-if-idle": lambda _request: control_socket.DeferredControlResult(
                result={"accepted": True, "padding": "x" * (600 * 1024)},
                after_drain=lambda: events.append("signal"),
                on_abort=lambda: events.append("cancel"),
            )
        },
    )

    prepared = server.prepare_request_line(b'{"verb":"restart-if-idle"}')

    assert json.loads(prepared.payload)["ok"] is False
    assert prepared.after_drain is None
    assert events == ["cancel"]


def test_server_wire_response_respects_exact_size_cap(home: Path, monkeypatch):
    import gateway.control_socket as control_socket

    server = GatewayControlServer(
        home, verb_handlers={"blob": lambda: {"padding": "x" * 256}}
    )
    request = b'{"verb":"blob","id":1,"protocol":1}'
    uncapped = server.prepare_request_line(request)
    encoded_size = len(uncapped.payload) - 1
    monkeypatch.setattr(control_socket, "_MAX_RESPONSE_BYTES", encoded_size)

    capped = server.prepare_request_line(request)

    assert len(capped.payload) <= encoded_size
    assert json.loads(capped.payload)["ok"] is False
    assert json.loads(capped.payload)["error"] == "response too large"


def test_default_identify_payload_shape(home: Path, monkeypatch):
    """The real identify handler carries the fleet-consumer contract fields."""
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def scenario():
        server = GatewayControlServer(home)  # default handlers
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    ident = _run(scenario())
    assert ident is not None
    assert ident["protocol"] == CONTROL_PROTOCOL_VERSION
    assert ident["pid"] == __import__("os").getpid()
    # contract keys exist even when values are None/absent-degradable
    for key in ("hermes_home", "supervisor", "kind", "start_time"):
        assert key in ident
    assert ident["supervisor"] in {
        "systemd",
        "launchd",
        "desktop",
        "external",
        "manual",
    }


# ---------------------------------------------------------------------------
# Consumer integration: fleet matrix + inventory prefer socket, fall back
# ---------------------------------------------------------------------------

def _fake_identity(pid: int, sha: str):
    return {
        "protocol": 1,
        "pid": pid,
        "code_sha": sha,
        "code_version": "9.9.9",
        "supervisor": "systemd",
        "kind": "hermes-gateway",
    }


def test_collect_fleet_versions_prefers_socket(tmp_path: Path, monkeypatch):
    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    # stale state file that would report a WRONG pid — socket must win
    (home / "gateway_state.json").write_text(
        json.dumps({"pid": 1, "code_sha": "stalefile", "kind": "hermes-gateway"})
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(31337, "HEADSHA"),
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    entry = fleet[0]
    assert entry["pid"] == 31337
    assert entry["state"] == "current"
    assert entry["source"] == "socket"


def test_collect_fleet_versions_falls_back_to_state_file(tmp_path: Path, monkeypatch):
    import os

    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway", lambda h, **kw: None
    )
    (home / "gateway_state.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # a live pid so _pid_exists passes
                "code_sha": "OLDSHA",
                "kind": "hermes-gateway",
            }
        )
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    assert fleet[0]["pid"] == os.getpid()
    assert fleet[0]["state"] == "stale"
    assert "source" not in fleet[0]


def test_runtime_inventory_dedupes_same_pid_across_homes(tmp_path: Path, monkeypatch):
    """One multiplex gateway answering identify for two profile homes must
    yield exactly ONE runtime record (reviewer point on #92447)."""
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()
    profiles_root = tmp_path / "profiles"
    (profiles_root / "coder").mkdir(parents=True)

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(777, "SHA777"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1, [r.__dict__ for r in gws]
    assert gws[0].pid == 777


def test_runtime_inventory_prefers_socket_supervisor(tmp_path: Path, monkeypatch):
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(555, "SHA555"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1
    assert gws[0].pid == 555
    # supervisor comes from the gateway's own declaration, not a PID scan
    assert gws[0].supervisor == "systemd"
    assert gws[0].code_sha == "SHA555"
