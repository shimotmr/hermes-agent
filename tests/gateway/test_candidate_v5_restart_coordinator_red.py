"""Candidate-v5 RED contract for production restart coordination.

These tests intentionally describe the next coordinator slice.  They use the
real admission barrier, cron bootstrap, plugin dispatch, restart snapshot
helpers, and control transport; only process/service signal boundaries are
replaced with recording callables.
"""
from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_restart_coordinator_from_relaunch_attestation(monkeypatch):
    """These tests exercise coordinator atomicity, not OS manager probing."""
    import gateway.restart_relaunch as relaunch

    monkeypatch.setattr(relaunch, "verify_relaunch_attestation", lambda *args, **kwargs: True)

from gateway.config import Platform
from gateway.control_socket import GatewayControlServer, query_gateway_control
from gateway.restart_runtime import (
    GatewayAdmissionBarrier,
    configured_platform_names,
    live_platform_writer_snapshot,
)
from hermes_cli.gateway_restart_contract import (
    ExpectedGatewayIdentity,
    authorize_restart_if_idle,
)


def _identity(home):
    return ExpectedGatewayIdentity(
        protocol=1,
        kind="hermes-gateway",
        pid=os.getpid(),
        start_time=1234,
        hermes_home=str(home.resolve()),
        code_sha="a" * 40,
        required_platforms=(),
    )


def _snapshot(home, *, configured=(), platforms=None):
    return {
        **_identity(home).to_mapping(),
        "answering_pid": os.getpid(),
        "signal_target_pid": os.getpid(),
        "supervisor_pid": os.getppid(),
        "supervisor": "launchd",
        "gateway_state": "running",
        "session_store": {"status": "ok"},
        "active_agents": 0,
        "configured_platforms": list(configured),
        "platforms": platforms or {},
        "obligations": {
            "delivery_queue": 0,
            "delivery_ledger": 0,
            "pending_final": 0,
            "drain": 0,
            "delegation_workers": 0,
            "updater": 0,
            "update_lock": 0,
            "oauth_refresh": 0,
            "oauth_token_lock": 0,
        },
    }


def test_authoritative_config_read_error_fails_closed(tmp_path):
    class BrokenConfig:
        def get_connected_platforms(self):
            raise RuntimeError("deterministic config read failure")

    configured = configured_platform_names(BrokenConfig())
    accepted, reason = authorize_restart_if_idle(
        _snapshot(tmp_path, configured=configured), _identity(tmp_path)
    )

    assert accepted is False
    assert reason == "configured-platforms-unavailable"


def test_live_writer_health_is_independent_of_adapter_presence(tmp_path):
    class RetryingAdapter:
        def restart_writer_health_probe(self):
            return {"state": "retrying", "writer_pid": None, "writer_start_time": None}

    config = SimpleNamespace(get_connected_platforms=lambda: [Platform.TELEGRAM])
    runner = SimpleNamespace(config=config, adapters={Platform.TELEGRAM: RetryingAdapter()})
    platforms = live_platform_writer_snapshot(
        runner, expected_pid=os.getpid(), expected_start_time=1234
    )

    assert platforms["telegram"] == {
        "state": "retrying",
        "writer_pid": None,
        "writer_start_time": None,
    }



def test_missing_writer_probe_cannot_authorize_synthetic_identity(tmp_path):
    class LegacyAdapter:
        send_path_degraded = False

    config = SimpleNamespace(get_connected_platforms=lambda: [Platform.TELEGRAM])
    runner = SimpleNamespace(config=config, adapters={Platform.TELEGRAM: LegacyAdapter()})
    platforms = live_platform_writer_snapshot(
        runner, expected_pid=os.getpid(), expected_start_time=1234
    )

    accepted, reason = authorize_restart_if_idle(
        _snapshot(tmp_path, configured=("telegram",), platforms=platforms),
        _identity(tmp_path),
    )

    assert accepted is False
    assert reason == "platform-telegram-unhealthy"


@pytest.mark.parametrize("probe_result", [RuntimeError("probe failed"), "malformed"])
def test_unavailable_writer_attestation_fails_closed(tmp_path, probe_result):
    class UnavailableProbeAdapter:
        def restart_writer_health_probe(self):
            if isinstance(probe_result, Exception):
                raise probe_result
            return probe_result

    config = SimpleNamespace(get_connected_platforms=lambda: [Platform.TELEGRAM])
    runner = SimpleNamespace(
        config=config,
        adapters={Platform.TELEGRAM: UnavailableProbeAdapter()},
    )
    platforms = live_platform_writer_snapshot(
        runner, expected_pid=os.getpid(), expected_start_time=1234
    )

    accepted, reason = authorize_restart_if_idle(
        _snapshot(tmp_path, configured=("telegram",), platforms=platforms),
        _identity(tmp_path),
    )

    assert accepted is False
    assert reason == "platform-telegram-unhealthy"


@pytest.mark.asyncio
async def test_in_process_cron_dispatch_is_inside_restart_admission(monkeypatch):
    import cron.scheduler_provider as scheduler_provider
    import gateway.run as gateway_run

    barrier = GatewayAdmissionBarrier()
    barrier.close("restart")
    captured = {}
    started = threading.Event()
    provider = scheduler_provider.InProcessCronScheduler()

    def start(_stop, **kwargs):
        captured.update(kwargs)
        started.set()

    monkeypatch.setattr(provider, "start", start)
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: provider)
    monkeypatch.setattr(scheduler_provider, "scheduler_for_profile_mode", lambda *_a, **_k: provider)
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", lambda *_a, **_k: None)
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=False),
        adapters={},
        _draining=False,
        _external_drain_active=False,
        _restart_admission_barrier=barrier,
    )

    stop, _, cron_thread, housekeeping_thread = gateway_run._start_gateway_start_cron_and_housekeeping(runner)
    assert started.wait(1)
    cron_thread.join(1)
    housekeeping_thread.join(1)
    stop.set()

    assert captured["can_dispatch"]() is False


@pytest.mark.asyncio
async def test_plugin_cross_thread_injection_holds_admission_until_routed():
    from gateway.run import GatewayRunner

    lookup_entered = asyncio.Event()
    release_lookup = asyncio.Event()
    barrier = GatewayAdmissionBarrier()

    class Store:
        async def lookup_by_session_key(self, _key):
            lookup_entered.set()
            await release_lookup.wait()
            return None

    runner = SimpleNamespace(
        _running=True,
        _draining=False,
        _restart_admission_barrier=barrier,
        async_session_store=Store(),
    )
    dispatch = GatewayRunner._dispatch_plugin_message_injection.__get__(runner)
    task = asyncio.create_task(dispatch(session_key="s", content="x", plugin_id="p"))
    await asyncio.wait_for(lookup_entered.wait(), 1)
    try:
        assert barrier.active_admissions() == 1
    finally:
        release_lookup.set()
        await task


@pytest.mark.asyncio
async def test_concurrent_restart_is_exactly_once_over_real_control_transport(tmp_path, monkeypatch):
    if not hasattr(asyncio, "start_unix_server"):
        pytest.skip("POSIX control transport required")

    proof = {"manager": "launchd", "service": "gui/501/test.gateway", "policy": "keepalive",
             "pid": os.getpid(), "start_time": 1234}
    monkeypatch.setattr("gateway.restart_relaunch.probe_gateway_relaunch", lambda *_: proof)
    signals = []
    snapshot = _snapshot(tmp_path)
    server = GatewayControlServer(
        tmp_path,
        restart_snapshot=lambda: snapshot,
        graceful_signal=lambda: signals.append("SIGUSR1"),
        sigusr1_supported=True,
        admission_barrier=GatewayAdmissionBarrier(),
    )
    assert await server.start() is True
    gate = threading.Barrier(2)

    def request():
        gate.wait()
        return query_gateway_control(
            tmp_path,
            "restart-if-idle",
            request_fields={"expected_identity": _identity(tmp_path).to_mapping(), "relaunch_attestation": proof},
            timeout=2,
        )

    try:
        first, second = await asyncio.gather(asyncio.to_thread(request), asyncio.to_thread(request))
    finally:
        await server.stop()

    replies = [first, second]
    assert sum(bool(reply and reply.get("accepted") is True) for reply in replies) == 1
    assert sorted(reply["reason"] for reply in replies if reply) == ["accepted", "restart-already-requested"]
    assert signals == ["SIGUSR1"]
