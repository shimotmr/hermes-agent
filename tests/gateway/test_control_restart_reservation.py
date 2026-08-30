import asyncio
import threading
from typing import Any, cast

from gateway.run import GatewayRunner, _call_on_loop_for_bool


def _runner(*, active_work: int) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._restart_task_started = False
    runner._stop_task = None
    runner._draining = False
    runner._external_drain_active = False
    runner._idle_restart_reserved = False
    runner._shutdown_event = asyncio.Event()
    runner._restart_requested = False
    runner._active_work_count = lambda: active_work
    runner._active_work_count_for_restart = lambda: active_work
    runner._update_runtime_status = lambda *_args, **_kwargs: None
    return runner


def test_idle_restart_reservation_uses_live_active_work_count():
    runner = _runner(active_work=1)

    assert runner.try_reserve_idle_restart() is False
    assert runner._idle_restart_reserved is False
    assert runner._draining is False


def test_idle_restart_rejects_unknown_cron_count(monkeypatch):
    import cron.scheduler

    runner = _runner(active_work=0)
    runner._active_work_count_for_restart = (
        GatewayRunner._active_work_count_for_restart.__get__(runner)
    )
    runner._running_agents = {}
    runner.adapters = {}

    def fail_reservation():
        raise RuntimeError("cron counter unavailable")

    monkeypatch.setattr(
        cron.scheduler, "reserve_restart_dispatch", fail_reservation
    )

    assert runner.try_reserve_idle_restart() is False
    assert runner._idle_restart_reserved is False
    assert runner._draining is False


def test_idle_restart_rejects_unknown_api_count(monkeypatch):
    import cron.scheduler
    from types import SimpleNamespace

    from gateway.config import Platform

    runner = _runner(active_work=0)
    runner._active_work_count_for_restart = (
        GatewayRunner._active_work_count_for_restart.__get__(runner)
    )
    runner._running_agents = {}
    runner.adapters = {
        Platform.API_SERVER: SimpleNamespace(
            active_agent_work_count=lambda: (_ for _ in ()).throw(
                RuntimeError("api counter unavailable")
            )
        )
    }
    monkeypatch.setattr(cron.scheduler, "get_running_job_ids", lambda: set())

    assert runner.try_reserve_idle_restart() is False
    assert runner._idle_restart_reserved is False
    assert runner._draining is False


def test_existing_api_work_is_not_rejected_by_a_transient_cron_pause(monkeypatch):
    import cron.scheduler
    from types import SimpleNamespace

    from gateway.config import Platform

    runner = _runner(active_work=0)
    runner._active_work_count_for_restart = (
        GatewayRunner._active_work_count_for_restart.__get__(runner)
    )
    runner._running_agents = {}
    runner.adapters = {
        Platform.API_SERVER: cast(
            Any, SimpleNamespace(strict_active_agent_work_count=lambda: 1)
        )
    }
    reserve_calls = []
    monkeypatch.setattr(
        cron.scheduler,
        "reserve_restart_dispatch",
        lambda: reserve_calls.append(True) or (object(), frozenset()),
    )

    assert runner.try_reserve_idle_restart() is False
    assert reserve_calls == []
    assert getattr(runner, "_idle_restart_cron_pause_token", None) is None
    assert runner._draining is False


def test_idle_restart_reservation_closes_dispatch_gate_before_count():
    runner = _runner(active_work=0)

    def count_after_gate_closed():
        assert runner._draining is True
        return 0

    runner._active_work_count = count_after_gate_closed

    assert runner.try_reserve_idle_restart() is True
    assert runner._idle_restart_reserved is True
    assert runner._draining is True


def test_status_write_failure_rolls_back_idle_restart_reservation(monkeypatch):
    import gateway.status

    runner = _runner(active_work=0)
    # Exercise the real best-effort status wrapper rather than replacing the
    # wrapper itself; the reservation boundary must observe its failure result.
    runner._update_runtime_status = GatewayRunner._update_runtime_status.__get__(runner)

    def fail_status(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(gateway.status, "write_runtime_status", fail_status)

    assert runner.try_reserve_idle_restart() is False
    assert runner._idle_restart_reserved is False
    assert runner._draining is False


def test_cancel_reservation_reopens_gateway_after_ack_failure():
    runner = _runner(active_work=0)
    assert runner.try_reserve_idle_restart() is True

    runner.cancel_idle_restart_reservation()

    assert runner._idle_restart_reserved is False
    assert runner._draining is False


def test_cancel_does_not_reopen_superseding_shutdown():
    runner = _runner(active_work=0)
    assert runner.try_reserve_idle_restart() is True
    runner._restart_task_started = True

    runner.cancel_idle_restart_reservation()

    assert runner._idle_restart_reserved is False
    assert runner._draining is True


def test_existing_stop_task_rejects_idle_restart_reservation():
    runner = _runner(active_work=0)
    runner._stop_task = cast(Any, object())

    assert runner.try_reserve_idle_restart() is False
    assert runner._idle_restart_reserved is False
    assert runner._draining is False


def test_confirm_reservation_rejects_superseding_stop_or_external_drain():
    for owner in ("stop", "external"):
        runner = _runner(active_work=0)
        assert runner.try_reserve_idle_restart() is True
        if owner == "stop":
            runner._stop_task = cast(Any, object())
        else:
            runner._external_drain_active = True

        assert runner.confirm_idle_restart_reservation() is False


def test_confirm_reservation_accepts_unchanged_owner():
    runner = _runner(active_work=0)
    assert runner.try_reserve_idle_restart() is True

    assert runner.confirm_idle_restart_reservation() is True


def test_cancel_does_not_reopen_gateway_after_stop_task_starts():
    runner = _runner(active_work=0)
    assert runner.try_reserve_idle_restart() is True
    runner._stop_task = cast(Any, object())

    runner.cancel_idle_restart_reservation()

    assert runner._idle_restart_reserved is False
    assert runner._draining is True


def test_cancel_during_external_drain_releases_only_reservation_gate():
    runner = _runner(active_work=0)
    statuses = []
    runner._update_runtime_status = (
        lambda gateway_state=None, exit_reason=None: statuses.append(gateway_state)
    )
    assert runner.try_reserve_idle_restart() is True
    runner._enter_external_drain()

    runner.cancel_idle_restart_reservation()

    assert runner._idle_restart_reserved is False
    assert runner._external_drain_active is True
    assert runner._draining is False
    assert statuses[-1] == "draining"

    runner._exit_external_drain()

    assert runner._external_drain_active is False
    assert runner._draining is False
    assert statuses[-1] == "running"


def test_cron_restart_pause_blocks_late_registration_atomically():
    import cron.scheduler as scheduler

    job_id = "restart-race-job"
    scheduler.release_running_job(job_id)
    token = None
    try:
        token, running = scheduler.reserve_restart_dispatch()
        assert job_id not in running
        assert scheduler.try_register_running_job(job_id) is False
        scheduler.release_restart_dispatch(token)
        token = None
        assert scheduler.try_register_running_job(job_id) is True
    finally:
        scheduler.release_running_job(job_id)
        if token is not None:
            scheduler.release_restart_dispatch(token)


def test_runner_holds_cron_pause_until_reservation_cancel():
    import cron.scheduler as scheduler

    runner = _runner(active_work=0)
    runner._active_work_count_for_restart = (
        GatewayRunner._active_work_count_for_restart.__get__(runner)
    )
    runner._running_agents = {}
    runner.adapters = {}
    job_id = "restart-runner-race-job"
    scheduler.release_running_job(job_id)
    try:
        assert runner.try_reserve_idle_restart() is True
        assert scheduler.try_register_running_job(job_id) is False
        assert runner.confirm_idle_restart_reservation() is True

        runner.cancel_idle_restart_reservation()

        assert scheduler.try_register_running_job(job_id) is True
    finally:
        runner.cancel_idle_restart_reservation()
        scheduler.release_running_job(job_id)


def test_direct_cron_fire_cannot_register_while_restart_paused(monkeypatch):
    import cron.scheduler as scheduler

    token = None
    job = {"id": "direct-fire-race", "fire_claim": {"by": "owner"}}
    entered = []

    def observe_entry(*args, **kwargs):
        entered.append(True)
        return True

    monkeypatch.setattr(
        scheduler, "_run_with_fire_claim_heartbeat", observe_entry
    )
    try:
        token, running = scheduler.reserve_restart_dispatch()
        assert running == frozenset()

        assert scheduler.run_one_job(job) is False
        assert entered == []
        assert job["id"] not in scheduler.get_running_job_ids()
    finally:
        if token is not None:
            scheduler.release_restart_dispatch(token)


def test_claimed_fire_rejection_closes_ledger_and_releases_exact_owner(
    tmp_path, monkeypatch
):
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    home = tmp_path / "home"
    cron_dir = home / "cron"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_dir / "executions.db")

    job = jobs.create_job(prompt="p", schedule="every 1h")
    provider = InProcessCronScheduler()
    claimed = provider.claim_fire(job["id"])
    assert claimed is not None
    owner = claimed["fire_claim"]["by"]
    before = next(item for item in jobs.load_jobs() if item["id"] == job["id"])
    entered = []
    monkeypatch.setattr(
        scheduler,
        "_run_with_fire_claim_heartbeat",
        lambda *_args, **_kwargs: entered.append(True) or True,
    )

    token = None
    try:
        token, _running = scheduler.reserve_restart_dispatch()
        assert provider.fire_claimed(claimed) is False
    finally:
        if token is not None:
            scheduler.release_restart_dispatch(token)

    assert entered == []
    assert job["id"] not in scheduler.get_running_job_ids()
    attempt = executions.list_executions(job_id=job["id"])[0]
    assert attempt["status"] == "failed"
    assert attempt["error"] == (
        "Execution was not started because gateway restart dispatch is paused."
    )
    after = next(item for item in jobs.load_jobs() if item["id"] == job["id"])
    assert after.get("fire_claim") is None
    for field in (
        "last_run_at",
        "last_status",
        "last_error",
        "failure_streak",
        "manual_run_at",
        "manual_run_prompt",
    ):
        assert after.get(field) == before.get(field)
    assert after.get("next_run_at") == before.get("next_run_at")

    # A stale rejection must not clear a replacement process's claim.
    persisted = jobs.load_jobs()
    current = next(item for item in persisted if item["id"] == job["id"])
    current["fire_claim"] = {"at": before["fire_claim"]["at"], "by": "replacement"}
    jobs.save_jobs(persisted)
    assert jobs.release_fire_claim(job["id"], expected_owner=owner) is False
    replacement = next(item for item in jobs.load_jobs() if item["id"] == job["id"])
    assert replacement["fire_claim"]["by"] == "replacement"


class _QueuedLoop:
    def __init__(self):
        self.callbacks = []

    def call_soon_threadsafe(self, callback):
        self.callbacks.append(callback)


def test_timed_out_loop_bridge_prevents_late_operation():
    loop = _QueuedLoop()
    calls = []

    assert _call_on_loop_for_bool(loop, lambda: calls.append("run") or True, timeout=0) is False
    loop.callbacks.pop()()

    assert calls == []


def test_timed_out_running_operation_rolls_back_late_success():
    started = threading.Event()
    release = threading.Event()
    rolled_back = []

    class ThreadLoop:
        def call_soon_threadsafe(self, callback):
            threading.Thread(target=callback).start()

    def operation():
        started.set()
        release.wait(timeout=1)
        return True

    timer = threading.Timer(0.02, release.set)
    timer.start()
    try:
        assert _call_on_loop_for_bool(
            ThreadLoop(),
            operation,
            timeout=0.005,
            rollback_late_true=lambda: rolled_back.append(True),
        ) is False
        assert started.wait(timeout=1)
        assert release.wait(timeout=1)
        for _ in range(100):
            if rolled_back:
                break
            threading.Event().wait(0.005)
    finally:
        timer.cancel()

    assert rolled_back == [True]
