"""Gateway-owned restart admission and live runtime probes."""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional

from gateway.config import Platform


_CONFIGURED_PLATFORMS_UNAVAILABLE = "__configured_platforms_unavailable__"


class GatewayAdmissionDenied(RuntimeError):
    """Raised when restart/drain coordination has closed new work admission."""


class GatewayAdmissionBarrier:
    """Serialize new-work admission with restart snapshot/authorization/signal."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._closed_reason: Optional[str] = None
        self._active_admissions = 0

    @contextlib.contextmanager
    def admission(self) -> Iterator[None]:
        with self._lock:
            if self._closed_reason is not None:
                raise GatewayAdmissionDenied(self._closed_reason)
            self._active_admissions += 1
            try:
                yield
            finally:
                self._active_admissions = max(0, self._active_admissions - 1)

    def active_admissions(self) -> int:
        with self._lock:
            return self._active_admissions

    def close(self, reason: str) -> None:
        with self._lock:
            self._closed_reason = reason

    def closed_reason(self) -> Optional[str]:
        with self._lock:
            return self._closed_reason

    def restart_if_idle(
        self,
        *,
        snapshot: Callable[[], Mapping[str, Any]],
        authorize: Callable[[Mapping[str, Any]], tuple[bool, str]],
        signal_restart: Callable[[], None],
        already_requested: Callable[[], bool],
        mark_requested: Callable[[], None],
    ) -> tuple[bool, str, Mapping[str, Any]]:
        with self._lock:
            if already_requested():
                return False, "restart-already-requested", {}
            live = snapshot()
            accepted, reason = authorize(live)
            if not accepted:
                return False, reason, live
            self._closed_reason = "restart"
            mark_requested()
            signal_restart()
            return True, "accepted", live


@dataclass(frozen=True)
class GatewayRestartTopology:
    gateway_pid: int
    supervisor_pid: int
    supervisor: str


def _count_sized(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return 0


def _count_unfinished_tasks(tasks: Any) -> int:
    count = 0
    for task in list(tasks or ()):
        try:
            if not task.done() and not getattr(task, "_hermes_supervised_watcher", False):
                count += 1
        except Exception:
            count += 1
    return count


def _threading_lock_held(lock: Any) -> bool:
    locked = getattr(lock, "locked", None)
    if callable(locked):
        try:
            return bool(locked())
        except Exception:
            return True
    return False


def count_owned_delivery_obligations() -> int:
    try:
        from gateway.delivery_ledger import count_active_owned_obligations
        return count_active_owned_obligations()
    except Exception:
        return 1


def build_gateway_restart_topology(supervisor: str) -> GatewayRestartTopology:
    return GatewayRestartTopology(
        gateway_pid=os.getpid(),
        supervisor_pid=os.getppid(),
        supervisor=supervisor,
    )


def configured_platform_names(config: Any) -> list[str]:
    names: list[str] = []
    try:
        platforms = config.get_connected_platforms()
    except Exception:
        return [_CONFIGURED_PLATFORMS_UNAVAILABLE]
    for platform in platforms:
        value = getattr(platform, "value", platform)
        if isinstance(value, str) and value:
            names.append(value)
    return sorted(set(names))


def live_platform_writer_snapshot(runner: Any, *, expected_pid: int, expected_start_time: Any) -> dict[str, dict[str, Any]]:
    platforms: dict[str, dict[str, Any]] = {}
    configured = configured_platform_names(getattr(runner, "config", None))
    adapters = getattr(runner, "adapters", {}) or {}
    for name in configured:
        try:
            platform_key = Platform(name)
        except ValueError:
            platform_key = name
        adapter = adapters.get(platform_key) or adapters.get(name)
        if adapter is None:
            platforms[name] = {"state": "missing", "writer_pid": None, "writer_start_time": None}
            continue
        probe = getattr(adapter, "restart_writer_health_probe", None)
        if not callable(probe):
            platforms[name] = {
                "state": "unavailable",
                "writer_pid": None,
                "writer_start_time": None,
            }
            continue
        try:
            attestation = probe()
        except Exception:
            attestation = None
        if not isinstance(attestation, Mapping):
            platforms[name] = {"state": "unavailable", "writer_pid": None, "writer_start_time": None}
            continue
        platforms[name] = {
            "state": attestation.get("state", "unavailable"),
            "writer_pid": attestation.get("writer_pid"),
            "writer_start_time": attestation.get("writer_start_time"),
        }
    return platforms


def live_session_store_snapshot(runner: Any) -> dict[str, str]:
    probe = getattr(getattr(runner, "session_store", None), "restart_health_probe", None)
    if callable(probe):
        try:
            result = probe()
            if isinstance(result, Mapping):
                state = str(result.get("status") or "")
                return {"status": state if state in {"ok", "unavailable", "retrying"} else "unknown"}
        except Exception:
            return {"status": "unavailable"}
    if getattr(runner, "_session_db_init_error", None):
        return {"status": "unavailable"}
    return {"status": "ok" if getattr(runner, "session_store", None) is not None else "unavailable"}


def restart_obligations(runner: Any) -> dict[str, int]:
    updater_task = getattr(runner, "_update_notification_task", None)
    obligations = {
        "delivery_queue": _count_unfinished_tasks(getattr(runner, "_background_tasks", ())),
        "delivery_ledger": count_owned_delivery_obligations(),
        "pending_final": _count_sized(getattr(runner, "_pending_final_deliveries", None)),
        "drain": int(bool(getattr(runner, "_draining", False) or getattr(runner, "_external_drain_active", False))),
        "delegation_workers": 0,
        "updater": int(updater_task is not None and not updater_task.done()),
        "update_lock": int(_threading_lock_held(getattr(runner, "_update_lock", None))),
        "oauth_refresh": _count_sized(getattr(runner, "_oauth_refreshes", None)),
        "oauth_token_lock": int(_threading_lock_held(getattr(runner, "_oauth_token_lock", None))),
    }
    try:
        from tools.async_delegation import active_count
        obligations["delegation_workers"] = max(0, int(active_count()))
    except Exception:
        obligations["delegation_workers"] = 1
    return obligations
