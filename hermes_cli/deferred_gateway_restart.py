"""Durable, non-agent completion of Gateway restarts deferred by active work."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

from hermes_cli.gateway_restart_contract import RestartProbe, ServingIdentity

_INTENT_NAME = "deferred_gateway_restart.json"
_RECEIPT_NAME = "deferred_gateway_restart_latest.json"
_ACTIVE_WORK_RE = re.compile(r"active-work:([1-9][0-9]*)\Z")
_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class ScheduleResult(NamedTuple):
    scheduled: bool
    reason: str
    path: Path | None


@dataclass(frozen=True)
class DeferredRestartIntent:
    schema: int
    state: str
    home: str
    port: int
    health_url: str
    expected_sha: str
    old_pid: int
    old_start_time: int
    created_at: float
    deadline_at: float
    reason: str
    worker_pid: int | None
    generation: str
    owner_token: str


def deferred_restart_intent_path(home: Path) -> Path:
    return home.resolve() / "logs" / "update_receipts" / _INTENT_NAME


def deferred_restart_receipt_path(home: Path) -> Path:
    return home.resolve() / "logs" / "update_receipts" / _RECEIPT_NAME


def is_schedulable_active_work_reason(reason: object) -> bool:
    """Return true only for an exact positive active-work authority result."""
    return isinstance(reason, str) and _ACTIVE_WORK_RE.fullmatch(reason) is not None


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _parse_intent(payload: object) -> DeferredRestartIntent:
    if not isinstance(payload, dict):
        raise ValueError("intent-not-object")
    required = {
        "schema",
        "state",
        "home",
        "port",
        "health_url",
        "expected_sha",
        "old_pid",
        "old_start_time",
        "created_at",
        "deadline_at",
        "reason",
        "worker_pid",
        "generation",
        "owner_token",
    }
    if set(payload) != required:
        raise ValueError("intent-fields-invalid")
    if payload.get("schema") != 1 or type(payload.get("schema")) is not int:
        raise ValueError("intent-schema-invalid")
    if payload.get("state") not in {"pending", "running", "restarting"}:
        raise ValueError("intent-state-invalid")
    home = payload.get("home")
    if not isinstance(home, str) or not home or not Path(home).is_absolute():
        raise ValueError("intent-home-invalid")
    port = payload.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("intent-port-invalid")
    health_url = payload.get("health_url")
    if health_url != f"http://127.0.0.1:{port}/health":
        raise ValueError("intent-health-url-invalid")
    expected_sha = payload.get("expected_sha")
    if not isinstance(expected_sha, str) or not _SHA_RE.fullmatch(expected_sha):
        raise ValueError("intent-sha-invalid")
    old_pid = payload.get("old_pid")
    old_start = payload.get("old_start_time")
    if type(old_pid) is not int or old_pid <= 0:
        raise ValueError("intent-old-pid-invalid")
    if type(old_start) is not int or old_start <= 0:
        raise ValueError("intent-old-start-invalid")
    created_at = payload.get("created_at")
    deadline_at = payload.get("deadline_at")
    if not _is_number(created_at) or not _is_number(deadline_at):
        raise ValueError("intent-time-invalid")
    if float(deadline_at) <= float(created_at):
        raise ValueError("intent-deadline-invalid")
    reason = payload.get("reason")
    if not isinstance(reason, str) or _ACTIVE_WORK_RE.fullmatch(reason) is None:
        raise ValueError("intent-reason-invalid")
    worker_pid = payload.get("worker_pid")
    if worker_pid is not None and (type(worker_pid) is not int or worker_pid <= 0):
        raise ValueError("intent-worker-pid-invalid")
    generation = payload.get("generation")
    owner_token = payload.get("owner_token")
    if not isinstance(generation, str) or not generation:
        raise ValueError("intent-generation-invalid")
    if not isinstance(owner_token, str) or not owner_token:
        raise ValueError("intent-owner-invalid")
    return DeferredRestartIntent(
        schema=1,
        state=str(payload["state"]),
        home=home,
        port=port,
        health_url=str(health_url),
        expected_sha=expected_sha,
        old_pid=old_pid,
        old_start_time=old_start,
        created_at=float(created_at),
        deadline_at=float(deadline_at),
        reason=reason,
        worker_pid=worker_pid,
        generation=generation,
        owner_token=owner_token,
    )


def _read_intent(path: Path) -> DeferredRestartIntent:
    return _parse_intent(json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _intent_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
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


def _cas_write_intent(
    path: Path,
    *,
    generation: str,
    owner_token: str,
    payload: dict[str, Any],
) -> bool:
    with _intent_lock(path):
        try:
            current = _read_intent(path)
        except Exception:
            return False
        if current.generation != generation or current.owner_token != owner_token:
            return False
        _atomic_write_json(path, payload)
        return True


def _claim_worker_intent(
    path: Path,
    *,
    generation: str,
    owner_token: str,
    worker_pid: int,
) -> DeferredRestartIntent | None:
    """Move only this scheduler-published worker from pending to running."""
    with _intent_lock(path):
        try:
            current = _read_intent(path)
        except Exception:
            return None
        if (
            current.generation != generation
            or current.owner_token != owner_token
            or current.state != "pending"
            or current.worker_pid != worker_pid
        ):
            return None
        payload = asdict(current)
        payload["state"] = "running"
        _atomic_write_json(path, payload)
        return _parse_intent(payload)


def _claim_restart_authority(
    path: Path, intent: DeferredRestartIntent, *, now: float
) -> bool:
    """Atomically claim the one restart side effect for this live worker."""
    with _intent_lock(path):
        try:
            current = _read_intent(path)
        except Exception:
            return False
        if (
            current.generation != intent.generation
            or current.owner_token != intent.owner_token
            or current.state != "running"
            or current.worker_pid != os.getpid()
            or current.deadline_at <= now
        ):
            return False
        payload = asdict(current)
        payload["state"] = "restarting"
        _atomic_write_json(path, payload)
        return True


def _create_intent_if_absent(path: Path, payload: dict[str, Any]) -> bool:
    with _intent_lock(path):
        if path.exists():
            return False
        _atomic_write_json(path, payload)
        return True


def _publish_spawned_worker(
    path: Path, *, generation: str, owner_token: str, worker_pid: int
) -> bool:
    """Publish a spawned PID without regressing a worker's running claim."""
    with _intent_lock(path):
        try:
            current = _read_intent(path)
        except Exception:
            return False
        if current.generation != generation or current.owner_token != owner_token:
            return False
        if current.state == "running" and current.worker_pid is not None:
            return True
        if current.state != "pending" or current.worker_pid is not None:
            return False
        payload = asdict(current)
        payload["worker_pid"] = worker_pid
        _atomic_write_json(path, payload)
        return True


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _spawn_worker(argv: list[str]) -> int:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return int(process.pid)


def schedule_deferred_restart(
    *,
    home: Path,
    port: int,
    expected_sha: str,
    old_identity: ServingIdentity,
    reason: str,
    spawn: Callable[[list[str]], int] = _spawn_worker,
    process_alive: Callable[[int], bool] = _process_alive,
    now: Callable[[], float] = time.time,
    timeout: float = 3600.0,
) -> ScheduleResult:
    """Persist and spawn an exact-target retry for verified positive active work."""
    if not is_schedulable_active_work_reason(reason):
        return ScheduleResult(False, "reason-not-schedulable", None)
    if (
        type(port) is not int
        or not 1 <= port <= 65535
        or not _SHA_RE.fullmatch(expected_sha)
        or type(old_identity.pid) is not int
        or old_identity.pid <= 0
        or type(old_identity.start_time) is not int
        or old_identity.start_time <= 0
        or old_identity.hermes_home.resolve() != home.resolve()
    ):
        return ScheduleResult(False, "target-invalid", None)
    created_at = float(now())
    if not math.isfinite(created_at) or not math.isfinite(timeout) or timeout <= 0:
        return ScheduleResult(False, "deadline-invalid", None)
    path = deferred_restart_intent_path(home)
    replacement = False
    replaced_generation: str | None = None
    replaced_owner: str | None = None
    if path.exists():
        try:
            current = _read_intent(path)
        except Exception:
            return ScheduleResult(False, "existing-intent-invalid", path)
        same_target = (
            current.home == str(home.resolve())
            and current.port == port
            and current.expected_sha == expected_sha
            and current.old_pid == old_identity.pid
            and current.old_start_time == old_identity.start_time
        )
        if current.worker_pid is not None and process_alive(current.worker_pid):
            if current.state == "restarting" or (
                same_target and current.deadline_at > created_at
            ):
                return ScheduleResult(
                    same_target,
                    "already-scheduled" if same_target else "different-intent-active",
                    path,
                )
        if not same_target and current.deadline_at > created_at:
            return ScheduleResult(False, "different-intent-active", path)
        replacement = True
        replaced_generation = current.generation
        replaced_owner = current.owner_token

    payload: dict[str, Any] = {
        "schema": 1,
        "state": "pending",
        "home": str(home.resolve()),
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
        "expected_sha": expected_sha,
        "old_pid": old_identity.pid,
        "old_start_time": old_identity.start_time,
        "created_at": created_at,
        "deadline_at": created_at + float(timeout),
        "reason": reason,
        "worker_pid": None,
        "generation": uuid.uuid4().hex,
        "owner_token": uuid.uuid4().hex,
    }
    if replacement:
        if not _cas_write_intent(
            path,
            generation=str(replaced_generation),
            owner_token=str(replaced_owner),
            payload=payload,
        ):
            return ScheduleResult(False, "intent-replaced-before-publish", path)
    else:
        if not _create_intent_if_absent(path, payload):
            return ScheduleResult(False, "intent-created-concurrently", path)
    argv = [
        sys.executable,
        "-m",
        "hermes_cli.deferred_gateway_restart",
        "--intent",
        str(path),
        "--generation",
        str(payload["generation"]),
        "--owner-token",
        str(payload["owner_token"]),
    ]
    try:
        worker_pid = spawn(argv)
    except Exception:
        return ScheduleResult(False, "worker-spawn-failed", path)
    if type(worker_pid) is not int or worker_pid <= 0:
        return ScheduleResult(False, "worker-pid-invalid", path)
    if not _publish_spawned_worker(
        path,
        generation=str(payload["generation"]),
        owner_token=str(payload["owner_token"]),
        worker_pid=worker_pid,
    ):
        return ScheduleResult(False, "intent-replaced-after-spawn", path)
    return ScheduleResult(True, "worker-replaced" if replacement else "scheduled", path)


def _terminal_payload(
    intent: DeferredRestartIntent,
    *,
    state: str,
    reason: str,
    finished_at: float,
    identity: ServingIdentity | None = None,
) -> dict[str, Any]:
    success = state == "completed" and identity is not None
    return {
        "schema": 1,
        "state": state,
        "reason": reason,
        "home": intent.home,
        "port": intent.port,
        "expected_sha": intent.expected_sha,
        "serving_sha": identity.code_sha if success else None,
        "old_pid": intent.old_pid,
        "old_start_time": intent.old_start_time,
        "new_pid": identity.pid if success else None,
        "new_start_time": identity.start_time if success else None,
        "listener_owner_pid": identity.pid if success else None,
        "control_health": "verified" if success else "unverified",
        "http_health": "verified" if success else "unverified",
        "finished_at": float(finished_at),
    }


def _validate_receipt(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("receipt-not-object")
    required = {
        "schema", "state", "reason", "home", "port", "expected_sha",
        "serving_sha", "old_pid", "old_start_time", "new_pid",
        "new_start_time", "listener_owner_pid", "control_health",
        "http_health", "finished_at",
    }
    if set(payload) != required or payload.get("schema") != 1:
        raise ValueError("receipt-shape-invalid")
    if payload.get("state") not in {"completed", "failed", "expired"}:
        raise ValueError("receipt-state-invalid")
    if not isinstance(payload.get("reason"), str):
        raise ValueError("receipt-reason-invalid")
    if not isinstance(payload.get("home"), str) or not Path(payload["home"]).is_absolute():
        raise ValueError("receipt-home-invalid")
    if type(payload.get("port")) is not int or not 1 <= payload["port"] <= 65535:
        raise ValueError("receipt-port-invalid")
    if not isinstance(payload.get("expected_sha"), str) or not _SHA_RE.fullmatch(payload["expected_sha"]):
        raise ValueError("receipt-expected-sha-invalid")
    if type(payload.get("old_pid")) is not int or type(payload.get("old_start_time")) is not int:
        raise ValueError("receipt-old-identity-invalid")
    if not _is_number(payload.get("finished_at")):
        raise ValueError("receipt-time-invalid")
    if payload["state"] == "completed":
        for key in ("new_pid", "new_start_time", "listener_owner_pid"):
            if type(payload.get(key)) is not int or payload[key] <= 0:
                raise ValueError("receipt-new-identity-invalid")
        if not isinstance(payload.get("serving_sha"), str) or not _SHA_RE.fullmatch(payload["serving_sha"]):
            raise ValueError("receipt-serving-sha-invalid")
        if payload["serving_sha"] != payload["expected_sha"]:
            raise ValueError("receipt-serving-sha-mismatch")
        if payload["listener_owner_pid"] != payload["new_pid"]:
            raise ValueError("receipt-listener-owner-mismatch")
        if payload.get("control_health") != "verified" or payload.get("http_health") != "verified":
            raise ValueError("receipt-health-unverified")
    else:
        if any(payload.get(key) is not None for key in ("serving_sha", "new_pid", "new_start_time", "listener_owner_pid")):
            raise ValueError("receipt-failure-claims-replacement")
    return dict(payload)


def read_deferred_restart_receipt(home: Path) -> dict[str, Any] | None:
    try:
        return _validate_receipt(
            json.loads(deferred_restart_receipt_path(home).read_text(encoding="utf-8"))
        )
    except Exception:
        return None


def _write_terminal(
    intent_path: Path,
    intent: DeferredRestartIntent,
    *,
    state: str,
    reason: str,
    finished_at: float,
    identity: ServingIdentity | None = None,
) -> bool:
    receipt = _terminal_payload(
        intent,
        state=state,
        reason=reason,
        finished_at=finished_at,
        identity=identity,
    )
    _validate_receipt(receipt)
    with _intent_lock(intent_path):
        try:
            current = _read_intent(intent_path)
        except Exception:
            return False
        if (
            current.generation != intent.generation
            or current.owner_token != intent.owner_token
        ):
            return False
        _atomic_write_json(deferred_restart_receipt_path(Path(intent.home)), receipt)
        try:
            intent_path.unlink()
        except FileNotFoundError:
            pass
        return True


def run_deferred_restart(
    intent_path: Path,
    *,
    generation: str,
    owner_token: str,
    probe: Callable[[DeferredRestartIntent], RestartProbe],
    restart: Callable[[DeferredRestartIntent], RestartProbe],
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = time.sleep,
    poll_interval: float = 2.0,
) -> int:
    """Wait for verified idle authority, then execute the existing restart contract."""
    try:
        intent = _read_intent(intent_path)
    except Exception:
        return 1
    canonical_path = deferred_restart_intent_path(Path(intent.home))
    supplied_path = Path(os.path.abspath(intent_path))
    if intent_path.is_symlink() or supplied_path != canonical_path:
        return 1
    intent = None
    for _attempt in range(500):
        intent = _claim_worker_intent(
            intent_path,
            generation=generation,
            owner_token=owner_token,
            worker_pid=os.getpid(),
        )
        if intent is not None:
            break
        try:
            waiting = _read_intent(intent_path)
        except Exception:
            return 1
        if (
            waiting.generation != generation
            or waiting.owner_token != owner_token
            or waiting.state != "pending"
            or waiting.worker_pid is not None
        ):
            return 1
        sleep(0.01)
    if intent is None:
        return 1

    while True:
        current_time = float(now())
        if current_time >= intent.deadline_at:
            _write_terminal(
                intent_path,
                intent,
                state="expired",
                reason="deadline-exceeded",
                finished_at=current_time,
            )
            return 1
        try:
            current = probe(intent)
        except Exception as exc:
            current = RestartProbe(False, f"probe-failed:{type(exc).__name__}", None)
        if current.ready:
            if current.identity is None:
                _write_terminal(
                    intent_path, intent, state="failed", reason="ready-without-identity",
                    finished_at=current_time,
                )
                return 1
            old_tuple = (intent.old_pid, intent.old_start_time)
            current_tuple = (current.identity.pid, current.identity.start_time)
            if current_tuple != old_tuple:
                if current.identity.code_sha == intent.expected_sha:
                    _write_terminal(
                        intent_path, intent, state="completed", reason="already-current",
                        finished_at=current_time, identity=current.identity,
                    )
                    return 0
                _write_terminal(
                    intent_path, intent, state="failed", reason="serving-identity-changed",
                    finished_at=current_time,
                )
                return 1
            claim_time = float(now())
            if not _claim_restart_authority(intent_path, intent, now=claim_time):
                return 1
            replacement = restart(_read_intent(intent_path))
            if (
                replacement.ready
                and replacement.identity is not None
                and (replacement.identity.pid, replacement.identity.start_time) != old_tuple
                and replacement.identity.code_sha == intent.expected_sha
            ):
                _write_terminal(
                    intent_path, intent, state="completed", reason=replacement.reason,
                    finished_at=float(now()), identity=replacement.identity,
                )
                return 0
            _write_terminal(
                intent_path,
                intent,
                state="failed",
                reason=replacement.reason if isinstance(replacement, RestartProbe) else "restart-result-invalid",
                finished_at=float(now()),
            )
            return 1
        if _ACTIVE_WORK_RE.fullmatch(current.reason) is None:
            _write_terminal(
                intent_path,
                intent,
                state="failed",
                reason=current.reason,
                finished_at=current_time,
            )
            return 1
        if current.identity is None or (
            current.identity.pid,
            current.identity.start_time,
        ) != (intent.old_pid, intent.old_start_time):
            _write_terminal(
                intent_path,
                intent,
                state="failed",
                reason="serving-identity-changed",
                finished_at=current_time,
            )
            return 1
        sleep(min(max(0.01, poll_interval), intent.deadline_at - current_time))


def _default_probe(intent: DeferredRestartIntent) -> RestartProbe:
    from hermes_cli.gateway_restart_contract import default_adapters, probe_current_gateway

    runtime = default_adapters()
    return probe_current_gateway(
        Path(intent.home),
        identify=runtime.identify,
        status=runtime.status,
        process_alive=runtime.process_alive,
        listener_owners=runtime.listener_owners,
        health_ok=runtime.health_ok,
        port=intent.port,
        health_url=intent.health_url,
        require_idle=True,
    )


def _default_restart(intent: DeferredRestartIntent) -> RestartProbe:
    from hermes_cli.gateway_restart_contract import perform_verified_graceful_restart

    return perform_verified_graceful_restart(
        Path(intent.home),
        port=intent.port,
        health_url=intent.health_url,
        expected_code_sha=intent.expected_sha,
        timeout=max(45.0, min(300.0, intent.deadline_at - time.time())),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--owner-token", required=True)
    args = parser.parse_args(argv)
    return run_deferred_restart(
        args.intent,
        generation=args.generation,
        owner_token=args.owner_token,
        probe=_default_probe,
        restart=_default_restart,
    )


if __name__ == "__main__":
    raise SystemExit(main())
