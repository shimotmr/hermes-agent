"""Fail-closed contracts for identifying and validating Gateway restarts.

The Gateway-owned control socket is the authority for serving identity. Runtime
status, listener ownership, process liveness, and HTTP health are independent
cross-checks; no single signal is sufficient to declare a replacement healthy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, cast


_MAX_STATUS_ANSWER_AGE_SECONDS = 10.0
_MAX_STATUS_ANSWER_FUTURE_SKEW_SECONDS = 1.0


# When executed by absolute path, Python puts ``hermes_cli/`` first on sys.path;
# that would make ``hermes_cli/gateway.py`` shadow the top-level gateway package.
if __package__ in (None, ""):
    repo_root = str(Path(__file__).resolve().parents[1])
    if sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)


class ServingIdentity(NamedTuple):
    pid: int
    start_time: int
    hermes_home: Path
    code_sha: str | None = None


class RuntimeAdapters(NamedTuple):
    identify: Callable[[Path], dict[str, Any] | None]
    status: Callable[[Path], dict[str, Any] | None]
    process_alive: Callable[[int], bool]
    listener_owners: Callable[[int], set[int]]
    health_ok: Callable[[str], bool]


class RestartProbe(NamedTuple):
    ready: bool
    reason: str
    identity: ServingIdentity | None = None


def parse_serving_identity(
    payload: dict[str, Any] | None, *, expected_home: Path
) -> ServingIdentity:
    if not isinstance(payload, dict):
        raise ValueError("control-identity-missing")
    if type(payload.get("protocol")) is not int or payload.get("protocol") != 1:
        raise ValueError("control-protocol-mismatch")
    if payload.get("kind") != "hermes-gateway":
        raise ValueError("control-kind-mismatch")
    pid = payload.get("pid")
    start_time = payload.get("start_time")
    if type(pid) is not int or pid <= 1:
        raise ValueError("invalid-serving-pid")
    if type(start_time) is not int or start_time <= 0:
        raise ValueError("invalid-serving-start-time")
    home = Path(str(payload.get("hermes_home", ""))).expanduser()
    if home != expected_home.expanduser():
        raise ValueError("runtime-home-mismatch")
    code_sha = payload.get("code_sha")
    if code_sha is not None and not isinstance(code_sha, str):
        raise ValueError("invalid-code-sha")
    return ServingIdentity(pid, start_time, home, code_sha)


def _live_status_payload(
    current: ServingIdentity, status: dict[str, Any]
) -> dict[str, Any]:
    """Project a legacy persisted status into the current writer inventory."""
    payload = dict(status)
    platforms = status.get("platforms")
    if isinstance(platforms, dict):
        payload["platforms"] = {
            name: record
            for name, record in platforms.items()
            if not (
                isinstance(record, dict)
                and type(record.get("writer_pid")) is int
                and record.get("writer_pid") != current.pid
                and type(record.get("writer_start_time")) is int
                and record.get("writer_start_time") != current.start_time
            )
        }
    return payload


def _validate_status(
    current: ServingIdentity,
    status: dict[str, Any],
    *,
    required_platforms: tuple[str, ...] | None = None,
) -> str | None:
    answered_at = status.get("answered_at")
    if type(answered_at) not in (int, float):
        return "status-answer-time-invalid"
    answer_timestamp = float(cast(int | float, answered_at))
    if not math.isfinite(answer_timestamp):
        return "status-answer-time-invalid"
    answer_age = time.time() - answer_timestamp
    if (
        answer_age > _MAX_STATUS_ANSWER_AGE_SECONDS
        or answer_age < -_MAX_STATUS_ANSWER_FUTURE_SKEW_SECONDS
    ):
        return "status-answer-stale"
    if status.get("gateway_state") != "running":
        return "gateway-not-running"
    if type(status.get("answering_pid")) is not int or status.get("answering_pid") != current.pid:
        return "status-answering-pid-mismatch"
    if type(status.get("pid")) is not int or status.get("pid") != current.pid:
        return "status-pid-mismatch"
    if type(status.get("start_time")) is not int or status.get("start_time") != current.start_time:
        return "status-start-time-mismatch"
    platforms = status.get("platforms")
    if not isinstance(platforms, dict):
        return "platform-state-missing"
    if required_platforms is None:
        if not platforms:
            return "platform-inventory-empty"
        required_platforms = tuple(platforms)
    for name in required_platforms:
        record = platforms.get(name)
        if not isinstance(record, dict) or record.get("state") != "connected":
            return f"platform-not-connected:{name}"
        if type(record.get("writer_pid")) is not int or record.get("writer_pid") != current.pid:
            return f"platform-writer-mismatch:{name}"
        if type(record.get("writer_start_time")) is not int or record.get("writer_start_time") != current.start_time:
            return f"platform-writer-start-time-mismatch:{name}"
    return None


def evaluate_current(
    *,
    current: ServingIdentity,
    status: dict[str, Any],
    process_alive: bool,
    listener_pids: set[int],
    health_ok: bool,
    require_idle: bool = False,
    required_platforms: tuple[str, ...] | None = None,
) -> RestartProbe:
    if not process_alive:
        return RestartProbe(False, "serving-process-not-alive", current)
    status_error = _validate_status(
        current, status, required_platforms=required_platforms
    )
    if status_error:
        return RestartProbe(False, status_error, current)
    if listener_pids != {current.pid}:
        return RestartProbe(False, "listener-owner-mismatch", current)
    if not health_ok:
        return RestartProbe(False, "health-check-failed", current)
    if require_idle:
        active_agents = status.get("active_agents")
        if type(active_agents) is not int or active_agents < 0:
            return RestartProbe(False, "active-work-unknown", current)
        if active_agents:
            return RestartProbe(False, f"active-work:{active_agents}", current)
    return RestartProbe(True, "gateway-healthy", current)


def evaluate_replacement(
    *,
    old: ServingIdentity,
    current: ServingIdentity,
    status: dict[str, Any],
    old_process_alive: bool,
    new_process_alive: bool,
    listener_pids: set[int],
    health_ok: bool,
    expected_code_sha: str | None = None,
    required_platforms: tuple[str, ...] | None = None,
) -> RestartProbe:
    if old_process_alive:
        return RestartProbe(False, "old-serving-process-still-alive", current)
    if (current.pid, current.start_time) == (old.pid, old.start_time):
        return RestartProbe(False, "serving-identity-did-not-change", current)
    if expected_code_sha is not None and current.code_sha != expected_code_sha:
        return RestartProbe(False, "replacement-code-sha-mismatch", current)
    current_probe = evaluate_current(
        current=current,
        status=status,
        process_alive=new_process_alive,
        listener_pids=listener_pids,
        health_ok=health_ok,
        required_platforms=required_platforms,
    )
    if not current_probe.ready:
        return current_probe
    return RestartProbe(True, "replacement-healthy", current)


def probe_current_gateway(
    home: Path,
    *,
    identify: Callable[[Path], dict[str, Any] | None],
    status: Callable[[Path], dict[str, Any] | None],
    process_alive: Callable[[int], bool],
    listener_owners: Callable[[int], set[int]],
    health_ok: Callable[[str], bool],
    port: int,
    health_url: str,
    require_idle: bool = False,
    required_platforms: tuple[str, ...] | None = None,
) -> RestartProbe:
    try:
        current = parse_serving_identity(identify(home), expected_home=home)
    except (OSError, TypeError, ValueError) as exc:
        return RestartProbe(False, str(exc), None)
    runtime_status = status(home)
    if not isinstance(runtime_status, dict):
        return RestartProbe(False, "control-status-missing", current)
    runtime_status = _live_status_payload(current, runtime_status)
    return evaluate_current(
        current=current,
        status=runtime_status,
        process_alive=process_alive(current.pid),
        listener_pids=listener_owners(port),
        health_ok=health_ok(health_url),
        require_idle=require_idle,
        required_platforms=required_platforms,
    )


def perform_verified_graceful_restart(
    home: Path,
    *,
    adapters: RuntimeAdapters | None = None,
    signal_process: Callable[[ServingIdentity], Any] | None = None,
    port: int = 8642,
    health_url: str = "http://127.0.0.1:8642/health",
    expected_code_sha: str | None = None,
    required_platforms: tuple[str, ...] | None = None,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> RestartProbe:
    """Request one graceful restart and verify the exact replacement.

    There is deliberately no force fallback. Two preflight snapshots must prove
    the same serving identity is idle before SIGUSR1 is sent. Unless supplied by
    the caller, the first status snapshot defines the enabled writer inventory
    that the replacement must restore.
    """

    runtime = adapters or default_adapters()

    def snapshot(
        *,
        require_idle: bool,
        platforms_required: tuple[str, ...] | None = None,
    ) -> tuple[RestartProbe, dict[str, Any] | None]:
        try:
            identity = parse_serving_identity(runtime.identify(home), expected_home=home)
            status_payload = runtime.status(home)
            process_alive = runtime.process_alive(identity.pid)
            listener_pids = runtime.listener_owners(port)
            is_healthy = runtime.health_ok(health_url)
        except Exception as exc:
            return RestartProbe(False, str(exc), None), None
        if not isinstance(status_payload, dict):
            return RestartProbe(False, "control-status-missing", identity), None
        status_payload = _live_status_payload(identity, status_payload)
        probe = evaluate_current(
            current=identity,
            status=status_payload,
            process_alive=process_alive,
            listener_pids=listener_pids,
            health_ok=is_healthy,
            require_idle=require_idle,
            required_platforms=platforms_required,
        )
        return probe, status_payload

    first, first_status = snapshot(
        require_idle=True, platforms_required=required_platforms
    )
    if not first.ready or first.identity is None or first_status is None:
        return first
    effective_platforms = required_platforms
    if effective_platforms is None:
        platforms = first_status.get("platforms")
        effective_platforms = tuple(platforms) if isinstance(platforms, dict) else ()

    second, _ = snapshot(
        require_idle=True, platforms_required=effective_platforms
    )
    if not second.ready or second.identity is None:
        return second
    if second.identity != first.identity:
        return RestartProbe(False, "serving-identity-changed-before-signal", second.identity)

    # Re-read the control-socket identity and idle state at the signal boundary.
    # Passing the full identity to the adapter prevents callers from discarding
    # the start-time PID-reuse fingerprint before delivering SIGUSR1.
    pre_signal, _ = snapshot(
        require_idle=True, platforms_required=effective_platforms
    )
    if (
        pre_signal.identity is not None
        and pre_signal.identity != second.identity
    ):
        return RestartProbe(
            False,
            "serving-identity-changed-before-signal",
            pre_signal.identity,
        )
    if not pre_signal.ready or pre_signal.identity is None:
        return pre_signal

    try:
        if signal_process is not None:
            signal_process(pre_signal.identity)
        else:
            from gateway.control_socket import query_gateway_control

            response = query_gateway_control(
                home,
                "restart-if-idle",
                request_fields={
                    "expected_pid": pre_signal.identity.pid,
                    "expected_start_time": pre_signal.identity.start_time,
                },
                timeout=max(1.0, min(5.0, timeout)),
            )
            response_identity = response.get("identity") if isinstance(response, dict) else None
            if not (
                isinstance(response, dict)
                and response.get("accepted") is True
                and isinstance(response_identity, dict)
                and type(response_identity.get("pid")) is int
                and response_identity.get("pid") == pre_signal.identity.pid
                and type(response_identity.get("start_time")) is int
                and response_identity.get("start_time") == pre_signal.identity.start_time
                and response.get("signal") == "SIGUSR1"
            ):
                return RestartProbe(
                    False,
                    "atomic-restart-request-rejected",
                    pre_signal.identity,
                )
    except Exception as exc:
        return RestartProbe(
            False, f"graceful-signal-failed:{type(exc).__name__}", second.identity
        )

    deadline = monotonic() + max(0.0, timeout)
    last = RestartProbe(False, "replacement-not-ready", second.identity)
    while True:
        try:
            current = parse_serving_identity(runtime.identify(home), expected_home=home)
            status_payload = runtime.status(home)
            if not isinstance(status_payload, dict):
                last = RestartProbe(False, "control-status-missing", current)
            else:
                status_payload = _live_status_payload(current, status_payload)
                last = evaluate_replacement(
                    old=second.identity,
                    current=current,
                    status=status_payload,
                    old_process_alive=runtime.process_alive(second.identity.pid),
                    new_process_alive=runtime.process_alive(current.pid),
                    listener_pids=runtime.listener_owners(port),
                    health_ok=runtime.health_ok(health_url),
                    expected_code_sha=expected_code_sha,
                    required_platforms=effective_platforms,
                )
                if last.ready:
                    return last
        except Exception as exc:
            last = RestartProbe(False, str(exc), None)
        now = monotonic()
        if now >= deadline:
            return RestartProbe(
                False, f"graceful-restart-timeout:{last.reason}", last.identity
            )
        sleep(min(max(0.0, poll_interval), deadline - now))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _listener_owners(port: int) -> set[int]:
    try:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if result.returncode not in (0, 1):
        return set()
    owners: set[int] = set()
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            return set()
        if pid > 1:
            owners.add(pid)
    return owners


def _health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - operator-provided localhost URL
            return response.status == 200
    except Exception:
        return False


def default_adapters() -> RuntimeAdapters:
    from gateway.control_socket import identify_gateway, query_gateway_control

    return RuntimeAdapters(
        identify=identify_gateway,
        status=lambda home: query_gateway_control(home, "status"),
        process_alive=_pid_alive,
        listener_owners=_listener_owners,
        health_ok=_health_ok,
    )


def _emit_probe(probe: RestartProbe) -> int:
    if probe.ready and probe.identity is not None:
        print(f"{probe.identity.pid} {probe.identity.start_time}")
        return 0
    print(probe.reason, file=sys.stderr)
    return 1


def main(
    argv: Sequence[str] | None = None, *, adapters: RuntimeAdapters | None = None
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("identity", "healthy", "ready"))
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--health-url", default="http://127.0.0.1:8642/health")
    parser.add_argument("--old-pid", type=int)
    parser.add_argument("--old-start", type=int)
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--required-platform", action="append", default=None)
    parser.add_argument("--expected-code-sha")
    args = parser.parse_args(argv)
    runtime = adapters or default_adapters()

    try:
        current = parse_serving_identity(runtime.identify(args.home), expected_home=args.home)
    except (OSError, TypeError, ValueError) as exc:
        return _emit_probe(RestartProbe(False, str(exc), None))

    if args.command == "identity":
        return _emit_probe(RestartProbe(True, "identity-valid", current))

    status_payload = runtime.status(args.home)
    if not isinstance(status_payload, dict):
        return _emit_probe(RestartProbe(False, "control-status-missing", current))
    status_payload = _live_status_payload(current, status_payload)

    if args.command == "healthy":
        return _emit_probe(
            evaluate_current(
                current=current,
                status=status_payload,
                process_alive=runtime.process_alive(current.pid),
                listener_pids=runtime.listener_owners(args.port),
                health_ok=runtime.health_ok(args.health_url),
                require_idle=args.require_idle,
                required_platforms=(
                    tuple(args.required_platform) if args.required_platform is not None else None
                ),
            )
        )

    if args.old_pid is None or args.old_start is None:
        parser.error("ready requires --old-pid and --old-start")
    old = ServingIdentity(args.old_pid, args.old_start, args.home)
    return _emit_probe(
        evaluate_replacement(
            old=old,
            current=current,
            status=status_payload,
            old_process_alive=runtime.process_alive(old.pid),
            new_process_alive=runtime.process_alive(current.pid),
            listener_pids=runtime.listener_owners(args.port),
            health_ok=runtime.health_ok(args.health_url),
            expected_code_sha=args.expected_code_sha,
            required_platforms=(
                tuple(args.required_platform) if args.required_platform is not None else None
            ),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())


# Atomic restart-if-idle authorization contract.
EXIT_OK = 0
EXIT_DENIED = 20
EXIT_TIMEOUT = 21
EXIT_UNSUPPORTED = 22

_REQUIRED_OBLIGATIONS = (
    "delivery_queue",
    "delivery_ledger",
    "pending_final",
    "drain",
    "delegation_workers",
    "updater",
    "update_lock",
    "oauth_refresh",
    "oauth_token_lock",
)

_SUPPORTED_RESTART_SUPERVISORS = frozenset({"launchd", "systemd", "desktop"})


@dataclass(frozen=True)
class ExpectedGatewayIdentity:
    protocol: int
    kind: str
    pid: int
    start_time: int | float | str
    hermes_home: str
    code_sha: str
    required_platforms: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExpectedGatewayIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("expected identity must be an object")
        required = ("protocol", "kind", "pid", "start_time", "hermes_home", "code_sha")
        if any(key not in value for key in required):
            raise ValueError("expected identity is missing required fields")
        if type(value["protocol"]) is not int or type(value["pid"]) is not int:
            raise ValueError("protocol and pid must be integers")
        if not isinstance(value["kind"], str) or not isinstance(value["code_sha"], str):
            raise ValueError("kind and code_sha must be strings")
        if not isinstance(value["hermes_home"], str) or not value["hermes_home"]:
            raise ValueError("hermes_home must be a non-empty string")
        if not isinstance(value["start_time"], (int, float, str)) or isinstance(value["start_time"], bool):
            raise ValueError("start_time has invalid type")
        platforms = value.get("required_platforms", ())
        if not isinstance(platforms, (list, tuple)) or any(not isinstance(p, str) or not p for p in platforms):
            raise ValueError("required_platforms must be a list of names")
        return cls(
            protocol=value["protocol"], kind=value["kind"], pid=value["pid"],
            start_time=value["start_time"], hermes_home=str(Path(value["hermes_home"]).expanduser().resolve(strict=False)),
            code_sha=value["code_sha"], required_platforms=tuple(platforms),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "kind": self.kind,
            "pid": self.pid,
            "start_time": self.start_time,
            "hermes_home": self.hermes_home,
            "code_sha": self.code_sha,
            "required_platforms": list(self.required_platforms),
        }


@dataclass(frozen=True)
class RestartResult:
    exit_code: int
    reason: str


def _canonical_home(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def authorize_restart_if_idle(snapshot: Mapping[str, Any], expected: ExpectedGatewayIdentity) -> tuple[bool, str]:
    """Authorize only an exact serving identity with a strictly idle snapshot."""
    if not isinstance(snapshot, Mapping):
        return False, "invalid-serving-snapshot"
    checks = (
        (type(snapshot.get("protocol")) is int and snapshot.get("protocol") == expected.protocol,
         "identity-protocol-mismatch"),
        (snapshot.get("kind") == expected.kind == "hermes-gateway", "identity-kind-mismatch"),
        (type(snapshot.get("pid")) is int and snapshot.get("pid") == expected.pid,
         "identity-pid-mismatch"),
        (isinstance(snapshot.get("start_time"), (int, float))
         and not isinstance(snapshot.get("start_time"), bool)
         and math.isfinite(float(snapshot["start_time"]))
         and snapshot.get("start_time") == expected.start_time, "identity-start-time-mismatch"),
        (_canonical_home(snapshot.get("hermes_home")) == _canonical_home(expected.hermes_home), "identity-home-mismatch"),
        (snapshot.get("code_sha") == expected.code_sha, "identity-sha-mismatch"),
        (snapshot.get("answering_pid") == expected.pid, "identity-answering-pid-mismatch"),
        (snapshot.get("signal_target_pid") == expected.pid, "identity-signal-target-pid-mismatch"),
        (type(snapshot.get("supervisor_pid")) is int and snapshot.get("supervisor_pid") != expected.pid,
         "identity-supervisor-pid-mismatch"),
        (snapshot.get("supervisor") in _SUPPORTED_RESTART_SUPERVISORS, "unsupported-supervisor"),
        (snapshot.get("gateway_state") == "running", "gateway-not-running"),
        (isinstance(snapshot.get("session_store"), Mapping) and snapshot["session_store"].get("status") == "ok", "session-store-unhealthy"),
        (type(snapshot.get("active_agents")) is int and snapshot.get("active_agents") == 0, "active-agents"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    obligations = snapshot.get("obligations")
    if not isinstance(obligations, Mapping):
        return False, "obligations-missing"
    for name in _REQUIRED_OBLIGATIONS:
        if type(obligations.get(name)) is not int or obligations.get(name) != 0:
            return False, f"obligation-{name}"
    for name, value in obligations.items():
        if name not in _REQUIRED_OBLIGATIONS and (type(value) is not int or value != 0):
            return False, f"obligation-{name}"
    platforms = snapshot.get("platforms")
    if not isinstance(platforms, Mapping):
        return False, "platforms-missing"
    configured_raw = snapshot.get("configured_platforms", ())
    if not isinstance(configured_raw, (list, tuple)):
        return False, "configured-platforms-missing"
    configured = tuple(str(name) for name in configured_raw if isinstance(name, str) and name)
    if "__configured_platforms_unavailable__" in configured:
        return False, "configured-platforms-unavailable"
    required_platforms = tuple(dict.fromkeys((*configured, *expected.required_platforms)))
    for name in required_platforms:
        platform = platforms.get(name)
        if not isinstance(platform, Mapping):
            return False, f"platform-{name}-missing"
        if platform.get("state") not in {"connected", "running", "ok"}:
            return False, f"platform-{name}-unhealthy"
        if platform.get("writer_pid") != expected.pid or platform.get("writer_start_time") != expected.start_time:
            return False, f"platform-{name}-stale-writer"
    return True, "accepted"


def restart_gateway_if_idle(
    expected: ExpectedGatewayIdentity,
    transport: Callable[..., Optional[dict[str, Any]]],
    *,
    timeout: float = 2.0,
) -> RestartResult:
    """Request one in-process atomic restart; unsupported/timeout/denial never fall back."""
    try:
        response = transport(
            "restart-if-idle",
            request_fields={"expected_identity": expected.to_mapping()},
            timeout=timeout,
        )
    except TimeoutError:
        return RestartResult(EXIT_TIMEOUT, "timeout")
    except Exception:
        return RestartResult(EXIT_DENIED, "transport-error")
    if not isinstance(response, Mapping):
        return RestartResult(EXIT_DENIED, "control-unavailable")
    raw_reason = response.get("reason")
    reason: str = raw_reason if isinstance(raw_reason, str) else "denied"
    if response.get("accepted") is True and reason == "accepted":
        return RestartResult(EXIT_OK, reason)
    if reason == "unsupported":
        return RestartResult(EXIT_UNSUPPORTED, reason)
    if reason == "timeout":
        return RestartResult(EXIT_TIMEOUT, reason)
    return RestartResult(EXIT_DENIED, reason)
