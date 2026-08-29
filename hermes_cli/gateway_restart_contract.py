"""Fail-closed contracts for identifying and validating Gateway restarts.

The Gateway-owned control socket is the authority for serving identity. Runtime
status, listener ownership, process liveness, and HTTP health are independent
cross-checks; no single signal is sufficient to declare a replacement healthy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence


REQUIRED_PLATFORMS = ("api_server", "webhook", "telegram")


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
    if payload.get("protocol") != 1:
        raise ValueError("control-protocol-mismatch")
    if payload.get("kind") != "hermes-gateway":
        raise ValueError("control-kind-mismatch")
    pid = payload.get("pid")
    start_time = payload.get("start_time")
    if not isinstance(pid, int) or pid <= 1:
        raise ValueError("invalid-serving-pid")
    if not isinstance(start_time, int) or start_time <= 0:
        raise ValueError("invalid-serving-start-time")
    home = Path(str(payload.get("hermes_home", ""))).expanduser()
    if home != expected_home.expanduser():
        raise ValueError("runtime-home-mismatch")
    code_sha = payload.get("code_sha")
    if code_sha is not None and not isinstance(code_sha, str):
        raise ValueError("invalid-code-sha")
    return ServingIdentity(pid, start_time, home, code_sha)


def _validate_status(
    current: ServingIdentity,
    status: dict[str, Any],
    *,
    required_platforms: tuple[str, ...] = REQUIRED_PLATFORMS,
) -> str | None:
    if status.get("gateway_state") != "running":
        return "gateway-not-running"
    if status.get("answering_pid") != current.pid:
        return "status-answering-pid-mismatch"
    platforms = status.get("platforms")
    if not isinstance(platforms, dict):
        return "platform-state-missing"
    for name in required_platforms:
        record = platforms.get(name)
        if not isinstance(record, dict) or record.get("state") != "connected":
            return f"platform-not-connected:{name}"
        if record.get("writer_pid") != current.pid:
            return f"platform-writer-mismatch:{name}"
    return None


def evaluate_current(
    *,
    current: ServingIdentity,
    status: dict[str, Any],
    process_alive: bool,
    listener_pids: set[int],
    health_ok: bool,
    require_idle: bool = False,
) -> RestartProbe:
    if not process_alive:
        return RestartProbe(False, "serving-process-not-alive", current)
    status_error = _validate_status(current, status)
    if status_error:
        return RestartProbe(False, status_error, current)
    if listener_pids != {current.pid}:
        return RestartProbe(False, "listener-owner-mismatch", current)
    if not health_ok:
        return RestartProbe(False, "health-check-failed", current)
    if require_idle:
        active_agents = status.get("active_agents")
        if not isinstance(active_agents, int) or active_agents < 0:
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
) -> RestartProbe:
    if old_process_alive:
        return RestartProbe(False, "old-serving-process-still-alive", current)
    if (current.pid, current.start_time) == (old.pid, old.start_time):
        return RestartProbe(False, "serving-identity-did-not-change", current)
    current_probe = evaluate_current(
        current=current,
        status=status,
        process_alive=new_process_alive,
        listener_pids=listener_pids,
        health_ok=health_ok,
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
) -> RestartProbe:
    try:
        current = parse_serving_identity(identify(home), expected_home=home)
    except (OSError, TypeError, ValueError) as exc:
        return RestartProbe(False, str(exc), None)
    runtime_status = status(home)
    if not isinstance(runtime_status, dict):
        return RestartProbe(False, "control-status-missing", current)
    return evaluate_current(
        current=current,
        status=runtime_status,
        process_alive=process_alive(current.pid),
        listener_pids=listener_owners(port),
        health_ok=health_ok(health_url),
        require_idle=require_idle,
    )


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

    if args.command == "healthy":
        return _emit_probe(
            evaluate_current(
                current=current,
                status=status_payload,
                process_alive=runtime.process_alive(current.pid),
                listener_pids=runtime.listener_owners(args.port),
                health_ok=runtime.health_ok(args.health_url),
                require_idle=args.require_idle,
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
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
