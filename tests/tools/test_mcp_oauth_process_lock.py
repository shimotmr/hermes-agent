"""Cross-process regression tests for rotating MCP OAuth refresh tokens.

The tests use two spawn-created Python processes, real Hermes token storage,
and the real MCP OAuth provider generator.  The parent process is a fake token
endpoint: it accepts the first use of the stale rotating refresh token and
rejects any second use.  No socket or external authorization flow is used.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue
import time
from pathlib import Path
from urllib.parse import parse_qs

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


def _seed_expired_oauth_state(hermes_home: Path) -> None:
    token_dir = hermes_home / "mcp-tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "srv.json").write_text(
        json.dumps(
            {
                "access_token": "stale-access",
                "token_type": "Bearer",
                "expires_in": 0,
                "expires_at": time.time() - 60,
                "refresh_token": "rotating-refresh-0",
            }
        )
    )
    (token_dir / "srv.client.json").write_text(
        json.dumps({"client_id": "test-client"})
    )
    (token_dir / "srv.meta.json").write_text(
        json.dumps(
            {
                "issuer": "http://127.0.0.1:9",
                "authorization_endpoint": "http://127.0.0.1:9/authorize",
                "token_endpoint": "http://127.0.0.1:9/token",
            }
        )
    )


def _oauth_flow_worker(
    hermes_home: str,
    worker_id: int,
    start,
    events,
    replies,
    *,
    lock_timeout: float | None = None,
) -> None:
    """Run one independent Hermes provider without performing network I/O."""
    os.environ["HERMES_HOME"] = hermes_home

    async def run() -> None:
        from mcp.shared.auth import OAuthClientMetadata
        from pydantic import AnyUrl

        from tools.mcp_oauth import HermesTokenStorage
        from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
        from tools.mcp_tool import sdk_httpx

        if lock_timeout is not None:
            import tools.mcp_oauth as oauth_module

            oauth_module._REFRESH_LOCK_TIMEOUT_SECONDS = lock_timeout

        assert _HERMES_PROVIDER_CLS is not None

        async def reject_redirect(_url: str) -> None:
            raise AssertionError("browser redirect must not be invoked")

        async def reject_callback() -> tuple[str, str | None]:
            raise AssertionError("browser callback must not be invoked")

        provider = _HERMES_PROVIDER_CLS(
            server_name="srv",
            server_url="http://127.0.0.1:9/mcp",
            client_metadata=OAuthClientMetadata(
                redirect_uris=[AnyUrl("http://127.0.0.1:9/callback")],
                client_name="Hermes race test",
            ),
            storage=HermesTokenStorage("srv"),
            redirect_handler=reject_redirect,
            callback_handler=reject_callback,
        )
        await provider._initialize()

        browser_calls = 0

        async def reject_browser_auth():
            nonlocal browser_calls
            browser_calls += 1
            raise AssertionError("browser authorization must not be invoked")

        provider._perform_authorization = reject_browser_auth
        httpx = sdk_httpx()
        original = httpx.Request("GET", "http://127.0.0.1:9/mcp")

        await asyncio.to_thread(start.wait)
        events.put((worker_id, "flow_started"))
        flow = provider.async_auth_flow(original)
        try:
            outgoing = await flow.__anext__()
            if outgoing is original:
                events.put(
                    (
                        worker_id,
                        "api_request",
                        outgoing.headers.get("Authorization"),
                    )
                )
                try:
                    await flow.asend(httpx.Response(200, request=outgoing))
                except StopAsyncIteration:
                    pass
            else:
                refresh_token = parse_qs(outgoing.content.decode())["refresh_token"][0]
                events.put((worker_id, "refresh_request", refresh_token))
                reply = await asyncio.to_thread(replies.get)
                if reply == "success":
                    response = httpx.Response(
                        200,
                        json={
                            "access_token": "winner-access",
                            "token_type": "Bearer",
                            "expires_in": 3600,
                            "refresh_token": "rotating-refresh-1",
                        },
                        request=outgoing,
                    )
                    next_request = await flow.asend(response)
                    assert next_request is original
                    events.put(
                        (
                            worker_id,
                            "api_request",
                            next_request.headers.get("Authorization"),
                        )
                    )
                    try:
                        await flow.asend(httpx.Response(200, request=next_request))
                    except StopAsyncIteration:
                        pass
                elif reply == "close":
                    response = httpx.Response(400, request=outgoing)
                    next_request = await flow.asend(response)
                    assert next_request is original
                    try:
                        await flow.asend(httpx.Response(200, request=next_request))
                    except StopAsyncIteration:
                        pass
                else:
                    response = httpx.Response(400, request=outgoing)
                    next_request = await flow.asend(response)
                    assert next_request is original
                    try:
                        await flow.asend(httpx.Response(200, request=next_request))
                    except StopAsyncIteration:
                        pass

            disk_tokens = json.loads(
                (Path(hermes_home) / "mcp-tokens" / "srv.json").read_text()
            )
            events.put(
                (
                    worker_id,
                    "done",
                    disk_tokens["access_token"],
                    browser_calls,
                )
            )
        except Exception as exc:
            events.put(
                (
                    worker_id,
                    "error",
                    type(exc).__name__,
                    str(exc),
                    browser_calls,
                )
            )
        finally:
            await flow.aclose()

    asyncio.run(run())


def _stop_processes(processes) -> None:
    for process in processes:
        if process.pid is not None:
            process.join(timeout=5)
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_two_processes_serialize_refresh_and_waiter_reuses_winner(tmp_path):
    """Only one process may consume a rotating refresh token.

    The waiter must reload after acquiring the per-server lock, observe the
    winner's fresh token, and send the API request directly.  A second stale
    refresh request is an independent proof that refresh rotation raced.
    """
    _seed_expired_oauth_state(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    events = ctx.Queue()
    replies = [ctx.Queue(), ctx.Queue()]
    processes = [
        ctx.Process(
            target=_oauth_flow_worker,
            args=(str(tmp_path), worker_id, start, events, replies[worker_id]),
        )
        for worker_id in range(2)
    ]

    try:
        for process in processes:
            process.start()
        start.set()

        refreshes = []
        observed = []
        winner_id = None
        while len(refreshes) < 1:
            event = events.get(timeout=10)
            observed.append(event)
            if event[1] == "refresh_request":
                refreshes.append(event)
                winner_id = event[0]

        # Hold the winner at the fake token endpoint long enough for an
        # unlocked peer to expose its competing stale refresh request.
        peer_deadline = time.monotonic() + 1
        while len(refreshes) == 1 and time.monotonic() < peer_deadline:
            try:
                event = events.get(timeout=max(peer_deadline - time.monotonic(), 0))
            except queue.Empty:
                break
            observed.append(event)
            if event[1] == "refresh_request":
                refreshes.append(event)

        assert winner_id is not None
        replies[winner_id].put("success")
        for event in refreshes[1:]:
            replies[event[0]].put("rejected-rotated-token")

        done = []
        deadline = time.monotonic() + 10
        while len(done) < 2 and time.monotonic() < deadline:
            try:
                event = events.get(timeout=1)
            except queue.Empty:
                continue
            observed.append(event)
            if event[1] == "refresh_request":
                refreshes.append(event)
                replies[event[0]].put("rejected-rotated-token")
            elif event[1] in {"done", "error"}:
                done.append(event)

        assert refreshes == [(winner_id, "refresh_request", "rotating-refresh-0")]
        assert sorted(event[2] for event in observed if event[1] == "api_request") == [
            "Bearer winner-access",
            "Bearer winner-access",
        ]
        assert sorted(done) == [
            (0, "done", "winner-access", 0),
            (1, "done", "winner-access", 0),
        ]
    finally:
        for reply in replies:
            reply.put("close")
        _stop_processes(processes)


def test_refresh_lock_timeout_fails_closed_without_browser_auth(tmp_path):
    """A bounded lock timeout must not refresh stale state or open a browser."""
    _seed_expired_oauth_state(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    events = ctx.Queue()
    holder_reply = ctx.Queue()
    contender_reply = ctx.Queue()
    holder = ctx.Process(
        target=_oauth_flow_worker,
        args=(str(tmp_path), 0, start, events, holder_reply),
    )
    contender = ctx.Process(
        target=_oauth_flow_worker,
        args=(str(tmp_path), 1, start, events, contender_reply),
        kwargs={"lock_timeout": 0.05},
    )

    try:
        holder.start()
        start.set()
        while True:
            event = events.get(timeout=10)
            if event[0] == 0 and event[1] == "refresh_request":
                break

        contender.start()
        timeout_event = events.get(timeout=10)
        while timeout_event[0] != 1 or timeout_event[1] == "flow_started":
            timeout_event = events.get(timeout=10)

        assert timeout_event[0:3] == (
            1,
            "error",
            "OAuthRefreshLockTimeout",
        )
        assert "refresh lock" in timeout_event[3].lower()
        assert timeout_event[4] == 0
    finally:
        holder_reply.put("close")
        contender_reply.put("close")
        _stop_processes([holder, contender])
