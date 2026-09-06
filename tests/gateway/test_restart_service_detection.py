"""/restart 的環境標記不構成 relaunch 授權；缺少 manager 證明保留 detached fallback。"""
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent, MessageType
from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _make_restart_event(update_id: int | None = 100) -> MessageEvent:
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
        platform_update_id=update_id,
    )


def _make_runner_with_mock_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    monkeypatch.setattr("gateway.restart_relaunch.probe_gateway_relaunch", lambda *_: None)
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    return runner


@pytest.mark.asyncio
async def test_restart_with_external_supervisor_marker_keeps_detached_path(
    tmp_path, monkeypatch
):
    """自行宣告的 supervisor 標記不能授權退出服務。"""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, "1")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "0", "false", "off"])
async def test_false_external_supervisor_marker_keeps_detached_path(
    value, tmp_path, monkeypatch
):
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, value)

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=True, via_service=False)
