"""實際 admission、健康與重啟邊界的回歸契約。"""
import asyncio
import json
import os
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.restart_runtime import GatewayAdmissionBarrier, GatewayAdmissionDenied
from gateway.session import SessionSource


class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        self._mark_connected()
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, **kwargs):
        return SendResult(success=True)


def adapter():
    return Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)


def event(text='hello'):
    return MessageEvent(text=text, source=SessionSource(platform=Platform.TELEGRAM, chat_id='1'))


@pytest.mark.asyncio
async def test_platform_admission_spans_topic_lookup_and_stop_bypasses(monkeypatch):
    a = adapter()
    barrier = a._restart_admission_barrier = GatewayAdmissionBarrier()
    entered, release = threading.Event(), threading.Event()
    routed = []

    def recover(_event):
        entered.set()
        assert release.wait(3)

    async def handler(evt):
        routed.append(evt.text)

    a.set_message_handler(handler)
    a._topic_recovery_fn = recover
    monkeypatch.setattr(a, '_apply_topic_recovery', recover)
    task = asyncio.create_task(a.handle_message(event()))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert barrier.active_admissions() == 1
    finally:
        release.set()
        await task
        await asyncio.gather(*a._background_tasks)
    barrier.close('restart')
    a._topic_recovery_fn = None
    await a.handle_message(event('blocked'))
    await a.handle_message(event('/retry'))
    await a.handle_message(event('/new'))
    await a.handle_message(event('/reset'))
    await a.handle_message(event('/stop'))
    await asyncio.gather(*a._background_tasks)
    assert routed == ['hello', '/stop']


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/new', '/reset'])
async def test_reset_commands_hold_admission_during_topic_recovery(monkeypatch, command):
    a = adapter()
    barrier = a._restart_admission_barrier = GatewayAdmissionBarrier()
    entered, release = threading.Event(), threading.Event()

    def recover(_event):
        entered.set()
        assert release.wait(3)

    async def handler(_event):
        return None

    a.set_message_handler(handler)
    a._topic_recovery_fn = recover
    monkeypatch.setattr(a, '_apply_topic_recovery', recover)
    task = asyncio.create_task(a.handle_message(event(command)))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert barrier.active_admissions() == 1
        result = await asyncio.to_thread(
            barrier.restart_if_idle, snapshot=lambda: {}, authorize=lambda _: (True, 'accepted'),
            signal_restart=lambda: None, already_requested=lambda: False,
            mark_requested=lambda: None)
        assert result[:2] == (False, 'admissions-active')
    finally:
        release.set()
        await task
        await asyncio.gather(*a._background_tasks)


@pytest.mark.asyncio
async def test_cross_thread_restart_rejects_live_lease_without_blocking_loop():
    barrier = GatewayAdmissionBarrier()
    signals = []
    with barrier.admission():
        result = await asyncio.wait_for(asyncio.to_thread(
            barrier.restart_if_idle, snapshot=lambda: {}, authorize=lambda _: (True, 'accepted'),
            signal_restart=lambda: signals.append(1), already_requested=lambda: False,
            mark_requested=lambda: None), 2)
        assert result[:2] == (False, 'admissions-active')
    assert signals == []
    assert barrier.closed_reason() is None


def test_final_snapshot_is_closed_even_to_reentrant_admission():
    barrier = GatewayAdmissionBarrier()
    signals = []

    def snapshot():
        with pytest.raises(GatewayAdmissionDenied):
            with barrier.admission():
                pass
        return {}

    assert barrier.restart_if_idle(
        snapshot=snapshot, authorize=lambda _: (True, 'accepted'),
        signal_restart=lambda: signals.append(1), already_requested=lambda: False,
        mark_requested=lambda: None)[0]
    assert signals == [1]


@pytest.mark.asyncio
async def test_api_creation_lease_spans_request_read_and_closed_returns_503(monkeypatch):
    from gateway.platforms import api_server, api_server_runs
    barrier = GatewayAdmissionBarrier()
    entered, release = asyncio.Event(), asyncio.Event()

    class Request:
        async def json(self):
            entered.set()
            await release.wait()
            raise ValueError('invalid request')

    a = SimpleNamespace(_restart_admission_barrier=barrier,
                        _parse_session_key_header=lambda _: (None, None))
    task = asyncio.create_task(api_server_runs._handle_runs(a, Request(), _api_server=api_server))
    await asyncio.wait_for(entered.wait(), 2)
    try:
        assert barrier.active_admissions() == 1
    finally:
        release.set()
        assert (await task).status == 400
    barrier.close('restart')
    assert (await api_server_runs._handle_runs(a, Request(), _api_server=api_server)).status == 503


def test_delegation_admission_rejects_before_durable_write(monkeypatch):
    from tools import async_delegation as delegation
    barrier = GatewayAdmissionBarrier()
    monkeypatch.setattr(delegation, '_restart_admission_barrier', barrier, raising=False)
    writes = []
    monkeypatch.setattr(delegation, '_persist_dispatch', lambda record: writes.append(record))
    barrier.close('restart')
    result = delegation.dispatch_async_delegation(
        goal='blocked', context=None, toolsets=None, role='leaf', model=None,
        session_key='test', runner=lambda: {})
    assert result['status'] == 'rejected'
    assert writes == []


@pytest.mark.asyncio
async def test_writer_identity_requires_live_health_and_process_start(monkeypatch):
    import gateway.status as status
    a = adapter()
    monkeypatch.setattr(status, 'get_process_start_time', lambda pid: 123 if pid == os.getpid() else None)
    assert a.restart_writer_health_probe()['writer_pid'] is None
    await a.connect()
    assert a.restart_writer_health_probe() == {
        'state': 'connected', 'writer_pid': os.getpid(), 'writer_start_time': 123}
    monkeypatch.setattr(status, 'get_process_start_time', lambda _: None)
    assert a.restart_writer_health_probe()['writer_pid'] is None
    monkeypatch.setattr(status, 'get_process_start_time', lambda _: 123)
    a._mark_degraded()
    assert a.restart_writer_health_probe()['writer_pid'] is None
    await a.connect()
    await a.disconnect()
    assert a.restart_writer_health_probe()['writer_pid'] is None


def test_spoofed_environment_cannot_authorize_service_restart(monkeypatch):
    from gateway.restart import is_gateway_supervisor_process
    for key in ('INVOCATION_ID', 'XPC_SERVICE_NAME', 'HERMES_S6_SUPERVISED_CHILD',
                'HERMES_GATEWAY_EXTERNAL_SUPERVISOR'):
        assert not is_gateway_supervisor_process({key: '1'})


@pytest.mark.asyncio
async def test_pause_without_contract_keeps_serving(tmp_path, monkeypatch):
    from gateway.run import _start_gateway_start_control_socket
    from gateway.control_socket import query_gateway_control
    calls = []
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = SimpleNamespace(_restart_admission_barrier=GatewayAdmissionBarrier(),
                             _build_restart_control_snapshot=lambda: {},
                             request_restart=lambda **kw: calls.append(kw) or True)
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        reply = await asyncio.to_thread(query_gateway_control, tmp_path, 'pause-for-update')
        assert reply['pausing'] is False
        assert calls == []
        assert runner._restart_admission_barrier.closed_reason() is None
    finally:
        await server.stop()


@pytest.mark.parametrize('change', [{}, {'pid': 99}, {'start_time': 99}, {'policy': 'never'}])
def test_control_requires_independent_matching_manager_attestation(tmp_path, monkeypatch, change):
    from gateway.control_socket import GatewayControlServer
    import gateway.restart_relaunch as restart
    proof = {'manager': 'launchd', 'service': 'gui/501/ai.hermes.gateway',
             'policy': 'keepalive', 'pid': 42, 'start_time': 1234}
    monkeypatch.setattr(restart, 'probe_gateway_relaunch', lambda *_: proof, raising=False)
    signals = []
    expected = {'protocol': 1, 'kind': 'hermes-gateway', 'pid': 42, 'start_time': 1234,
                'hermes_home': str(tmp_path), 'code_sha': 'a' * 40}
    snapshot = {**expected, 'answering_pid': 42, 'signal_target_pid': 42,
                'supervisor_pid': 24, 'supervisor': 'launchd', 'gateway_state': 'running',
                'session_store': {'status': 'ok'}, 'active_agents': 0,
                'configured_platforms': [], 'platforms': {},
                'obligations': dict.fromkeys(('delivery_queue', 'delivery_ledger', 'pending_final',
                    'drain', 'delegation_workers', 'updater', 'update_lock', 'oauth_refresh', 'oauth_token_lock'), 0)}
    server = GatewayControlServer(tmp_path, restart_snapshot=lambda: snapshot,
                                  graceful_signal=lambda: signals.append(1), sigusr1_supported=True,
                                  admission_barrier=GatewayAdmissionBarrier())
    request = {'protocol': 1, 'verb': 'restart-if-idle', 'expected_identity': expected}
    if change:
        request['relaunch_attestation'] = {**proof, **change}
    reply = json.loads(server.handle_request_line(json.dumps(request).encode()))['result']
    assert reply['accepted'] is False
    assert signals == []


@pytest.mark.asyncio
async def test_relay_handshake_and_transport_loss_control_writer_identity(monkeypatch):
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor, CONTRACT_VERSION
    from gateway.relay.ws_transport import WebSocketRelayTransport
    import gateway.status as status

    desc = CapabilityDescriptor(contract_version=CONTRACT_VERSION, platform='telegram', label='Telegram',
        max_message_length=4096, supports_draft_streaming=False, supports_edit=True,
        supports_threads=True, markdown_dialect='plain', len_unit='utf16', emoji='', platform_hint='', pii_safe=False)
    entered, release = asyncio.Event(), asyncio.Event()

    class Transport:
        is_connected = True
        fail = False

        def set_inbound_handler(self, handler):
            self.handler = handler

        async def connect(self):
            return True

        async def handshake(self):
            entered.set()
            await release.wait()
            if self.fail:
                raise OSError('lost handshake')
            return desc

        async def disconnect(self, **kwargs):
            self.is_connected = False

    transport = Transport()
    a = RelayAdapter(PlatformConfig(), desc, transport=transport)
    monkeypatch.setattr(status, 'get_process_start_time', lambda _: 123)
    task = asyncio.create_task(a.connect())
    await entered.wait()
    assert a.restart_writer_health_probe()['writer_pid'] is None
    release.set()
    assert await task
    assert a.restart_writer_health_probe()['writer_pid'] == os.getpid()
    transport.is_connected = False
    assert a.restart_writer_health_probe()['writer_pid'] is None
    transport.is_connected = True
    transport.fail = True
    assert not await a.connect(is_reconnect=True)
    assert a.restart_writer_health_probe()['writer_pid'] is None
    await a.disconnect()
    assert not a.is_connected
    # production transport: a handshake alone cannot attest a dead reader.
    ws = object.__new__(WebSocketRelayTransport)
    ws._closing, ws._ws = False, object()
    ws._descriptor_ready = asyncio.get_running_loop().create_future()
    ws._descriptor_ready.set_result(desc)
    ws._reader = asyncio.create_task(asyncio.Event().wait())
    try:
        assert ws.is_connected
    finally:
        ws._reader.cancel()
        await asyncio.gather(ws._reader, return_exceptions=True)
    assert not ws.is_connected


@pytest.mark.asyncio
@pytest.mark.parametrize('platform', ['matrix', 'mattermost'])
async def test_plugin_writer_teardown_resets_inherited_health(monkeypatch, platform):
    from importlib import import_module
    cls = getattr(import_module(f'plugins.platforms.{platform}.adapter'),
                  {'matrix': 'MatrixAdapter', 'mattermost': 'MattermostAdapter'}[platform])
    a = cls(PlatformConfig())
    a._mark_connected()
    await a.disconnect()
    assert not a.is_connected
    assert a.restart_writer_health_probe()['writer_pid'] is None


@pytest.mark.asyncio
async def test_matrix_dead_receive_task_cannot_claim_healthy_writer(monkeypatch):
    from plugins.platforms.matrix.adapter import MatrixAdapter
    import gateway.status as status
    a = MatrixAdapter(PlatformConfig())
    monkeypatch.setattr(status, 'get_process_start_time', lambda _: 123)
    a._mark_connected()
    a._sync_task = asyncio.create_task(asyncio.sleep(0))
    await a._sync_task
    assert a.restart_writer_health_probe()['writer_pid'] is None


@pytest.mark.asyncio
async def test_mattermost_unproven_websocket_cannot_claim_healthy_writer(monkeypatch):
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    import gateway.status as status
    a = MattermostAdapter(PlatformConfig())
    monkeypatch.setattr(status, 'get_process_start_time', lambda _: 123)
    a._mark_connected()
    assert a.restart_writer_health_probe()['writer_pid'] is None


def test_sync_delegation_spawn_is_closed_but_control_bypasses(monkeypatch):
    from tools import async_delegation, delegate_tool
    barrier = GatewayAdmissionBarrier()
    monkeypatch.setattr(async_delegation, '_restart_admission_barrier', barrier)
    barrier.close('restart')
    parent = SimpleNamespace()
    result = json.loads(delegate_tool.delegate_task(goal='blocked', parent_agent=parent))
    assert '重啟' in result['error']
    calls = []
    monkeypatch.setattr(delegate_tool, '_handle_control_action', lambda *args: calls.append(args) or '{}')
    assert delegate_tool.delegate_task(action='stop', parent_agent=parent) == '{}'
    assert len(calls) == 1


@pytest.mark.macos_only
@pytest.mark.asyncio
async def test_live_runner_writer_and_caller_manager_proof_over_control_socket(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _start_gateway_start_control_socket
    from gateway.control_socket import query_gateway_control
    from gateway.status import get_process_start_time
    from hermes_cli.gateway_restart_contract import ExpectedGatewayIdentity, restart_gateway_if_idle
    import gateway.restart_relaunch as relaunch

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('XPC_SERVICE_NAME', 'ai.hermes.test')
    pid = os.getpid()
    started = get_process_start_time(pid)
    commands = []

    def manager_read(args):
        commands.append(args)
        if args == ['launchctl', 'list']:
            return f'{pid}\t0\tai.hermes.test\n'
        assert args == ['launchctl', 'print', f'gui/{os.getuid()}/ai.hermes.test']
        return f'pid = {pid}\nproperties = keepalive | runatload\n'

    monkeypatch.setattr(relaunch, '_read_manager', manager_read)
    runner = GatewayRunner(GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}))
    runner._running = True
    a = adapter()
    runner._wire_adapter_handlers(a)
    runner.adapters[Platform.TELEGRAM] = a
    await a.connect()
    assert a._restart_admission_barrier is runner._restart_admission_barrier
    snapshot = runner._build_restart_control_snapshot()
    assert snapshot['platforms']['telegram']['writer_start_time'] == started
    assert snapshot['session_store']['status'] == 'ok'
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    signals = []
    server._graceful_signal = lambda: signals.append(pid)
    try:
        expected = ExpectedGatewayIdentity.from_mapping(snapshot)
        result = await asyncio.to_thread(restart_gateway_if_idle, expected,
            lambda verb, **kw: query_gateway_control(tmp_path, verb, **kw))
        assert result.exit_code == 0, result.reason
        assert signals == [pid]
        assert runner._restart_admission_barrier.closed_reason() == 'restart'
        assert sum(cmd == ['launchctl', 'list'] for cmd in commands) == 2
    finally:
        await server.stop()
        await a.disconnect()


@pytest.mark.macos_only
@pytest.mark.parametrize('pid_offset,policy', [(1, 'keepalive'), (0, 'runatload'), (0, 'keepalive')])
def test_manager_policy_and_pid_are_read_only_and_start_time_is_rechecked(monkeypatch, pid_offset, policy):
    from gateway.status import get_process_start_time
    import gateway.restart_relaunch as relaunch
    import gateway.status as status
    pid, calls = os.getpid(), []
    started = get_process_start_time(pid)

    def read(args):
        calls.append(args)
        if args == ['launchctl', 'list']:
            return f'{pid}\t0\tai.hermes.test\n'
        assert args[0:2] == ['launchctl', 'print']
        return f'pid = {pid + pid_offset}\nproperties = {policy}\n'

    monkeypatch.setattr(relaunch, '_read_manager', read)
    if not pid_offset and policy == 'keepalive':
        starts = iter((started, started + 1))
        monkeypatch.setattr(status, 'get_process_start_time', lambda _: next(starts))
    assert relaunch.probe_gateway_relaunch(pid, started) is None
    assert calls


def test_owned_delivery_count_tracks_durable_pending_until_ack(tmp_path, monkeypatch):
    from gateway import delivery_ledger as ledger
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    assert ledger.count_active_owned_obligations() == 0
    ledger.record_obligation(obligation_id='ours', session_key='s', platform='telegram',
                             chat_id='1', thread_id=None, content='reply')
    assert ledger.count_active_owned_obligations() == 1
    ledger.mark_attempting('ours')
    assert ledger.count_active_owned_obligations() == 1
    ledger.mark_failed('ours', 'offline')
    assert ledger.count_active_owned_obligations() == 1
    ledger.mark_delivered('ours')
    assert ledger.count_active_owned_obligations() == 0


@pytest.mark.parametrize('restart,prevent,success,accepted', [
    ('always', '78', '', True), ('on-failure', '', '', True),
    ('always', '75', '', False), ('on-failure', '', '75', False),
    ('no', '', '', False), ('on-failure', '', 'unknown', False),
])
def test_systemd_exit75_contract_respects_loaded_exit_status_policy(restart, prevent, success, accepted):
    from gateway.restart_relaunch import _systemd_exit75_policy
    policy = _systemd_exit75_policy({'Restart': restart, 'RestartPreventExitStatus': prevent,
                                    'SuccessExitStatus': success})
    assert (policy is not None) is accepted
