"""唯讀查核 service manager 的即時 PID 與 graceful-exit 重啟政策。"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


def _read_manager(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=3, check=False)
    return result.stdout if result.returncode == 0 else ""


def _systemd_exit75_policy(props: Mapping[str, str]) -> dict[str, str] | None:
    """以已載入的狀態碼分類判斷 exit 75，未知格式拒絕。"""
    policy = props.get('Restart')
    if policy not in {'always', 'on-failure'}:
        return None
    statuses = {}
    for key in ('RestartPreventExitStatus', 'SuccessExitStatus'):
        raw = props.get(key)
        if raw is None or any(not token.isdecimal() for token in raw.split()):
            return None
        statuses[key] = {int(token) for token in raw.split()}
    if 75 in statuses['RestartPreventExitStatus']:
        return None
    if policy == 'on-failure' and 75 in statuses['SuccessExitStatus']:
        return None
    return {'policy': policy, 'prevent_exit_status': props['RestartPreventExitStatus'],
            'success_exit_status': props['SuccessExitStatus']}


def _systemd_contract(pid: int) -> dict[str, Any] | None:
    # cgroup 來自核心；環境變數與 gateway 自行寫入的狀態不能指定服務身分。
    cgroups = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    units = {part for line in cgroups.splitlines() for part in line.split(':', 2)[-1].split('/')
             if part.endswith('.service')}
    for unit in sorted(units):
        for scope in (['--user'], []):
            output = _read_manager(['systemctl', *scope, 'show', unit,
                '--property=MainPID,ActiveState,Restart,RestartPreventExitStatus,SuccessExitStatus'])
            props = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
            if props.get('MainPID') != str(pid) or props.get('ActiveState') != 'active':
                continue
            policy = _systemd_exit75_policy(props)
            if policy is None:
                return None
            return {'manager': 'systemd', 'service': unit,
                    'scope': 'user' if scope else 'system', **policy}
    return None


def _launchd_contract(pid: int) -> dict[str, Any] | None:
    # list 的 PID 欄位決定 label；print 驗證「已載入」政策，不能讀磁碟 plist 代替。
    for line in _read_manager(['launchctl', 'list']).splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[0] != str(pid):
            continue
        label = fields[2]
        for domain in (f'gui/{os.getuid()}', f'user/{os.getuid()}', 'system'):
            service = f'{domain}/{label}'
            output = _read_manager(['launchctl', 'print', service])
            current = re.search(r'^\s*pid = (\d+)\s*$', output, re.MULTILINE)
            props = re.search(r'^\s*properties = (.+)$', output, re.MULTILINE)
            if current is None or int(current[1]) != pid or props is None:
                continue
            if 'keepalive' not in {p.strip() for p in props[1].split('|')}:
                return None
            return {'manager': 'launchd', 'service': service, 'policy': 'keepalive'}
    return None


def probe_gateway_relaunch(pid: int, start_time: Any) -> dict[str, Any] | None:
    """只回傳目前 OS manager 足以保證 exit 75 重啟的契約；未知即拒絕。"""
    from gateway.status import get_process_start_time

    if type(pid) is not int or pid <= 1 or type(start_time) not in (int, float) or start_time <= 0:
        return None
    probe = {'darwin': _launchd_contract, 'linux': _systemd_contract}.get(sys.platform)
    if probe is None:
        return None
    try:
        if get_process_start_time(pid) != start_time:
            return None
        contract = probe(pid)
        if contract is None or get_process_start_time(pid) != start_time:
            return None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return {**contract, 'pid': pid, 'start_time': start_time}


def verify_relaunch_attestation(attestation: Any, *, pid: int, start_time: Any,
                               manager: str) -> bool:
    """呼叫端證明須與服務端獨立重讀的 manager 結果完全相符。"""
    if not isinstance(attestation, Mapping) or attestation.get('manager') != manager:
        return False
    if type(attestation.get('pid')) is not int or attestation.get('pid') != pid:
        return False
    if type(attestation.get('start_time')) not in (int, float) or attestation.get('start_time') != start_time:
        return False
    live = probe_gateway_relaunch(pid, start_time)
    return live is not None and dict(attestation) == live
