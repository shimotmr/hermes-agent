from __future__ import annotations

import subprocess

import pytest


def test_live_system_guard_blocks_launchctl_gateway_mutation() -> None:
    harmless_shell_command = [
        "/bin/sh",
        "-c",
        ": # launchctl bootout gui/501/ai.hermes.gateway",
    ]

    with pytest.raises(RuntimeError, match="launchctl"):
        subprocess.run(harmless_shell_command, check=True)


def test_live_system_guard_allows_read_only_launchctl_text() -> None:
    harmless_shell_command = [
        "/bin/sh",
        "-c",
        ": # launchctl print gui/501/ai.hermes.gateway",
    ]

    subprocess.run(harmless_shell_command, check=True)
