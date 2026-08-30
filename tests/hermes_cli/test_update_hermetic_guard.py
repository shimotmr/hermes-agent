"""Fail-closed proof for updater/service orchestration test boundaries."""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.update_orchestration


def test_update_scope_guard_is_autouse_and_blocks_live_backup(request) -> None:
    if "_update_scope_boundary_guard" not in request.fixturenames:
        pytest.fail("updater scope guard is not autouse")

    from hermes_cli import main as hermes_main

    with pytest.raises(AssertionError, match="live updater boundary called"):
        getattr(hermes_main, "_run_pre_update_backup")()


def test_update_scope_guard_blocks_package_install_subprocess(request) -> None:
    if "_update_scope_boundary_guard" not in request.fixturenames:
        pytest.fail("updater scope guard is not autouse")

    with pytest.raises(AssertionError, match="package install subprocess"):
        subprocess.run(["python", "-m", "pip", "install", "example.invalid"])
