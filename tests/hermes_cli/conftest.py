"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_update_orchestrator(monkeypatch):
    """Hermetic boundary set for complete ``cmd_update`` orchestration tests."""
    from types import SimpleNamespace

    from hermes_cli import gateway as hermes_gateway
    from hermes_cli import main as hermes_main
    from hermes_cli import managed_uv, profiles, update_cmd, update_inventory
    from hermes_cli import update_receipt
    from tools import skills_sync

    forbidden: list[str] = []

    def forbid(name):
        def blocked(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError(f"live updater boundary called: {name}")

        return blocked

    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(hermes_main, "_capture_active_lazy_features", lambda: [])
    monkeypatch.setattr(hermes_main, "_capture_active_tool_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *a, **k: None)
    monkeypatch.setattr(
        hermes_main,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(hermes_main, "_refresh_active_lazy_features", lambda *a, **k: True)
    monkeypatch.setattr(hermes_main, "_restore_active_tool_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_editable_install_is_current", lambda *a, **k: True)
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda *a, **k: [])
    monkeypatch.setattr(
        update_cmd, "_sync_with_upstream_if_needed", lambda *a, **k: False
    )
    monkeypatch.setattr(managed_uv, "update_managed_uv", lambda *a, **k: None)
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda *a, **k: "/usr/bin/true")
    monkeypatch.setattr(update_cmd, "_restart_macos_launchd_gateways", lambda *a, **k: None)
    monkeypatch.setattr(
        update_inventory,
        "collect_runtime_inventory",
        lambda: SimpleNamespace(runtimes=()),
    )
    monkeypatch.setattr(update_inventory, "record_plan_in_receipt", lambda *a, **k: None)
    monkeypatch.setattr(
        skills_sync,
        "sync_skills",
        lambda **k: {
            key: []
            for key in ("copied", "updated", "user_modified", "cleaned", "relocated")
        },
    )
    monkeypatch.setattr(profiles, "list_profiles", lambda: [])
    monkeypatch.setattr(profiles, "backfill_profile_envs", lambda **k: [])
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **k: [])
    monkeypatch.setattr(hermes_gateway, "_get_service_pids", lambda **k: set())
    monkeypatch.setattr(hermes_gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )
    for boundary in (
        "_request_gateway_self_restart",
        "_graceful_restart_via_sigusr1",
        "_spawn_gateway_restart_watcher",
        "stop_profile_gateway",
    ):
        if hasattr(hermes_gateway, boundary):
            monkeypatch.setattr(hermes_gateway, boundary, forbid(boundary))
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions", lambda **k: []
    )
    monkeypatch.setattr(update_receipt, "begin_update_receipt", lambda *a, **k: None)
    monkeypatch.setattr(update_receipt, "record_plan", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(update_receipt, "record_step", lambda *a, **k: None)
    monkeypatch.setattr(update_receipt, "finalize_update_receipt", lambda *a, **k: None)

    yield forbidden

    assert forbidden == []


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
