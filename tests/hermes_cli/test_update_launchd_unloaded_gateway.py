"""macOS update restart contract: fail closed without launchctl force paths.

On macOS the update's launchd branch guarded the restart behind
``launchctl list <label>`` exiting 0. A job that has been *booted out* of
launchd exits non-zero there, so the whole restart branch was skipped — with
no ``else`` and no message. The update printed ``✓ Update complete!`` and
exited 0 while the gateway was stopped *and* deregistered, which ``KeepAlive``
cannot recover because the job definition is gone. Messaging adapters and
cron stayed dark until someone manually ran ``hermes gateway restart``.

``launchctl list`` is not a serving-identity oracle. A plist-present gateway
may only restart through its control socket after verified idle; missing or
unloaded identity is deferred and never bootstrapped by the updater.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import update_cmd
from hermes_cli.gateway_restart_contract import RestartProbe


class _FakePlist:
    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


@pytest.fixture
def launchd(monkeypatch):
    """Stub hermes_cli.gateway so no real launchctl call is made."""
    calls: list[str] = []
    state = {
        "plist": _FakePlist(True),
        "probe": RestartProbe(False, "control-identity-missing", None),
        "probe_exc": None,
    }
    subprocess_calls: list[list] = []

    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "get_launchd_label", lambda: "ai.hermes.gateway", raising=False)
    monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: state["plist"], raising=False)
    monkeypatch.setattr(
        gateway_mod,
        "_launchd_service_registered",
        lambda _label: True,
        raising=False,
    )
    monkeypatch.setattr(
        gateway_mod,
        "_locate_launchd_gateway_service",
        lambda _label: ("gui/501", 4321),
        raising=False,
    )

    def forbidden_restart():
        raise AssertionError("launchd_restart force path was called")

    monkeypatch.setattr(
        gateway_mod, "launchd_restart", forbidden_restart, raising=False
    )

    def fake_contract():
        calls.append("contract")
        if state["probe_exc"] is not None:
            raise state["probe_exc"]
        return state["probe"]

    monkeypatch.setattr(
        update_cmd,
        "_verified_graceful_restart_current_launchd_gateway",
        fake_contract,
    )

    def fake_run(*args, **kwargs):
        subprocess_calls.append(args[0] if args else [])
        return subprocess.CompletedProcess(args=args[0] if args else [], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    return calls, state, subprocess_calls


class TestLaunchdRestartAfterUpdate:
    def test_plist_present_uses_verified_contract_without_classifying(
        self, launchd, capsys
    ):
        calls, state, subprocess_calls = launchd
        state["probe"] = RestartProbe(True, "replacement-healthy", None)

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == (["ai.hermes.gateway"], [])
        assert calls == ["contract"]
        # No `launchctl list` classification happens in this helper.
        assert subprocess_calls == []
        assert "NOT running" not in capsys.readouterr().out

    def test_missing_identity_is_deferred_without_force_fallback(self, launchd, capsys):
        calls, state, _ = launchd

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == ([], ["ai.hermes.gateway"])
        out = capsys.readouterr().out
        assert calls == ["contract"]
        assert "control-identity-missing" in out
        assert "No force fallback" in out

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError("launchctl"),
            subprocess.TimeoutExpired(cmd=["launchctl", "kickstart"], timeout=90),
        ],
    )
    def test_contract_exception_is_not_swallowed(self, launchd, capsys, exc):
        calls, state, _ = launchd
        state["probe_exc"] = exc

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == ([], ["ai.hermes.gateway"])
        assert calls == ["contract"]
        out = capsys.readouterr().out
        assert "Could not verify a graceful gateway restart" in out
        assert "No force fallback" in out

    def test_no_plist_is_not_a_launchd_install(self, launchd, capsys):
        """No service definition → nothing to restart, and nothing to warn about."""
        calls, state, _ = launchd
        state["plist"] = _FakePlist(False)

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == ([], [])
        assert calls == []
        assert capsys.readouterr().out == ""


# `launchctl print gui/<uid>/<label>` excerpt for a running service, matching
# the real output shape (tab-indented, lowercase `pid = <N>`).
_PRINT_OUTPUT_RUNNING = """\
ai.hermes.gateway = {
\tactive count = 1
\tpath = /Users/u/Library/LaunchAgents/ai.hermes.gateway.plist
\tstate = running
\tpid = 59038
\tprogram = /Users/u/.hermes/bin/hermes
}
"""


class TestServicePidSweepExclusion:
    """Regression for the PR #75021 review: `_get_service_pids()` must not
    rely on `launchctl list` alone.

    In the session-scoped failure state (`list` exits non-zero while the
    domain-qualified `print` reports a positive PID) the launchd-owned
    gateway PID was missing from the exclusion set, so the post-update
    manual-gateway sweep could kill the process launchd just (re)started.
    """

    @pytest.fixture
    def macos_launchd(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod

        state = {"list_rc": 1, "print_rc": 0, "print_out": _PRINT_OUTPUT_RUNNING}

        monkeypatch.setattr(gateway_mod, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_mod, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_mod, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_mod, "_launchd_domain", lambda: "gui/501")

        def fake_run(argv, **kwargs):
            if argv[:2] == ["launchctl", "list"]:
                return subprocess.CompletedProcess(argv, state["list_rc"], stdout="", stderr="")
            if argv[:2] == ["launchctl", "print"]:
                if argv[2].startswith("user/501/"):
                    return subprocess.CompletedProcess(
                        argv, 113, stdout="", stderr=""
                    )
                return subprocess.CompletedProcess(
                    argv, state["print_rc"], stdout=state["print_out"], stderr=""
                )
            raise AssertionError(f"unexpected subprocess call: {argv}")

        monkeypatch.setattr(gateway_mod.subprocess, "run", fake_run)
        return state

    def test_list_failure_falls_back_to_domain_print(self, macos_launchd):
        """`list` rc=1, `print` reports pid 59038 → the PID is still excluded."""
        from hermes_cli.gateway import _get_service_pids

        assert 59038 in _get_service_pids()

    def test_both_interfaces_negative_means_no_pid(self, macos_launchd):
        macos_launchd["print_rc"] = 113  # job genuinely not found in the domain

        from hermes_cli.gateway import _get_service_pids

        assert _get_service_pids() == set()

    def test_registered_but_not_running_has_no_pid_line(self, macos_launchd):
        macos_launchd["print_out"] = _PRINT_OUTPUT_RUNNING.replace("\tpid = 59038\n", "")

        from hermes_cli.gateway import _get_service_pids

        assert _get_service_pids() == set()


class TestParseLaunchdPidFromPrintOutput:
    def test_running_service(self):
        from hermes_cli.gateway import _parse_launchd_pid_from_print_output

        assert _parse_launchd_pid_from_print_output(_PRINT_OUTPUT_RUNNING) == 59038

    def test_no_pid_line(self):
        from hermes_cli.gateway import _parse_launchd_pid_from_print_output

        assert _parse_launchd_pid_from_print_output("state = not running\n") is None

    def test_nonpositive_pid_is_ignored(self):
        from hermes_cli.gateway import _parse_launchd_pid_from_print_output

        assert _parse_launchd_pid_from_print_output("\tpid = -1\n") is None
