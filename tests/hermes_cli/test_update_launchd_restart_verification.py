"""Regression for #88848 - a launchd restart the update never verified.

``hermes update`` on macOS printed ``Update complete!`` and exited 0 while the
``ai.hermes.gateway`` LaunchAgent sat deregistered for 36 minutes.  The restart
phase treated "``launchd_restart()`` returned without raising" as success and
appended the label to ``restarted_services``.  Both of launchd_restart's normal
outcomes are asynchronous - the ``_request_gateway_self_restart`` branch returns
the instant the running gateway is *asked* to restart, and a plist reload is
handed to a detached helper - so a helper that died before its first bootstrap
was invisible to the caller.

The systemd branch of the same phase has never drawn that inference: it polls
``_wait_for_service_active`` before recording the unit and routes a unit that
never came back into ``failed_or_stale_units``, which is what makes the update
exit non-zero.  These tests pin the same contract for launchd.

No macOS hardware is involved: every case drives the seam through mocked
``launchctl`` outcomes.
"""

from __future__ import annotations

import subprocess

import pytest

import hermes_cli.gateway as gateway_cli
import hermes_cli.update_cmd as update_cmd
from hermes_cli.gateway_restart_contract import RestartProbe, ServingIdentity
from hermes_cli.update_cmd import _warn_incomplete_gateway_fleet_restart

LABEL = "ai.hermes.gateway"


@pytest.mark.parametrize("returncode", [3, 113])
def test_launchd_print_not_found_is_definitely_absent(monkeypatch, returncode):
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, "", "not found"),
    )

    assert gateway_cli._launchd_print_service_pid("gui/501", LABEL) == (False, None)


@pytest.mark.parametrize("returncode", [1, 125])
def test_launchd_print_query_error_is_unknown(monkeypatch, returncode):
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, "", "denied"),
    )

    with pytest.raises(subprocess.CalledProcessError):
        gateway_cli._launchd_print_service_pid("gui/501", LABEL)


def test_launchd_locator_uses_positive_domain_after_other_domain_error(monkeypatch):
    calls = iter(
        [subprocess.CalledProcessError(125, ["launchctl"]), (True, 4200)]
    )

    def probe(_domain, _label):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(gateway_cli, "_launchd_print_service_pid", probe)

    assert gateway_cli._locate_launchd_gateway_service(LABEL) == (
        f"user/{gateway_cli.os.getuid()}",
        4200,
    )


def test_launchd_locator_error_plus_absent_is_unknown(monkeypatch):
    calls = iter([subprocess.CalledProcessError(125, ["launchctl"]), (False, None)])

    def probe(_domain, _label):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(gateway_cli, "_launchd_print_service_pid", probe)

    with pytest.raises(subprocess.CalledProcessError):
        gateway_cli._locate_launchd_gateway_service(LABEL)


def test_launchd_registration_query_error_is_unknown(monkeypatch):
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 125, "", "denied"),
    )

    with pytest.raises(subprocess.CalledProcessError):
        gateway_cli._launchd_service_registered(LABEL)


class _FakeClock:
    """Monotonic clock that only advances when the code under test sleeps.

    Keeps the poll loop's wall-clock budget honest without spending it: a real
    20s verification timeout would otherwise make this file the slowest in the
    suite, and shortening the timeout would stop testing the throttle window.
    """

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(gateway_cli.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(gateway_cli.time, "sleep", fake.sleep)
    return fake


@pytest.fixture(autouse=True)
def _no_detached_fallback(monkeypatch):
    """Default every test to "launchd can manage this domain"."""
    monkeypatch.setattr(
        gateway_cli, "_launchd_unsupported_marker_exists", lambda: False
    )


def _supervision_returning(*results):
    """Fake ``_launchctl_label_supervising_process`` yielding ``results`` in order.

    The final value repeats, so a test can say "False twenty times, then True
    from then on".
    """
    seq = list(results)
    calls = []

    def probe(label):
        calls.append(label)
        return seq[min(len(calls) - 1, len(seq) - 1)]

    probe.calls = calls
    return probe


class TestWaitForLaunchdGatewaySupervision:
    def test_returns_true_when_already_supervised(self, monkeypatch, clock):
        """The common case must not cost a single sleep."""
        monkeypatch.setattr(
            gateway_cli,
            "_launchctl_label_supervising_process",
            _supervision_returning(True),
        )

        assert gateway_cli.wait_for_launchd_gateway_supervision(label=LABEL) is True
        assert clock.slept == []

    def test_waits_out_the_launchd_respawn_throttle(self, monkeypatch, clock):
        """A pid that only appears after ~10s is a SUCCESS, not a failure.

        launchd will not relaunch a KeepAlive job more than about once per 10
        seconds, so a gateway that exits promptly leaves the label registered
        with no pid for most of that window.  A one-shot check - or any budget
        shorter than the throttle - would report a perfectly healthy restart as
        a silent failure, which is a worse bug than the one being fixed.
        """
        # 0.5s poll interval: 20 misses is ~10s of throttle, then the pid lands.
        probe = _supervision_returning(*([False] * 20 + [True]))
        monkeypatch.setattr(
            gateway_cli, "_launchctl_label_supervising_process", probe
        )

        assert gateway_cli.wait_for_launchd_gateway_supervision(label=LABEL) is True
        assert sum(clock.slept) == pytest.approx(10.0)

    def test_gives_up_at_the_deadline(self, monkeypatch, clock):
        """A job that never comes back must fail, and must fail bounded."""
        probe = _supervision_returning(False)
        monkeypatch.setattr(
            gateway_cli, "_launchctl_label_supervising_process", probe
        )

        assert (
            gateway_cli.wait_for_launchd_gateway_supervision(
                label=LABEL, timeout=20.0
            )
            is False
        )
        assert sum(clock.slept) <= 20.0
        # The deadline is enforced by wall clock, not by a probe count.
        assert len(probe.calls) == 41

    def test_detached_fallback_is_not_a_failure(self, monkeypatch, clock):
        """On a host where launchd cannot manage the domain, no pid is correct.

        ``_launchd_fallback_to_detached`` is a legitimate outcome (macOS 26+
        unmanageable domains); the gateway runs unsupervised by design there.
        Reporting that as an incomplete update would fail every update on those
        hosts.
        """
        monkeypatch.setattr(
            gateway_cli, "_launchd_unsupported_marker_exists", lambda: True
        )
        probe = _supervision_returning(False)
        monkeypatch.setattr(
            gateway_cli, "_launchctl_label_supervising_process", probe
        )

        assert gateway_cli.wait_for_launchd_gateway_supervision(label=LABEL) is True
        assert probe.calls == []


def _patch_launchd_env(
    monkeypatch,
    *,
    plist_exists=True,
    registered=True,
    restart=None,
    supervised=True,
    running_pid: int | None = 4321,
    domain: str | None = "gui/501",
):
    """Drive ``_restart_macos_launchd_gateways`` through the invoking profile only.

    ``launchd_gateway_labels_for_install`` is pinned to the current label so the
    sibling loop is a no-op: this file is about the invoking profile, which is
    the one branch that was never verified.
    """

    class _Plist:
        def exists(self):
            return plist_exists

    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: _Plist())
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(
        gateway_cli, "_launchd_service_registered", lambda label: registered
    )
    monkeypatch.setattr(
        gateway_cli, "launchd_gateway_labels_for_install", lambda: [LABEL]
    )
    monkeypatch.setattr(
        gateway_cli,
        "_locate_launchd_gateway_service",
        lambda _label: (domain, running_pid),
    )

    calls = {"restart": 0, "verify": 0, "label": None}

    def _restart():
        calls["restart"] += 1
        if restart is not None:
            raise restart

    monkeypatch.setattr(gateway_cli, "launchd_restart", _restart)

    def _verify(label, *, old_pid, timeout, domain, expected_pid=None):
        calls["verify"] += 1
        calls["label"] = label
        calls["expected_pid"] = expected_pid
        return supervised

    monkeypatch.setattr(
        gateway_cli, "_wait_for_launchd_service_pid", _verify
    )
    identity = ServingIdentity(
        9876, 12345, update_cmd.get_hermes_home(), "verified-code"
    )
    monkeypatch.setattr(
        update_cmd,
        "_verified_graceful_restart_current_launchd_gateway",
        lambda: RestartProbe(True, "replacement-healthy", identity),
    )
    return calls


def _run_fleet_restart():
    """Run the real update-path helper and return its two accounting lists."""
    restarted: list = []
    failed_or_stale: list = []
    update_cmd._restart_macos_launchd_gateways(restarted, failed_or_stale, 5.0)
    return restarted, failed_or_stale


class TestInvokingProfileIsVerifiedLikeItsSiblings:
    """The sibling loop already polls for a fresh supervised pid before
    counting a label as restarted.  The invoking profile did not, so the two
    halves of the same function disagreed about what "restarted" means."""

    def test_active_gateway_is_deferred_without_launchd_fallback(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch, supervised=True)
        identity = ServingIdentity(4321, 9876, update_cmd.get_hermes_home(), "old")
        monkeypatch.setattr(
            update_cmd,
            "_verified_graceful_restart_current_launchd_gateway",
            lambda: RestartProbe(False, "active-work:1", identity),
            raising=False,
        )

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == []
        assert failed_or_stale == [LABEL]
        assert calls["restart"] == 0
        assert calls["verify"] == 0

    def test_idle_gateway_uses_verified_contract_without_force_fallback(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch, supervised=True)
        identity = ServingIdentity(9876, 12345, update_cmd.get_hermes_home(), "new")
        monkeypatch.setattr(
            update_cmd,
            "_verified_graceful_restart_current_launchd_gateway",
            lambda: RestartProbe(True, "replacement-healthy", identity),
            raising=False,
        )

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == [LABEL]
        assert failed_or_stale == []
        assert calls["restart"] == 0
        assert calls["verify"] == 1

    def test_reports_restarted_only_after_supervision_is_confirmed(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch, supervised=True)

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == [LABEL]
        assert failed_or_stale == []
        assert calls["restart"] == 0
        assert calls["verify"] == 1
        assert calls["label"] == LABEL
        assert calls["expected_pid"] == 9876

    def test_supervision_of_wrong_fresh_pid_is_not_accepted(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch, supervised=False)

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == []
        assert failed_or_stale == [LABEL]
        assert calls["expected_pid"] == 9876

    def test_unverified_restart_is_not_reported_as_restarted(
        self, monkeypatch, capsys
    ):
        """THE regression (#88848).

        ``launchd_restart()`` returns normally - that is the reported failure's
        exact shape, since the ``_request_gateway_self_restart`` branch returns
        without raising while the reload helper dies afterwards.  Before the
        fix this appended the label to ``restarted_services`` and the update
        reported the gateway as restarted, and exited 0, over a job that was
        deregistered from launchd.
        """
        _patch_launchd_env(monkeypatch, supervised=False)

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == []
        # Routed into failed_or_stale_units, which sets
        # gateway_fleet_restart_incomplete and makes the update exit 1.
        assert failed_or_stale == [LABEL]
        assert LABEL in capsys.readouterr().out

    def test_verification_budget_clears_the_respawn_throttle(self):
        """A budget under launchd's ~10s respawn throttle would false-alarm.

        The call site takes the helper's default, so the default is the
        contract that has to stay above the throttle.
        """
        assert gateway_cli.LAUNCHD_SUPERVISION_VERIFY_TIMEOUT >= 15.0

    def test_contract_exception_is_not_verified_and_is_reported(
        self, monkeypatch, capsys
    ):
        """A failed contract probe is loud and never reaches supervision."""
        calls = _patch_launchd_env(monkeypatch)

        def fail_contract():
            raise OSError("boom")

        monkeypatch.setattr(
            update_cmd,
            "_verified_graceful_restart_current_launchd_gateway",
            fail_contract,
        )

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == []
        assert failed_or_stale == [LABEL]
        assert calls["verify"] == 0
        assert "boom" in capsys.readouterr().out

    def test_no_plist_means_the_gateway_is_not_launchd_managed(self, monkeypatch):
        calls = _patch_launchd_env(monkeypatch, plist_exists=False)

        assert _run_fleet_restart() == ([], [])
        assert calls["restart"] == 0
        assert calls["verify"] == 0

    def test_unregistered_label_is_not_force_bootstrapped(self, monkeypatch):
        """Missing serving identity fails closed instead of bootstrapping."""
        calls = _patch_launchd_env(
            monkeypatch, registered=False, running_pid=None, domain=None
        )
        monkeypatch.setattr(
            update_cmd,
            "_verified_graceful_restart_current_launchd_gateway",
            lambda: RestartProbe(False, "control-identity-missing", None),
        )

        assert _run_fleet_restart() == ([], [])
        assert calls["restart"] == 0
        assert calls["verify"] == 0

    def test_launchctl_list_failure_does_not_hide_live_current_gateway(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(
            monkeypatch, registered=False, running_pid=4321, domain="gui/501"
        )

        restarted, failed = _run_fleet_restart()

        assert (restarted, failed) == ([LABEL], [])
        assert calls["verify"] == 1

    def test_registered_but_unlocatable_current_is_failed_unknown(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(
            monkeypatch, registered=True, running_pid=None, domain=None
        )

        restarted, failed = _run_fleet_restart()

        assert (restarted, failed) == ([], [LABEL])
        assert calls["restart"] == 0

    def test_unexpected_current_contract_exception_is_isolated(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch)

        def explode():
            raise RuntimeError("adapter exploded")

        monkeypatch.setattr(
            update_cmd,
            "_verified_graceful_restart_current_launchd_gateway",
            explode,
        )

        restarted, failed = _run_fleet_restart()

        assert (restarted, failed) == ([], [LABEL])
        assert calls["verify"] == 0

    def test_loaded_but_stopped_current_profile_skips_contract_and_signal(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch, registered=True, running_pid=None)

        restarted, failed = _run_fleet_restart()

        assert (restarted, failed) == ([], [])
        assert calls["restart"] == 0
        assert calls["verify"] == 0


class TestIncompleteFleetWarningIsPlatformCorrect:
    def test_macos_recovery_instructions_use_safe_retry(self, monkeypatch, capsys):
        """A launchd label must not be handed force or systemd commands."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)

        _warn_incomplete_gateway_fleet_restart([LABEL])

        out = capsys.readouterr().out
        assert "hermes update --gateway" in out
        assert "launchctl bootstrap" not in out
        assert "launchctl kickstart -k" not in out
        assert "systemctl" not in out

    def test_linux_recovery_instructions_are_unchanged(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        _warn_incomplete_gateway_fleet_restart(["hermes-gateway.service"])

        out = capsys.readouterr().out
        assert "systemctl --user restart <unit>" in out
        assert "launchctl" not in out
