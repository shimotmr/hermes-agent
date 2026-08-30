"""Regression for #41403 — ``hermes update`` must restart ALL macOS launchd gateways.

The macOS branch of the update's fleet-restart step only restarted the
invoking profile's LaunchAgent (``get_launchd_label()`` is profile-scoped).
Sibling ``ai.hermes.gateway-<profile>`` services kept running pre-update
modules cached in ``sys.modules`` and died on their next agent turn once the
new code lazily imported a symbol the old module generation didn't have
(``ImportError: cannot import name ...`` — or, with a wider version gap,
``TypeError``/``AttributeError`` on changed call signatures with garbled
tracebacks, because the source files on disk no longer match the loaded
code objects).

Also covers the launchd-domain review feedback on PR #41403: every sibling
interaction (liveness discovery, kickstart, fresh-PID verification) must be
domain-explicit — ``_launchd_domain()`` caches the *current* profile's
domain, and a sibling bootstrapped in the other supported domain
(``gui/<uid>`` vs ``user/<uid>``) would otherwise be probed or kickstarted
in a domain it does not live in.
"""

from __future__ import annotations

import json
import subprocess
import sys

import yaml

import pytest

import hermes_cli.gateway as gw
import hermes_cli.profiles
import hermes_cli.update_cmd as update_cmd
from hermes_cli.gateway_restart_contract import RestartProbe, ServingIdentity
from hermes_cli.gateway import (
    _locate_launchd_gateway_service,
    _parse_launchd_pid_from_print_output,
    _probe_launchd_domain_for_label,
    launchd_gateway_labels_for_install,
)
from hermes_cli.update_cmd import (
    _restart_macos_launchd_gateways,
    _warn_incomplete_gateway_fleet_restart,
)


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="launchd fleet restart is macOS-only; helpers use POSIX os.getuid",
)

UID = 501

PRINT_RUNNING = (
    "system/com.example = {\n"
    "\tactive count = 1\n"
    "\tstate = running\n"
    "\tpid = 4242\n"
    "\tprogram = /usr/bin/true\n"
    "}\n"
)
PRINT_LOADED_NOT_RUNNING = (
    "system/com.example = {\n"
    "\tactive count = 0\n"
    "\tstate = not running\n"
    "\tprogram = /usr/bin/true\n"
    "}\n"
)


@pytest.fixture(autouse=True)
def _fixed_uid(monkeypatch):
    monkeypatch.setattr(gw.os, "getuid", lambda: UID)


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


class _Profile:
    def __init__(self, name, is_default=False):
        self.name = name
        self.is_default = is_default


class TestLaunchdGatewayLabelsForInstall:
    def test_labels_derive_from_this_installs_profiles(self, monkeypatch):
        """The fleet is THIS install's profiles, root first — never a glob of
        the shared per-user LaunchAgents dir. A sandboxed HERMES_HOME (tests,
        side-by-side installs) must not enumerate — and restart — another
        install's services, and the hermetic test suite must not see the dev
        machine's real fleet."""
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [
                _Profile("tfl-wiki"),
                _Profile("default", is_default=True),
                _Profile("merit-ops"),
                _Profile("Bad Name!"),  # cannot map to a service suffix — skipped
            ],
        )
        assert launchd_gateway_labels_for_install() == [
            "ai.hermes.gateway",
            "ai.hermes.gateway-merit-ops",
            "ai.hermes.gateway-tfl-wiki",
        ]

    def test_no_profiles_means_no_fleet(self, monkeypatch):
        monkeypatch.setattr(hermes_cli.profiles, "list_profiles", lambda: [])
        assert launchd_gateway_labels_for_install() == []


class TestParseLaunchdPidFromPrintOutput:
    def test_running_service_pid(self):
        assert _parse_launchd_pid_from_print_output(PRINT_RUNNING) == 4242

    def test_loaded_but_not_running_has_no_pid(self):
        assert _parse_launchd_pid_from_print_output(PRINT_LOADED_NOT_RUNNING) is None


class TestLocateLaunchdGatewayService:
    def test_domains_resolve_per_label_not_from_cache(self, monkeypatch):
        """The #41403 review defect: sibling domains are independent."""
        gui_loaded = {"ai.hermes.gateway-a"}

        def fake_run(cmd, **kwargs):
            assert cmd[:2] == ["launchctl", "print"]
            domain, _, label = cmd[2].rpartition("/")
            in_gui = domain == f"gui/{UID}" and label in gui_loaded
            in_user = domain == f"user/{UID}" and label not in gui_loaded
            if in_gui or in_user:
                return _completed(0, PRINT_RUNNING)
            return _completed(113)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        # Simulate a prior current-profile resolution having populated the
        # process-wide cache — per-label lookups must not consult it.
        monkeypatch.setattr(gw, "_resolved_launchd_domain", f"gui/{UID}")

        assert _locate_launchd_gateway_service("ai.hermes.gateway-a") == (
            f"gui/{UID}",
            4242,
        )
        assert _locate_launchd_gateway_service("ai.hermes.gateway-b") == (
            f"user/{UID}",
            4242,
        )

    def test_loaded_without_live_process(self, monkeypatch):
        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd[2])
            if cmd[2].startswith(f"gui/{UID}/"):
                return _completed(0, PRINT_LOADED_NOT_RUNNING)
            return _completed(113)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        assert _locate_launchd_gateway_service("ai.hermes.gateway-x") == (
            f"gui/{UID}",
            None,
        )
        assert len(calls) == 2

    def test_cross_domain_multiple_registration_fails_closed(self, monkeypatch):
        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd[2])
            if cmd[2].startswith(f"gui/{UID}/"):
                return _completed(0, PRINT_LOADED_NOT_RUNNING)
            return _completed(0, PRINT_RUNNING)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="multiple launchd domains"):
            _locate_launchd_gateway_service("ai.hermes.gateway-x")
        assert len(calls) == 2

    @pytest.mark.parametrize("error_domain", ["gui", "user"])
    def test_stopped_plus_query_error_fails_closed(
        self, monkeypatch, error_domain
    ):
        def fake_run(cmd, **_kwargs):
            is_error_domain = cmd[2].startswith(f"{error_domain}/{UID}/")
            if is_error_domain:
                return _completed(1)
            return _completed(0, PRINT_LOADED_NOT_RUNNING)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)

        with pytest.raises(subprocess.CalledProcessError):
            _locate_launchd_gateway_service("ai.hermes.gateway-x")

    def test_not_loaded_in_either_domain(self, monkeypatch):
        monkeypatch.setattr(gw.subprocess, "run", lambda *a, **k: _completed(113))
        assert _locate_launchd_gateway_service("ai.hermes.gateway-x") == (None, None)

    def test_timeout_propagates_to_caller(self, monkeypatch):
        """A wedged launchctl must surface as a failure, not read as
        'unloaded' — the update path owns per-label failure accounting."""

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        with pytest.raises(subprocess.TimeoutExpired):
            _locate_launchd_gateway_service("ai.hermes.gateway-x")


class TestProbeLaunchdDomainForLabel:
    def test_unloaded_label_falls_back_to_managername(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "print"]:
                raise subprocess.CalledProcessError(113, cmd)
            if cmd == ["launchctl", "managername"]:
                return _completed(0, "Aqua\n")
            raise AssertionError(f"unexpected command {cmd}")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        assert _probe_launchd_domain_for_label("ai.hermes.gateway-x") == f"gui/{UID}"

    def test_unloaded_label_defaults_to_user_domain(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "print"]:
                raise subprocess.CalledProcessError(113, cmd)
            if cmd == ["launchctl", "managername"]:
                return _completed(0, "Background\n")
            raise AssertionError(f"unexpected command {cmd}")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        assert _probe_launchd_domain_for_label("ai.hermes.gateway-x") == f"user/{UID}"


class TestGetServicePidsScoping:
    def _wire(self, monkeypatch):
        monkeypatch.setattr(gw, "is_macos", lambda: True)
        monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(
            gw, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(lambda *_a, **_k: _completed())})
        )
        monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(
            gw,
            "launchd_gateway_labels_for_install",
            lambda: ["ai.hermes.gateway", "ai.hermes.gateway-a", "ai.hermes.gateway-b"],
        )
        located = {
            "ai.hermes.gateway": (f"gui/{UID}", 100),
            "ai.hermes.gateway-a": (f"gui/{UID}", 200),
            "ai.hermes.gateway-b": (None, None),  # not bootstrapped
        }
        monkeypatch.setattr(
            gw, "_locate_launchd_gateway_service", lambda label: located[label]
        )

    def test_all_profiles_returns_every_gateway_service_pid(self, monkeypatch):
        """The update sweep's exclude-set must protect ALL freshly-restarted
        services, not only the invoking profile's (else the sweep SIGTERMs
        gateways launchd just respawned)."""
        self._wire(monkeypatch)
        assert gw._get_service_pids(all_profiles=True) == {100, 200}

    def test_default_stays_scoped_to_current_profile(self, monkeypatch):
        """Regression guard: default-scope callers (gateway status, cron,
        stop_profile_gateway's orphan reaper) must NOT start seeing sibling
        service PIDs — the reaper SIGTERM/SIGKILLs what they feed it."""
        self._wire(monkeypatch)
        assert gw._get_service_pids() == {100}

    def test_find_gateway_pids_passes_profile_scope_through(self, monkeypatch):
        calls: list[bool] = []
        monkeypatch.setattr(
            gw,
            "_get_service_pids",
            lambda all_profiles=False: (calls.append(all_profiles), set())[1],
        )
        monkeypatch.setattr(gw, "_scan_gateway_pids", lambda *a, **k: [])
        monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)

        gw.find_gateway_pids(all_profiles=False)
        gw.find_gateway_pids(all_profiles=True)
        assert calls == [False, True]


class TestSiblingVerifiedRestartContract:
    """Sibling profiles must use their own control socket, never PID fallback."""

    @staticmethod
    def _wire(monkeypatch, tmp_path, *, probe):
        from types import SimpleNamespace

        current = "ai.hermes.gateway"
        sibling = "ai.hermes.gateway-research"
        sibling_home = tmp_path / "profiles" / "research"
        sibling_home.mkdir(parents=True)
        target = SimpleNamespace(label=sibling, home=sibling_home, port=9412)
        waits = []

        monkeypatch.setattr(
            update_cmd,
            "_restart_launchd_gateway_after_update",
            lambda **_kw: ([current], []),
        )
        monkeypatch.setattr(gw, "get_launchd_label", lambda: current)
        monkeypatch.setattr(
            gw, "launchd_gateway_labels_for_install", lambda: [current, sibling]
        )
        monkeypatch.setattr(
            gw,
            "_locate_launchd_gateway_service",
            lambda label: (f"gui/{UID}", 4200) if label == sibling else (None, None),
        )
        monkeypatch.setattr(
            update_cmd,
            "_resolve_launchd_gateway_contract_target",
            lambda label: target,
            raising=False,
        )
        monkeypatch.setattr(
            update_cmd,
            "_verified_graceful_restart_launchd_target",
            lambda resolved, **_kw: probe,
            raising=False,
        )

        def wait_for_pid(label, *, old_pid, timeout, domain, expected_pid=None):
            waits.append((label, old_pid, timeout, domain, expected_pid))
            return True

        monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", wait_for_pid)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("force-capable launchd fallback was called")

        monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", forbidden)
        monkeypatch.setattr(gw, "_launchd_kickstart", forbidden)
        return current, sibling, target, waits

    def test_active_sibling_fails_closed_without_pid_or_kickstart_fallback(
        self, monkeypatch, tmp_path
    ):
        old = ServingIdentity(4200, 42, tmp_path / "profiles" / "research", "old")
        current, sibling, _target, waits = self._wire(
            monkeypatch,
            tmp_path,
            probe=RestartProbe(False, "active-work:1", old),
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, 5.0)

        assert restarted == [current]
        assert failed == [sibling]
        assert waits == []

    def test_idle_sibling_requires_verified_replacement_and_supervision(
        self, monkeypatch, tmp_path
    ):
        new = ServingIdentity(4300, 43, tmp_path / "profiles" / "research", "new")
        current, sibling, _target, waits = self._wire(
            monkeypatch,
            tmp_path,
            probe=RestartProbe(True, "replacement-healthy", new),
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, 5.0)

        assert restarted == [current, sibling]
        assert failed == []
        assert waits == [(sibling, 4200, 10.0, f"gui/{UID}", 4300)]

    def test_ready_sibling_without_identity_fails_closed(
        self, monkeypatch, tmp_path
    ):
        current, sibling, _target, waits = self._wire(
            monkeypatch,
            tmp_path,
            probe=RestartProbe(True, "replacement-healthy", None),
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, 5.0)

        assert restarted == [current]
        assert failed == [sibling]
        assert waits == []

    @pytest.mark.parametrize(
        ("registered", "expected_failed"),
        [(False, []), (True, ["ai.hermes.gateway-research"])],
    )
    def test_unlocatable_sibling_distinguishes_stopped_from_unknown(
        self, monkeypatch, tmp_path, registered, expected_failed
    ):
        current, sibling, _target, waits = self._wire(
            monkeypatch,
            tmp_path,
            probe=RestartProbe(False, "must-not-run", None),
        )
        checks: list[str] = []
        monkeypatch.setattr(
            gw, "_locate_launchd_gateway_service", lambda _label: (None, None)
        )
        monkeypatch.setattr(
            gw,
            "_launchd_service_registered",
            lambda label: (checks.append(label), registered)[1],
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, 5.0)

        assert restarted == [current]
        assert failed == expected_failed
        assert checks == [sibling]
        assert waits == []

    def test_profile_resolution_exception_is_failed_not_silently_skipped(
        self, monkeypatch, tmp_path
    ):
        current, sibling, _target, waits = self._wire(
            monkeypatch,
            tmp_path,
            probe=RestartProbe(False, "unused", None),
        )

        def fail_resolution(_label):
            raise ImportError("profiles unavailable")

        monkeypatch.setattr(
            update_cmd,
            "_resolve_launchd_gateway_contract_target",
            fail_resolution,
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, 5.0)

        assert restarted == [current]
        assert failed == [sibling]
        assert waits == []


class TestLaunchdGatewayContractTargetResolution:
    def test_label_maps_to_profile_home_and_env_port(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "research"
        home.mkdir(parents=True)
        (home / ".env").write_text(
            "API_SERVER_KEY=test-only-key-123456\nAPI_SERVER_PORT=9412\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="research", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-research"
        )

        assert target.label == "ai.hermes.gateway-research"
        assert target.home == home.resolve()
        assert target.port == 9412

    def test_config_port_is_used_when_env_does_not_set_one(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "ops"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "platforms:\n  api_server:\n    extra:\n      port: 9513\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="ops", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-ops"
        )

        assert target.home == home.resolve()
        assert target.port == 9513

    def test_malformed_config_matches_gateway_default_without_mutation(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "broken"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("platforms: [unterminated", encoding="utf-8")
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="broken", path=home, is_default=False)],
        )

        before = sorted(path.name for path in home.iterdir())
        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-broken"
        )
        after = sorted(path.name for path in home.iterdir())

        assert target is not None
        assert target.port == 8642
        assert after == before

    @pytest.mark.parametrize(
        ("yaml_text", "expected"),
        [
            ("platforms:\n  api_server:\n    port: 9511\n", 9511),
            ("gateway:\n  api_server:\n    port: 9512\n", 9512),
            ("gateway:\n  platforms:\n    api_server:\n      port: 9513\n", 9513),
            ("api_server:\n  enabled: true\n  port: 9514\n", 9514),
        ],
    )
    def test_all_startup_yaml_shapes_resolve_api_server_port(
        self, monkeypatch, tmp_path, yaml_text, expected
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "shapes"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml_text, encoding="utf-8")
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="shapes", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-shapes"
        )

        assert target.port == expected

    def test_dotenv_port_overrides_all_yaml_shapes_without_mutating_process_env(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "precedence"
        home.mkdir(parents=True)
        (home / ".env").write_text(
            "API_SERVER_KEY=test-only-key-123456\nAPI_SERVER_PORT=9611\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            "platforms:\n  api_server:\n    port: 9511\n", encoding="utf-8"
        )
        monkeypatch.setenv("API_SERVER_PORT", "7777")
        before = dict(update_cmd.os.environ)
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="precedence", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-precedence"
        )

        assert target.port == 9611
        assert dict(update_cmd.os.environ) == before

    def test_dotenv_port_without_usable_dotenv_key_does_not_override_yaml(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "yaml-key"
        home.mkdir(parents=True)
        (home / ".env").write_text("API_SERVER_PORT=9611\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "api_server:\n  key: config-only-key-123456\n  port: 9513\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="yaml-key", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-yaml-key"
        )

        assert target is not None
        assert target.port == 9513

    def test_dotenv_port_without_key_is_adapter_fallback_when_config_has_no_port(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "env-fallback"
        home.mkdir(parents=True)
        (home / ".env").write_text("API_SERVER_PORT=9611\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "api_server:\n  key: config-only-key-123456\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [
                SimpleNamespace(name="env-fallback", path=home, is_default=False)
            ],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-env-fallback"
        )

        assert target is not None
        assert target.port == 9611

    def test_config_port_placeholder_matches_adapter_default_not_dotenv_expansion(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "literal-port"
        home.mkdir(parents=True)
        (home / ".env").write_text("CUSTOM_PORT=9611\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "api_server:\n  key: config-only-key-123456\n  port: ${CUSTOM_PORT}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [
                SimpleNamespace(name="literal-port", path=home, is_default=False)
            ],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-literal-port"
        )

        assert target is not None
        assert target.port == 8642

    def test_legacy_gateway_json_port_matches_startup(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "legacy"
        home.mkdir(parents=True)
        (home / "gateway.json").write_text(
            '{"platforms":{"api_server":{"extra":{"port":9515}}}}',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="legacy", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-legacy"
        )

        assert target is not None
        assert target.port == 9515

    def test_managed_overlay_port_matches_startup(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "managed"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="managed", path=home, is_default=False)],
        )
        monkeypatch.setattr(
            "hermes_cli.managed_scope.apply_managed_overlay",
            lambda _raw: {"api_server": {"port": 9516}},
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-managed"
        )

        assert target is not None
        assert target.port == 9516

    def test_invalid_dotenv_port_matches_startup_yaml_fallback(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "bad-env"
        home.mkdir(parents=True)
        (home / ".env").write_text("API_SERVER_PORT=not-a-port\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "api_server:\n  port: 9517\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="bad-env", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-bad-env"
        )
        assert target is not None
        assert target.port == 9517

    @pytest.mark.parametrize("env_port", [0, -1, 65536, 70000])
    def test_integer_dotenv_port_matches_startup_without_range_normalization(
        self, monkeypatch, tmp_path, env_port
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "range-env"
        home.mkdir(parents=True)
        (home / ".env").write_text(
            f"API_SERVER_KEY={'x' * 16}\nAPI_SERVER_PORT={env_port}\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            "api_server:\n  port: 9517\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="range-env", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-range-env"
        )
        assert target is not None
        assert target.port == env_port

    @pytest.mark.parametrize("source", ["yaml", "legacy", "managed"])
    @pytest.mark.parametrize(
        ("configured_port", "expected_port"),
        [
            ("not-a-port", 8642),
            (0, 0),
            (-1, -1),
            (65536, 65536),
            (70000, 70000),
            (9413.0, 9413),
            (True, 1),
            (False, 0),
        ],
    )
    def test_explicit_config_port_matches_adapter_coercion(
        self, monkeypatch, tmp_path, source, configured_port, expected_port
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / source
        home.mkdir(parents=True)
        platform = {"enabled": True, "port": configured_port}
        if source == "yaml":
            (home / "config.yaml").write_text(
                yaml.safe_dump({"platforms": {"api_server": platform}}),
                encoding="utf-8",
            )
        elif source == "legacy":
            legacy_platform = {"enabled": True, "extra": {"port": configured_port}}
            (home / "gateway.json").write_text(
                json.dumps({"platforms": {"api_server": legacy_platform}}),
                encoding="utf-8",
            )
        else:
            (home / "config.yaml").write_text("{}\n", encoding="utf-8")
            monkeypatch.setattr(
                "hermes_cli.managed_scope.apply_managed_overlay",
                lambda _raw: {"platforms": {"api_server": platform}},
            )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name=source, path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            f"ai.hermes.gateway-{source}"
        )

        assert target is not None
        assert target.port == expected_port

    def test_platform_merge_preserves_earlier_extra_port(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "merge"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "gateway": {
                        "platforms": {"api_server": {"extra": {"port": 9001}}}
                    },
                    "platforms": {"api_server": {"enabled": True, "port": 9002}},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(name="merge", path=home, is_default=False)],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-merge"
        )

        assert target is not None
        assert target.port == 9001

    @pytest.mark.parametrize(
        ("legacy", "raw", "expected_port"),
        [
            (
                {"platforms": {"api_server": {"extra": {"port": 9100}}}},
                {"platforms": {"api_server": {"port": 9102}}},
                9102,
            ),
            (
                {},
                {
                    "gateway": {"api_server": {"extra": {"port": 9002}}},
                    "platforms": {"api_server": {"port": 9001}},
                },
                9001,
            ),
        ],
    )
    def test_shared_key_bridge_reapplies_selected_nested_direct_port(
        self, monkeypatch, tmp_path, legacy, raw, expected_port
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / f"shared-bridge-{expected_port}"
        home.mkdir(parents=True)
        if legacy:
            (home / "gateway.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
        (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(
                name=f"shared-bridge-{expected_port}", path=home, is_default=False
            )],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            f"ai.hermes.gateway-shared-bridge-{expected_port}"
        )

        assert target is not None
        assert target.port == expected_port

    @pytest.mark.parametrize(
        ("case", "expected_port"),
        [
            ("gateway-platforms", 9101),
            ("platforms", 9102),
            ("top-level", 9201),
            ("legacy", 8642),
        ],
    )
    def test_malformed_extra_matches_loader_fallback(
        self, monkeypatch, tmp_path, case, expected_port
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / f"malformed-{case}"
        home.mkdir(parents=True)
        legacy_port = 9101 if case == "gateway-platforms" else 9102
        legacy_extra = [1] if case == "legacy" else {"port": legacy_port}
        (home / "gateway.json").write_text(
            json.dumps(
                {
                    "platforms": {
                        "api_server": {"enabled": True, "extra": legacy_extra}
                    }
                }
            ),
            encoding="utf-8",
        )
        if case == "gateway-platforms":
            raw = {"gateway": {"platforms": {"api_server": {"extra": [1]}}}}
        elif case == "platforms":
            raw = {"platforms": {"api_server": {"extra": [1]}}}
        elif case == "top-level":
            raw = {"api_server": {"enabled": True, "port": 9201, "extra": [1]}}
        else:
            raw = {"platforms": {"api_server": {"port": 9301}}}
        (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(
                name=f"malformed-{case}", path=home, is_default=False
            )],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            f"ai.hermes.gateway-malformed-{case}"
        )

        assert target is not None
        assert target.port == expected_port

    @pytest.mark.parametrize("legacy_root", [[1], 1, True, "x"])
    def test_malformed_legacy_root_matches_gateway_default(
        self, monkeypatch, tmp_path, legacy_root
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "malformed-legacy-root"
        home.mkdir(parents=True)
        (home / "gateway.json").write_text(json.dumps(legacy_root), encoding="utf-8")
        (home / "config.yaml").write_text(
            yaml.safe_dump({"platforms": {"api_server": {"port": 9300}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(
                name="malformed-legacy-root", path=home, is_default=False
            )],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-malformed-legacy-root"
        )

        assert target is not None
        assert target.port == 8642

    def test_malformed_yaml_falls_back_without_mutating_profile(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "malformed-yaml"
        home.mkdir(parents=True)
        (home / "gateway.json").write_text(
            json.dumps({"platforms": {"api_server": {"extra": {"port": 9400}}}}),
            encoding="utf-8",
        )
        config_path = home / "config.yaml"
        config_path.write_text("platforms: [\n", encoding="utf-8")
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(
                name="malformed-yaml", path=home, is_default=False
            )],
        )

        before = sorted(path.name for path in home.iterdir())
        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-malformed-yaml"
        )
        after = sorted(path.name for path in home.iterdir())

        assert target is not None
        assert target.port == 9400
        assert after == before

    def test_malformed_managed_root_falls_back_to_legacy(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "malformed-managed"
        home.mkdir(parents=True)
        (home / "gateway.json").write_text(
            json.dumps({"platforms": {"api_server": {"extra": {"port": 8123}}}}),
            encoding="utf-8",
        )
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            "hermes_cli.managed_scope.apply_managed_overlay", lambda _raw: [1]
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(
                name="malformed-managed", path=home, is_default=False
            )],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-malformed-managed"
        )

        assert target is not None
        assert target.port == 8123

    def test_late_malformed_platform_merge_retains_earlier_partial_merge(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        home = tmp_path / "profiles" / "partial-merge"
        home.mkdir(parents=True)
        (home / "gateway.json").write_text(
            json.dumps({"platforms": {"api_server": {"extra": {"port": 9100}}}}),
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "gateway": {
                        "platforms": {"api_server": {"extra": {"port": 9200}}}
                    },
                    "platforms": {"api_server": {"extra": [1]}},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [SimpleNamespace(
                name="partial-merge", path=home, is_default=False
            )],
        )

        target = update_cmd._resolve_launchd_gateway_contract_target(
            "ai.hermes.gateway-partial-merge"
        )

        assert target is not None
        assert target.port == 9200


def test_missing_checkout_sha_fails_before_contract_can_signal(monkeypatch, tmp_path):
    target = update_cmd._LaunchdGatewayContractTarget(
        "ai.hermes.gateway", tmp_path, 8642
    )
    calls = []
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.gateway_restart_contract.perform_verified_graceful_restart",
        lambda *_args, **_kwargs: calls.append("contract"),
    )

    result = update_cmd._verified_graceful_restart_launchd_target(target)

    assert result == RestartProbe(False, "checkout-code-sha-unavailable", None)
    assert calls == []


def _fleet(monkeypatch, tmp_path, *, current, labels, located,
           registered=None, plist_exists=True,
           drain_results=None, kick_errors=None, wait_results=None,
           current_supervised=True):
    """Wire a fake launchd fleet through hermes_cli.gateway seams.

    ``located`` maps label -> (domain, pid) as ``_locate_launchd_gateway_service``
    would return it (values may also be exceptions to raise). ``registered``
    maps label -> bool for the current-profile ``launchctl list`` gate and
    defaults to "located in some domain". Returns a SimpleNamespace of
    recorder lists: rec.kickstarts, rec.drains, rec.current_restarts, rec.waits, locates,
    registered_checks.
    """
    from types import SimpleNamespace

    rec = SimpleNamespace(
        kickstarts=[], drains=[], contracts=[], current_restarts=[], waits=[],
        locates=[], registered_checks=[], current_verifies=[],
    )

    plist = tmp_path / f"{current}.plist"
    if plist_exists:
        plist.write_text("<plist/>")

    def fake_locate(label):
        rec.locates.append(label)
        value = located[label]
        if isinstance(value, Exception):
            raise value
        return value

    def fake_registered(label):
        rec.registered_checks.append(label)
        if registered is not None:
            return registered[label]
        value = located.get(label)
        return (
            value is not None
            and not isinstance(value, Exception)
            and value[0] is not None
        )

    monkeypatch.setattr(gw, "get_launchd_label", lambda: current)
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gw, "launchd_gateway_labels_for_install", lambda: list(labels))
    monkeypatch.setattr(gw, "_locate_launchd_gateway_service", fake_locate)
    monkeypatch.setattr(gw, "_launchd_service_registered", fake_registered)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("force-capable launchd fallback was called")

    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", forbidden)
    monkeypatch.setattr(gw, "_launchd_kickstart", forbidden)

    def fake_wait(label, old_pid, timeout, domain, expected_pid=None):
        rec.waits.append((f"{domain}/{label}", expected_pid))
        return (wait_results or {}).get(label, True)

    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", fake_wait)

    def fake_current_restart(*, supervision_verify=True):
        if not plist_exists:
            return [], []
        rec.current_restarts.append(current)
        if current_supervised:
            rec.current_verifies.append(current)
            return [current], []
        return [], [current]

    monkeypatch.setattr(
        update_cmd, "_restart_launchd_gateway_after_update", fake_current_restart
    )

    def target_for(label):
        home = tmp_path / "homes" / label
        home.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(label=label, home=home, port=8642)

    monkeypatch.setattr(
        update_cmd, "_resolve_launchd_gateway_contract_target", target_for
    )

    pid_by_label = {
        label: value[1]
        for label, value in located.items()
        if not isinstance(value, Exception) and value[1] is not None
    }

    def verified_restart(target, **_kw):
        rec.contracts.append(target.label)
        error = (kick_errors or {}).get(target.label)
        if isinstance(error, subprocess.TimeoutExpired):
            raise error
        old_pid = pid_by_label.get(target.label, 1)
        home = target.home.resolve()
        if error is not None:
            return RestartProbe(False, f"contract-failed:{type(error).__name__}", None)
        return RestartProbe(
            True,
            "replacement-healthy",
            ServingIdentity(old_pid + 10000, old_pid + 1, home, "new"),
        )

    monkeypatch.setattr(
        update_cmd, "_verified_graceful_restart_launchd_target", verified_restart
    )
    return rec


class TestRestartMacosLaunchdGateways:
    def test_current_and_siblings_use_verified_contracts_in_own_domains(
        self, monkeypatch, tmp_path
    ):
        """Each running sibling is contract-restarted, then supervision-checked."""
        current = "ai.hermes.gateway-merit-ops"
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current=current,
            labels=["ai.hermes.gateway", current, "ai.hermes.gateway-user-scoped"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                current: (f"gui/{UID}", 200),
                "ai.hermes.gateway-user-scoped": (f"user/{UID}", 300),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.current_restarts == [current]
        assert rec.kickstarts == []
        assert rec.contracts == [
            "ai.hermes.gateway",
            "ai.hermes.gateway-user-scoped",
        ]
        assert rec.waits == [
            (f"gui/{UID}/ai.hermes.gateway", 10100),
            (f"user/{UID}/ai.hermes.gateway-user-scoped", 10300),
        ]
        assert restarted == [
            current,
            "ai.hermes.gateway",
            "ai.hermes.gateway-user-scoped",
        ]
        assert failed == []
        assert rec.drains == []

    def test_current_profile_without_plist_makes_no_launchctl_calls(
        self, monkeypatch, tmp_path
    ):
        """Upstream gate order preserved: no plist → the current profile is
        skipped without ANY launchctl interaction (no registered probe, no
        locate) — and definitely without inventing a failure. Siblings are
        still processed."""
        current = "ai.hermes.gateway"
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current=current,
            labels=[current, "ai.hermes.gateway-a"],
            located={"ai.hermes.gateway-a": (f"gui/{UID}", 200)},
            plist_exists=False,
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.current_restarts == []
        assert current not in rec.registered_checks
        assert current not in rec.locates
        assert restarted == ["ai.hermes.gateway-a"]
        assert failed == []

    def test_current_profile_registered_but_unlocatable_still_restarts(
        self, monkeypatch, tmp_path
    ):
        """macOS-26 quirk: a label can be `launchctl list`-registered while
        both explicit gui/user `launchctl print` probes fail (domain doesn't
        support service management). The gate must use the registered
        predicate and hand off to launchd_restart(), which owns the
        domain-unsupported fallback — locate is for siblings only."""
        current = "ai.hermes.gateway"
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current=current,
            labels=[current],
            located={current: (None, None)},
            registered={current: True},
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.current_restarts == [current]
        assert current not in rec.locates
        assert restarted == [current]
        assert failed == []

    def test_unbootstrapped_sibling_is_skipped_not_failed(
        self, monkeypatch, tmp_path
    ):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-idle"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-idle": (None, None),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.kickstarts == []
        assert restarted == ["ai.hermes.gateway"]
        assert failed == []

    def test_loaded_but_not_running_sibling_is_left_stopped(
        self, monkeypatch, tmp_path
    ):
        """Updater does not start an intentionally stopped sibling."""
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-dormant"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-dormant": (f"gui/{UID}", None),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.drains == []
        assert rec.kickstarts == []
        assert rec.contracts == []
        assert restarted == ["ai.hermes.gateway"]
        assert failed == []

    def test_verified_contract_replacement_skips_kickstart(
        self, monkeypatch, tmp_path
    ):
        """A verified replacement is supervision-checked without a hard restart."""
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-a"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-a": (f"gui/{UID}", 200),
            },
            drain_results={200: True},
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.drains == []
        assert rec.kickstarts == []
        assert rec.contracts == ["ai.hermes.gateway-a"]
        assert rec.waits == [(f"gui/{UID}/ai.hermes.gateway-a", 10200)]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-a"]
        assert failed == []

    def test_contract_failure_is_recorded_and_rest_continue(
        self, monkeypatch, tmp_path
    ):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=[
                "ai.hermes.gateway",
                "ai.hermes.gateway-bad",
                "ai.hermes.gateway-good",
            ],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-bad": (f"gui/{UID}", 200),
                "ai.hermes.gateway-good": (f"gui/{UID}", 300),
            },
            kick_errors={
                "ai.hermes.gateway-bad": subprocess.CalledProcessError(
                    5, ["launchctl", "kickstart"]
                )
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert failed == ["ai.hermes.gateway-bad"]
        assert rec.kickstarts == []
        assert rec.contracts == [
            "ai.hermes.gateway-bad",
            "ai.hermes.gateway-good",
        ]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-good"]

    def test_timeout_during_discovery_is_failed_and_rest_continue(
        self, monkeypatch, tmp_path
    ):
        """A wedged launchctl during liveness discovery must be accounted as
        a failure (the sibling may still be on old code), not silently
        skipped — and must not abort the remaining fleet (#68523 parity)."""
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=[
                "ai.hermes.gateway",
                "ai.hermes.gateway-wedged",
                "ai.hermes.gateway-after",
            ],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-wedged": subprocess.TimeoutExpired(
                    cmd=["launchctl", "print"], timeout=5
                ),
                "ai.hermes.gateway-after": (f"gui/{UID}", 300),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert failed == ["ai.hermes.gateway-wedged"]
        assert rec.kickstarts == []
        assert rec.contracts == ["ai.hermes.gateway-after"]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-after"]

    def test_timeout_during_contract_is_failed_and_rest_continue(
        self, monkeypatch, tmp_path
    ):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=[
                "ai.hermes.gateway",
                "ai.hermes.gateway-wedged",
                "ai.hermes.gateway-after",
            ],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-wedged": (f"gui/{UID}", 200),
                "ai.hermes.gateway-after": (f"gui/{UID}", 300),
            },
            kick_errors={
                "ai.hermes.gateway-wedged": subprocess.TimeoutExpired(
                    cmd=["launchctl", "kickstart"], timeout=90
                )
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert failed == ["ai.hermes.gateway-wedged"]
        assert rec.kickstarts == []
        assert rec.contracts == [
            "ai.hermes.gateway-wedged",
            "ai.hermes.gateway-after",
        ]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-after"]

    def test_sibling_that_never_comes_back_is_failed(self, monkeypatch, tmp_path):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-zombie"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-zombie": (f"gui/{UID}", 200),
            },
            wait_results={"ai.hermes.gateway-zombie": False},
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert restarted == ["ai.hermes.gateway"]
        assert failed == ["ai.hermes.gateway-zombie"]


class TestWaitForLaunchdServicePid:
    def test_returns_true_once_pid_changes(self, monkeypatch):
        pids = iter([200, 200, 4242])
        monkeypatch.setattr(
            gw,
            "_launchd_print_service_pid",
            lambda domain, label: (True, next(pids)),
        )
        monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
        assert gw._wait_for_launchd_service_pid(
            "ai.hermes.gateway-x", old_pid=200, timeout=5.0, domain=f"gui/{UID}"
        )

    def test_returns_false_when_pid_never_changes(self, monkeypatch):
        clock = iter(float(i) for i in range(100))
        monkeypatch.setattr(gw.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            gw,
            "_launchd_print_service_pid",
            lambda domain, label: (True, 200),
        )
        assert not gw._wait_for_launchd_service_pid(
            "ai.hermes.gateway-x", old_pid=200, timeout=3.0, domain=f"gui/{UID}"
        )

    def test_rejects_fresh_but_wrong_pid_when_verified_pid_is_required(
        self, monkeypatch
    ):
        clock = iter(float(i) for i in range(100))
        monkeypatch.setattr(gw.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            gw,
            "_launchd_print_service_pid",
            lambda domain, label: (True, 4242),
        )

        assert not gw._wait_for_launchd_service_pid(
            "ai.hermes.gateway-x",
            old_pid=200,
            expected_pid=4300,
            timeout=3.0,
            domain=f"gui/{UID}",
        )


class TestIncompleteWarningMentionsLaunchctl:
    def test_launchd_labels_get_safe_retry_hint(self, monkeypatch, capsys):
        monkeypatch.setattr(gw, "is_macos", lambda: True)
        _warn_incomplete_gateway_fleet_restart(["ai.hermes.gateway-merit-ops"])
        out = capsys.readouterr().out
        assert "Update incomplete" in out
        assert "launchctl kickstart -k" not in out
        assert "retry" in out.lower()

    def test_systemd_units_keep_systemctl_hint(self, monkeypatch, capsys):
        monkeypatch.setattr(gw, "is_macos", lambda: False)
        _warn_incomplete_gateway_fleet_restart(["hermes-gateway-coder"])
        out = capsys.readouterr().out
        assert "systemctl" in out
        assert "launchctl" not in out
