from pathlib import Path
import subprocess
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd

pytestmark = [
    pytest.mark.update_orchestration,
    pytest.mark.usefixtures("isolated_update_orchestrator"),
]


# ---------------------------------------------------------------------------
# Managed-uv compatibility for tests that patch shutil.which
# ---------------------------------------------------------------------------
# The production code now uses ``ensure_uv()`` / ``update_managed_uv()``
# instead of ``shutil.which("uv")``.  Many tests in this file patch
# ``shutil.which`` to control whether uv is "available" — these autouse
# fixtures make the managed_uv functions delegate to the patched
# ``shutil.which`` so the existing test setup keeps working without
# per-test changes.
@pytest.fixture(autouse=True)
def _patch_managed_uv(request):
    """Make managed_uv helpers follow shutil.which mocking in tests."""
    import shutil

    # resolve_uv delegates to shutil.which("uv") so that test patches
    # on shutil.which flow through naturally.
    def _fake_resolve_uv(**kwargs):
        return shutil.which("uv")

    def _fake_ensure_uv(**kwargs):
        return shutil.which("uv")

    def _fake_update_managed_uv(**kwargs):
        return None  # never actually self-update in tests

    with patch("hermes_cli.managed_uv.resolve_uv", side_effect=_fake_resolve_uv), \
         patch("hermes_cli.managed_uv.ensure_uv", side_effect=_fake_ensure_uv), \
         patch("hermes_cli.managed_uv.update_managed_uv", side_effect=_fake_update_managed_uv):
        yield


@pytest.fixture(autouse=True)
def _patch_gateway_discovery():
    """Keep cmd_update's gateway auto-restart phase off this machine's gateways.

    Tests in this file that reach the full success path (e.g. the #87694
    orphan-history rescue-ref tests) would otherwise hit real gateway
    discovery: an unmocked ``find_gateway_pids`` on a box with a live gateway
    reaches the conftest live-system guard and turns into a spurious
    ``sys.exit(1)`` (#78574). Discovery returning nothing makes the phase a
    clean no-op — none of the tests here assert on gateway restarts.

    ``_purge_stale_hermes_modules`` must also be stubbed: it evicts
    ``hermes_cli.gateway`` from ``sys.modules`` mid-update, and the restart
    phase's fresh ``from hermes_cli.gateway import ...`` then loads an
    UNPATCHED copy of the module — silently discarding every mock here and
    letting real gateway discovery (and real ``os.kill``) run on the dev box.
    """
    with patch("hermes_cli.gateway.find_gateway_pids", return_value=[]), \
         patch("hermes_cli.gateway.supports_systemd_services", return_value=False), \
         patch("hermes_cli.gateway.find_profile_gateway_processes", return_value=[]), \
         patch("hermes_cli.update_inventory.collect_runtime_inventory", return_value=None), \
         patch("hermes_cli.update_inventory.report_unaccounted_runtimes", return_value=False), \
         patch.object(hermes_main, "_fleet_probe_expected_runtimes", lambda *a, **kw: False), \
         patch.object(hermes_main, "_purge_stale_hermes_modules", lambda *a, **kw: None), \
         patch("hermes_cli.update_receipt.collect_fleet_versions", return_value=[]):
        yield















# ---------------------------------------------------------------------------
# Update uses .[all] with fallback to .
# ---------------------------------------------------------------------------

def _setup_update_mocks(monkeypatch, tmp_path):
    """Common setup for cmd_update tests."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", lambda *a, **kw: True)
    monkeypatch.setattr(hermes_config, "get_missing_env_vars", lambda required_only=True: [])
    monkeypatch.setattr(hermes_config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(hermes_config, "check_config_version", lambda **_kwargs: (5, 5))
    monkeypatch.setattr(hermes_config, "migrate_config", lambda **kw: {"env_added": [], "config_added": []})
    monkeypatch.setattr(hermes_main, "_upgrade_pip_before_lazy_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_refresh_active_lazy_features", lambda *a, **kw: True)


def test_drop_owned_stash_revalidates_selector_before_mutation(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-3:] == ["stash", "list", "--format=%gd %H"]:
            return subprocess.CompletedProcess(command, 0, "stash@{0} owned\n", "")
        if command[-2:] == ["rev-parse", "stash@{0}"]:
            return subprocess.CompletedProcess(command, 0, "replacement\n", "")
        raise AssertionError(f"unsafe mutation attempted: {command}")

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    assert update_cmd._drop_owned_stash(["git"], tmp_path, "owned") is False
    assert not any(command[-2:] == ["drop", "stash@{0}"] for command in calls)


def test_drop_owned_stash_fails_closed_when_refs_stash_changes_before_mutation(
    monkeypatch, tmp_path
):
    """A concurrent stash must never be deleted through selector drift."""
    real_run = subprocess.run

    def git(*args):
        return real_run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Release Gate Test")
    git("config", "user.email", "release-gate@example.invalid")
    (tmp_path / "tracked.txt").write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-qm", "base")
    (tmp_path / "tracked.txt").write_text("owned\n", encoding="utf-8")
    git("stash", "push", "-qm", "owned")
    owned = git("rev-parse", "refs/stash")
    injected = False

    def race_before_mutation(command, **kwargs):
        nonlocal injected
        is_drop = command[-2:-1] == ["drop"]
        is_ref_delete = command[-4:-3] == ["update-ref"] and "-d" in command
        if not injected and (is_drop or is_ref_delete):
            injected = True
            (tmp_path / "tracked.txt").write_text("foreign\n", encoding="utf-8")
            git("stash", "push", "-qm", "foreign")
        return real_run(command, **kwargs)

    monkeypatch.setattr(update_cmd.subprocess, "run", race_before_mutation)

    assert update_cmd._drop_owned_stash(["git"], tmp_path, owned) is False
    remaining = git("stash", "list", "--format=%H")
    assert owned in remaining
    assert len(remaining.splitlines()) == 2


def test_restore_guidance_never_emits_mutable_selector_after_concurrent_push(
    monkeypatch, tmp_path, capsys
):
    """CAS failure guidance must identify our stash only by immutable OID."""
    real_run = subprocess.run

    def git(*args):
        return real_run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Updater Test")
    git("config", "user.email", "updater@example.invalid")
    (tmp_path / "tracked.txt").write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-qm", "base")
    (tmp_path / "tracked.txt").write_text("owned\n", encoding="utf-8")
    git("stash", "push", "-qm", "owned")
    owned = git("rev-parse", "refs/stash")
    injected = False

    def push_before_cas(command, **kwargs):
        nonlocal injected
        if not injected and command[-4:-3] == ["update-ref"] and "-d" in command:
            injected = True
            (tmp_path / "foreign.txt").write_text("foreign\n", encoding="utf-8")
            git("stash", "push", "-uqm", "foreign")
        return real_run(command, **kwargs)

    monkeypatch.setattr(update_cmd.subprocess, "run", push_before_cas)

    assert update_cmd._restore_stashed_changes(["git"], tmp_path, owned) is True

    output = capsys.readouterr().out
    assert owned in output
    assert "stash@{" not in output
    assert "git stash drop" not in output
    assert git("rev-parse", "refs/stash") != owned
    assert owned in git("stash", "list", "--format=%H").splitlines()




def test_refresh_active_memory_provider_dependencies_reinstalls_active_provider(monkeypatch):
    """#53272/#70636: update must re-run the active provider's dep install."""
    recorded = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mem0"}},
    )
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda provider_name, force=False: recorded.append((provider_name, force)),
    )

    hermes_main._refresh_active_memory_provider_dependencies()

    assert recorded == [("mem0", True)]




def test_reload_updated_runtime_modules_restores_new_hermes_constants_symbol(monkeypatch):
    """A pre-pull module object missing a new helper is repaired by reload."""
    import hermes_constants

    monkeypatch.delattr(hermes_constants, "apply_subprocess_home_env", raising=False)
    assert not hasattr(hermes_constants, "apply_subprocess_home_env")

    hermes_main._reload_updated_runtime_modules()

    assert callable(hermes_constants.apply_subprocess_home_env)






# ---------------------------------------------------------------------------
# ff-only fallback to reset --hard on diverged history
# ---------------------------------------------------------------------------

def _make_update_side_effect(
    current_branch="main",
    commit_count="3",
    ff_only_fails=False,
    reset_fails=False,
    fetch_fails=False,
    fetch_stderr="",
    local_only_count="0",
    merge_base_exists=True,
    update_ref_fails=False,
    pre_pull_sha_unavailable=False,
    existing_rescue_refs=None,
):
    """Build a subprocess.run side_effect for cmd_update tests.

    ``merge_base_exists`` controls the ``git merge-base HEAD origin/<branch>``
    probe used by the ff-only-fallback orphan-history guard (#87694): True
    (default) simulates ordinary divergence (a common ancestor exists, e.g.
    upstream force-push), False simulates orphan/unrelated-history divergence
    (no common ancestor at all).

    ``update_ref_fails`` simulates ``git update-ref`` itself failing (disk
    full, permissions) when writing the orphan rescue ref.

    ``pre_pull_sha_unavailable`` simulates ``_capture_head_sha`` being unable
    to resolve HEAD before the pull (empty rev-parse output) — the rescue-ref
    guard requires a truthy ``pre_pull_sha`` and must degrade gracefully
    without one.

    ``existing_rescue_refs`` simulates the refs already present under
    ``refs/hermes-update-backups/orphan-<branch>-*`` (oldest first) so the
    ``_prune_orphan_rescue_refs`` cleanup pass has something to trim.
    """
    recorded = []
    head_sha_calls = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)
        if "fetch" in joined and "origin" in joined:
            if fetch_fails:
                return SimpleNamespace(stdout="", stderr=fetch_stderr, returncode=128)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-parse" in joined and "^{commit}" in joined:
            return SimpleNamespace(stdout=f"{'f' * 40}\n", stderr="", returncode=0)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(stdout=f"{current_branch}\n", stderr="", returncode=0)
        if "show-current" in joined:
            return SimpleNamespace(stdout=f"{current_branch}\n", stderr="", returncode=0)
        if "rev-parse" in joined and "HEAD" in joined:
            # First call = pre-pull HEAD, every later call = post-pull HEAD
            # (issue #79678's "did HEAD actually move" guard depends on these
            # differing after a successful reset/merge).
            head_sha_calls.append(1)
            if len(head_sha_calls) == 1:
                if pre_pull_sha_unavailable:
                    return SimpleNamespace(stdout="", stderr="", returncode=0)
                return SimpleNamespace(
                    stdout="1111111111111111111111111111111111111beef\n", stderr="", returncode=0
                )
            return SimpleNamespace(
                stdout="2222222222222222222222222222222222222cafe\n", stderr="", returncode=0
            )
        if "checkout" in joined and "main" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-list" in joined and "..HEAD" in joined:
            return SimpleNamespace(
                stdout=f"{local_only_count}\n", stderr="", returncode=0
            )
        if "rev-list" in joined:
            return SimpleNamespace(stdout=f"{commit_count}\n", stderr="", returncode=0)
        if "merge-base" in joined:
            if merge_base_exists:
                return SimpleNamespace(stdout="abc123deadbeef\n", stderr="", returncode=0)
            return SimpleNamespace(
                stdout="", stderr="fatal: Not a valid commit name origin/main\n", returncode=1
            )
        if "for-each-ref" in joined:
            refs = existing_rescue_refs or []
            return SimpleNamespace(stdout="\n".join(refs) + ("\n" if refs else ""), stderr="", returncode=0)
        if "update-ref" in joined and "-d" in cmd:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "update-ref" in joined:
            if update_ref_fails:
                return SimpleNamespace(
                    stdout="", stderr="fatal: unable to write ref\n", returncode=128
                )
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "--ff-only" in joined:
            if ff_only_fails:
                return SimpleNamespace(
                    stdout="",
                    stderr="fatal: Not possible to fast-forward, aborting.\n",
                    returncode=128,
                )
            return SimpleNamespace(stdout="Updating abc..def\n", stderr="", returncode=0)
        if "reset" in joined and "--hard" in joined:
            if reset_fails:
                return SimpleNamespace(stdout="", stderr="error: unable to write\n", returncode=1)
            return SimpleNamespace(stdout="HEAD is now at abc123\n", stderr="", returncode=0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect, recorded


# ---------------------------------------------------------------------------
# Same-branch local commits must survive divergence fallback
# ---------------------------------------------------------------------------


def test_same_branch_divergence_detects_local_only_commits(monkeypatch, tmp_path):
    """A positive reachability count authorizes merge-over-reset."""
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="1\n",
            stderr="",
            returncode=0,
        ),
    )

    assert update_cmd._same_branch_has_local_only_commits(
        ["git"], tmp_path, "origin/main"
    ) is True


def test_same_branch_divergence_distinguishes_remote_rewrite(monkeypatch, tmp_path):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0\n",
            stderr="",
            returncode=0,
        ),
    )

    assert update_cmd._same_branch_has_local_only_commits(
        ["git"], tmp_path, "origin/main"
    ) is False


def test_same_branch_divergence_fails_closed_when_reachability_is_unverifiable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="",
            stderr="fatal: shallow history",
            returncode=128,
        ),
    )

    assert update_cmd._same_branch_has_local_only_commits(
        ["git"], tmp_path, "origin/main"
    ) is None


def test_merge_abort_verification_requires_restored_head_and_clean_index(
    monkeypatch, tmp_path
):
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="before-sha\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **kw: next(responses))

    ok, detail = update_cmd._abort_merge_and_verify(
        ["git"], tmp_path, "before-sha"
    )

    assert ok is True
    assert detail == ""


@pytest.mark.parametrize(
    ("responses", "expected_detail"),
    [
        (
            [SimpleNamespace(returncode=1, stdout="", stderr="abort failed")],
            "merge_abort_failed",
        ),
        (
            [
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout="different-sha\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
            ],
            "head_not_restored",
        ),
        (
            [
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout="before-sha\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="conflicted.py\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
            ],
            "unmerged_entries_remain",
        ),
        (
            [
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout="before-sha\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout=" M tracked.py\n", stderr=""),
            ],
            "tracked_changes_remain",
        ),
    ],
)
def test_merge_abort_verification_fails_closed(
    monkeypatch, tmp_path, responses, expected_detail
):
    queued = iter(responses)
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **kw: next(queued))

    ok, detail = update_cmd._abort_merge_and_verify(
        ["git"], tmp_path, "before-sha"
    )

    assert ok is False
    assert detail == expected_detail


def test_same_branch_local_commit_survives_real_git_merge(tmp_path):
    """Real Git oracle: local fix + remote update both survive the fallback."""
    import subprocess

    origin = tmp_path / "origin.git"
    local = tmp_path / "local"
    peer = tmp_path / "peer"

    def run(*args, cwd=None):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )

    run("init", "-q", "--bare", str(origin))
    run("clone", "-q", str(origin), str(local))
    run("config", "user.email", "test@example.com", cwd=local)
    run("config", "user.name", "Test", cwd=local)
    run("checkout", "-q", "-b", "main", cwd=local)
    (local / "base.txt").write_text("base\n")
    run("add", "base.txt", cwd=local)
    run("commit", "-qm", "base", cwd=local)
    run("push", "-q", "-u", "origin", "main", cwd=local)

    (local / "desktop-fix.txt").write_text("preserved\n")
    run("add", "desktop-fix.txt", cwd=local)
    run("commit", "-qm", "local desktop fix", cwd=local)

    run("clone", "-q", "--branch", "main", str(origin), str(peer))
    run("config", "user.email", "test@example.com", cwd=peer)
    run("config", "user.name", "Test", cwd=peer)
    (peer / "upstream.txt").write_text("update\n")
    run("add", "upstream.txt", cwd=peer)
    run("commit", "-qm", "upstream update", cwd=peer)
    run("push", "-q", cwd=peer)

    run("fetch", "-q", "origin", cwd=local)
    assert update_cmd._same_branch_has_local_only_commits(
        ["git"], local, "origin/main"
    ) is True
    run("merge", "--no-edit", "origin/main", cwd=local)

    assert (local / "desktop-fix.txt").read_text() == "preserved\n"
    assert (local / "upstream.txt").read_text() == "update\n"


def test_merge_abort_verification_real_git_conflict(tmp_path):
    """A real conflicted merge must restore HEAD, index, and the local tree."""
    import subprocess

    origin = tmp_path / "origin.git"
    local = tmp_path / "local"
    peer = tmp_path / "peer"

    def run(*args, cwd=None, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )

    run("init", "-q", "--bare", str(origin))
    run("clone", "-q", str(origin), str(local))
    run("config", "user.email", "test@example.com", cwd=local)
    run("config", "user.name", "Test", cwd=local)
    run("checkout", "-q", "-b", "main", cwd=local)
    (local / "conflict.txt").write_text("base\n")
    run("add", "conflict.txt", cwd=local)
    run("commit", "-qm", "base", cwd=local)
    run("push", "-q", "-u", "origin", "main", cwd=local)

    run("clone", "-q", "--branch", "main", str(origin), str(peer))
    run("config", "user.email", "test@example.com", cwd=peer)
    run("config", "user.name", "Test", cwd=peer)
    (peer / "conflict.txt").write_text("remote\n")
    run("add", "conflict.txt", cwd=peer)
    run("commit", "-qm", "remote change", cwd=peer)
    run("push", "-q", cwd=peer)

    (local / "conflict.txt").write_text("local\n")
    run("add", "conflict.txt", cwd=local)
    run("commit", "-qm", "local change", cwd=local)
    pre_merge = run("rev-parse", "HEAD", cwd=local).stdout.strip()
    run("fetch", "-q", "origin", cwd=local)
    conflict = run("merge", "--no-edit", "origin/main", cwd=local, check=False)
    assert conflict.returncode != 0
    assert run(
        "diff", "--name-only", "--diff-filter=U", cwd=local
    ).stdout.strip() == "conflict.txt"

    restored, detail = update_cmd._abort_merge_and_verify(
        ["git"], local, pre_merge
    )

    assert restored is True
    assert detail == ""
    assert run("rev-parse", "HEAD", cwd=local).stdout.strip() == pre_merge
    assert run(
        "status", "--porcelain", "--untracked-files=no", cwd=local
    ).stdout == ""
    assert (local / "conflict.txt").read_text() == "local\n"


def test_same_branch_local_merge_commit_is_not_invisible(tmp_path):
    """Regression: `git cherry` omits merges; reachability must preserve them."""
    import subprocess

    origin = tmp_path / "origin.git"
    local = tmp_path / "local"
    peer = tmp_path / "peer"

    def run(*args, cwd=None):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )

    run("init", "-q", "--bare", str(origin))
    run("clone", "-q", str(origin), str(local))
    run("config", "user.email", "test@example.com", cwd=local)
    run("config", "user.name", "Test", cwd=local)
    run("checkout", "-q", "-b", "main", cwd=local)
    (local / "base.txt").write_text("base\n")
    run("add", "base.txt", cwd=local)
    run("commit", "-qm", "base", cwd=local)
    base = run("rev-parse", "HEAD", cwd=local).stdout.strip()
    (local / "remote-parent.txt").write_text("parent\n")
    run("add", "remote-parent.txt", cwd=local)
    run("commit", "-qm", "remote parent", cwd=local)
    remote_parent = run("rev-parse", "HEAD", cwd=local).stdout.strip()
    run("push", "-q", "-u", "origin", "main", cwd=local)

    # Build a local-only merge commit whose parents are both already reachable
    # from origin/main. Git cherry prints nothing for this commit even though its
    # tree contains a unique Desktop fix.
    (local / "desktop-merge-fix.txt").write_text("must survive\n")
    run("add", "desktop-merge-fix.txt", cwd=local)
    tree = run("write-tree", cwd=local).stdout.strip()
    merge_commit = run(
        "commit-tree",
        tree,
        "-p",
        remote_parent,
        "-p",
        base,
        "-m",
        "local merge fix",
        cwd=local,
    ).stdout.strip()
    run("update-ref", "refs/heads/main", merge_commit, cwd=local)
    assert run("cherry", "origin/main", "HEAD", cwd=local).stdout == ""

    run("clone", "-q", "--branch", "main", str(origin), str(peer))
    run("config", "user.email", "test@example.com", cwd=peer)
    run("config", "user.name", "Test", cwd=peer)
    (peer / "upstream-next.txt").write_text("next\n")
    run("add", "upstream-next.txt", cwd=peer)
    run("commit", "-qm", "upstream next", cwd=peer)
    run("push", "-q", cwd=peer)
    run("fetch", "-q", "origin", cwd=local)

    assert run(
        "rev-list", "--count", "origin/main..HEAD", cwd=local
    ).stdout.strip() == "1"
    assert update_cmd._same_branch_has_local_only_commits(
        ["git"], local, "origin/main"
    ) is True


# ---------------------------------------------------------------------------
# Non-main branch → auto-checkout main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fetch failure — friendly error messages
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reset --hard failure — don't attempt stash restore
# ---------------------------------------------------------------------------

def test_cmd_update_skips_stash_restore_when_reset_fails(monkeypatch, tmp_path, capsys):
    """When reset --hard fails, stash restore is skipped with a helpful message."""
    _setup_update_mocks(monkeypatch, tmp_path)
    # Re-enable stash so it actually returns a ref
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )

    side_effect, _ = _make_update_side_effect(ff_only_fails=True, reset_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    # Stash restore should NOT have been called
    assert len(restore_calls) == 0

    out = capsys.readouterr().out
    assert "preserved in stash" in out


# ---------------------------------------------------------------------------
# #87694: orphan/unrelated-history divergence must be backed up to a rescue
# ref before `reset --hard` discards it (ordinary divergence is unaffected).
# ---------------------------------------------------------------------------

def test_cmd_update_orphan_history_backs_up_before_reset(monkeypatch, tmp_path, capsys):
    """No common ancestor with origin/<branch> → HEAD is parked behind a
    ``refs/hermes-update-backups/orphan-*`` ref before the reset proceeds."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert len(update_ref_calls) == 1
    ref_name = update_ref_calls[0][update_ref_calls[0].index("update-ref") + 1]
    assert ref_name.startswith("refs/hermes-update-backups/orphan-main-")
    assert update_ref_calls[0][update_ref_calls[0].index("update-ref") + 2] == (
        "1111111111111111111111111111111111111beef"
    )
    # Ref name carries the pre-pull SHA, not just a second-resolution
    # timestamp, so two updates racing within the same second don't collide.
    assert ref_name.endswith("-111111111111")

    out = capsys.readouterr().out
    assert "orphan divergence" in out
    assert ref_name in out
    # The user is told the backup is temporary and when it expires.
    assert f"expires after {hermes_main._ORPHAN_RESCUE_REF_MAX_AGE_DAYS} days" in out


def test_cmd_update_orphan_rescue_ref_write_failure_message_is_honest(monkeypatch, tmp_path, capsys):
    """When ``git update-ref`` fails, the printed message must not claim a
    backup exists — it should say the write was attempted and failed."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, update_ref_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    out = capsys.readouterr().out
    assert "orphan divergence" in out
    assert "backup write failed" in out
    assert "backed up current HEAD" not in out


def test_cmd_update_orphan_rescue_refs_pruned_beyond_keep_limit(monkeypatch, tmp_path, capsys):
    """Recent orphan rescue refs beyond the retention count are deleted so a
    repeatedly corrupted install doesn't pin unbounded objects against gc."""
    from datetime import datetime, timedelta, timezone

    _setup_update_mocks(monkeypatch, tmp_path)

    # All refs are recent (within the age window) so only the count cap
    # applies — the age-expiry path is exercised separately below.
    now = datetime.now(timezone.utc)
    total = hermes_main._ORPHAN_RESCUE_REFS_TO_KEEP + 2
    stale_refs = [
        "refs/hermes-update-backups/orphan-main-"
        f"{(now - timedelta(hours=total - i)).strftime('%Y%m%d-%H%M%S')}-abc"
        for i in range(total)
    ]
    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, existing_rescue_refs=stale_refs,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    delete_calls = [
        c for c in recorded
        if "update-ref" in " ".join(str(x) for x in c) and "-d" in c
    ]
    assert len(delete_calls) == total - hermes_main._ORPHAN_RESCUE_REFS_TO_KEEP
    deleted_refs = {c[c.index("-d") + 1] for c in delete_calls}
    assert deleted_refs == set(stale_refs[: total - hermes_main._ORPHAN_RESCUE_REFS_TO_KEEP])


def test_cmd_update_orphan_rescue_refs_expired_by_age(monkeypatch, tmp_path, capsys):
    """Rescue refs older than the max-age window are deleted even when the
    count cap alone would have kept them — the age expiry is what bounds a
    multi-GB snapshot's lifetime for a user who never runs another orphan
    incident past the count cap."""
    from datetime import datetime, timedelta, timezone

    _setup_update_mocks(monkeypatch, tmp_path)

    now = datetime.now(timezone.utc)
    old = now - timedelta(days=hermes_main._ORPHAN_RESCUE_REF_MAX_AGE_DAYS + 5)
    fresh = now - timedelta(days=1)
    expired_ref = (
        "refs/hermes-update-backups/orphan-main-"
        f"{old.strftime('%Y%m%d-%H%M%S')}-old1"
    )
    fresh_ref = (
        "refs/hermes-update-backups/orphan-main-"
        f"{fresh.strftime('%Y%m%d-%H%M%S')}-new1"
    )
    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False,
        existing_rescue_refs=[expired_ref, fresh_ref],
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    delete_calls = [
        c for c in recorded
        if "update-ref" in " ".join(str(x) for x in c) and "-d" in c
    ]
    deleted_refs = {c[c.index("-d") + 1] for c in delete_calls}
    assert deleted_refs == {expired_ref}


def test_prune_orphan_rescue_refs_leaves_unparseable_names_alone():
    """A ref whose timestamp segment doesn't parse must never be age-deleted
    (it can still fall to the count cap, but not to a guessed age)."""
    from types import SimpleNamespace as NS
    from unittest.mock import patch as mock_patch

    weird = "refs/hermes-update-backups/orphan-main-not-a-timestamp-xyz"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "for-each-ref" in cmd:
            return NS(stdout=weird + "\n", stderr="", returncode=0)
        return NS(stdout="", stderr="", returncode=0)

    with mock_patch.object(hermes_main.subprocess, "run", side_effect=fake_run):
        hermes_main._prune_orphan_rescue_refs(["git"], ".", "main")

    delete_calls = [c for c in calls if "update-ref" in c and "-d" in c]
    assert delete_calls == []


def test_cmd_update_ordinary_divergence_skips_rescue_ref(monkeypatch, tmp_path, capsys):
    """Common ancestor still exists (e.g. upstream force-push) → no rescue
    ref, no orphan messaging, behavior identical to before #87694."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert update_ref_calls == []

    out = capsys.readouterr().out
    assert "orphan divergence" not in out
    assert "Fast-forward not possible (history diverged), resetting to match remote" in out


def test_cmd_update_orphan_rescue_ref_write_failure_is_non_fatal(monkeypatch, tmp_path, capsys):
    """#87694 stress test: ``git update-ref`` itself fails (disk full,
    permissions) while parking the orphan rescue ref. The backup attempt is
    best-effort — so the reset must still proceed and the update succeed."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, update_ref_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert len(update_ref_calls) == 1

    reset_calls = [
        c for c in recorded if "reset" in " ".join(str(x) for x in c) and "--hard" in c
    ]
    assert len(reset_calls) == 1

    out = capsys.readouterr().out
    assert "orphan divergence" in out


def test_cmd_update_orphan_guard_skips_rescue_ref_when_pre_pull_sha_missing(
    monkeypatch, tmp_path, capsys
):
    """#87694 stress test: if capturing the pre-pull HEAD SHA itself fails
    (empty ``rev-parse HEAD`` output), the rescue-ref guard requires a
    truthy ``pre_pull_sha`` and must skip the backup rather than writing a
    ref pointing at nothing — the reset must still proceed without crashing.
    """
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, pre_pull_sha_unavailable=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert update_ref_calls == []

    out = capsys.readouterr().out
    assert "orphan divergence" not in out


def test_cmd_update_orphan_rescue_ref_persists_when_reset_fails(monkeypatch, tmp_path, capsys):
    """#87694 stress test: even when the subsequent ``reset --hard`` itself
    fails, the rescue ref must already have been written — the backup is
    not lost just because the overall update aborts."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, reset_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(SimpleNamespace())
    assert exc_info.value.code == 1

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert len(update_ref_calls) == 1

    out = capsys.readouterr().out
    assert "orphan divergence" in out
    assert "Failed to reset to origin/main" in out


# ---------------------------------------------------------------------------
# Non-interactive update.non_interactive_local_changes setting
# (chat app / gateway): "discard" throws stashed changes away, "stash"
# (default) restores them. Interactive terminal updates ignore the setting
# and always go through the restore path.
# ---------------------------------------------------------------------------

def _setup_setting_test(monkeypatch, tmp_path, mode):
    """Common wiring: real stash returns a ref, restore + discard are
    recorded, and load_config reports the given non_interactive_local_changes
    mode."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    discard_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_discard_stashed_changes",
        lambda *a, **kw: discard_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **kw: {"updates": {"non_interactive_local_changes": mode}},
    )
    side_effect, recorded = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    return restore_calls, discard_calls, recorded


# ---------------------------------------------------------------------------
# --keep-stash (desktop updater): stash for the update, never re-apply.
# ---------------------------------------------------------------------------

def _setup_keep_stash_test(monkeypatch, tmp_path):
    """Wiring for --keep-stash tests: stash returns a ref; restore, discard,
    and park are all recorded."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    discard_calls = []
    park_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_discard_stashed_changes",
        lambda *a, **kw: discard_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_park_stashed_changes",
        lambda *a, **kw: park_calls.append(a) or None,
    )
    # Keep the update flow away from the real gateway fleet on this machine —
    # a live gateway PID would trip the test-suite kill guard and turn the
    # run into exit 1 (gateway_fleet_restart_incomplete).
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids", lambda **kw: [], raising=False
    )
    return restore_calls, discard_calls, park_calls


def test_update_keep_stash_parks_instead_of_restoring(monkeypatch, tmp_path):
    """--keep-stash: after a successful update, the autostash is parked (left
    in git stash) — never re-applied, never discarded."""
    restore_calls, discard_calls, park_calls = _setup_keep_stash_test(monkeypatch, tmp_path)
    side_effect, _ = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace(yes=True, keep_stash=True))

    assert len(park_calls) == 1
    assert park_calls[0][0] == "abc123deadbeef"
    assert restore_calls == []
    assert discard_calls == []


def test_update_without_keep_stash_still_restores(monkeypatch, tmp_path):
    """Regression guard: default behavior (no --keep-stash) is unchanged —
    the autostash is auto-restored under --yes."""
    restore_calls, discard_calls, park_calls = _setup_keep_stash_test(monkeypatch, tmp_path)
    side_effect, _ = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace(yes=True, keep_stash=False))

    assert restore_calls == [1]
    assert park_calls == []
    assert discard_calls == []


def test_update_keep_stash_failure_path_still_preserves(monkeypatch, tmp_path, capsys):
    """--keep-stash + failed update: neither restore nor park runs; the
    existing preserved-in-stash message fires (working tree unknown)."""
    restore_calls, discard_calls, park_calls = _setup_keep_stash_test(monkeypatch, tmp_path)
    side_effect, _ = _make_update_side_effect(ff_only_fails=True, reset_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace(yes=True, keep_stash=True))

    assert restore_calls == []
    assert park_calls == []
    assert discard_calls == []
    assert "preserved in stash" in capsys.readouterr().out


def test_update_parser_accepts_keep_stash():
    """The flag parses and defaults off."""
    import argparse

    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_update_parser(subparsers, cmd_update=lambda args: None)

    args = parser.parse_args(["update", "--keep-stash"])
    assert args.keep_stash is True
    args = parser.parse_args(["update"])
    assert args.keep_stash is False






def test_bootstrap_marker_not_autostashed_by_update(tmp_path):
    """#38529: the Desktop bootstrap marker must be git-ignored so that
    ``hermes update``'s ``git stash push --include-untracked`` does not sweep it
    into an autostash on every run.

    Behavioral + hermetic: build a throwaway repo that adopts the project's real
    ``.gitignore`` (the contract under test), drop the marker, and confirm the
    same stash invocation the updater uses leaves it untouched.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    repo_gitignore = Path(hermes_main.__file__).resolve().parents[1] / ".gitignore"

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text(repo_gitignore.read_text())
    (tmp_path / "tracked.txt").write_text("x\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    marker = tmp_path / ".hermes-bootstrap-complete"
    marker.write_text("")

    # Exact flags used by hermes update (hermes_cli/main.py).
    git("stash", "push", "--include-untracked", "-m", "hermes-update-autostash")

    assert marker.exists(), (
        ".hermes-bootstrap-complete was swept into the update autostash — it must "
        "be listed in .gitignore so `git stash -u` skips it (#38529)."
    )
    # It must not even register as a dirty/untracked change.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert ".hermes-bootstrap-complete" not in status


# ---------------------------------------------------------------------------
# Permission-denied autostash class: undeletable untracked files (root-owned
# packaging/ etc.) must not abort the update when the stash entry was created.
# ---------------------------------------------------------------------------






def test_update_autostash_survives_undeletable_untracked_dir(tmp_path):
    """Behavioral E2E of the whole permission-denied class with real git:
    root-owned-style undeletable untracked dir → stash succeeds, update-style
    reset works, restore round-trips, nothing lost. (#70127 follow-up)"""
    import os
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    if os.name == "nt":
        pytest.skip("POSIX permission semantics")
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write bits")

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    (tmp_path / "tracked.txt").write_text("v2 local change\n")
    pkg = tmp_path / "packaging" / "homebrew"
    pkg.mkdir(parents=True)
    (pkg / "hermes-agent.rb").write_text("formula\n")
    os.chmod(pkg, 0o555)  # undeletable contents, like a root-owned dir
    try:
        stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
        assert stash_ref

        # The tracked change is stashed; simulate the updater's checkout window.
        assert (tmp_path / "tracked.txt").read_text() == "v1\n"

        restored = hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )
        assert restored is True
        assert (tmp_path / "tracked.txt").read_text() == "v2 local change\n"
        assert (pkg / "hermes-agent.rb").read_text() == "formula\n"
    finally:
        os.chmod(pkg, 0o755)


def test_partial_stash_failure_does_not_reset_for_concurrent_foreign_stash(
    monkeypatch, tmp_path
):
    """A raced refs/stash update is not proof that this invocation saved edits."""
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        joined = " ".join(command)
        if "status --porcelain" in joined:
            return SimpleNamespace(returncode=0, stdout=" M tracked.txt\n", stderr="")
        if "ls-files --unmerged" in joined:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "stash push" in joined:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="warning: failed to remove foreign-owned file",
                args=command,
            )
        if "stash list" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{'b' * 40}\x00On main: somebody-elses-stash\n",
                stderr="",
            )
        if "reset --hard" in joined:
            pytest.fail("foreign stash must never authorize destructive reset")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(update_cmd.subprocess, "run", run)

    with pytest.raises(CalledProcessError):
        update_cmd._stash_local_changes_if_needed(["git"], tmp_path)

    assert not any("reset --hard" in " ".join(command) for command in commands)
def test_restore_rejects_invalid_python_and_keeps_clean_updated_tree(
    monkeypatch, tmp_path, capsys
):
    """A cleanly-applied stash must not be allowed to brick every agent turn."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    source = tmp_path / "tools" / "terminal_tool.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    source.write_text("<<<<<<< Updated upstream\nVALUE = 2\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ())

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )

    assert exc_info.value.code == 1
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git("status", "--porcelain").stdout == ""
    assert git("stash", "list").stdout.strip()
    output = capsys.readouterr().out
    assert "made the Hermes agent unexecutable" in output
    assert "gateway was not restarted" in output
    assert f"git stash apply {stash_ref}" in output


def test_restore_rejects_new_import_time_failure_and_preserves_stash(
    monkeypatch, tmp_path, capsys
):
    """A valid-Python stash must not introduce a critical import failure."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    source = tmp_path / "consumer.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    source.write_text("raise RuntimeError('restored local failure')\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )

    assert exc_info.value.code == 1
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git("status", "--porcelain").stdout == ""
    assert git("stash", "list").stdout.strip()
    output = capsys.readouterr().out
    assert "agent import consumer" in output
    assert "restored local failure" in output
    assert "gateway was not restarted" in output


def test_restore_allows_preexisting_import_time_failure(monkeypatch, tmp_path):
    """A restore may proceed when it does not worsen an environment failure."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "consumer.py").write_text(
        "raise RuntimeError('missing local config')\n", encoding="utf-8"
    )
    local_file = tmp_path / "local.txt"
    local_file.write_text("original\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    local_file.write_text("restored\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    assert hermes_main._restore_stashed_changes(
        ["git"], tmp_path, stash_ref, prompt_user=False
    )
    assert local_file.read_text(encoding="utf-8") == "restored\n"
    assert git("stash", "list").stdout.strip() == ""


def test_restore_rejects_later_failure_masked_by_preexisting_failure(
    monkeypatch, tmp_path, capsys
):
    """Every critical module must be compared, not only the first failure."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "first.py").write_text(
        "raise RuntimeError('missing local config')\n", encoding="utf-8"
    )
    second = tmp_path / "second.py"
    second.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    second.write_text("raise RuntimeError('restored later failure')\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("first", "second"))

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )

    assert exc_info.value.code == 1
    assert second.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git("status", "--porcelain").stdout == ""
    assert git("stash", "list").stdout.strip()
    output = capsys.readouterr().out
    assert "agent import second" in output
    assert "restored later failure" in output
    assert "gateway was not restarted" in output


def test_restore_rejects_system_exit_masked_by_preexisting_failure(
    monkeypatch, tmp_path, capsys
):
    """A terminating import must be compared instead of hiding the marker."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "first.py").write_text(
        "raise RuntimeError('missing local config')\n", encoding="utf-8"
    )
    second = tmp_path / "second.py"
    second.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    second.write_text("raise SystemExit('restored exit')\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("first", "second"))

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )

    assert exc_info.value.code == 1
    assert second.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git("status", "--porcelain").stdout == ""
    assert git("stash", "list").stdout.strip()
    output = capsys.readouterr().out
    assert "agent import second" in output
    assert "restored exit" in output
    assert "gateway was not restarted" in output


def test_restore_rejects_probe_termination(monkeypatch, tmp_path, capsys):
    """A stash cannot bypass import validation by terminating the probe."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    source = tmp_path / "consumer.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    source.write_text("import os\nos._exit(7)\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )

    assert exc_info.value.code == 1
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git("status", "--porcelain").stdout == ""
    assert git("stash", "list").stdout.strip()
    output = capsys.readouterr().out
    assert "critical-module probe" in output
    assert "exit code 7" in output
    assert "gateway was not restarted" in output


def test_restore_stays_parked_when_untracked_baseline_is_unknown(
    monkeypatch, tmp_path, capsys
):
    """Unknown cleanup scope must not turn into a destructive empty baseline."""
    from hermes_cli import update_cmd

    monkeypatch.setattr(update_cmd, "_git_untracked_paths", lambda *_args: None)

    restored = hermes_main._restore_stashed_changes(
        ["git"], tmp_path, "stash@{0}", prompt_user=False
    )

    assert restored is False
    output = capsys.readouterr().out
    assert "cleanup baseline is unknown" in output
    assert "git stash apply stash@{0}" in output


def test_reject_does_not_claim_cleanup_when_git_state_is_unknown(
    monkeypatch, tmp_path, capsys
):
    """Cleanup failures must not be reported as a restored clean tree."""
    from hermes_cli import update_cmd

    monkeypatch.setattr(update_cmd, "_git_untracked_paths", lambda *_args: None)

    with pytest.raises(SystemExit):
        update_cmd._reject_unsafe_stash_restore(
            ["git"], tmp_path, "stash@{0}", set(), "consumer.py", "invalid"
        )

    output = capsys.readouterr().out
    assert "could not be fully restored automatically" in output
    assert "The clean updated tree has been restored" not in output


def test_restore_rejects_unknown_restored_python_paths(
    monkeypatch, tmp_path, capsys
):
    """A failed post-apply path query cannot skip restored syntax validation."""
    import subprocess
    from hermes_cli import update_cmd

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    source = tmp_path / "consumer.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ())
    monkeypatch.setattr(update_cmd, "_restored_python_paths", lambda *_args: None)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )

    assert exc_info.value.code == 1
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git("status", "--porcelain").stdout == ""
    assert git("stash", "list").stdout.strip()
    output = capsys.readouterr().out
    assert "restored Python source discovery" in output
    assert "gateway was not restarted" in output


def test_gateway_restore_prompt_defaults_to_keep_stash(tmp_path, capsys):
    prompts = []

    restored = hermes_main._restore_stashed_changes(
        ["git"],
        tmp_path,
        "stash@{0}",
        prompt_user=True,
        input_fn=lambda prompt, default: prompts.append((prompt, default)) or "",
    )

    assert restored is False
    assert prompts == [("Restore local changes now? [y/N]", "n")]
    assert "still preserved in git stash" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# #87694: real-git sanity check for the merge-base premise the orphan guard
# relies on — two independently-initialized repos (no shared history) must
# report no merge-base, while an ordinary branch divergence must report one.
# ---------------------------------------------------------------------------

def test_merge_base_detects_orphan_vs_ordinary_divergence_with_real_git(tmp_path):
    """Anchors the assumption behind the #87694 orphan-history guard: `git
    merge-base` fails/empties on truly unrelated histories, and succeeds on
    ordinary (e.g. force-pushed) divergence. If a future git version changes
    this contract, this test breaks instead of the guard silently going
    inert."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(cwd, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
        )

    # Ordinary divergence: two branches of the SAME repo, one reset to an
    # earlier point then given a new commit (simulates an upstream force-push).
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    git(ordinary, "init", "-q", "-b", "main")
    git(ordinary, "config", "user.email", "t@example.com")
    git(ordinary, "config", "user.name", "t")
    (ordinary / "f.txt").write_text("v1\n")
    git(ordinary, "add", "-A")
    git(ordinary, "commit", "-qm", "init")
    git(ordinary, "checkout", "-qb", "origin-main")
    (ordinary / "f.txt").write_text("v2\n")
    git(ordinary, "add", "-A")
    git(ordinary, "commit", "-qm", "upstream")
    git(ordinary, "checkout", "-q", "main")
    result = git(ordinary, "merge-base", "HEAD", "origin-main", check=False)
    assert result.returncode == 0
    assert result.stdout.strip()

    # Orphan divergence: two independently-init'd repos wired as remotes,
    # sharing zero history.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    git(orphan, "init", "-q", "-b", "main")
    git(orphan, "config", "user.email", "t@example.com")
    git(orphan, "config", "user.name", "t")
    (orphan / "f.txt").write_text("local\n")
    git(orphan, "add", "-A")
    git(orphan, "commit", "-qm", "local init")

    remote = tmp_path / "orphan-remote"
    remote.mkdir()
    git(remote, "init", "-q", "-b", "main")
    git(remote, "config", "user.email", "t@example.com")
    git(remote, "config", "user.name", "t")
    (remote / "f.txt").write_text("remote\n")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "remote init")

    git(orphan, "remote", "add", "origin", str(remote))
    git(orphan, "fetch", "-q", "origin")
    result = git(orphan, "merge-base", "HEAD", "origin/main", check=False)
    assert result.returncode != 0
    assert not result.stdout.strip()


def test_prune_orphan_rescue_refs_with_real_git_unpins_objects(tmp_path):
    """End-to-end with real git: an orphan rescue ref pins a snapshot's
    objects against gc; pruning the ref (age-expired) makes them collectable.
    This is the sabotage/size test for the #87745 bounded-growth mitigation."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "gc.auto", "0")
    (tmp_path / "f.txt").write_text("base\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    # Snapshot commit carrying a "large" payload (scaled down for CI).
    import os

    (tmp_path / "big.bin").write_bytes(os.urandom(512 * 1024))
    git("add", "-A")
    git("commit", "-qm", "snapshot")
    snap_sha = git("rev-parse", "HEAD").stdout.strip()

    # Rewind, park the snapshot behind an AGE-EXPIRED rescue ref.
    git("reset", "-q", "--hard", "HEAD~1")
    old_ref = "refs/hermes-update-backups/orphan-main-20200101-000000-" + snap_sha[:12]
    git("update-ref", old_ref, snap_sha)

    # With the ref present, gc cannot drop the snapshot objects.
    git("reflog", "expire", "--expire=now", "--all")
    git("gc", "-q", "--prune=now")
    assert git("cat-file", "-e", snap_sha, check=False).returncode == 0

    # Prune (the ref's 2020 timestamp is way past the age window) → ref gone.
    hermes_main._prune_orphan_rescue_refs(["git"], tmp_path, "main")
    remaining = git("for-each-ref", "refs/hermes-update-backups/").stdout
    assert old_ref not in remaining

    # And gc can now reclaim the snapshot's objects.
    git("gc", "-q", "--prune=now")
    assert git("cat-file", "-e", snap_sha, check=False).returncode != 0
