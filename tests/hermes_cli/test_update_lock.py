"""Cross-process update mutual exclusion (``hermes_cli.update_lock``).

Three surfaces can start an update of one install tree: a terminal ``hermes
update``, the dashboard's Update button (which spawns that same command
detached), and the desktop's Update button (Tauri updater → install-mode
bootstrap on its failure screen). Before the shared lock, two of them could run
concurrently and rewrite source under a live interpreter — observed in the wild
as an installer ``git checkout`` rewinding the checkout ~9k commits while a
dashboard-spawned ``hermes update`` was mid-``npm install``, which then failed
against the rewound tree's manifests.

These exercise the real marker file against a temp home — no mocks — because
the contract that matters is what the Rust updater and the Electron gate see on
disk.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.update_lock import (
    HANDOFF_PID_ENV,
    HANDOFF_TOKEN_ENV,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateLock,
    describe_holder,
    read_live_update,
    update_marker_path,
)

# A pid no live process owns. os.kill(pid, 0) must report it dead so a crashed
# updater can never wedge every future update. Deliberately larger than any
# platform's pid_t so it also covers the corrupt-marker path (OverflowError).
DEAD_PID = 4294967294


@pytest.fixture
def marker(tmp_path):
    return tmp_path / ".hermes-update-in-progress"


def test_marker_path_follows_process_hermes_home(tmp_path, monkeypatch):
    """The lock must land where the Rust updater and Electron gate look.

    All three resolve the *process* HERMES_HOME; a profile-scoped path would
    put the lock somewhere the other two owners never read.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert update_marker_path() == tmp_path / ".hermes-update-in-progress"


def test_acquire_writes_pid_and_start_time(marker):
    lock = UpdateLock(path=marker)

    assert lock.acquire() is True
    assert lock.acquired is True

    lines = marker.read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) == os.getpid(), "the Electron gate probes this pid for liveness"
    assert int(lines[1]) == pytest.approx(time.time(), abs=5)
    assert len(lines) == 3, "wire format is pid + started_at + claim token"
    assert lines[2], "token binds release and handoff authority"


def test_second_acquire_is_refused_while_the_first_is_live(marker):
    """The bug: two updaters mutating one checkout at the same time."""
    first = UpdateLock(path=marker)
    assert first.acquire() is True

    second = UpdateLock(path=marker)
    assert second.acquire() is False
    assert second.holder is not None
    assert second.holder.pid == os.getpid()
    assert second.acquired is False


def test_simultaneous_claims_have_exactly_one_owner(marker, monkeypatch):
    """Two claimants that both observe no marker must still serialize."""
    import hermes_cli.update_lock as update_lock_module

    real_read_live_update = update_lock_module.read_live_update
    first_reads = 0
    first_reads_lock = threading.Lock()
    both_observed_absent = threading.Barrier(2)

    def synchronized_initial_read(*, path=None):
        nonlocal first_reads
        with first_reads_lock:
            first_reads += 1
            synchronize = first_reads <= 2
        if synchronize:
            assert not marker.exists()
            both_observed_absent.wait(timeout=2)
            return None
        return real_read_live_update(path=path)

    monkeypatch.setattr(update_lock_module, "read_live_update", synchronized_initial_read)
    locks = [UpdateLock(path=marker), UpdateLock(path=marker)]
    results: list[bool] = []
    threads = [threading.Thread(target=lambda lock=lock: results.append(lock.acquire())) for lock in locks]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(results) == [False, True]


def test_refused_lock_does_not_delete_the_live_owners_marker(marker):
    first = UpdateLock(path=marker)
    first.acquire()

    second = UpdateLock(path=marker)
    second.acquire()
    second.release()

    assert marker.exists(), "a refused claimant must never clear the live owner's lock"

    first.release()
    assert not marker.exists()


def test_release_leaves_a_marker_a_handoff_partner_now_owns(marker):
    """The desktop writes the marker, then the Tauri updater takes ownership.

    Releasing must not delete a marker whose pid is no longer ours — that would
    reopen the gate while the partner is still mid-update.
    """
    lock = UpdateLock(path=marker)
    lock.acquire()

    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")
    lock.release()

    assert marker.exists(), "the partner's marker is not ours to remove"


def test_release_does_not_delete_replacement_after_owner_read(
    marker, monkeypatch
):
    """A claimant replacing our marker in the read/unlink window keeps it."""
    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    replacement = f"{DEAD_PID}\n{int(time.time())}\n"
    real_read_text = Path.read_text
    replaced = False

    def replace_after_read(path, *args, **kwargs):
        nonlocal replaced
        raw = real_read_text(path, *args, **kwargs)
        if path == marker and not replaced:
            replaced = True
            marker.unlink()
            marker.write_text(replacement, encoding="utf-8")
        return raw

    monkeypatch.setattr(Path, "read_text", replace_after_read)

    lock.release()

    assert real_read_text(marker, encoding="utf-8") == replacement


def test_release_does_not_delete_same_inode_handoff_during_unlink(
    marker, monkeypatch
):
    """An Electron-style in-place owner rewrite at unlink time must survive."""
    import hermes_cli.update_lock as update_lock_module

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    replacement = f"{DEAD_PID}\n{int(time.time())}\n"
    real_replace = update_lock_module.os.replace
    rewrote = False

    def rewrite_during_detach(source, destination, *args, **kwargs):
        nonlocal rewrote
        result = real_replace(source, destination, *args, **kwargs)
        if Path(source) == marker and not rewrote:
            rewrote = True
            Path(destination).write_text(replacement, encoding="utf-8")
        return result

    monkeypatch.setattr(update_lock_module.os, "replace", rewrite_during_detach)

    lock.release()

    assert marker.read_text(encoding="utf-8") == replacement


def test_dead_owner_is_reclaimed_not_honored(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


def test_owner_past_the_age_ceiling_is_never_reclaimed_while_alive(marker):
    """A long-running Windows updater remains authoritative while alive."""
    long_ago = int(time.time()) - UPDATE_MARKER_MAX_AGE_SECONDS - 60
    marker.write_text(f"{os.getpid()}\n{long_ago}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is False
    assert lock.holder is not None
    assert lock.holder.pid == os.getpid()
    assert marker.exists()


def test_cleanup_never_opens_a_publish_window_or_loses_live_foreign_claim(
    marker, monkeypatch
):
    """A revalidated live inode must remain canonical throughout cleanup."""
    import hermes_cli.update_lock as update_lock_module

    original = f"{os.getpid()}\n{int(time.time())}\n"
    marker.write_text(original, encoding="utf-8")
    real_replace = update_lock_module.os.replace
    contender_result = None

    def contend_after_detach(source, destination, *args, **kwargs):
        nonlocal contender_result
        result = real_replace(source, destination, *args, **kwargs)
        if Path(source) == marker:
            contender_result = UpdateLock(path=marker).acquire()
        return result

    monkeypatch.setattr(update_lock_module.os, "replace", contend_after_detach)

    assert update_lock_module._remove_marker_if(marker, lambda _raw: False) is False
    assert contender_result is False, "cleanup must hold the shared operation guard"
    assert marker.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "body",
    ["", "not-a-pid\n123\n", "\n\n", "12345"],
    ids=["empty", "garbage-pid", "blank-lines", "no-start-time"],
)
def test_malformed_markers_fail_closed(marker, body):
    marker.write_text(body, encoding="utf-8")

    assert read_live_update(path=marker) is not None
    assert UpdateLock(path=marker).acquire() is False


def test_stale_marker_is_removed_on_read(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    assert read_live_update(path=marker) is None
    assert not marker.exists(), "whoever notices a stale marker clears it"


def test_stale_cleanup_does_not_delete_a_replacement_claim(marker, monkeypatch):
    import hermes_cli.update_lock as update_lock_module

    stale = f"{DEAD_PID}\n{int(time.time())}\n"
    replacement = f"{os.getpid()}\n{int(time.time())}\n"
    marker.write_text(stale, encoding="utf-8")

    calls = 0

    def replace_while_validating(_pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            marker.unlink()
            marker.write_text(replacement, encoding="utf-8")
            return update_lock_module._Liveness.DEAD
        return update_lock_module._Liveness.ALIVE

    monkeypatch.setattr(update_lock_module, "_pid_liveness", replace_while_validating)

    assert read_live_update(path=marker) is not None
    assert marker.read_text(encoding="utf-8") == replacement


def test_stale_cleanup_does_not_delete_same_inode_handoff_during_unlink(
    marker, monkeypatch
):
    """Stale cleanup must not unlink a same-inode owner handoff."""
    import hermes_cli.update_lock as update_lock_module

    stale = f"{DEAD_PID}\n{int(time.time())}\n"
    replacement = f"{os.getpid()}\n{int(time.time())}\n"
    marker.write_text(stale, encoding="utf-8")
    real_replace = update_lock_module.os.replace
    rewrote = False

    def rewrite_during_detach(source, destination, *args, **kwargs):
        nonlocal rewrote
        result = real_replace(source, destination, *args, **kwargs)
        if Path(source) == marker and not rewrote:
            rewrote = True
            Path(destination).write_text(replacement, encoding="utf-8")
        return result

    monkeypatch.setattr(update_lock_module.os, "replace", rewrite_during_detach)

    assert read_live_update(path=marker) is not None
    assert marker.read_text(encoding="utf-8") == replacement


def test_absent_marker_reports_no_live_update(marker):
    assert read_live_update(path=marker) is None


def test_marker_read_error_fails_closed(marker, monkeypatch):
    real_read_text = Path.read_text

    def deny_marker_read(path, *args, **kwargs):
        if path == marker:
            raise PermissionError("simulated marker ACL denial")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_marker_read)

    holder = read_live_update(path=marker)
    assert holder is not None
    assert holder.pid == 0


def test_dangling_marker_symlink_fails_closed(marker):
    marker.symlink_to(marker.with_name("missing-marker-target"))

    holder = read_live_update(path=marker)
    assert holder is not None
    assert holder.pid == 0


def test_guard_inspection_error_fails_closed_for_absent_marker(marker, monkeypatch):
    import hermes_cli.update_lock as update_lock_module

    guard = update_lock_module._operation_guard_path(marker)
    real_lstat = Path.lstat

    def deny_guard_inspection(path, *args, **kwargs):
        if path == guard:
            raise PermissionError("simulated guard ACL denial")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", deny_guard_inspection)

    holder = read_live_update(path=marker)
    assert holder is not None
    assert holder.pid == 0


def test_operation_guard_is_not_visible_until_owner_identity_is_complete(marker, monkeypatch):
    """A crash while building the private guard must not publish ambiguity."""
    import hermes_cli.update_lock as update_lock_module

    canonical = update_lock_module._operation_guard_path(marker)
    real_link = update_lock_module.os.link

    def crash_before_publish(source, destination, *args, **kwargs):
        if Path(destination) == canonical:
            assert Path(source).read_text(encoding="ascii").strip() == str(os.getpid())
            raise OSError("simulated crash before guard publication")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(update_lock_module.os, "link", crash_before_publish)

    with update_lock_module._marker_operation(marker) as guarded:
        assert guarded is False

    assert not canonical.exists()


def test_operation_guard_publish_survives_unsupported_directory_fsync(marker, monkeypatch):
    """Windows may reject opening a directory even after its owner is durable."""
    import hermes_cli.update_lock as update_lock_module

    real_open = update_lock_module.os.open

    def reject_directory_open(path, flags, *args, **kwargs):
        if Path(path).is_dir():
            raise PermissionError("directory fsync unsupported")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(update_lock_module.os, "open", reject_directory_open)

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    lock.release()


def test_operation_guard_reclaim_preserves_aba_replacement(marker, monkeypatch):
    """A live guard published after stale detach must remain canonical."""
    import hermes_cli.update_lock as update_lock_module

    guard = update_lock_module._operation_guard_path(marker)
    guard.write_text(f"{DEAD_PID}\n", encoding="ascii")
    replacement = f"{os.getpid()}\n"
    real_replace = update_lock_module.os.replace
    replaced = False
    guard_detaches = 0

    def publish_after_detach(source, destination, *args, **kwargs):
        nonlocal guard_detaches, replaced
        result = real_replace(source, destination, *args, **kwargs)
        if Path(source) == guard:
            guard_detaches += 1
            if not replaced:
                replaced = True
                guard.write_text(replacement, encoding="ascii")
        return result

    monkeypatch.setattr(update_lock_module.os, "replace", publish_after_detach)

    with update_lock_module._marker_operation(marker) as operation:
        assert operation is False

    assert guard_detaches == 1
    assert guard.read_text(encoding="ascii") == replacement


def test_operation_guard_restore_failure_keeps_canonical_namespace_blocked(
    marker, monkeypatch
):
    import hermes_cli.update_lock as update_lock_module

    guard = update_lock_module._operation_guard_path(marker)
    guard.write_text("malformed\n", encoding="ascii")
    real_link = update_lock_module.os.link
    real_mkdir = Path.mkdir

    def deny_quarantine_restore(source, destination, *args, **kwargs):
        if Path(destination) == guard and ".stale-" in Path(source).name:
            raise PermissionError("simulated ACL denial")
        return real_link(source, destination, *args, **kwargs)

    def deny_sentinel_mkdir(path, *args, **kwargs):
        if path == guard:
            raise PermissionError("simulated namespace denial")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(update_lock_module.os, "link", deny_quarantine_restore)
    monkeypatch.setattr(Path, "mkdir", deny_sentinel_mkdir)

    assert update_lock_module._reclaim_dead_guard(guard) is False
    assert not guard.exists()
    assert list(guard.parent.glob(f".{guard.name}.stale-*"))
    holder = read_live_update(path=marker)
    assert holder is not None
    assert holder.pid == 0
    with update_lock_module._marker_operation(marker) as operation:
        assert operation is False


def test_guard_presence_blocks_reader_when_marker_is_temporarily_absent(marker):
    import hermes_cli.update_lock as update_lock_module

    guard = update_lock_module._operation_guard_path(marker)
    guard.mkdir()
    (guard / "owner").write_text(f"{os.getpid()}\n", encoding="ascii")

    holder = read_live_update(path=marker)

    assert holder is not None
    assert holder.pid == 0
    assert UpdateLock(path=marker).acquire() is False


def test_restore_failure_keeps_canonical_namespace_blocked(marker, monkeypatch):
    import hermes_cli.update_lock as update_lock_module

    original = f"{os.getpid()}\n{int(time.time())}\nforeign-token\n"
    marker.write_text(original, encoding="utf-8")
    real_link = update_lock_module.os.link

    def deny_restore(source, destination, *args, **kwargs):
        if Path(destination) == marker:
            raise PermissionError("simulated ACL denial")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(update_lock_module.os, "link", deny_restore)

    assert update_lock_module._remove_marker_if(marker, lambda _raw: False) is False
    assert marker.exists() or update_lock_module._operation_guard_path(marker).exists()
    assert UpdateLock(path=marker).acquire() is False
    quarantines = list(marker.parent.glob(f".{marker.name}.remove-*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_text(encoding="utf-8") == original


def test_context_manager_releases_even_on_exception(marker):
    with pytest.raises(RuntimeError):
        with UpdateLock(path=marker) as lock:
            assert lock.acquired is True
            raise RuntimeError("update blew up mid-flight")

    assert not marker.exists(), "a crashed update must not strand the lock"


def test_describe_holder_names_the_pid_and_elapsed_time(marker):
    lock = UpdateLock(path=marker)
    lock.acquire()

    holder = read_live_update(path=marker)
    assert holder is not None
    message = describe_holder(holder)

    assert str(os.getpid()) in message, "the user needs the pid to find the other update"
    assert "already running" in message


def test_unwritable_marker_location_fails_closed(tmp_path):
    """An updater must not mutate a checkout when lock ownership is unknowable."""
    lock = UpdateLock(path=tmp_path / "nonexistent-file" / "marker")
    (tmp_path / "nonexistent-file").write_text("i am a file, not a dir", encoding="utf-8")

    assert lock.acquire() is False
    assert lock.acquired is False, "nothing was written, so there is nothing to release"


class TestHandoffFromOrchestratingUpdater:
    """The Tauri updater holds the marker, then spawns ``hermes update``.

    The regression: the child saw its own parent's live marker and exited 2,
    so every GUI update failed with "Hermes is still running" and retrying
    just re-ran the same self-deadlock. The parent names its pid in
    HANDOFF_PID_ENV; a live holder matching it is our own orchestrator.
    """

    def test_legacy_parent_claim_without_token_is_refused(self, marker, monkeypatch):
        # Stand in for the parent updater with our own (live) pid.
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.acquired is False

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()

    def test_legacy_two_line_claim_allows_only_exact_direct_parent_migration(
        self, marker, monkeypatch
    ):
        parent_pid = os.getpid()
        marker.write_text(f"{parent_pid}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, str(parent_pid))
        monkeypatch.delenv(HANDOFF_TOKEN_ENV, raising=False)
        monkeypatch.setattr(os, "getppid", lambda: parent_pid)

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        lines = marker.read_text(encoding="utf-8").splitlines()
        assert int(lines[0]) == os.getpid()
        assert len(lines) == 3
        assert lines[2], "migration mints token authority for all later handoffs"

    def test_legacy_two_line_claim_rejects_non_parent_pid(self, marker, monkeypatch):
        owner_pid = os.getpid()
        marker.write_text(f"{owner_pid}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, str(owner_pid))
        monkeypatch.delenv(HANDOFF_TOKEN_ENV, raising=False)
        monkeypatch.setattr(os, "getppid", lambda: owner_pid + 1)

        assert UpdateLock(path=marker).acquire() is False

    def test_handoff_pid_that_is_not_the_live_holder_grants_nothing(self, marker, monkeypatch):
        """The env var alone must not bypass the lock."""
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid() + 1))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None

    def test_token_handoff_requires_expected_owner_and_token(self, marker, monkeypatch):
        token = "handoff-token"
        marker.write_text(
            f"{os.getpid()}\n{int(time.time())}\n{token}\n", encoding="utf-8"
        )
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid() + 1))
        monkeypatch.setenv(HANDOFF_TOKEN_ENV, token)
        assert UpdateLock(path=marker).acquire() is False

        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))
        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        lines = marker.read_text(encoding="utf-8").splitlines()
        assert int(lines[0]) == os.getpid()
        assert lines[2] == token
        assert lock.acquired is True, "the child must maintain the transferred claim"
        lock.release()
        assert not marker.exists(), "the transferred child owns release authority"

    def test_legacy_ancestry_without_token_fails_closed(self, marker, monkeypatch):
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))

        assert UpdateLock(path=marker).acquire() is False

    @pytest.mark.parametrize("value", ["", "not-a-pid", "-1", "0"], ids=["empty", "garbage", "negative", "zero"])
    def test_malformed_handoff_values_fall_back_to_refusal(self, marker, monkeypatch, value):
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, value)

        assert UpdateLock(path=marker).acquire() is False

    def test_handoff_env_with_no_marker_claims_normally(self, marker, monkeypatch):
        """A handoff pid must not stop us writing our own claim when unlocked."""
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is True
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


class TestLegacyHandoff:
    """Pid or ancestry without the opaque token cannot transfer authority."""

    @pytest.fixture(autouse=True)
    def _liveness_pinned_true(self, monkeypatch):
        from hermes_cli.update_lock import _Liveness

        monkeypatch.setattr(
            "hermes_cli.update_lock._pid_liveness", lambda pid: _Liveness.ALIVE
        )

    def test_marker_owned_by_our_parent_process_without_token_is_refused(self, marker):
        marker.write_text(f"{os.getppid()}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.acquired is False

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getppid()

    def test_live_non_ancestor_holder_is_still_refused(self, marker):
        """Ancestry must not open the lock to unrelated concurrent updaters."""
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None
        assert lock.holder.pid == os.getpid()
