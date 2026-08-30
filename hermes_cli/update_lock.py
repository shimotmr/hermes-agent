"""Cross-process mutual exclusion for in-flight Hermes updates.

Three different surfaces can start an update of the same install tree:

* ``hermes update`` from a terminal,
* the dashboard's Update button (``POST /api/hermes/update`` →
  ``_spawn_hermes_action(["update"])``, detached),
* the desktop's Update button, which hands off to the Tauri
  ``hermes-setup --update`` and, on its failure screen, to install-mode
  bootstrap (``install.ps1`` / ``install.sh``).

Until now only the Tauri updater published an "update in progress" marker
(``UpdateMarkerGuard`` in ``apps/bootstrap-installer/src-tauri/src/update.rs``),
and only the Electron desktop consumed it (``electron/update-marker.ts``, to
gate local backend startup). Nothing stopped two *updaters* from running at
once — so a dashboard-spawned ``hermes update`` and an installer-driven
``git checkout`` could mutate the same checkout concurrently, rewriting source
under a live interpreter and leaving the tree half-updated.

This module makes that same marker the single lock for **all** update
entrypoints instead of adding a fourth mechanism. The first two lines remain
backward-compatible; a third opaque token binds release/handoff authority:

    <HERMES_HOME>/.hermes-update-in-progress   body: "<pid>\\n<started_at_unix>\\n<token>"

A marker counts as live whenever its pid is alive. Age is diagnostic only:
Windows updates can legitimately be quiet for 40+ minutes, and PID identity
uncertainty fails closed. Confirmed-dead markers are removed under the shared
operation guard.

One layering wrinkle: the Tauri updater holds this marker for its WHOLE run and
then spawns ``hermes update`` as a child stage. Without a handoff the child
sees its own parent's live marker and refuses — the GUI update deadlocks
against itself on every attempt ("Hermes is still running", retry forever).
The orchestrating parent exports a pid + opaque token pair:

* The updater exports :data:`HANDOFF_PID_ENV` naming its own pid, and
  ``acquire`` treats a live holder matching that pid as the lock we are
  already running under. The env var alone grants nothing: the pid must also
  be the live marker owner, so a stale or forged value cannot bypass the lock.
Legacy ancestry without that token cannot prove safe ownership transfer and is
therefore refused; the parent must keep the canonical claim intact.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in
# apps/desktop/electron/update-marker.ts — the same marker is read by both, and
# retained for wire/API compatibility and elapsed-time tests. It is not an
# eviction authority while the owner pid remains alive.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

MARKER_NAME = ".hermes-update-in-progress"

# Set by an orchestrating updater (the Tauri `hermes-setup --update` flow) to
# its own pid before spawning `hermes update` as a child stage. The parent
# holds the marker for its whole run, so without this the child refuses its
# own parent's lock and the GUI update can never complete. See update_child_env
# in apps/bootstrap-installer/src-tauri/src/update.rs — keep the name in sync.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"
HANDOFF_TOKEN_ENV = "HERMES_UPDATE_CLAIM_TOKEN"

# Exit code meaning "another updater/instance owns this install right now".
# Already the de-facto contract: the Windows shim + venv-holder guards in
# _cmd_update_impl exit 2, and the Tauri updater matches on it
# (UPDATE_EXIT_CONCURRENT in apps/bootstrap-installer/src-tauri/src/update.rs)
# to show "Hermes is still running" instead of a generic failure. Naming it
# here keeps the concurrent-update refusal on that same understood contract.
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Uses the *process* Hermes home (never the context-local profile override):
    the Rust updater resolves ``$HERMES_HOME`` or the platform default, and the
    desktop pins that same value into the updater's env. A profile-scoped path
    here would put the lock somewhere the other two owners never look.
    """
    from hermes_constants import get_process_hermes_home

    return get_process_hermes_home() / MARKER_NAME


class _Liveness(Enum):
    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


def _pid_liveness(pid: int) -> _Liveness:
    """True when a process with ``pid`` currently exists.

    Delegates to :func:`gateway.status._pid_exists`, the project's existing
    no-kill probe. Do NOT hand-roll this with ``os.kill(pid, 0)``: on Windows
    that is not a no-op — CPython routes ``sig=0`` to
    ``GenerateConsoleCtrlEvent``, which Ctrl+C's the target's whole console
    process group (bpo-14484). A liveness check that killed the updater it was
    asking about would be a spectacular way to fix a concurrency bug.

    An unevaluable positive pid fails closed as alive. PID reuse cannot be
    distinguished portably across every publisher, so a live numeric identity
    is never evicted merely because the claim is old.
    """
    if pid <= 0:
        return _Liveness.UNKNOWN
    try:
        from gateway.status import _pid_exists

        return _Liveness.ALIVE if _pid_exists(pid) else _Liveness.DEAD
    except Exception as exc:
        # Import/probe failure leaves authority uncertain. Fail closed.
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return _Liveness.UNKNOWN


def _pid_alive(pid: int) -> bool:
    """Compatibility predicate: unknown authority is deliberately blocking."""
    return _pid_liveness(pid) is not _Liveness.DEAD


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us, if any.

    Read from :data:`HANDOFF_PID_ENV`. Malformed values count as absent —
    a broken handoff must fall back to the normal refusal, never crash.
    """
    raw = os.environ.get(HANDOFF_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


@dataclass(frozen=True)
class UpdateHolder:
    """A confirmed-live update currently holding the lock."""

    pid: int
    age_seconds: float


def _marker_owner(raw: str) -> int | None:
    try:
        return int(raw.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _marker_token(raw: str) -> str | None:
    lines = raw.splitlines()
    if len(lines) < 3:
        return None
    token = lines[2].strip()
    return token or None


def _operation_guard_path(marker: Path) -> Path:
    return marker.with_name(f"{marker.name}.lock")


def _guard_owner_path(guard: Path) -> Path:
    """Owner inode for the current file guard or a legacy directory guard."""
    return guard / "owner" if guard.is_dir() else guard


def _guard_quarantine_present(guard: Path) -> bool:
    """Treat any orphaned cross-runtime stale quarantine as authoritative."""
    prefixes = (f".{guard.name}.stale-", f"{guard.name}.stale-")
    try:
        return any(entry.name.startswith(prefixes) for entry in guard.parent.iterdir())
    except OSError:
        return True


def _restore_quarantined_guard(guard: Path, quarantine: Path, owner_file: Path) -> bool:
    """Restore detached guard authority without replacing a newer claimant."""
    try:
        os.link(owner_file, guard)
    except FileExistsError:
        # A newer canonical guard wins. Never delete or replace it.
        return True
    except OSError:
        # If the exact inode cannot be restored, publish a no-replace directory
        # sentinel.  Even if its owner link also fails, the malformed canonical
        # guard blocks every runtime while the original stays in quarantine.
        try:
            guard.mkdir()
        except FileExistsError:
            return True
        except OSError:
            return False
        try:
            os.link(owner_file, guard / "owner")
        except OSError:
            pass
        return False
    return True


def _reclaim_dead_guard(guard: Path) -> bool:
    """Detach one guard identity, then reclaim only that confirmed-dead inode."""
    quarantine = guard.with_name(
        f".{guard.name}.stale-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        os.replace(guard, quarantine)
    except OSError:
        return False

    owner_file = quarantine / "owner" if quarantine.is_dir() else quarantine
    try:
        guard_pid = int(owner_file.read_text(encoding="ascii").splitlines()[0].strip())
    except (OSError, ValueError, IndexError):
        if not _restore_quarantined_guard(guard, quarantine, owner_file):
            return False
        _remove_guard_quarantine(quarantine)
        return False

    if _pid_liveness(guard_pid) is not _Liveness.DEAD:
        if not _restore_quarantined_guard(guard, quarantine, owner_file):
            return False
        _remove_guard_quarantine(quarantine)
        return False

    _remove_guard_quarantine(quarantine)
    # A replacement published after our detach is authoritative. Do not enter
    # another retry that would detach that newer guard as if it were stale.
    return not quarantine.exists() and not guard.exists()


def _remove_guard_quarantine(quarantine: Path) -> None:
    try:
        if quarantine.is_dir():
            (quarantine / "owner").unlink(missing_ok=True)
            quarantine.rmdir()
        else:
            quarantine.unlink(missing_ok=True)
    except OSError:
        pass


@contextmanager
def _marker_operation(marker: Path):
    """Serialize every canonical marker mutation across all updater runtimes."""
    guard = _operation_guard_path(marker)
    private_guard: Path | None = None
    acquired = False
    preserve = False
    if _guard_quarantine_present(guard):
        yield False
        return
    for _attempt in range(3):
        try:
            private_guard = Path(
                tempfile.mkdtemp(prefix=f".{guard.name}.", dir=guard.parent)
            )
            owner_file = private_guard / "owner"
            descriptor = os.open(owner_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                directory_descriptor = os.open(private_guard, os.O_RDONLY)
            except OSError:
                directory_descriptor = None
            if directory_descriptor is not None:
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            os.link(owner_file, guard)
            acquired = True
            break
        except FileExistsError:
            # Never decide from a pathname and then unlink that pathname: another
            # runtime can replace it in between (ABA). Detach one identity into
            # quarantine, revalidate there, and restore with no replacement.
            if not _reclaim_dead_guard(guard):
                yield False
                return
        except OSError:
            yield False
            return
        finally:
            if private_guard is not None:
                try:
                    (private_guard / "owner").unlink(missing_ok=True)
                    private_guard.rmdir()
                except OSError:
                    pass
                private_guard = None
    if not acquired:
        yield False
        return
    try:
        class OperationLease:
            def preserve(self) -> None:
                nonlocal preserve
                preserve = True

            def __bool__(self) -> bool:
                return True

        yield OperationLease()
    finally:
        if not preserve:
            try:
                if guard.is_dir():
                    (guard / "owner").unlink(missing_ok=True)
                    guard.rmdir()
                else:
                    guard.unlink()
            except OSError:
                # A stranded guard fails closed; never delete uncertain authority.
                pass


def _restore_quarantined_marker(marker: Path, quarantine: Path) -> bool:
    """Restore a marker without replacing a claim published after our rename."""
    try:
        os.link(quarantine, marker)
    except FileExistsError:
        # A newer claimant already occupies the canonical pathname. Keeping that
        # marker is the fail-closed result; the quarantined copy is redundant.
        return True
    except OSError:
        # Ownership could not be restored safely. Retain the private marker for
        # diagnosis rather than deleting an inode we do not own.
        return False
    try:
        quarantine.unlink()
    except OSError:
        pass
    return True


def _remove_marker_if(marker: Path, predicate) -> bool:
    """Atomically detach, verify, and delete one marker claim.

    Writers publish a fresh inode with atomic replacement. Renaming the pathname
    into a private quarantine therefore linearizes deletion against every
    publish: a claim published before the rename is re-verified in quarantine;
    one published after it remains at ``marker`` and cannot be deleted here.
    """
    with _marker_operation(marker) as operation:
        if not operation:
            return False
        quarantine = marker.with_name(
            f".{marker.name}.remove-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            os.replace(marker, quarantine)
        except OSError:
            return False
        try:
            raw = quarantine.read_text(encoding="utf-8")
            removable = bool(predicate(raw))
        except Exception:
            removable = False
        if not removable:
            if not _restore_quarantined_marker(marker, quarantine):
                operation.preserve()
            return False
        try:
            quarantine.unlink()
        except OSError:
            if not _restore_quarantined_marker(marker, quarantine):
                operation.preserve()
            return False
        return True


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the live update holding the lock, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``. Only an
    absent marker with no operation guard, or a marker whose process is
    explicitly dead and safely removed, is clear. Unknown/corrupt authority
    blocks. Never raises.
    """
    marker = path or update_marker_path()
    try:
        raw = marker.read_text(encoding="utf-8")
    except OSError:
        if _operation_guard_path(marker).exists():
            return UpdateHolder(pid=0, age_seconds=float("inf"))
        return None

    lines = raw.splitlines()
    try:
        pid = int(lines[0].strip())
    except (IndexError, ValueError):
        pid = -1
    try:
        started_at = float(lines[1].strip())
    except (IndexError, ValueError):
        return UpdateHolder(pid=0, age_seconds=float("inf"))

    age = time.time() - started_at
    liveness = _pid_liveness(pid)
    if liveness is _Liveness.DEAD:
        def still_stale(candidate: str) -> bool:
            candidate_lines = candidate.splitlines()
            try:
                candidate_pid = int(candidate_lines[0].strip())
                candidate_started = float(candidate_lines[1].strip())
            except (IndexError, ValueError):
                return True
            return _pid_liveness(candidate_pid) is _Liveness.DEAD

        if _remove_marker_if(marker, still_stale):
            return None
        return UpdateHolder(pid=0, age_seconds=age)

    if liveness is _Liveness.UNKNOWN:
        return UpdateHolder(pid=pid if pid > 0 else 0, age_seconds=age)

    return UpdateHolder(pid=pid, age_seconds=age)


def describe_holder(holder: UpdateHolder) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    minutes, seconds = divmod(int(max(holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"✗ Another Hermes update is already running (PID {holder.pid}, "
        f"started {elapsed} ago).\n"
        "\n"
        "  Two updates mutating the same checkout corrupt it: one rewrites\n"
        "  source while the other is mid-install. Wait for it to finish, or\n"
        "  close the window/dashboard tab that started it, then retry."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update already holds it — callers
    decide whether that's a hard refusal (CLI/dashboard) or a wait. Releasing
    only removes the marker when *we* still own it, so a marker rewritten by a
    handoff partner (the Tauri updater overwrites it with its own pid) is never
    deleted out from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self.token: str | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder matching the expected pid + token is atomically
        transferred to this child. During the staged-updater rollout, an exact
        two-line marker owned by this process's direct parent is the only
        tokenless compatibility path; it is immediately upgraded with a token.
        """
        for _attempt in range(3):
            existing = read_live_update(path=self.path)
            if existing is not None:
                try:
                    existing_raw = self.path.read_text(encoding="utf-8")
                except OSError:
                    return False
                handoff_token = os.environ.get(HANDOFF_TOKEN_ENV, "").strip()
                existing_lines = existing_raw.splitlines()
                token_handoff = (
                    handoff_token
                    and existing.pid == _handoff_pid()
                    and _marker_token(existing_raw) == handoff_token
                )
                # Two-phase rollout for staged updaters that already publish
                # HANDOFF_PID but predate claim tokens.  The proof is deliberately
                # narrower than the removed ancestry bypass: exact two-line legacy
                # wire format + exact live marker owner + exact direct OS parent.
                legacy_direct_parent_handoff = (
                    not handoff_token
                    and len(existing_lines) == 2
                    and _marker_token(existing_raw) is None
                    and existing.pid == _handoff_pid()
                    and existing.pid == os.getppid()
                )
                if token_handoff or legacy_direct_parent_handoff:
                    transfer_token = handoff_token or uuid.uuid4().hex
                    with _marker_operation(self.path) as guarded:
                        if not guarded:
                            return False
                        try:
                            current = self.path.read_text(encoding="utf-8")
                        except OSError:
                            return False
                        lines = current.splitlines()
                        current_is_expected = (
                            _marker_owner(current) == _handoff_pid()
                            and len(lines) >= 2
                            and (
                                (_marker_token(current) == handoff_token)
                                if token_handoff
                                else (
                                    len(lines) == 2
                                    and _marker_token(current) is None
                                    and _handoff_pid() == os.getppid()
                                )
                            )
                        )
                        if not current_is_expected:
                            return False
                        temporary_path: str | None = None
                        try:
                            descriptor, temporary_path = tempfile.mkstemp(
                                prefix=f".{self.path.name}.handoff-", dir=self.path.parent
                            )
                            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                                handle.write(
                                    f"{os.getpid()}\n{lines[1].strip()}\n{transfer_token}\n"
                                )
                                handle.flush()
                                os.fsync(handle.fileno())
                            os.replace(temporary_path, self.path)
                        except OSError:
                            return False
                        finally:
                            if temporary_path is not None:
                                try:
                                    Path(temporary_path).unlink()
                                except OSError:
                                    pass
                    self.token = transfer_token
                    self.acquired = True
                    return True
                self.holder = existing
                return False

            # ``None`` means either genuinely absent or unreadable/stale. A
            # marker entry that remains after the read cannot be reclaimed
            # safely, so destructive update ownership is unprovable.
            try:
                self.path.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.debug("Could not inspect update marker %s: %s", self.path, exc)
                return False
            else:
                return False

            with _marker_operation(self.path) as guarded:
                if not guarded:
                    return False
                # Revalidate absence while holding the protocol shared by all
                # publishers and cleanup paths.
                if self.path.exists():
                    continue
                temporary_path: str | None = None
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    descriptor, temporary_path = tempfile.mkstemp(
                        prefix=f".{self.path.name}.", dir=self.path.parent
                    )
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        self.token = uuid.uuid4().hex
                        handle.write(f"{os.getpid()}\n{int(time.time())}\n{self.token}\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.link(temporary_path, self.path)
                except FileExistsError as exc:
                    if temporary_path is None:
                        logger.debug("Could not write update marker %s: %s", self.path, exc)
                        return False
                    continue
                except OSError as exc:
                    logger.debug("Could not write update marker %s: %s", self.path, exc)
                    return False
                finally:
                    if temporary_path is not None:
                        try:
                            Path(temporary_path).unlink()
                        except OSError:
                            pass

            self.acquired = True
            return True

        # Repeated replacement means ownership cannot be proven. Fail closed
        # instead of allowing two update processes to mutate the checkout.
        self.holder = read_live_update(path=self.path)
        return False

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        owner = _marker_owner(raw)
        token = _marker_token(raw)
        if owner != os.getpid() or (self.token is not None and token != self.token):
            # A handoff partner took ownership (e.g. the Tauri updater wrote
            # its own pid). Leave it alone — it's still a live update.
            return
        _remove_marker_if(
            self.path,
            lambda candidate: _marker_owner(candidate) == os.getpid()
            and (self.token is None or _marker_token(candidate) == self.token),
        )

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
