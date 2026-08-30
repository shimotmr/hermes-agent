/**
 * In-app update mutual-exclusion marker (#50238).
 *
 * The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
 * duration of an `--update` run (see apps/bootstrap-installer/src-tauri/src/
 * update.rs `UpdateMarkerGuard`). The marker body is two lines: the updater's
 * pid and the unix-seconds it started.
 *
 * Why: if the user relaunches the desktop mid-update — the window vanished with
 * no progress and looks crashed — a fresh instance must NOT spawn its own local
 * backend. That backend re-locks the venv shim, the updater's straggler cleanup
 * (`force_kill_other_hermes`, taskkill /IM hermes.exe) kills it, the launch
 * fails with the 45s "backend didn't come up" timeout, and the user relaunches
 * into the same trap — an infinite respawn/kill loop. The desktop gates local
 * backend startup on this marker and parks until the update finishes.
 *
 * This module holds the PURE, side-effect-light logic (path, pid liveness,
 * parse + staleness) so it is unit-testable without booting Electron. The
 * polling/boot-progress wrapper lives in main.ts where the boot-progress and
 * log sinks are.
 */

import crypto from 'node:crypto'
import fs from 'fs'
import path from 'path'

// Retained for API compatibility and elapsed-time diagnostics. A live pid is
// never evicted solely by age: Windows updates can be quiet for 40+ minutes,
// and PID reuse/identity uncertainty must fail closed.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

export function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

function atomicReplaceMarker(file: string, body: string) {
  const temporary = `${file}.write-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`

  try {
    fs.writeFileSync(temporary, body, { encoding: 'utf8', flag: 'wx' })
    fs.renameSync(temporary, file)
  } finally {
    try {
      fs.unlinkSync(temporary)
    } catch {
      void 0
    }
  }
}

function markerOperationGuard(file: string, action: () => boolean) {
  const guard = `${file}.lock`
  const ownerFile = path.join(guard, 'owner')

  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      fs.mkdirSync(guard)
      try {
        fs.writeFileSync(ownerFile, `${process.pid}\n`, { encoding: 'ascii', flag: 'wx' })
      } catch {
        void 0
      }
      try {
        return action()
      } finally {
        try {
          fs.unlinkSync(ownerFile)
        } catch {
          void 0
        }
        try {
          fs.rmdirSync(guard)
        } catch {
          void 0
        }
      }
    } catch (err) {
      if (!err || (err as NodeJS.ErrnoException).code !== 'EEXIST') return false
      let guardPid = 0
      try {
        guardPid = Number.parseInt(fs.readFileSync(ownerFile, 'ascii').trim(), 10)
      } catch {
        return false
      }
      if (isPidAlive(guardPid)) return false
      try {
        fs.unlinkSync(ownerFile)
        fs.rmdirSync(guard)
      } catch {
        return false
      }
    }
  }
  return false
}

function removeMarkerIfUnchanged(file: string, expected: string) {
  return markerOperationGuard(file, () => removeMarkerIfUnchangedGuarded(file, expected))
}

function removeMarkerIfUnchangedGuarded(file: string, expected: string) {
  const quarantine = `${file}.remove-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`

  try {
    fs.renameSync(file, quarantine)
  } catch {
    return false
  }

  let removable = false

  try {
    removable = fs.readFileSync(quarantine, 'utf8') === expected
  } catch {
    removable = false
  }

  if (!removable) {
    try {
      fs.linkSync(quarantine, file)
    } catch {
      void 0
    }
  }

  let removed = false

  try {
    fs.unlinkSync(quarantine)
    removed = true
  } catch {
    void 0
  }

  if (removable && !removed) {
    try {
      fs.linkSync(quarantine, file)
    } catch {
      void 0
    }
  }

  return removable && removed
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
export function isPidAlive(pid, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err) {
    return Boolean(err && (err as NodeJS.ErrnoException).code !== 'ESRCH')
  }
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` only when an update is GENUINELY still running
 * (parseable pid that is alive). Returns `null` for every "no live update"
 * case — absent, unreadable, malformed, or confirmed-dead pid — and, when a
 * stale marker file exists, deletes it so it
 * cannot strand future launches.
 *
 * Pure-ish: file I/O against the given path, plus an injectable pid probe and
 * clock for tests.
 */
export function readLiveUpdateMarker(
  hermesHome,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
  } = {}
) {
  const file = markerPath(hermesHome)
  let raw

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return null // absent or unreadable => no live update
  }

  const [pidLine, startedLine] = String(raw).split('\n')
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1000 : Infinity
  const alive = Number.isInteger(pid) && isPidAlive(pid, kill)

  if (!alive) {
    const removed = removeMarkerIfUnchanged(file, raw)
    if (!removed && fs.existsSync(file)) {
      // A cross-runtime handoff/cleanup operation currently owns the guard.
      // Fail closed until its canonical marker transition completes.
      return { pid: Number.isInteger(pid) ? pid : 0, ageMs }
    }

    return null
  }

  return { pid, ageMs }
}

function parseClaim(raw: string) {
  const [pidLine, startedLine, tokenLine] = raw.split('\n')
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const token = (tokenLine || '').trim()

  return Number.isInteger(pid) && Number.isInteger(startedAt) && token ? { pid, startedAt, token } : null
}

export function claimUpdateMarker(
  hermesHome: string,
  pid: number,
  { now = Date.now, startedAt, kill }: { now?: () => number; startedAt?: number; kill?: typeof process.kill } = {}
) {
  const file = markerPath(hermesHome)
  const acquiredAt = Number.isInteger(startedAt) ? startedAt! : Math.floor(now() / 1000)
  const token = crypto.randomUUID()
  const body = `${pid}\n${acquiredAt}\n${token}\n`
  let claimed = false

  const guarded = markerOperationGuard(file, () => {
    if (fs.existsSync(file)) {
      let existingPid = 0
      try {
        existingPid = Number.parseInt(fs.readFileSync(file, 'utf8').split('\n')[0].trim(), 10)
      } catch {
        return false
      }
      if (isPidAlive(existingPid, kill)) return false
      const stale = `${file}.stale-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`
      try {
        fs.renameSync(file, stale)
        fs.unlinkSync(stale)
      } catch {
        return false
      }
    }
    try {
      fs.writeFileSync(file, body, { encoding: 'utf8', flag: 'wx' })
      claimed = true
      return true
    } catch {
      return false
    }
  })

  return guarded && claimed ? { pid, startedAt: acquiredAt, token } : null
}

export function handoffUpdateMarker(hermesHome: string, expectedPid: number, expectedToken: string, nextPid: number) {
  const file = markerPath(hermesHome)

  return markerOperationGuard(file, () => {
    let raw = ''
    try {
      raw = fs.readFileSync(file, 'utf8')
    } catch {
      return false
    }
    const claim = parseClaim(raw)
    if (!claim || claim.pid !== expectedPid || claim.token !== expectedToken) return false
    try {
      atomicReplaceMarker(file, `${nextPid}\n${claim.startedAt}\n${claim.token}\n`)
      return true
    } catch {
      return false
    }
  })
}

export function releaseUpdateMarker(hermesHome: string, expectedPid: number, expectedToken: string) {
  const file = markerPath(hermesHome)

  return markerOperationGuard(file, () => {
    let raw = ''
    try {
      raw = fs.readFileSync(file, 'utf8')
    } catch {
      return false
    }
    const claim = parseClaim(raw)
    if (!claim || claim.pid !== expectedPid || claim.token !== expectedToken) return false
    return removeMarkerIfUnchangedGuarded(file, raw)
  })
}

/**
 * Write the update-in-progress marker *from the desktop* before handing off
 * to the detached updater.
 *
 * The Tauri-based hermes-setup.exe takes several seconds to initialise its
 * window and reach the Rust `run_update` entry point where it writes the
 * marker itself. During that gap the desktop's `app.quit()` teardown kills
 * the backend child, the renderer's WebSocket drops, and the renderer
 * immediately calls `ensureBackend()` → `waitForUpdateToFinish()`. Because
 * the updater hasn't written the marker yet, the gate sees no live update
 * and spawns a *new* backend — which re-locks `.pyd` files in the venv.
 * When the updater finally reaches the venv-rebuild stage it finds those
 * files locked and the update bricks.
 *
 * Compatibility wrapper for older call sites. New hand-offs first call
 * `claimUpdateMarker`, then transfer ownership with `handoffUpdateMarker`'s
 * expected-pid + token compare-and-swap. This wrapper therefore performs only
 * an atomic no-replace initial claim; it never overwrites another publisher.
 */
export function writeUpdateMarker(
  hermesHome,
  pid,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    startedAt
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    startedAt?: number
  } = {}
) {
  const file = markerPath(hermesHome)
  const nowMs = now()
  void file
  void kill
  void maxAgeMs
  return claimUpdateMarker(hermesHome, pid, { now: () => nowMs, startedAt, kill })
}

/**
 * Whether a NEW updater hand-off must be refused because a different,
 * already-alive updater currently owns the marker (#75778).
 *
 * This remains a read-only diagnostic helper for user-facing conflict text.
 * Actual exclusion is provided by `claimUpdateMarker` under the shared marker
 * operation guard, so no check-then-publish window remains.
 *
 * Returns the live foreign owner (with a ready-to-show message) when the
 * hand-off must be refused, or `null` when it's safe to spawn — no marker,
 * or the existing one is stale/dead and self-heals via
 * `readLiveUpdateMarker`.
 */
export function updateHandoffConflict(
  hermesHome,
  opts: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
  } = {}
) {
  const owner = readLiveUpdateMarker(hermesHome, opts)

  if (!owner) {
    return null
  }

  const mins = Math.floor(owner.ageMs / 60_000)
  const secs = Math.floor((owner.ageMs % 60_000) / 1000)
  const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return {
    pid: owner.pid,
    ageMs: owner.ageMs,
    message: `An update is already running (PID ${owner.pid}, started ${elapsed} ago). Wait for it to finish, then try again.`
  }
}
