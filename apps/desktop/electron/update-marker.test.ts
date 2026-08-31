/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report a live update only when the
 * updater pid is alive AND the marker is fresh, (b) treat absent/malformed/
 * dead-pid/expired markers as "no live update" so a crashed updater can't
 * strand future launches, and (c) self-heal by deleting a stale marker file.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import os from 'os'
import path from 'path'

import { test, vi } from 'vitest'

import {
  isPidAlive,
  claimUpdateMarker,
  handoffUpdateMarker,
  markerPath,
  readLiveUpdateMarker,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))

  return dir
}

function writeMarker(home, pid, startedAtSec) {
  fs.writeFileSync(markerPath(home), `${pid}\n${startedAtSec}`)
}

const ALIVE: typeof process.kill = () => true // injected kill that "succeeds" => pid alive

const DEAD: typeof process.kill = () => {
  const err = new Error('no such process')

  ;(err as any).code = 'ESRCH'
  throw err
}

test('absent marker => no live update', () => {
  const home = tmpHome('absent')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
})

test('operation guard blocks backend gate while canonical marker is absent', () => {
  const home = tmpHome('guard-only')
  const guard = `${markerPath(home)}.lock`
  fs.mkdirSync(guard)
  fs.writeFileSync(path.join(guard, 'owner'), `${process.pid}\n`)

  const blocked = readLiveUpdateMarker(home, { kill: ALIVE })

  assert.ok(blocked)
  assert.equal(blocked.pid, 0)
})

test('operation guard appearing during marker read still blocks backend gate', () => {
  const home = tmpHome('guard-race')
  const file = markerPath(home)
  const guard = `${file}.lock`
  const read = vi.spyOn(fs, 'readFileSync').mockImplementation(candidate => {
    assert.equal(candidate, file)
    fs.writeFileSync(guard, `${process.pid}\n`)
    const err = new Error('marker temporarily absent')

    ;(err as NodeJS.ErrnoException).code = 'ENOENT'
    throw err
  })

  try {
    assert.ok(readLiveUpdateMarker(home, { kill: ALIVE }))
  } finally {
    read.mockRestore()
  }
})

test('unknown or corrupt operation guard blocks backend gate', () => {
  const home = tmpHome('guard-corrupt')
  fs.mkdirSync(`${markerPath(home)}.lock`)

  assert.ok(readLiveUpdateMarker(home, { kill: DEAD }))
})

test('operation guard reclaim preserves an ABA replacement published after detach', () => {
  const home = tmpHome('guard-aba')
  const guard = `${markerPath(home)}.lock`
  fs.writeFileSync(guard, '999999\n')
  const replacement = `${process.pid}\n`
  const renameSync = fs.renameSync.bind(fs)
  const rename = vi.spyOn(fs, 'renameSync').mockImplementation((source, destination) => {
    renameSync(source, destination)
    if (source === guard) fs.writeFileSync(guard, replacement)
  })

  try {
    assert.equal(claimUpdateMarker(home, 4242, { kill: pid => (pid === process.pid ? true : DEAD(pid, 0)) }), null)
    assert.equal(
      rename.mock.calls.filter((call: any[]) => call[0] === guard).length,
      1,
      'the replacement guard must never be detached'
    )
    assert.equal(fs.readFileSync(guard, 'ascii'), replacement)
  } finally {
    rename.mockRestore()
  }
})

test('operation guard restore failure keeps the canonical namespace blocked', () => {
  const home = tmpHome('guard-restore-failure')
  const guard = `${markerPath(home)}.lock`
  fs.writeFileSync(guard, 'malformed\n')
  const linkSync = fs.linkSync.bind(fs)
  const mkdirSync = fs.mkdirSync.bind(fs)
  const link = vi.spyOn(fs, 'linkSync').mockImplementation((source, destination) => {
    if (destination === guard && String(source).includes('.stale-')) {
      const err = new Error('simulated ACL denial')
      ;(err as NodeJS.ErrnoException).code = 'EACCES'
      throw err
    }
    return linkSync(source, destination)
  })
  const mkdir = vi.spyOn(fs, 'mkdirSync').mockImplementation((candidate, options) => {
    if (candidate === guard) {
      const err = new Error('simulated namespace denial')
      ;(err as NodeJS.ErrnoException).code = 'EACCES'
      throw err
    }
    return mkdirSync(candidate, options as any)
  })

  try {
    assert.equal(claimUpdateMarker(home, 4242, { kill: DEAD }), null)
    assert.equal(fs.existsSync(guard), false)
    assert.ok(fs.readdirSync(path.dirname(guard)).some(name => name.includes(`${path.basename(guard)}.stale-`)))
    assert.equal(claimUpdateMarker(home, 4242, { kill: DEAD }), null, 'orphan quarantine must block reacquire')
  } finally {
    link.mockRestore()
    mkdir.mockRestore()
  }
})

test('live pid within age ceiling => live update reported', () => {
  const home = tmpHome('live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5) // 5s old
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'a fresh, alive marker is a live update')
  assert.equal(res.pid, 4242)
  assert.ok(res.ageMs >= 0 && res.ageMs < 10_000)
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('dead pid => no live update and marker is pruned', () => {
  const home = tmpHome('dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(readLiveUpdateMarker(home, { kill: DEAD }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'a dead-pid marker self-heals (deleted)')
})

test('expired marker with a live pid remains authoritative', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  const owner = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(owner)
  assert.equal(owner.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'age alone must never evict a live updater')
})

test('claim is atomic no-replace and handoff requires expected owner and token', () => {
  const home = tmpHome('claim-cas')
  const first = claimUpdateMarker(home, 1010, { now: () => 1_000_000_000_000, kill: ALIVE })
  assert.ok(first, 'the first claimant atomically publishes')

  assert.equal(
    claimUpdateMarker(home, 2020, { now: () => 1_000_000_001_000, kill: ALIVE }),
    null,
    'a second claimant must not replace the existing claim'
  )
  assert.equal(handoffUpdateMarker(home, 1010, 'wrong-token', 2020), false)
  assert.equal(handoffUpdateMarker(home, 9999, first.token, 2020), false)
  assert.equal(handoffUpdateMarker(home, 1010, first.token, 2020), true)

  const lines = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(lines[0], 10), 2020)
  assert.equal(lines[2], first.token)
})

test('malformed marker fails closed', () => {
  const home = tmpHome('malformed')
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')
  assert.ok(readLiveUpdateMarker(home, { kill: ALIVE }))
  assert.ok(fs.existsSync(markerPath(home)))
})

test('isPidAlive: own pid is alive, impossible pid is dead', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(-1), false)
  assert.equal(isPidAlive(0), false)
  assert.equal(isPidAlive(NaN), false)
})

test('isPidAlive: EPERM counts as alive (process owned by another user)', () => {
  const eperm = () => {
    const err = new Error('operation not permitted')

    ;(err as any).code = 'EPERM'
    throw err
  }

  assert.equal(isPidAlive(4242, eperm), true)
})

test('isPidAlive: unknown probe errors fail closed', () => {
  const unknown = () => {
    const err = new Error('probe failed')

    ;(err as any).code = 'EIO'
    throw err
  }

  assert.equal(isPidAlive(4242, unknown), true)
})

test('writeUpdateMarker writes a marker that readLiveUpdateMarker accepts', () => {
  const home = tmpHome('write')
  const now = 1_000_000_000_000
  writeUpdateMarker(home, 4242, { now: () => now })
  // The marker should be readable and report the same pid.
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'marker written by writeUpdateMarker should be detected as live')
  assert.equal(res.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'marker file should exist after write')
})

test('writeUpdateMarker preserves a live holder age across pid hand-off', () => {
  const home = tmpHome('write-handoff-age')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  const claim = claimUpdateMarker(home, 1010, { now: () => now, startedAt })
  assert.ok(claim)
  const oldHandle = fs.openSync(markerPath(home), 'r')

  try {
    assert.equal(handoffUpdateMarker(home, 1010, claim.token, 2020), true)

    const detachedBody = Buffer.alloc(128)
    const detachedLength = fs.readSync(oldHandle, detachedBody, 0, detachedBody.length, 0)

    assert.match(
      detachedBody.subarray(0, detachedLength).toString('utf8'),
      /^1010\n/,
      'handoff publishes a new inode; an already-open old claim is never rewritten in place'
    )
  } finally {
    fs.closeSync(oldHandle)
  }

  const [pidLine, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(pidLine, 10), 2020, 'the hand-off records the new owner')
  assert.equal(Number.parseInt(startedLine, 10), startedAt, 'the holder age must not restart during hand-off')
})

test('writeUpdateMarker uses the acquisition time passed to a detached script', () => {
  const home = tmpHome('write-script-acquired-at')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeUpdateMarker(home, 2020, { now: () => now, startedAt })

  const [, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(startedLine, 10), startedAt)
})

test('writeUpdateMarker is best-effort (no throw on bad path)', () => {
  // A non-existent directory should not throw.
  const badHome = path.join(os.tmpdir(), 'hermes-marker-nonexistent-' + Date.now())
  assert.doesNotThrow(() => writeUpdateMarker(badHome, 4242))
})

test('writeUpdateMarker + dead pid => self-heals on read', () => {
  const home = tmpHome('write-dead')
  writeUpdateMarker(home, 999999, { now: () => Date.now() })
  // PID 999999 is almost certainly not alive.
  const res = readLiveUpdateMarker(home, { kill: DEAD })
  assert.equal(res, null, 'a dead-pid marker from writeUpdateMarker self-heals')
  assert.ok(!fs.existsSync(markerPath(home)), 'marker file is pruned')
})

// ---------------------------------------------------------------------------
// updateHandoffConflict (#75778)
//
// A retried "Update" click must not spawn a second updater over a still-live
// one. The diagnostic helper and atomic claim must agree that the original
// updater remains authoritative.
// ---------------------------------------------------------------------------

test('no marker => hand-off is not blocked', () => {
  const home = tmpHome('conflict-none')
  assert.equal(updateHandoffConflict(home, { kill: ALIVE }), null)
})

test('a different live updater already owns the marker => hand-off is blocked', () => {
  const home = tmpHome('conflict-live')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 6) // 6s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict, 'a live foreign updater must block a new hand-off')
  assert.equal(conflict.pid, 1010)
  assert.match(conflict.message, /already running/)
  assert.match(conflict.message, /PID 1010/)
  assert.match(conflict.message, /6s/)
})

test('a dead-pid marker does not block a hand-off (self-heals)', () => {
  const home = tmpHome('conflict-dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(updateHandoffConflict(home, { kill: DEAD }), null)
})

test('an old marker still blocks a hand-off while its pid is alive', () => {
  const home = tmpHome('conflict-expired')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.ok(updateHandoffConflict(home, { kill: ALIVE, now: () => now }))
})

test('minutes-scale elapsed time is formatted as "Nm Ss"', () => {
  const home = tmpHome('conflict-minutes')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 125) // 2m 5s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict)
  assert.match(conflict.message, /2m 5s/)
})
