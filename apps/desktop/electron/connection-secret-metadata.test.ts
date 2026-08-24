import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registryTokenMetadata, resolveRemoteTokenMetadata } from './connection-secret-metadata'

const stored = { encoding: 'safeStorage', value: 'opaque-ciphertext-bytes' }

test('local connection metadata preserves the set flag without decrypting a dormant remote token', () => {
  let decryptCalls = 0

  const metadata = resolveRemoteTokenMetadata('local', stored, () => {
    decryptCalls += 1

    return 'plaintext-secret'
  })

  assert.equal(decryptCalls, 0)
  assert.deepEqual(metadata, { remoteToken: '', remoteTokenSet: true })
})

test('remote connection metadata decrypts the selected token', () => {
  let decryptCalls = 0

  const metadata = resolveRemoteTokenMetadata('remote', stored, value => {
    decryptCalls += 1
    assert.equal(value, stored)

    return 'plaintext-secret'
  })

  assert.equal(decryptCalls, 1)
  assert.deepEqual(metadata, { remoteToken: 'plaintext-secret', remoteTokenSet: true })
})

test('remote connection metadata treats invalid ciphertext as unset', () => {
  let decryptCalls = 0

  const metadata = resolveRemoteTokenMetadata('remote', stored, () => {
    decryptCalls += 1

    return ''
  })

  assert.equal(decryptCalls, 1)
  assert.deepEqual(metadata, { remoteToken: '', remoteTokenSet: false })
})

test('registry metadata neither decrypts nor exposes token bytes or a plaintext suffix', () => {
  let decryptCalls = 0

  const metadata = registryTokenMetadata(stored, () => {
    decryptCalls += 1

    return 'plaintext-secret'
  })

  assert.equal(decryptCalls, 0)
  assert.deepEqual(metadata, { tokenSet: true, tokenPreview: '' })
  assert.equal(JSON.stringify(metadata).includes(stored.value), false)
  assert.equal(JSON.stringify(metadata).includes('plaintext-secret'), false)
})
