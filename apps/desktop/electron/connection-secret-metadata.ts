type StoredSecret = unknown

type DecryptSecret = (value: StoredSecret) => string

function storedSecretIsSet(value: StoredSecret): boolean {
  return Boolean(value && typeof value === 'object' && 'value' in value && (value as { value?: unknown }).value)
}

function resolveRemoteTokenMetadata(mode: string, storedSecret: StoredSecret, decrypt: DecryptSecret) {
  if (mode === 'local') {
    return {
      remoteToken: '',
      remoteTokenSet: storedSecretIsSet(storedSecret)
    }
  }

  const remoteToken = decrypt(storedSecret)

  return {
    remoteToken,
    remoteTokenSet: Boolean(remoteToken)
  }
}

function registryTokenMetadata(storedSecret: StoredSecret, _decrypt: DecryptSecret) {
  return {
    tokenSet: storedSecretIsSet(storedSecret),
    tokenPreview: '' as const
  }
}

export { registryTokenMetadata, resolveRemoteTokenMetadata }
