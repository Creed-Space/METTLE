# Vault holder deployment

`render.yaml` defines a one-worker private service. It needs a persistent PostgreSQL database, a TLS-protected Vault Transit endpoint, and six Render secret files. Do not point it at Vault development mode.

## Required secret files

| File | Purpose |
|---|---|
| `mettle-holder-vault-ca.pem` | Private Vault CA certificate used for explicit TLS verification |
| `mettle-holder-vault-public-key.pem` | Public key for the explicitly pinned Vault key version |
| `mettle-holder-vault-token` | Short-lived or renewable runtime token with only `transit/sign/mettle-holder` update permission |
| `mettle-holder-policy.json` | Issuer keyrings, audiences, and record budgets |
| `mettle-holder-control-token` | Bearer token used by the calling METTLE service |
| `mettle-holder-state-hmac-key` | At least 32 bytes of random material for authenticating PostgreSQL snapshots |

The Vault token policy is in `../vault/mettle-holder-sign.hcl`. Rotation uses a separate administrative identity with `../vault/mettle-holder-rotate.hcl`.

The runtime token must be renewable. The holder renews it before the first signing operation and halfway through each returned lease, capped at one renewal per hour. Renewal and signing both use the pinned CA, reject redirects, ignore proxy environment variables, bound response bodies, and fail closed on malformed replies.

## Singleton deployment fence

The service must retain the one-gigabyte `mettle-holder-singleton-fence` disk declared in `render.yaml`. Render normally starts a replacement private-service instance before stopping the live instance. That overlap conflicts with the holder's PostgreSQL advisory lock and correctly prevents the replacement from starting. Attaching a disk selects Render's supported stop-first deployment sequence, so the old process releases its lock before the replacement starts.

The disk is deployment fencing only. Do not store Vault keys, tokens, holder snapshots, or other security state on it. Vault and PostgreSQL remain authoritative. A working deployment shows the old holder exiting before the replacement acquires the same holder ID, with no `Another holder instance owns the persistence lock` error.

## Key version pinning

`METTLE_HOLDER_VAULT_KEY_VERSION` is mandatory. The service sends that version with every Vault signing request and rejects a response from any other version. The public key file must contain the public key for the same version.

This prevents a Vault rotation from silently changing the holder identity. A stored holder state is bound to its public-key fingerprint and will fail closed if it is opened with another key.

## Automated rotation sequence

1. Rotate the Transit key with the separate rotation identity. Keep the current service pinned to its existing version.
2. Run the persistent soak against the rotated Vault. The existing holder ID must still sign with the old pinned version after both PostgreSQL and Vault restarts.
3. Start a new private service identity with a new `METTLE_HOLDER_ID`, the new public key file, and the new explicit key version.
4. Route new sessions and credentials to the new service only after its health, authorization, replay, persistence, split-brain, and concurrency probes pass.
5. Keep the old service available for credentials bound to its public key until their permitted presentation lifetime ends. Then revoke its runtime token and retire its service.

Never reuse a PostgreSQL holder ID across public-key versions. Never replace the public key file in place while retaining the same holder ID.

The local proof command is `scripts/testing/run_holder_service_soak.py`; its caller supplies a Vault and PostgreSQL instance. A passing proof verifies concurrent Vault signatures at one pinned Transit key version, issuer-key overlap, persistence close and reopen, stable idempotent replay, conflicting replay rejection, singleton locking, and concurrent requests. It does not restart Vault or PostgreSQL and does not rotate the Transit key. Those remain separate operational drills against the release candidate.
