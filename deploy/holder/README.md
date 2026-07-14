# Vault holder deployment

`render.yaml` defines a one-worker private service. It needs a persistent PostgreSQL database, a production Vault Transit endpoint, and five Render secret files. Do not point it at Vault development mode.

## Required secret files

| File | Purpose |
|---|---|
| `mettle-holder-vault-public-key.pem` | Public key for the explicitly pinned Vault key version |
| `mettle-holder-vault-token` | Short-lived or renewable runtime token with only `transit/sign/mettle-holder` update permission |
| `mettle-holder-policy.json` | Issuer keyrings, audiences, and record budgets |
| `mettle-holder-control-token` | Bearer token used by the calling METTLE service |
| `mettle-holder-state-hmac-key` | At least 32 bytes of random material for authenticating PostgreSQL snapshots |

The Vault token policy is in `../vault/mettle-holder-sign.hcl`. Rotation uses a separate administrative identity with `../vault/mettle-holder-rotate.hcl`.

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

The local proof command is `scripts/testing/run_holder_service_soak.py`; the repository integration harness supplies an external Vault and PostgreSQL instance. A passing proof records versions `[1, 1, 1, 2]`: before restart, after Vault restart, old-version continuity after rotation, and fresh-version cutover.
