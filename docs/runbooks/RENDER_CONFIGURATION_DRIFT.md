# Render configuration drift

METTLE keeps its working production services under direct Render ownership so
their public identities and URLs remain stable. `render.yaml` is the reviewed
provisioning contract. `deploy/render-production.json` binds each declaration
to its existing service and workspace.

## Read-only check

Read the API token from the macOS Keychain at the point of use. The checker
performs only Render `GET` requests. It compares service identity, workspace,
repository, branch, deployment mode, runtime, plan, region, instance count,
health path, build and start configuration, and every environment key. A
separate encrypted GitHub secret supplies approved SHA-256 fingerprints for
provider secret values and managed secret-file contents. The checker also
requires the exact approved secret-file name set. The receipt reports only
match, mismatch, or missing; it never emits a secret, file content, or digest.

```bash
security find-generic-password -s render -a production -w \
  | python3 scripts/check_render_drift.py --token-stdin \
      --output output/evidence/render-drift.json
```

Exit `0` means the provider matches the reviewed contract. Exit `1` means
drift. Exit `2` means the check could not establish a verdict. Any nonzero
result blocks deployment acceptance.

`RENDER_SECRET_FINGERPRINTS` must contain the exact approved secret-key set and
secret-file fingerprint set described by `scripts/check_render_drift.py`. A
nonempty substitute, omitted holder file, renamed file, or unexpected extra file
does not pass.

## Proxy identity boundary

The API start command enables Uvicorn proxy handling and explicitly sets
`METTLE_FORWARDED_ALLOW_IPS=10.0.0.0/8`. Render's ingress reaches the service
from its private `10/8` network. Uvicorn therefore reduces the forwarded chain
to the rightmost untrusted public hop. For `mettle.sh`, that hop is Cloudflare,
so `CloudflareClientIPMiddleware` restores the single `CF-Connecting-IP` value
only when the public hop belongs to Cloudflare's authoritative IPv4 or IPv6
networks. A direct non-Cloudflare caller cannot make the application trust that
header. Review the pinned networks against `https://www.cloudflare.com/ips-v4`
and `/ips-v6` whenever Cloudflare announces a range change.

Never restore wildcard proxy trust. A deployed `v0.4.2` probe demonstrated that
it lets callers rotate `X-Forwarded-For` values. A `v0.4.3` probe then showed
that stopping at the Cloudflare hop makes one caller appear as rotating edge
addresses and still defeats a per-caller budget. The two-stage Render plus
Cloudflare resolution is required. If either provider changes, replace its
trust boundary and repeat the 61-request live spoof-resistance probe before
acceptance.

## Reconciliation

1. Preserve all existing secret values.
2. Review each reported nonsecret mismatch against `render.yaml`.
3. Change only the intended provider field.
4. Rerun the check and retain its JSON receipt.
5. Deploy the exact source revision and repeat public API and MCP acceptance.

Do not create replacement services or move any bound service between workspaces
without an approved URL continuity and rollback plan.

Working if: a deliberate fixture mismatch fails tests, live production yields
`result: match`, and no provider secret value appears in output or logs.
