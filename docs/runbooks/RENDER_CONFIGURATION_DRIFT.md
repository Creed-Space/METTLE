# Render configuration drift

METTLE keeps its working production services under direct Render ownership so
their public identities and URLs remain stable. `render.yaml` is the reviewed
provisioning contract. `deploy/render-production.json` binds each declaration
to its existing service and workspace.

## Read-only check

Read the API token from the macOS Keychain at the point of use. The checker
performs only Render `GET` requests. It compares service identity, workspace,
repository, branch, deployment mode, runtime, plan, region, instance count,
health path, build and start configuration, and every environment key. Secret
values are reduced to `present` or `missing` before output.

```bash
security find-generic-password -s render -a production -w \
  | python3 scripts/check_render_drift.py --token-stdin \
      --output output/evidence/render-drift.json
```

Exit `0` means the provider matches the reviewed contract. Exit `1` means
drift. Exit `2` means the check could not establish a verdict. Any nonzero
result blocks deployment acceptance.

## Reconciliation

1. Preserve all existing secret values.
2. Review each reported nonsecret mismatch against `render.yaml`.
3. Change only the intended provider field.
4. Rerun the check and retain its JSON receipt.
5. Deploy the exact source revision and repeat public API and MCP acceptance.

Do not create replacement services or move either service between workspaces
without an approved URL continuity and rollback plan.

Working if: a deliberate fixture mismatch fails tests, live production yields
`result: match`, and no provider secret value appears in output or logs.
