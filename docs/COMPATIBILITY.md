# METTLE compatibility contract

## Supported consumers

The Python package declares Python 3.10 and later. CI runs release-authoritative
tests on Python 3.11 and clean wheel smokes on Python 3.10 through 3.14. A green
local Python 3.14 run does not substitute for the hosted matrix.

JavaScript and Rust are verifier examples, not published SDKs. Their acceptance
contract is the same signed fixture corpus used by Python.

## Credential fixtures

`fixtures/credentials/v1.json` contains five deterministic, public cases:

1. valid Bronze credential;
2. valid Unicode metadata;
3. signed metadata tampering;
4. expiry at the exclusive boundary;
5. an explicit unsupported suite policy.

The deterministic fixture private seed exists only inside the generator and is
labelled test-only. It must never be configured as an issuer key.

Run:

```bash
python3 scripts/generate_credential_fixtures.py
node examples/verify_credential_fixture.js
```

For Rust, create an isolated Cargo project using the dependency comment in
`examples/verify_credential_fixture.rs`, then pass the fixture path. The release
gate compiles and runs this example in a clean temporary project. The repository
does not commit Cargo build output.

## OpenAPI

`docs/openapi-v1.json` is the public snapshot. The compatibility checker rejects
removed paths, methods, response statuses, schemas, properties, and newly required
request fields. It also rejects type changes. Additive optional fields remain
compatible.

Run:

```bash
python3 scripts/check_openapi_compatibility.py
python3 scripts/smoke_openapi_client.py
```

The second command generates a bounded client from snapshot operation IDs and
smokes safe GET and POST operations against the live ASGI application. Updating
the snapshot is a reviewed protocol action, never an automatic response to a
failed compatibility check.

## Compatibility change procedure

1. Establish whether a change is additive, deprecated, or breaking.
2. Increment credential schema or suite policy when interpretation changes.
3. Add positive and negative fixture cases before implementation is accepted.
4. Preserve verify-only keys and historical decoding for the documented window.
5. Update examples in Python, JavaScript, and Rust together.
6. Record the change and limitations in the release manifest.

Working if: all language consumers reach identical verdicts on every fixture,
the schema-derived client reaches the current application, and an intentional
break cannot ship through an unreviewed snapshot overwrite.
