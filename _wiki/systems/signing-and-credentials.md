# Signing and Credentials

## Current State

METTLE issues credentials only through server-owned signers. Quick sessions earn Bronze or Silver. Authenticated suite sessions earn Bronze through Platinum only when every suite in the contiguous tier range passes. Single, partial, failed, or LLM-only results cannot mint a tier (`main.py`; `mettle/vcp.py`).

Caller-supplied VCP governance metadata remains `source_verified=false`, all operational governance flags remain false, and no attestation signature is created (`mettle/router.py:487-561`). Digest allowlists and environment switches cannot promote it (`tests/test_attestation_security.py`).

The legacy badge verification endpoint exists only to validate or revoke historical JWTs. Current session result paths clear badge fields and issue nothing (`main.py:748-760`; `main.py`, Historical Badge Verification Endpoints).

## Future Requirement

A future credential needs a separately typed trusted-execution proof and authoritative verifier. It cannot be enabled by suite scores, model judgments, boolean switches, environment flags, or token digests (`README.md`, Assurance Boundary).
