# Security Scan Reconciliation, 2026-07-13

This document records the remediation boundary applied after deep scan `e10ef725-9526-4d0e-9dee-4d24a2c8c480` and the subsequent residual-risk closure.

## Credential Boundary Closure

The initial remediation disabled credential issuance to close unsafe source-to-signer paths. Product review then established that METTLE is intentionally a probabilistic reverse CAPTCHA, comparable to conventional CAPTCHA rather than an identity or trusted-execution proof. Issuance was restored behind these explicit boundaries:

* `verified` records whether the configured METTLE challenge policy passed;
* public quick sessions issue one stable, signed, time-limited Bronze or Silver badge;
* public `entity_id` values are marked `self_asserted` inside the badge;
* authenticated suite credentials require complete contiguous ranges: Bronze 1 through 5, Silver 1 through 7, Gold 1 through 9, and Platinum 1 through 11;
* Suite 12 is supplemental and never raises a tier;
* partial, cherry-picked, failed, or LLM-only results cannot reach the signer;
* callers cannot provide signing functions or issuer keys;
* MCP and CLI auto-solving surfaces remain removed;
* local CLI results remain unsigned and cannot impersonate server credentials;
* raw VCP governance metadata remains unverified and cannot promote governance claims or tiers;
* digest allowlists and environment flags cannot promote caller metadata.

This preserves the reverse-CAPTCHA product while closing credential confusion, self-signing, automatic solving, and metadata-promotion paths.

## Other Remediated Boundaries

The preceding remediation also established complete challenge-set enforcement, authoritative timing, independent session bearer tokens, bounded payloads and in-memory capacity, fail-closed persistence behavior, atomic Redis transitions and quota reservation, answer separation, feedback non-disclosure, LLM role separation and bounded scores, webhook destination controls, admin-key header authentication, stable historical badge identifiers, full-envelope historical signature binding, and regression coverage for those controls. It previously attempted to bind a one-step operator commitment; the 2026-08-14 remediation retired that field because a self-supplied signature could not establish the claimed operator identity or contact.

## Final Engineering Closure

The follow-up engineering pass also closed these operational issues:

* badge verification now accepts credentials through `POST /api/badge/verify`, keeping bearer tokens out of request URLs; the transitional URL-token endpoint described by this July record was removed on 2026-08-14;
* dependency floors were raised above the audited vulnerable releases, including the FastAPI and Starlette stack, cryptography, PyJWT, IDNA, Click, and python-dotenv;
* the current Python environment and declared requirement sets pass `pip-audit` with no known vulnerabilities;
* committed local transcript backups were removed from the product tree and added to `.gitignore` after secret scanning found credential-shaped material in them;
* the active tracked and untracked source tree passes the secret scanner, and every retained baseline item is classified as a reviewed false positive.

The removed transcript files remain present in earlier Git history until an explicitly authorized history rewrite is completed. Any live credential that appeared in those historical transcripts must be rotated by its owner. These two actions are operational incident response, not application-code changes.

## Remaining Research Limitations

These are residual limitations of a probabilistic gate:

1. Behavioral heuristics can be imitated, reverse engineered, relayed, or solved by unintended respondents.
2. LLM semantic judgment remains probabilistic and prompt-injection-sensitive.
3. VCP governance claims are self-asserted unless an external provenance system verifies them.
4. METTLE does not accept an operator commitment or authenticate an operator identity or contact. Operator evidence requires an external provenance system outside this protocol.

The credential therefore asserts a bounded fact: a METTLE session passed at the stated tier, under the stated policy, at the stated time. Relying services decide what access that result permits and should add controls proportionate to the value behind the gate.
