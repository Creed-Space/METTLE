# METTLE retry and idempotency contract

METTLE does not accept an `Idempotency-Key` header. Retry safety therefore comes
from endpoint semantics, session tokens, challenge identifiers, Redis locks, and
single-winner credential caching.

| Operation | Duplicate behavior | Safe client action |
|---|---|---|
| Start quick session | Every accepted request creates a new session and token. | Do not retry blindly after an ambiguous network failure. Start again only if an extra session is acceptable. |
| Start authenticated session | Every accepted request creates a new session and reserves quota. Failed or cancelled construction releases its reservation. | Query or retain the returned session ID. An ambiguous creation is not idempotent. |
| Submit quick answer | The current challenge ID and session token must match. A committed answer advances state; replay of the old challenge is rejected. | After an ambiguous response, read session state before resubmitting. |
| Submit authenticated suite or round | Session ownership, expected suite or round, and a distributed session lock serialize mutation. Completed work cannot be applied twice. | Read session status or feedback before retry. |
| Read result | Read-only. | Retry freely with the same authorization. |
| Request signed authenticated credential | The first signed envelope wins through Redis `SET NX`; concurrent and later result reads return the same credential. Unsigned evidence is not cached as issuance. | Retry result reads freely. |
| Read quick result and badge | The first issued quick badge is persisted with session progress and reused. | Retry the result read with the same session token. |
| Verify badge or credential | Read-only except for status metrics. | Retry freely. |
| Create presentation challenge | Every accepted request creates a fresh nonce. | Do not assume two responses identify the same presentation. |
| Verify presentation | A successful call consumes the challenge exactly once. Replays fail. A failed holder signature does not consume a valid challenge. | Retry only after checking whether the first call succeeded. If uncertain, request a fresh challenge. |
| Revoke badge or Presence JTI | Repeating a known revocation does not restore authority. The response may identify it as already revoked. | Retry with the same target and evidence after an ambiguous transport failure. |
| Register webhook | The authenticated entity ID is the key. A repeat replaces URL, events, secret, and creation time. | Send the complete desired configuration on every retry. |
| Unregister webhook | First success removes it; a later call returns not found. | Treat not found after an ambiguous first response as consistent with success. |
| Register API key | Every success creates a distinct secret. | Never retry blindly. Revoke unintended duplicates through the admin path. |
| Rotate Ed25519 key | Configuration is declarative, but a changed private key under an existing ID is invalid. | Use a new key ID and verify overlap before removing the old public key. |
| Render deployment | `render.yaml` `autoDeploy` is the sole repository deployment authority. | Correlate one main commit to one provider deploy. Do not add a second webhook trigger. |

Working if: duplicate answer and presentation tests fail closed, concurrent
credential reads return byte-equivalent envelopes, cancellation restores quota,
and operational retries never create untracked keys, sessions, or deploys.
