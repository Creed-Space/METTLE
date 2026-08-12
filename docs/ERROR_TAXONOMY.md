# METTLE error taxonomy

HTTP failures preserve the historical `detail` field and add a stable `code`.
Clients should branch on status and code, then display detail as bounded context.
They must not parse exception class names or English substrings.

| Status | Code | Meaning |
|---|---|---|
| 400 | `invalid_request` | Request state or value cannot be accepted. |
| 401 | `authentication_required` | Required credential is missing or invalid. |
| 403 | `forbidden` | Authenticated authority does not own or permit the operation. |
| 404 | `not_found` | Resource is absent, expired, or intentionally concealed. |
| 405 | `method_not_allowed` | Route exists but the HTTP method is unsupported. |
| 409 | `conflict` | Current state conflicts with the requested transition. |
| 413 | `request_too_large` | Bounded input limit was exceeded. |
| 422 | `validation_error` | Schema validation failed. `detail` is a bounded list. |
| 429 | `rate_limited` | A request or authorization attempt limit was reached. |
| 500 | `internal_error` | Unexpected server failure. Internal cause is not returned. |
| 502 | `upstream_error` | An upstream returned an invalid failure. |
| 503 | `dependency_unavailable` | Required storage, signer, or service is unavailable. |
| 504 | `upstream_timeout` | Required upstream did not complete within its bound. |

Unknown HTTP status values use `request_error`. The `X-Request-ID` response
header correlates client and structured server evidence. Incoming request IDs are
bounded and sanitized. Logs record event category, status, duration, and internal
exception type where needed, while redacting secrets and participant content.

Broad catches are allowed only at a boundary that deliberately converts an
internal failure into a fail-closed response or completes cleanup. The code must
retain the cause with exception chaining or structured logging, and a test must
exercise the boundary.

Working if: representative 404, validation, rate-limit, and storage failures
return stable codes; request IDs correlate logs; and no client body contains a
stack trace, database URL, secret, raw answer, token, or internal exception text.
