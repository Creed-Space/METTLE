# METTLE: Production Deployment & Remaining Hardening

*Continuation prompt for fresh context - 2026-02-04*

---

## Current State

METTLE v0.2 has completed 3 rounds of security red-teaming. All critical and high-severity vulnerabilities are fixed. **161 tests pass.**

### Security Fixes Applied
- Badge forgery prevention (requires SECRET_KEY)
- Admin authentication on sensitive endpoints
- SSRF protection with DNS rebinding defense
- Race condition fixes (atomic challenge consumption)
- Timing attack mitigations (constant-time comparison)
- Brute force protection (exponential backoff)
- Human bypass prevention (reduced timing to 2-3.5s)
- Answer leakage prevention
- HTTPS enforcement for production webhooks

**Audit conclusion: PRODUCTION READY with proper configuration.**

---

## Task 1: Production Environment Configuration (REQUIRED)

### Render Environment Variables

Set these in Render dashboard for the METTLE service:

```
METTLE_SECRET_KEY=<generate 32+ byte random secret>
METTLE_ADMIN_API_KEY=<generate strong random key>
METTLE_ENVIRONMENT=production
```

**Generate secrets:**
```bash
# Generate SECRET_KEY (64 hex chars = 32 bytes)
python3 -c "import secrets; print(secrets.token_hex(32))"

# Generate ADMIN_API_KEY
python3 -c "import secrets; print(f'mtl_admin_{secrets.token_hex(24)}')"
```

### Verify Production Config

After deployment, verify:
```bash
# Should return signed JWT badge (starts with eyJ)
curl -X POST https://mettle.sh/api/session/start -H "Content-Type: application/json" -d '{"difficulty":"basic"}'

# Admin endpoints should require auth
curl https://mettle.sh/api/badge/revocations
# Expected: 401 Unauthorized
```

---

## Task 2: Enable Database Persistence (RECOMMENDED)

Currently using in-memory storage. For production reliability:

### Option A: PostgreSQL on Render

1. Create PostgreSQL database on Render
2. Set environment variables:
```
METTLE_USE_DATABASE=true
METTLE_DATABASE_URL=<postgresql connection string from Render>
```

### Option B: SQLite (simpler, less durable)

```
METTLE_USE_DATABASE=true
METTLE_DATABASE_URL=sqlite:///data/mettle.db
```

**Note:** Requires persistent disk on Render.

---

## Task 3: Add Memory Limits for In-Memory Stores (LOW PRIORITY)

Prevent DoS via unbounded memory growth. Add to `main.py`:

### Files to Modify

**main.py** - Add constants near top:
```python
# Memory limits for in-memory stores
MAX_VERIFICATION_GRAPH_SIZE = 10000
MAX_REVOCATION_AUDIT_SIZE = 10000
MAX_WEBHOOKS = 1000
MAX_API_KEYS = 10000
MAX_SESSIONS = 5000
```

### Stores to Cap

| Store | Location | Current Limit | Recommended |
|-------|----------|---------------|-------------|
| `sessions` | line 83 | None | 5000 |
| `challenges` | line 84 | None | 10000 |
| `verification_graph` | line 85 | None | 10000 |
| `verification_timestamps` | line 86 | 1000 | OK |
| `revoked_badges` | line 87 | None | 10000 |
| `revocation_audit` | line 88 | None | 10000 |
| `api_keys` | line 89 | None | 10000 |
| `webhooks` | line 90 | None | 1000 |
| `_admin_auth_failures` | line 261 | None | 10000 |

### Implementation Pattern

```python
def add_with_limit(store: dict, key: str, value: Any, max_size: int) -> None:
    """Add to dict with LRU-style eviction when full."""
    if len(store) >= max_size:
        # Remove oldest (first) item
        oldest_key = next(iter(store))
        del store[oldest_key]
    store[key] = value
```

---

## Task 4: Add Request IDs to Error Responses (LOW PRIORITY)

For debugging production issues.

### Implementation

**main.py** - Add middleware:
```python
import uuid

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

Update HTTPException raises to include request_id in detail.

---

## Task 5: Audit Webhook Registrations (LOW PRIORITY)

Add logging for webhook registration/deletion.

### Implementation

In `register_webhook()` and `unregister_webhook()`:
```python
logger.info(
    "webhook_registered",
    entity_id=body.entity_id,
    url=body.url[:50],
    events=body.events,
    ip_address=get_remote_address(request),
)
```

---

## Task 6: Monitoring Setup (RECOMMENDED)

### Key Metrics to Track

1. **Failed admin auth attempts** - Alert on spike
2. **Collusion flags** - Monitor `collusion_flagged` log events
3. **Badge revocations** - Unusual patterns
4. **Session creation rate** - DoS indicator
5. **Verification pass rate** - Quality metric

### Suggested Alerts

| Metric | Threshold | Action |
|--------|-----------|--------|
| Failed admin auth | >10/min | Investigate |
| Collusion flags | >50/hour | Review |
| Session rate | >100/min | Rate limit |
| Memory usage | >80% | Scale up |

---

## Key Files Reference

```
mettle-api/
├── main.py                 # Main FastAPI app (~1700 lines)
│   ├── Lines 260-320      # Admin auth + brute force protection
│   ├── Lines 770-870      # Challenge answer flow
│   ├── Lines 1030-1100    # Badge generation
│   ├── Lines 1150-1250    # Badge revocation (admin auth)
│   ├── Lines 1340-1380    # Collusion stats
│   ├── Lines 1450-1530    # Webhook delivery (DNS rebinding protection)
│   └── Lines 1540-1600    # Webhook URL validation (SSRF)
├── config.py               # Settings with env var mapping
├── mettle/
│   ├── challenger.py       # Challenge generation (timing limits)
│   ├── verifier.py         # Answer verification (no answer leakage)
│   └── models.py           # Pydantic models
├── static/
│   ├── index.html          # Landing page
│   ├── docs.html           # Documentation
│   └── pricing.html        # Pricing
└── tests/                  # 161 tests
```

---

## Validation Checklist

Before marking production-ready:

- [ ] `METTLE_SECRET_KEY` set (verify: badges are JWTs)
- [ ] `METTLE_ADMIN_API_KEY` set (verify: admin endpoints return 401)
- [ ] `METTLE_ENVIRONMENT=production` set
- [ ] All 161 tests pass
- [ ] Badge verification works: `curl https://mettle.sh/api/badge/verify/<token>`
- [ ] HTTPS enforced for webhook registration
- [ ] Monitoring configured (optional but recommended)

---

## Quick Commands

```bash
# Run tests
cd /path/to/mettle-api
python3 -m pytest tests/ -q

# Lint
ruff check . --ignore E501

# Generate secrets
python3 -c "import secrets; print(secrets.token_hex(32))"

# Test production config locally
export METTLE_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
export METTLE_ADMIN_API_KEY=$(python3 -c "import secrets; print(f'mtl_admin_{secrets.token_hex(24)}')")
export METTLE_ENVIRONMENT=production
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

*Created: 2026-02-04*
*Previous: 3 rounds of security red-teaming completed*
*Status: Production ready with configuration*
