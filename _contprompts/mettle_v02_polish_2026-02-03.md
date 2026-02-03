# METTLE v0.2: Polish & Documentation Enhancement

*Continuation prompt for fresh context - 2026-02-03*

---

## Current State

METTLE is live at **https://mettle.sh** (also https://mettle.is)

### What's Deployed
- Landing page with interactive demo
- Documentation at `/static/docs.html`
- Pricing at `/pricing`
- About at `/about`
- Full API at `/api` and `/docs` (OpenAPI)

### Recent Changes (This Session)
1. Fixed CI lint errors (unused imports)
2. Created docs, pricing, about pages
3. Improved 5 Questions section (icons, 5-column grid, hover effects)
4. Improved Use Cases section (2x2 grid, horizontal cards)
5. Fixed flaky integration tests (lenient timing for CI)
6. Changed version from v2.0 → v0.2
7. Removed "24h expiry" emphasis, replaced with "Fresh Badges"

---

## Tasks for This Session

### 1. Visual Review
Check how the site looks now at https://mettle.sh:
- [ ] Landing page layout and styling
- [ ] 5 Questions section (should be 5 columns with icons)
- [ ] Use Cases section (should be 2x2 balanced grid)
- [ ] Mobile responsiveness
- [ ] Any CSS issues or visual glitches

### 2. Emphasize Configurability

The badge expiry is configurable via `badge_expiry_seconds` (default 86400 = 24h). Update messaging to make this clear:

**Files to update:**
- `static/index.html` - Landing page
- `static/docs.html` - Documentation
- `static/pricing.html` - Pricing tiers (Pro/Enterprise can configure expiry)

**Key messages:**
- "Configurable badge expiry (default 24h)"
- "Enterprise: Custom expiry periods"
- "Fresh verification proves current state, not historical"

### 3. Documentation Enhancement

Current docs at `/static/docs.html` need:

**Missing sections:**
- [ ] Configuration options (all env vars)
- [ ] Badge expiry configuration
- [ ] Rate limiting details per tier
- [ ] Error codes and handling
- [ ] MCP server integration (there's an mcp_server.py)

**Improvements:**
- [ ] Add configuration section showing all settings
- [ ] Expand security model with more detail
- [ ] Add troubleshooting section
- [ ] Add changelog/version history

### 4. Unresolved Items

**From previous contprompt (`_contprompts/mettle_v2_hardening_and_website.md`):**

| Item | Status | Notes |
|------|--------|-------|
| Badge expiry | ✅ Done | JWT with configurable `exp` claim |
| Revocation registry | ✅ Done | `/api/badge/revoke` endpoint |
| Freshness nonces | ✅ Done | In badge payload |
| Collusion detection | ✅ Done | `CollusionDetector` class |
| Model fingerprinting | ✅ Done | Basic implementation |
| Rate limiting tiers | ✅ Done | Free/Pro/Enterprise |
| Webhooks | ✅ Done | Registration + delivery |
| Session cleanup | ✅ Done | Background task |
| Domain setup | ✅ Done | mettle.sh verified |

**Still TODO:**
- [ ] Freshness challenge-response endpoint (nonce exists but no verification endpoint)
- [ ] Database persistence (optional, currently in-memory)
- [ ] MCP server documentation

### 5. Code Quality

Run validation:
```bash
cd /Users/nellwatson/Documents/GitHub/Rewind/mettle-api
ruff check . --ignore E501
pytest tests/ -v
```

---

## Key Files

```
mettle-api/
├── main.py                 # FastAPI app (1600+ lines)
├── config.py               # Settings (badge_expiry_seconds, etc.)
├── mettle/
│   ├── challenger.py       # Challenge generation
│   ├── verifier.py         # Response verification
│   └── models.py           # Pydantic models
├── static/
│   ├── index.html          # Landing page
│   ├── docs.html           # Documentation
│   ├── pricing.html        # Pricing
│   ├── about.html          # About
│   ├── style.css           # Main styles
│   └── docs.css            # Docs-specific styles
├── mcp_server.py           # MCP server for Claude integration
└── tests/                  # Test suite
```

---

## Configuration Reference

From `config.py`:

```python
class Settings(BaseSettings):
    api_title: str = "METTLE API"
    api_version: str = "0.2.0"
    environment: str = "development"
    secret_key: str | None = None  # Required in production for JWT signing
    badge_expiry_seconds: int = 86400  # 24 hours default
    allowed_origins: str = "*"
    rate_limit_sessions: str = "100/minute"
    rate_limit_answers: str = "300/minute"
    use_database: bool = False
    database_url: str | None = None
    admin_api_key: str | None = None
```

**Key point:** `badge_expiry_seconds` is configurable. Document this prominently.

---

## Visual Design Targets

**5 Questions Section:**
- 5 equal columns on desktop
- Icons: microchip, unlock, fingerprint, user-check, shield-heart
- Gradient header bar
- Hover lift effect

**Use Cases Section:**
- 2x2 grid (no orphan cards)
- Horizontal layout with icon box on left
- Left-aligned text

**Color Palette:**
- Primary: #6366f1 (indigo)
- Background: #0f172a (dark slate)
- Cards: #1e293b
- Success: #22c55e
- Error: #ef4444

---

## API Endpoints Summary

**Sessions:**
- `POST /api/session/start` - Start verification
- `POST /api/session/answer` - Submit answer
- `GET /api/session/{id}` - Get status
- `GET /api/session/{id}/result` - Get final result
- `POST /api/session/batch` - Batch start (Pro+)

**Badges:**
- `GET /api/badge/verify/{token}` - Verify badge
- `POST /api/badge/revoke` - Revoke badge
- `GET /api/badge/revocations` - List revocations

**Security:**
- `GET /api/security/collusion` - Collusion stats
- `POST /api/security/fingerprint` - Model fingerprinting

**Webhooks:**
- `POST /api/webhooks/register` - Register webhook
- `DELETE /api/webhooks/{entity_id}` - Unregister
- `GET /api/webhooks/events` - List event types

---

## Execution Checklist

1. [ ] Open https://mettle.sh and review visually
2. [ ] Check mobile view
3. [ ] Update configurability messaging in landing page
4. [ ] Update pricing page to show configurable expiry for Pro/Enterprise
5. [ ] Add configuration section to docs
6. [ ] Add MCP server section to docs
7. [ ] Run `ruff check` and `pytest`
8. [ ] Commit and push changes

---

*Created: 2026-02-03 12:45*
*Previous session: Fixed CI, added pages, improved styling, version to v0.2*
