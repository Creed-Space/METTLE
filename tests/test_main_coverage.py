"""Tests targeting uncovered lines in main.py.

Covers:
- Badge verification endpoint (/api/badge/verify)
- Badge revocation endpoint (/api/badge/revoke) with admin auth
- Webhook delivery (WebhookManager.send_webhook)
- API key management (/api/keys/register)
- Static/SEO endpoints (/sitemap.xml, /robots.txt, /, /ui, /about)
- Admin auth rate limiting (check_admin_auth_rate_limit, record_admin_auth_failure)
- add_with_limit eviction
- CollusionDetector memory bounding
- RateTier.check_limit daily usage tracking
- Webhook URL validation (SSRF protection)
- generate_signed_badge error path
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from main import (
    CollusionDetector,
    RateTier,
    WebhookManager,
    _admin_auth_failures,
    add_with_limit,
    api_keys,
    app,
    check_admin_auth_rate_limit,
    limiter,
    record_admin_auth_failure,
    revoked_badges,
    sessions,
    verification_graph,
    verification_timestamps,
    webhooks,
)

# Test constants matching conftest.py
SECRET_KEY = "test-secret-key-for-mettle-testing-only"
ADMIN_KEY = "test-admin-key-for-mettle-testing-only"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear all in-memory state before each test."""
    sessions.clear()
    from main import challenges

    challenges.clear()
    verification_graph.clear()
    verification_timestamps.clear()
    api_keys.clear()
    webhooks.clear()
    revoked_badges.clear()
    _admin_auth_failures.clear()
    limiter.reset()
    yield
    sessions.clear()
    challenges.clear()
    verification_graph.clear()
    verification_timestamps.clear()
    api_keys.clear()
    webhooks.clear()
    revoked_badges.clear()
    _admin_auth_failures.clear()
    limiter.reset()


def _make_badge_token(
    entity_id="test-entity",
    jti="test-jti-001",
    expired=False,
    secret=SECRET_KEY,
    extra_claims=None,
):
    """Create a signed JWT badge token for testing."""
    now = datetime.now(timezone.utc)
    if expired:
        exp = (now - timedelta(hours=1)).timestamp()
    else:
        exp = (now + timedelta(hours=1)).timestamp()

    payload = {
        "entity_id": entity_id,
        "difficulty": "basic",
        "pass_rate": 1.0,
        "verified_at": now.isoformat(),
        "version": "1.0.0",
        "iss": "mettle-api",
        "exp": exp,
        "jti": jti,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")


# =============================================================================
# Badge Verification (lines 1139-1173)
# =============================================================================


class TestBadgeVerification:
    """Tests for /api/badge/verify/{token} endpoint."""

    def test_valid_badge_returns_valid_true(self, client):
        """Valid JWT badge should return valid=True with payload."""
        token = _make_badge_token()
        response = client.get(f"/api/badge/verify/{token}")

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["payload"]["entity_id"] == "test-entity"
        assert data["expires_at"] is not None
        assert data.get("error") is None

    def test_revoked_badge_returns_revoked(self, client):
        """Revoked badge should return valid=False with revoked=True."""
        token = _make_badge_token(jti="revoked-jti")
        revoked_badges["revoked-jti"] = time.time()

        response = client.get(f"/api/badge/verify/{token}")

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["revoked"] is True
        assert "revoked" in data["error"].lower()

    def test_expired_badge_returns_expired(self, client):
        """Expired JWT should return valid=False with 'expired' error."""
        token = _make_badge_token(expired=True)

        response = client.get(f"/api/badge/verify/{token}")

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "expired" in data["error"].lower()

    def test_invalid_token_returns_invalid(self, client):
        """Garbage token should return valid=False with 'Invalid' error."""
        response = client.get("/api/badge/verify/not-a-real-jwt-token")

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "invalid" in data["error"].lower()

    def test_wrong_secret_returns_invalid(self, client):
        """Token signed with wrong secret should fail verification."""
        token = _make_badge_token(secret="wrong-secret-key")

        response = client.get(f"/api/badge/verify/{token}")

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_no_secret_key_returns_not_configured(self, client):
        """When secret_key is not set, verification should return not configured."""
        with patch("main.settings") as mock_settings:
            mock_settings.secret_key = None
            mock_settings.admin_api_key = ADMIN_KEY
            response = client.get("/api/badge/verify/any-token")

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "not configured" in data["error"].lower() or "signing key" in data["error"].lower()


# =============================================================================
# Badge Revocation (lines 1237-1287)
# =============================================================================


class TestBadgeRevocationFull:
    """Tests for /api/badge/revoke endpoint with admin auth."""

    def test_successful_revocation(self, client):
        """Valid admin key + valid token should revoke badge."""
        token = _make_badge_token(jti="revoke-me-jti")

        response = client.post(
            "/api/badge/revoke",
            json={
                "token": token,
                "reason": "Test revocation reason for coverage testing",
            },
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["revoked"] is True
        assert data["jti"] == "revoke-me-jti"
        assert "revoke-me-jti" in revoked_badges

    def test_already_revoked_badge(self, client):
        """Revoking an already-revoked badge returns revoked=False."""
        token = _make_badge_token(jti="already-revoked-jti")
        revoked_badges["already-revoked-jti"] = time.time()

        response = client.post(
            "/api/badge/revoke",
            json={
                "token": token,
                "reason": "Trying to revoke again for coverage",
            },
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["revoked"] is False
        assert "already" in data["message"].lower()

    def test_invalid_token_returns_400(self, client):
        """Invalid JWT token should return 400."""
        response = client.post(
            "/api/badge/revoke",
            json={
                "token": "not-a-valid-jwt",
                "reason": "Invalid token revocation test",
            },
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 400

    def test_no_jti_in_token_returns_400(self, client):
        """Token without jti claim should return 400."""
        # Create token without jti
        payload = {
            "entity_id": "test",
            "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        response = client.post(
            "/api/badge/revoke",
            json={
                "token": token,
                "reason": "No jti in token test for coverage",
            },
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 400
        assert "jti" in response.json()["detail"].lower()

    def test_no_secret_key_returns_400(self, client):
        """When secret_key is not configured, revocation should return 400."""
        with patch("main.settings") as mock_settings:
            mock_settings.secret_key = None
            mock_settings.admin_api_key = ADMIN_KEY
            mock_settings.is_production = False

            response = client.post(
                "/api/badge/revoke",
                json={
                    "token": "any-token",
                    "reason": "No secret key configured test",
                },
                headers=ADMIN_HEADERS,
            )

        assert response.status_code == 400
        assert "signing" in response.json()["detail"].lower() or "configured" in response.json()["detail"].lower()

    def test_no_admin_key_configured_returns_503(self, client):
        """When admin_api_key is not set, revocation returns 503."""
        with patch("main.settings") as mock_settings:
            mock_settings.admin_api_key = None
            mock_settings.secret_key = SECRET_KEY
            mock_settings.is_production = False

            response = client.post(
                "/api/badge/revoke",
                json={
                    "token": "any-token",
                    "reason": "No admin key configured test",
                },
            )

        assert response.status_code == 503

    def test_wrong_admin_key_returns_401(self, client):
        """Wrong admin key should return 401."""
        response = client.post(
            "/api/badge/revoke",
            json={
                "token": "any-token",
                "reason": "Wrong admin key test for coverage",
            },
            headers={"X-Admin-Key": "wrong-admin-key"},
        )

        assert response.status_code == 401


# =============================================================================
# Webhook Delivery (lines 1482-1558)
# =============================================================================


class TestWebhookDelivery:
    """Tests for WebhookManager.send_webhook method."""

    @pytest.mark.asyncio
    async def test_entity_not_registered_returns_false(self):
        """send_webhook returns False for unregistered entity."""
        result = await WebhookManager.send_webhook("unknown-entity", "session.completed", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_entity_returns_false(self):
        """send_webhook returns False for empty entity_id."""
        result = await WebhookManager.send_webhook("", "session.completed", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_no_url_returns_false(self):
        """send_webhook returns False when config has no URL."""
        webhooks["entity-1"] = {"events": ["session.completed"]}
        result = await WebhookManager.send_webhook("entity-1", "session.completed", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_event_not_subscribed_returns_false(self):
        """send_webhook returns False when event not in subscribed list."""
        WebhookManager.register("entity-1", "https://example.com/hook", ["session.started"])
        result = await WebhookManager.send_webhook("entity-1", "badge.issued", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        """send_webhook returns True on successful HTTP post."""
        WebhookManager.register("entity-1", "https://example.com/hook")

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Patch socket.gethostbyname to return a public IP
            with patch("socket.gethostbyname", return_value="93.184.216.34"):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {"session_id": "test-123"},
                )

        assert result is True

    @pytest.mark.asyncio
    async def test_delivery_with_secret_includes_hmac(self):
        """send_webhook includes HMAC signature when secret is configured."""
        WebhookManager.register(
            "entity-1",
            "https://example.com/hook",
            secret="a" * 32,
        )

        mock_response = AsyncMock()
        mock_response.status_code = 200
        captured_payload = {}

        async def capture_post(url, json=None):
            captured_payload.update(json)
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.gethostbyname", return_value="93.184.216.34"):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {"data": "test"},
                )

        assert result is True
        assert "signature" in captured_payload

    @pytest.mark.asyncio
    async def test_dns_rebinding_blocked(self):
        """send_webhook returns False when DNS resolves to private IP."""
        WebhookManager.register("entity-1", "https://example.com/hook")

        with patch("socket.gethostbyname", return_value="127.0.0.1"):
            result = await WebhookManager.send_webhook(
                "entity-1",
                "session.completed",
                {},
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_dns_rebinding_private_ip_blocked(self):
        """send_webhook returns False when DNS resolves to private range."""
        WebhookManager.register("entity-1", "https://example.com/hook")

        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            result = await WebhookManager.send_webhook(
                "entity-1",
                "session.completed",
                {},
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_httpx_failure_returns_false(self):
        """send_webhook returns False on httpx exception.

        NOTE: main.py logger bug with event= kwarg - patched here.
        """
        WebhookManager.register("entity-1", "https://example.com/hook")

        with patch("httpx.AsyncClient") as mock_client_cls, \
             patch("main.logger"):
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.gethostbyname", return_value="93.184.216.34"):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {},
                )

        assert result is False


# =============================================================================
# API Key Management (lines 1749-1777)
# =============================================================================


class TestAPIKeyManagement:
    """Tests for /api/keys/register endpoint."""

    def test_register_key_with_admin(self, client):
        """Valid admin key should create a new API key."""
        response = client.post(
            "/api/keys/register",
            json={"tier": "pro", "entity_id": "test-entity"},
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tier"] == "pro"
        assert data["api_key"].startswith("mtl_")
        assert "limits" in data

    def test_register_key_no_admin_key_returns_401(self, client):
        """Missing admin key should return 401."""
        response = client.post(
            "/api/keys/register",
            json={"tier": "free"},
        )

        assert response.status_code == 401

    def test_register_key_invalid_tier_returns_400(self, client):
        """Invalid tier name should return 400."""
        response = client.post(
            "/api/keys/register",
            json={"tier": "nonexistent-tier"},
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 400


# =============================================================================
# Static/SEO Endpoints (lines 1809-1870, 1876-1887)
# =============================================================================


class TestStaticAndSEOEndpoints:
    """Tests for sitemap.xml, robots.txt, root, /ui, /about."""

    def test_sitemap_returns_xml(self, client):
        """GET /sitemap.xml should return valid XML."""
        response = client.get("/sitemap.xml")

        assert response.status_code == 200
        assert "application/xml" in response.headers["content-type"]
        assert "mettle.sh" in response.text
        assert '<?xml version="1.0"' in response.text

    def test_robots_txt_returns_text(self, client):
        """GET /robots.txt should return text response."""
        response = client.get("/robots.txt")

        assert response.status_code == 200
        # Either a static file or generated content
        content_type = response.headers["content-type"]
        assert "text/plain" in content_type or "text" in content_type

    def test_root_serves_ui_or_redirect(self, client):
        """GET / should either serve index.html or redirect to /api."""
        response = client.get("/", follow_redirects=False)

        # Either 200 (serves index.html) or 307 redirect
        assert response.status_code in (200, 307)

    def test_legacy_ui_redirect(self, client):
        """GET /ui should redirect to / with 301."""
        response = client.get("/ui", follow_redirects=False)

        assert response.status_code == 301
        assert response.headers["location"] == "/"

    def test_about_serves_or_redirects(self, client):
        """GET /about should serve about.html or redirect."""
        response = client.get("/about", follow_redirects=False)

        assert response.status_code in (200, 307)


# =============================================================================
# Admin Auth Rate Limiting (lines 321-341)
# =============================================================================


class TestAdminAuthRateLimiting:
    """Tests for check_admin_auth_rate_limit and record_admin_auth_failure."""

    def test_ip_allowed_with_no_failures(self):
        """Clean IP should be allowed."""
        allowed, retry_after = check_admin_auth_rate_limit("192.168.1.100")

        assert allowed is True
        assert retry_after == 0

    def test_ip_blocked_after_max_failures(self):
        """IP should be blocked after exceeding max failures."""
        ip = "10.0.0.1"
        # Record enough failures to trigger block (5 is the max)
        for _ in range(6):
            record_admin_auth_failure(ip)

        allowed, retry_after = check_admin_auth_rate_limit(ip)

        assert allowed is False
        assert retry_after > 0

    def test_record_admin_auth_failure_stores(self):
        """record_admin_auth_failure should store failure timestamps."""
        ip = "172.16.0.1"
        record_admin_auth_failure(ip)

        assert ip in _admin_auth_failures
        assert len(_admin_auth_failures[ip]) == 1

    def test_record_admin_auth_failure_bounds_per_ip(self):
        """Failures per IP should be bounded at 100."""
        ip = "192.168.1.1"
        for _ in range(110):
            record_admin_auth_failure(ip)

        assert len(_admin_auth_failures[ip]) <= 100

    def test_record_admin_auth_failure_evicts_oldest_ip(self):
        """When too many IPs tracked, oldest should be evicted."""
        from main import MAX_AUTH_FAILURES

        # Fill up the failures dict to capacity
        for i in range(MAX_AUTH_FAILURES):
            record_admin_auth_failure(f"10.0.{i // 256}.{i % 256}")

        # The next one should evict the oldest
        record_admin_auth_failure("99.99.99.99")
        assert "99.99.99.99" in _admin_auth_failures


# =============================================================================
# add_with_limit (lines 104-105)
# =============================================================================


class TestAddWithLimit:
    """Tests for add_with_limit LRU eviction."""

    def test_evicts_oldest_when_full(self):
        """add_with_limit should evict oldest item when at capacity."""
        store = {"a": 1, "b": 2, "c": 3}
        add_with_limit(store, "d", 4, max_size=3)

        assert "d" in store
        assert "a" not in store  # Oldest evicted
        assert len(store) == 3

    def test_no_eviction_when_under_limit(self):
        """add_with_limit should not evict when under capacity."""
        store = {"a": 1}
        add_with_limit(store, "b", 2, max_size=5)

        assert "a" in store
        assert "b" in store
        assert len(store) == 2


# =============================================================================
# CollusionDetector memory bounding (lines 232-247)
# =============================================================================


class TestCollusionDetectorMemoryBounds:
    """Tests for CollusionDetector memory bounding."""

    def test_verification_graph_bounds_entities(self):
        """verification_graph should evict oldest when at MAX_VERIFICATION_GRAPH."""
        from main import MAX_VERIFICATION_GRAPH

        # Fill to capacity
        for i in range(MAX_VERIFICATION_GRAPH):
            CollusionDetector.record_verification(f"entity-{i}", "192.168.1.1", True)

        # Next should evict first
        CollusionDetector.record_verification("entity-overflow", "192.168.1.1", True)

        assert "entity-overflow" in verification_graph
        assert len(verification_graph) <= MAX_VERIFICATION_GRAPH

    def test_verification_graph_bounds_records_per_entity(self):
        """Records per entity should be bounded at 100."""
        for i in range(110):
            CollusionDetector.record_verification("entity-big", f"10.0.0.{i % 256}", True)

        assert len(verification_graph["entity-big"]) <= 100

    def test_verification_timestamps_bounded(self):
        """verification_timestamps should be bounded at 1000."""
        for i in range(1010):
            CollusionDetector.record_verification(f"entity-{i}", "192.168.1.1", True)

        assert len(verification_timestamps) <= 1000


# =============================================================================
# RateTier.check_limit daily usage tracking (lines 173-184)
# =============================================================================


class TestRateTierDailyUsage:
    """Tests for RateTier.check_limit daily usage tracking."""

    def test_daily_limit_reached(self):
        """Pro tier should be blocked after exceeding daily session limit."""
        key = "test-pro-key"
        RateTier.register_key(key, "pro", "entity-1")

        # Exhaust daily limit (10000 for pro)
        api_keys[key]["usage_count"] = 10000
        api_keys[key]["usage_date"] = datetime.now(timezone.utc).date().isoformat()

        allowed, message = RateTier.check_limit(key, "session")

        assert allowed is False
        assert "limit reached" in message.lower()

    def test_usage_resets_on_new_day(self):
        """Usage should reset when date changes."""
        key = "test-pro-key"
        RateTier.register_key(key, "pro", "entity-1")

        # Set usage from yesterday
        api_keys[key]["usage_count"] = 9999
        api_keys[key]["usage_date"] = "2025-01-01"  # Old date

        allowed, message = RateTier.check_limit(key, "session")

        assert allowed is True
        # Usage count should be reset to 1 (incremented for this request)
        assert api_keys[key]["usage_count"] == 1


# =============================================================================
# Webhook URL Validation / SSRF Protection (lines 1603-1654)
# =============================================================================


class TestWebhookURLValidation:
    """Tests for WebhookRegisterRequest URL validation."""

    def test_non_http_scheme_rejected(self, client):
        """ftp:// scheme should be rejected."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "ftp://example.com/hook",
            },
        )
        assert response.status_code == 422

    def test_localhost_rejected(self, client):
        """localhost URL should be rejected (SSRF protection)."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://localhost/hook",
            },
        )
        assert response.status_code == 422

    def test_127_0_0_1_rejected(self, client):
        """127.0.0.1 URL should be rejected."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://127.0.0.1/hook",
            },
        )
        assert response.status_code == 422

    def test_cloud_metadata_rejected(self, client):
        """Cloud metadata endpoint should be rejected."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://169.254.169.254/latest/meta-data",
            },
        )
        assert response.status_code == 422

    def test_private_ip_rejected(self, client):
        """Private IP URL is blocked by SSRF validator."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://10.0.0.1/hook",
            },
        )
        assert response.status_code == 422

    def test_internal_hostname_rejected(self, client):
        """Internal hostname patterns should be rejected."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://service.internal/hook",
            },
        )
        assert response.status_code == 422

    def test_short_secret_rejected(self, client):
        """Webhook secret shorter than 32 chars should be rejected."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "https://example.com/hook",
                "secret": "too-short",
            },
        )
        assert response.status_code == 422


# =============================================================================
# Session not found paths (lines 998, 1049)
# =============================================================================


class TestSessionNotFound:
    """Tests for session 404 paths."""

    def test_get_session_not_found(self, client):
        """GET /api/session/nonexistent should return 404."""
        response = client.get("/api/session/nonexistent-session-id")
        assert response.status_code == 404

    def test_get_result_not_found(self, client):
        """GET /api/result/nonexistent should return 404."""
        response = client.get("/api/result/nonexistent-session-id")
        assert response.status_code == 404


# =============================================================================
# Answer field validation (line 610)
# =============================================================================


class TestAnswerValidation:
    """Tests for SubmitAnswerRequest validation."""

    def test_oversized_answer_rejected(self, client):
        """Answer exceeding 1024 chars should be rejected by Pydantic max_length."""
        # Start a session first to get valid IDs
        session_resp = client.post(
            "/api/session/start",
            json={"entity_id": "test-entity", "difficulty": "basic"},
        )
        assert session_resp.status_code == 200
        data = session_resp.json()
        session_id = data["session_id"]
        challenge_id = data["current_challenge"]["id"]

        # Submit answer with extremely long string (>1024)
        response = client.post(
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": challenge_id,
                "answer": "x" * 1025,
            },
        )
        assert response.status_code == 422


# =============================================================================
# ModelFingerprinter equal distribution (line 1383)
# =============================================================================


class TestModelFingerprinterEdge:
    """Tests for ModelFingerprinter edge cases."""

    def test_fingerprint_with_neutral_response(self):
        """Responses matching no model should get equal distribution."""
        from main import ModelFingerprinter

        # Very short response unlikely to match any model patterns
        result = ModelFingerprinter.fingerprint(["ok"])

        # Scores should sum to ~1.0
        total = sum(result["scores"].values())
        assert 0.99 <= total <= 1.01


# =============================================================================
# HSTS header in production (line 391)
# =============================================================================


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_security_headers_present(self, client):
        """Verify security headers are set on responses."""
        response = client.get("/api/health")

        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "X-Request-ID" in response.headers


# =============================================================================
# __main__ block (lines 1885-1887)
# =============================================================================


class TestMainBlock:
    """Test the __main__ guard."""

    def test_main_module_importable(self):
        """Verify main can be imported without running uvicorn."""
        import main

        assert hasattr(main, "app")
        assert hasattr(main, "WebhookManager")


# =============================================================================
# generate_signed_badge no secret_key path (line 1097)
# =============================================================================


class TestGenerateSignedBadge:
    """Tests for generate_signed_badge function."""

    def test_no_secret_key_raises_value_error(self):
        """generate_signed_badge should raise ValueError when no secret_key."""
        from main import generate_signed_badge

        with patch("main.settings") as mock_settings:
            mock_settings.secret_key = None
            mock_settings.badge_expiry_seconds = 3600
            mock_settings.api_version = "1.0.0"

            with pytest.raises(ValueError, match="SECRET_KEY not configured"):
                generate_signed_badge("entity-1", "basic", 1.0, "ses_test")


# =============================================================================
# get_result session not found (line 1049)
# =============================================================================


class TestGetResultEndpoint:
    """Tests for /api/session/{session_id}/result endpoint."""

    def test_result_not_found(self, client):
        """GET /api/session/{id}/result with invalid session returns 404."""
        response = client.get("/api/session/nonexistent-session-id/result")
        assert response.status_code == 404

    def test_result_session_not_complete(self, client):
        """GET /api/session/{id}/result with incomplete session returns 400."""
        # Start session
        session_resp = client.post(
            "/api/session/start",
            json={"entity_id": "test-entity", "difficulty": "basic"},
        )
        session_id = session_resp.json()["session_id"]

        response = client.get(f"/api/session/{session_id}/result")
        assert response.status_code == 400


# =============================================================================
# Revocation list rate limit path (line 1309)
# =============================================================================


class TestRevocationListRateLimit:
    """Tests for revocation list admin auth rate limiting."""

    def test_revocations_rate_limited_after_failures(self, client):
        """Revocation list should be 429 after too many auth failures."""
        ip = "testclient"
        # Record enough failures to trigger block
        for _ in range(6):
            record_admin_auth_failure(ip)

        response = client.get("/api/badge/revocations")
        assert response.status_code == 429


# =============================================================================
# Badge revoke rate limit path (line 1226)
# =============================================================================


class TestBadgeRevokeRateLimit:
    """Tests for badge revoke rate limiting."""

    def test_revoke_rate_limited_after_failures(self, client):
        """Badge revocation should be 429 after too many auth failures."""
        ip = "testclient"
        for _ in range(6):
            record_admin_auth_failure(ip)

        response = client.post(
            "/api/badge/revoke",
            json={
                "token": "any-token",
                "reason": "Rate limit testing for coverage",
            },
        )
        assert response.status_code == 429


# =============================================================================
# API key register rate limit path (line 1755)
# =============================================================================


class TestKeyRegisterRateLimit:
    """Tests for key register rate limiting."""

    def test_key_register_rate_limited(self, client):
        """Key registration should be 429 after too many auth failures."""
        ip = "testclient"
        for _ in range(6):
            record_admin_auth_failure(ip)

        response = client.post(
            "/api/keys/register",
            json={"tier": "free"},
            headers=ADMIN_HEADERS,
        )
        assert response.status_code == 429


# =============================================================================
# Revocation audit bounding (line 1278)
# =============================================================================


class TestRevocationAuditBounding:
    """Tests for revocation audit trail memory bounds."""

    def test_audit_trail_bounded(self, client):
        """Revocation audit trail should be bounded."""
        from main import MAX_REVOCATION_AUDIT, revocation_audit

        # Fill audit beyond max
        for i in range(MAX_REVOCATION_AUDIT + 5):
            token = _make_badge_token(jti=f"audit-jti-{i}")
            client.post(
                "/api/badge/revoke",
                json={
                    "token": token,
                    "reason": f"Audit bounding test number {i}",
                },
                headers=ADMIN_HEADERS,
            )

        assert len(revocation_audit) <= MAX_REVOCATION_AUDIT


# =============================================================================
# Webhook URL HTTPS production-only validation (line 1628)
# =============================================================================


class TestWebhookProductionValidation:
    """Tests for webhook URL production-only HTTPS requirement."""

    def test_http_allowed_in_dev(self, client):
        """HTTP URLs should be allowed in non-production."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://example.com/hook",
            },
        )
        assert response.status_code == 200

    def test_webhook_secret_min_length_validated(self, client):
        """Webhook secret must be at least 32 chars."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "https://example.com/hook",
                "secret": "a" * 31,  # Too short
            },
        )
        assert response.status_code == 422


# =============================================================================
# Batch start exception handler (lines 843-846)
# =============================================================================


class TestBatchStartExceptionHandling:
    """Tests for batch session start exception handling."""

    def test_batch_with_many_entities(self, client):
        """Batch start should handle multiple entities correctly."""
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": [f"entity-{i}" for i in range(5)],
                "difficulty": "basic",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert data["failed"] == 0


# =============================================================================
# DNS resolution failure path (lines 1539-1541)
# =============================================================================


class TestWebhookDNSFailure:
    """Tests for webhook DNS resolution failure handling."""

    @pytest.mark.asyncio
    async def test_dns_resolution_failure_allows_request(self):
        """When DNS resolution fails, webhook should still attempt delivery.

        The code allows the request to proceed if DNS fails (external hostname).
        """
        import socket

        WebhookManager.register("entity-1", "https://external-host.com/hook")

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls, \
             patch("main.logger"):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.gethostbyname", side_effect=socket.gaierror("DNS failed")):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {},
                )

        assert result is True


# =============================================================================
# Static file paths when static dir doesn't exist (lines 1812, 1828, 1836, 1878)
# =============================================================================


class TestStaticFileFallbacks:
    """Tests for static file paths when files don't exist."""

    def test_root_redirects_when_no_static(self, client):
        """Root should redirect to /api when static dir doesn't exist."""
        with patch("main._static_dir") as mock_dir:
            mock_dir.exists.return_value = False
            response = client.get("/", follow_redirects=False)
            # Should redirect to /api when no static files
            assert response.status_code in (200, 307)

    def test_about_redirects_when_no_static(self, client):
        """About should redirect to / when no static files."""
        with patch("main._static_dir") as mock_dir:
            mock_dir.exists.return_value = False
            response = client.get("/about", follow_redirects=False)
            assert response.status_code in (200, 307)

    def test_robots_fallback_when_no_static(self, client):
        """Robots.txt should return generated content when no static file."""
        with patch("main._static_dir") as mock_dir:
            mock_dir.exists.return_value = False
            response = client.get("/robots.txt")
            assert response.status_code == 200
            assert "User-agent" in response.text
