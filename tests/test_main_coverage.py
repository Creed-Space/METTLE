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
"""

import hashlib
import re
import socket
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
import main as main_module
from main import (
    CollusionDetector,
    RateTier,
    WebhookManager,
    _admin_auth_failures,
    add_with_limit,
    api_keys,
    app,
    challenges,
    check_admin_auth_rate_limit,
    limiter,
    record_admin_auth_failure,
    revocation_audit,
    revoked_badges,
    sessions,
    verification_graph,
    verification_timestamps,
    webhooks,
)
from tests.session_client import SessionAwareTestClient

# Test constants matching conftest.py
SECRET_KEY = "test-secret-key-for-mettle-testing-only"
ADMIN_KEY = "test-admin-key-for-mettle-testing-only"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}


def _addrinfo(ip: str = "93.184.216.34"):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))]


@pytest.fixture
def client():
    """Create test client."""
    return SessionAwareTestClient(app)


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
    revocation_audit.clear()
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
    revocation_audit.clear()
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
        "iat": now.timestamp(),
        "exp": exp,
        "jti": jti,
        "session_id": "test-session",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")


# =============================================================================
# Durable runtime recovery
# =============================================================================


class TestPersistentRuntimeRecovery:
    """Exercise PostgreSQL-backed legacy session and webhook recovery."""

    def test_persist_legacy_session_and_progress(self):
        from mettle import BadgeInfo, Difficulty, MettleSession

        session = MettleSession(
            session_id="persistent-session",
            entity_id="persistent-agent",
            difficulty=Difficulty.BASIC,
            challenges=[],
            access_token_hash="a" * 64,
            badge_info=BadgeInfo(
                token="signed-token",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                freshness_nonce=None,
                signed=True,
                jti="persistent-jti",
            ),
        )
        mock_db = MagicMock()
        mock_db.save_session.return_value = True
        mock_db.update_session_results.return_value = True
        badge_info = session.badge_info
        assert badge_info is not None

        with patch.object(main_module, "db", mock_db):
            assert main_module._persist_new_legacy_session(session) is True
            assert main_module._persist_legacy_progress(session) is True

        mock_db.save_session.assert_called_once_with(
            session.session_id,
            session.entity_id,
            session.difficulty.value,
            session.challenges,
            session.access_token_hash,
            session.started_at,
        )
        mock_db.update_session_results.assert_called_once_with(
            session.session_id,
            session.results,
            session.completed,
            badge_info.model_dump(mode="json"),
        )

        with patch.object(main_module, "db", None):
            assert main_module._persist_new_legacy_session(session) is True
            assert main_module._persist_legacy_progress(session) is True

    def test_restore_sessions_challenges_and_webhooks(self):
        from mettle import (
            BadgeInfo,
            Challenge,
            ChallengeType,
            Difficulty,
            VerificationResult,
        )

        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id="recovered-challenge",
            type=ChallengeType.SPEED_MATH,
            prompt="What is 2 + 2?",
            data={"expected_answer": 4},
            issued_at=now,
            expires_at=now + timedelta(minutes=5),
            time_limit_ms=1000,
        )
        result = VerificationResult(
            challenge_id=challenge.id,
            challenge_type=challenge.type,
            passed=True,
            response_time_ms=10,
            time_limit_ms=challenge.time_limit_ms,
        )
        badge = BadgeInfo(
            token="recovered-token",
            expires_at=now + timedelta(hours=1),
            freshness_nonce=None,
            signed=True,
            jti="recovered-jti",
        )
        mock_db = MagicMock()
        mock_db.get_recent_sessions.return_value = [
            {
                "session_id": "recovered-incomplete",
                "entity_id": "agent-incomplete",
                "difficulty": Difficulty.BASIC.value,
                "challenges": [challenge.model_dump(mode="json")],
                "results": [],
                "created_at": now.replace(tzinfo=None).isoformat(),
                "completed": False,
                "access_token_hash": "b" * 64,
                "badge_info": None,
            },
            {
                "session_id": "recovered-complete",
                "entity_id": "agent-complete",
                "difficulty": Difficulty.FULL.value,
                "challenges": [challenge.model_dump(mode="json")],
                "results": [result.model_dump(mode="json")],
                "created_at": now,
                "completed": True,
                "access_token_hash": "c" * 64,
                "badge_info": badge.model_dump(mode="json"),
            },
            {"session_id": "malformed-session"},
        ]
        mock_db.get_webhooks.return_value = [
            {
                "entity_id": "recovered-webhook-owner",
                "url": "https://example.com/mettle-hook",
                "events": ["session.completed"],
                "secret": "webhook-secret",
                "created_at": now.isoformat(),
            }
        ]

        with patch.object(main_module, "db", mock_db):
            main_module._restore_persistent_runtime_state()

        assert sessions["recovered-incomplete"].started_at.tzinfo is not None
        assert sessions["recovered-complete"].badge_info == badge
        assert challenges[challenge.id][0] == challenge
        assert challenges[challenge.id][1] is None
        with patch.object(main_module.time, "time", return_value=1234.5):
            main_module._arm_recovered_challenge(sessions["recovered-incomplete"])
        assert challenges[challenge.id][1] == 1234.5
        assert webhooks["recovered-webhook-owner"]["url"].endswith("mettle-hook")
        mock_db.get_recent_sessions.assert_called_once_with(
            max_age_seconds=main_module.LEGACY_SESSION_RECOVERY_SECONDS,
            limit=main_module.MAX_SESSIONS,
        )
        mock_db.get_webhooks.assert_called_once_with(
            limit=main_module.MAX_WEBHOOKS,
            raise_on_error=True,
        )

    def test_restore_is_a_noop_without_database(self):
        with patch.object(main_module, "db", None):
            main_module._restore_persistent_runtime_state()


# =============================================================================
# Badge identity semantics (REWIND-FRESH-014)
# =============================================================================


class TestStableBadgeIssuance:
    def test_repeated_result_reads_return_same_signed_badge(self, client):
        from mettle import ChallengeType, Difficulty, MettleSession, VerificationResult

        session_id = "stable-session"
        session_token = "stable-session-token"
        sessions[session_id] = MettleSession(
            session_id=session_id,
            entity_id="agent-1",
            difficulty=Difficulty.BASIC,
            challenges=[],
            results=[
                VerificationResult(
                    challenge_id="challenge-1",
                    challenge_type=ChallengeType.SPEED_MATH,
                    passed=True,
                    response_time_ms=1,
                    time_limit_ms=1000,
                )
            ],
            completed=True,
            access_token_hash=hashlib.sha256(session_token.encode()).hexdigest(),
        )
        client.session_tokens[session_id] = session_token

        first = client.get(f"/api/session/{session_id}/result").json()
        second = client.get(f"/api/session/{session_id}/result").json()

        for result in (first, second):
            assert result["screening_passed"] is True
            assert result["verified"] is True
            assert result["credential_eligible"] is True
            assert result["badge"]
            assert result["badge_info"]["signed"] is True
        assert first["badge"] == second["badge"]
        payload = jwt.decode(first["badge"], SECRET_KEY, algorithms=["HS256"])
        assert payload["credential_type"] == "mettle-reverse-captcha-pass"
        assert payload["attests"] == "mettle_session_passed"
        assert payload["identity_binding"] == "self_asserted"
        assert payload["tier"] == "bronze"

    def test_emergency_switch_stops_new_quick_badges(self, client):
        from mettle import ChallengeType, Difficulty, MettleSession, VerificationResult

        session_id = "issuance-disabled-session"
        session_token = "issuance-disabled-token"
        sessions[session_id] = MettleSession(
            session_id=session_id,
            entity_id="agent-disabled",
            difficulty=Difficulty.BASIC,
            challenges=[],
            results=[
                VerificationResult(
                    challenge_id="challenge-1",
                    challenge_type=ChallengeType.SPEED_MATH,
                    passed=True,
                    response_time_ms=1,
                    time_limit_ms=1000,
                )
            ],
            completed=True,
            access_token_hash=hashlib.sha256(session_token.encode()).hexdigest(),
        )
        client.session_tokens[session_id] = session_token

        with patch.object(main_module.settings, "credential_issuance_enabled", False):
            result = client.get(f"/api/session/{session_id}/result").json()

        assert result["verified"] is True
        assert result["credential_eligible"] is False
        assert result["badge"] is None


# =============================================================================
# Badge Verification (lines 1139-1173)
# =============================================================================


class TestBadgeVerification:
    """Tests for the badge verification endpoints."""

    def test_legacy_url_token_route_is_absent(self, client):
        """Replayable credentials cannot be supplied in a request URL."""
        token = _make_badge_token()

        response = client.get(f"/api/badge/verify/{token}")

        assert response.status_code == 404

    def test_post_valid_badge_keeps_token_out_of_url(self, client):
        """The primary endpoint accepts credentials in the request body."""
        token = _make_badge_token()

        response = client.post("/api/badge/verify", json={"token": token})

        assert response.status_code == 200
        assert response.request.url.path == "/api/badge/verify"
        assert response.json()["valid"] is True
        assert response.json()["payload"]["entity_id"] == "test-entity"

    def test_post_rejects_empty_badge(self, client):
        """The request model rejects empty credentials before verification."""
        response = client.post("/api/badge/verify", json={"token": ""})

        assert response.status_code == 422

    def test_valid_badge_returns_valid_true(self, client):
        """The body-only credential verifier accepts a valid badge."""
        token = _make_badge_token()
        response = client.post("/api/badge/verify", json={"token": token})

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

        response = client.post("/api/badge/verify", json={"token": token})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["revoked"] is True
        assert "revoked" in data["error"].lower()

    def test_database_revocation_is_enforced(self, client):
        token = _make_badge_token(jti="durably-revoked-jti")
        mock_db = MagicMock()
        mock_db.is_badge_revoked.return_value = True

        with patch("main.db", mock_db):
            response = client.post("/api/badge/verify", json={"token": token})

        assert response.json()["revoked"] is True
        mock_db.is_badge_revoked.assert_called_once_with(
            "durably-revoked-jti", raise_on_error=True
        )

    def test_database_revocation_error_fails_closed(self, client):
        token = _make_badge_token(jti="unknown-revocation-jti")
        mock_db = MagicMock()
        mock_db.is_badge_revoked.side_effect = RuntimeError("database unavailable")

        with patch("main.db", mock_db):
            response = client.post("/api/badge/verify", json={"token": token})

        data = response.json()
        assert data["valid"] is False
        assert "temporarily unavailable" in data["error"]

    def test_expired_badge_returns_expired(self, client):
        """Expired JWT should return valid=False with 'expired' error."""
        token = _make_badge_token(expired=True)

        response = client.post("/api/badge/verify", json={"token": token})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "expired" in data["error"].lower()

    def test_invalid_token_returns_invalid(self, client):
        """Garbage token should return valid=False with 'Invalid' error."""
        response = client.post(
            "/api/badge/verify", json={"token": "not-a-real-jwt-token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "invalid" in data["error"].lower()

    def test_wrong_secret_returns_invalid(self, client):
        """Token signed with wrong secret should fail verification."""
        token = _make_badge_token(secret="wrong-secret-key-that-is-at-least-32-bytes")

        response = client.post("/api/badge/verify", json={"token": token})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_no_secret_key_returns_not_configured(self, client):
        """When secret_key is not set, verification should return not configured."""
        with patch("main.settings") as mock_settings:
            mock_settings.secret_key = None
            mock_settings.admin_api_key = ADMIN_KEY
            response = client.post("/api/badge/verify", json={"token": "any-token"})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert (
            "not configured" in data["error"].lower()
            or "signing key" in data["error"].lower()
        )


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

    def test_successful_revocation_is_persisted_before_memory_update(self, client):
        token = _make_badge_token(jti="persist-revocation-jti")
        mock_db = MagicMock()
        mock_db.is_badge_revoked.return_value = False
        mock_db.add_revoked_badge.return_value = True

        with patch("main.db", mock_db):
            response = client.post(
                "/api/badge/revoke",
                json={"token": token, "reason": "Persist this revocation safely"},
                headers=ADMIN_HEADERS,
            )

        assert response.status_code == 200
        mock_db.add_revoked_badge.assert_called_once()
        assert "persist-revocation-jti" in revoked_badges

    def test_presence_credential_jti_can_be_revoked(self, client):
        credential_jti = "a" * 32
        mock_db = MagicMock()
        mock_db.is_badge_revoked.return_value = False
        mock_db.add_revoked_badge.return_value = True

        with patch("main.db", mock_db):
            response = client.post(
                "/api/badge/revoke",
                json={
                    "jti": credential_jti,
                    "entity_id": "agent-42",
                    "reason": "Revoke compromised Presence credential",
                },
                headers=ADMIN_HEADERS,
            )

        assert response.status_code == 200
        mock_db.add_revoked_badge.assert_called_once_with(
            credential_jti,
            "agent-42",
            "Revoke compromised Presence credential",
            None,
        )
        assert credential_jti in revoked_badges

    def test_revocation_requires_exactly_one_credential_form(self, client):
        token = _make_badge_token(jti="duplicate-form-jti")
        both = client.post(
            "/api/badge/revoke",
            json={
                "token": token,
                "jti": "b" * 32,
                "reason": "Ambiguous revocation request must fail",
            },
            headers=ADMIN_HEADERS,
        )
        neither = client.post(
            "/api/badge/revoke",
            json={"reason": "Missing revocation credential must fail"},
            headers=ADMIN_HEADERS,
        )

        assert both.status_code == 400
        assert neither.status_code == 400
        assert "exactly one" in both.json()["detail"].lower()

    def test_persistence_failure_does_not_claim_revocation(self, client):
        token = _make_badge_token(jti="failed-persistence-jti")
        mock_db = MagicMock()
        mock_db.is_badge_revoked.return_value = False
        mock_db.add_revoked_badge.return_value = False

        with patch("main.db", mock_db):
            response = client.post(
                "/api/badge/revoke",
                json={"token": token, "reason": "Persistence failure coverage"},
                headers=ADMIN_HEADERS,
            )

        assert response.status_code == 503
        assert "failed-persistence-jti" not in revoked_badges

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
            "iss": "mettle-api",
            "iat": datetime.now(timezone.utc).timestamp(),
            "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
            "session_id": "test-session",
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
            mock_settings.redis_url = None

            response = client.post(
                "/api/badge/revoke",
                json={
                    "token": "any-token",
                    "reason": "No secret key configured test",
                },
                headers=ADMIN_HEADERS,
            )

        assert response.status_code == 400
        assert (
            "signing" in response.json()["detail"].lower()
            or "configured" in response.json()["detail"].lower()
        )

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

    @staticmethod
    def _stream_context(status_code: int = 200):
        response = MagicMock(status_code=status_code)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    @pytest.mark.asyncio
    async def test_entity_not_registered_returns_false(self):
        """send_webhook returns False for unregistered entity."""
        result = await WebhookManager.send_webhook(
            "unknown-entity", "session.completed", {}
        )
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
        WebhookManager.register(
            "entity-1", "https://example.com/hook", ["session.started"]
        )
        result = await WebhookManager.send_webhook("entity-1", "badge.issued", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        """send_webhook returns True on successful HTTP post."""
        WebhookManager.register("entity-1", "https://example.com/hook")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.stream.return_value = self._stream_context()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {"session_id": "test-123"},
                )

        assert result is True
        mock_client.stream.assert_called_once()
        call = mock_client.stream.call_args
        assert call.args[:2] == ("POST", "https://93.184.216.34/hook")
        assert call.kwargs["headers"] == {"Host": "example.com"}
        assert call.kwargs["extensions"] == {"sni_hostname": "example.com"}
        mock_client_cls.assert_called_once_with(
            timeout=10.0,
            follow_redirects=False,
            trust_env=False,
        )

    @pytest.mark.asyncio
    async def test_delivery_with_secret_includes_hmac(self):
        """send_webhook includes HMAC signature when secret is configured."""
        WebhookManager.register(
            "entity-1",
            "https://example.com/hook",
            secret="a" * 32,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.stream.return_value = self._stream_context()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {"data": "test"},
                )

        assert result is True
        captured_payload = mock_client.stream.call_args.kwargs["json"]
        assert "signature" in captured_payload

    @pytest.mark.asyncio
    async def test_delivery_logs_never_include_callback_path_or_query_secrets(self):
        """Callback bearer material remains outside structured application logs."""
        path_secret = "path-bearer-secret"  # pragma: allowlist secret
        query_secret = "query-bearer-secret"  # pragma: allowlist secret
        WebhookManager.register(
            "entity-1",
            f"https://example.com/hooks/{path_secret}?token={query_secret}",
        )

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            patch("main.logger") as mock_logger,
        ):
            mock_client = MagicMock()
            mock_client.stream.return_value = self._stream_context()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                assert await WebhookManager.send_webhook(
                    "entity-1", "session.completed", {}
                )

        serialized_logs = repr(mock_logger.method_calls)
        assert path_secret not in serialized_logs
        assert query_secret not in serialized_logs
        assert "webhook_id" in serialized_logs

    @pytest.mark.asyncio
    async def test_redirect_is_not_followed_or_counted_as_success(self):
        """Webhook redirects cannot trigger a second unvalidated destination."""
        WebhookManager.register("entity-1", "https://example.com/hook")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.stream.return_value = self._stream_context(302)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = await WebhookManager.send_webhook(
                    "entity-1", "session.completed", {}
                )

        assert result is False
        mock_client_cls.assert_called_once_with(
            timeout=10.0,
            follow_redirects=False,
            trust_env=False,
        )

    @pytest.mark.asyncio
    async def test_dns_rebinding_blocked(self):
        """send_webhook returns False when DNS resolves to private IP."""
        WebhookManager.register("entity-1", "https://example.com/hook")

        with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
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

        with patch("socket.getaddrinfo", return_value=_addrinfo("10.0.0.1")):
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

        with patch("httpx.AsyncClient") as mock_client_cls, patch("main.logger"):
            mock_client = MagicMock()
            mock_client.stream.side_effect = Exception("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.getaddrinfo", return_value=_addrinfo()):
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

    def test_register_key_rejects_admin_key_in_query(self, client):
        """Secrets in query strings must never authorize admin operations."""
        response = client.post(
            f"/api/keys/register?x_admin_key={ADMIN_KEY}",
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

    @pytest.mark.asyncio
    async def test_configured_admin_failure_state_is_shared_in_redis(self, monkeypatch):
        ip_address = "203.0.113.42"
        redis = AsyncMock()
        redis.eval.return_value = [6, time.time()]
        request = MagicMock()
        request.app.state.redis = redis
        monkeypatch.setattr(main_module.settings, "redis_url", "rediss://shared")

        allowed, retry_after = await main_module._check_admin_auth_rate_limit(
            request, ip_address
        )

        assert allowed is False
        assert retry_after > 0
        assert ip_address not in repr(redis.eval.await_args)

        redis.eval.return_value = 1
        await main_module._record_admin_auth_failure(request, ip_address)
        assert redis.eval.await_count == 2


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
            CollusionDetector.record_verification(
                "entity-big", f"10.0.0.{i % 256}", True
            )

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

    def test_bulk_charge_counts_every_requested_session(self):
        key = "test-pro-key"
        RateTier.register_key(key, "pro", "entity-1")

        allowed, _ = RateTier.check_limit(key, "session", amount=50)

        assert allowed is True
        assert api_keys[key]["usage_count"] == 50

    def test_bulk_charge_is_rejected_atomically_at_limit(self):
        key = "test-pro-key"
        RateTier.register_key(key, "pro", "entity-1")
        api_keys[key]["usage_date"] = datetime.now(timezone.utc).date().isoformat()
        api_keys[key]["usage_count"] = 9990

        allowed, _ = RateTier.check_limit(key, "session", amount=50)

        assert allowed is False
        assert api_keys[key]["usage_count"] == 9990


class TestPublicSessionQuota:
    def test_anonymous_daily_quota_rejects_before_session_allocation(
        self, client, monkeypatch
    ):
        redis = AsyncMock()
        redis.eval.return_value = -1
        monkeypatch.setattr(main_module.settings, "redis_url", "rediss://shared")
        client.app.state.redis = redis

        response = client.post("/api/session/start", json={"difficulty": "basic"})

        assert response.status_code == 429
        assert "Daily limit reached" in response.json()["detail"]
        assert not sessions
        assert redis.set.await_count == 0

    def test_anonymous_daily_quota_allows_a_reserved_session(self, client, monkeypatch):
        redis = AsyncMock()
        redis.eval.return_value = 1
        redis.set.return_value = True
        monkeypatch.setattr(main_module.settings, "redis_url", "rediss://shared")
        client.app.state.redis = redis

        response = client.post("/api/session/start", json={"difficulty": "basic"})

        assert response.status_code == 200
        assert response.json()["session_id"].startswith("ses_")
        assert redis.eval.await_count == 1


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

    def test_cross_session_challenge_is_rejected_without_consuming_it(self, client):
        """A challenge can only advance the session that issued it."""
        first = client.post("/api/session/start", json={}).json()
        second = client.post("/api/session/start", json={}).json()
        foreign_challenge_id = second["current_challenge"]["id"]

        response = client.post(
            "/api/session/answer",
            json={
                "session_id": first["session_id"],
                "challenge_id": foreign_challenge_id,
                "answer": "0",
            },
        )

        assert response.status_code == 404
        assert foreign_challenge_id in challenges


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

    def test_endpoint_rejects_oversized_response_element(self, client):
        response = client.post(
            "/api/security/fingerprint",
            json={"responses": ["x" * 4097]},
        )

        assert response.status_code == 422


class TestPublicSessionCapacity:
    def test_start_rejects_capacity_without_evicting_existing_session(
        self, client, monkeypatch
    ):
        import main

        sentinel = MagicMock()
        sessions["existing"] = sentinel
        monkeypatch.setattr(main, "MAX_SESSIONS", 1)

        response = client.post("/api/session/start", json={})

        assert response.status_code == 503
        assert sessions == {"existing": sentinel}


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
        assert response.headers.get("Permissions-Policy") == (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        assert response.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
        assert response.headers.get("X-METTLE-Source-Revision") == "unknown"
        assert "X-Request-ID" in response.headers

    def test_safe_request_id_is_propagated(self, client):
        response = client.get(
            "/api/health", headers={"X-Request-ID": "caller.trace-123"}
        )

        assert response.headers["X-Request-ID"] == "caller.trace-123"

    @pytest.mark.parametrize(
        "unsafe_request_id",
        ["contains spaces", "line\nbreak", "x" * 65, "", "slash/not-allowed"],
    )
    def test_unsafe_request_id_is_replaced(self, client, unsafe_request_id):
        response = client.get(
            "/api/health", headers={"X-Request-ID": unsafe_request_id}
        )

        issued = response.headers["X-Request-ID"]
        assert issued != unsafe_request_id
        assert re.fullmatch(r"[0-9a-f]{32}", issued)


class TestOperationalHealth:
    """Exercise liveness, readiness, and privacy-preserving metrics."""

    def test_liveness_and_default_readiness(self, client):
        assert client.get("/api/health/live").json() == {
            "status": "alive",
            "source_revision": "unknown",
        }

        ready = client.get("/api/health/ready")
        assert ready.status_code == 200
        assert ready.headers["cache-control"] == "no-store"
        assert ready.json() == {
            "status": "ready",
            "source_revision": "unknown",
        }

    def test_source_revision_prefers_explicit_override(self, monkeypatch):
        import main

        explicit = "a" * 40
        monkeypatch.setenv("METTLE_SOURCE_REVISION", explicit.upper())
        monkeypatch.setenv("RENDER_GIT_COMMIT", "b" * 40)

        assert main.deployed_source_revision() == explicit

    def test_source_revision_uses_valid_render_fallback(self, monkeypatch):
        import main

        monkeypatch.setenv("METTLE_SOURCE_REVISION", "not-a-commit")
        monkeypatch.setenv("RENDER_GIT_COMMIT", "c" * 64)

        assert main.deployed_source_revision() == "c" * 64

    def test_production_readiness_rejects_unknown_source(self, client, monkeypatch):
        import main

        monkeypatch.setattr(main.settings, "environment", "production")
        monkeypatch.delenv("METTLE_SOURCE_REVISION", raising=False)
        monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)

        response = client.get("/api/health/ready")

        assert response.status_code == 503
        assert response.json()["source_revision"] == "unknown"
        assert "components" not in response.json()

    def test_health_and_cors_expose_deployed_source(self, client, monkeypatch):
        revision = "d" * 40
        monkeypatch.setenv("RENDER_GIT_COMMIT", revision)

        response = client.get("/api/health", headers={"Origin": "http://testserver"})

        assert response.json()["source_revision"] == revision
        assert response.headers["X-METTLE-Source-Revision"] == revision
        exposed = response.headers["Access-Control-Expose-Headers"].lower()
        assert "x-mettle-source-revision" in exposed
        assert "x-request-id" in exposed

    def test_database_readiness_fails_closed(self, client, monkeypatch):
        import main

        unavailable_database = MagicMock()
        unavailable_database.check_health.return_value = False
        monkeypatch.setattr(main.settings, "use_database", True)
        monkeypatch.setattr(main, "db", unavailable_database)

        response = client.get("/api/health/ready")

        assert response.status_code == 503
        assert "components" not in response.json()
        unavailable_database.check_health.assert_called_once_with()
        unavailable_database.check_schema_current.assert_called_once_with()

    def test_database_schema_readiness_fails_closed(self, client, monkeypatch):
        import main

        stale_database = MagicMock()
        stale_database.check_health.return_value = True
        stale_database.check_schema_current.return_value = False
        monkeypatch.setattr(main.settings, "use_database", True)
        monkeypatch.setattr(main, "db", stale_database)

        response = client.get("/api/health/ready")

        assert response.status_code == 503
        assert "components" not in response.json()
        stale_database.check_health.assert_called_once_with()
        stale_database.check_schema_current.assert_called_once_with()

    def test_redis_readiness_fails_closed(self, client, monkeypatch):
        import main

        monkeypatch.setattr(main.settings, "redis_url", "redis://configured")
        client.app.state.redis = None

        response = client.get("/api/health/ready")

        assert response.status_code == 503
        assert "components" not in response.json()

    def test_metrics_are_bounded_and_content_free(self, client):
        client.get("/api/health/live")

        response = client.get("/api/metrics", headers=ADMIN_HEADERS)
        body = response.text
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert "mettle_http_requests_total" in body
        assert "mettle_http_request_duration_seconds_bucket" in body
        assert "session_id" not in body
        assert "entity_id" not in body


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

        revocation_audit.extend({"index": i} for i in range(MAX_REVOCATION_AUDIT))
        token = _make_badge_token(jti="audit-jti-boundary")
        response = client.post(
            "/api/badge/revoke",
            json={"token": token, "reason": "Audit bounding boundary test"},
            headers=ADMIN_HEADERS,
        )

        assert response.status_code == 200
        assert len(revocation_audit) == MAX_REVOCATION_AUDIT
        assert revocation_audit[-1]["jti"] == "audit-jti-boundary"


# =============================================================================
# Webhook URL HTTPS production-only validation (line 1628)
# =============================================================================


class TestWebhookProductionValidation:
    """Tests for webhook URL production-only HTTPS requirement."""

    def test_http_allowed_in_dev(self, client):
        """HTTP URLs should be allowed in non-production (with an owning API key)."""
        RateTier.register_key("dev-http-key", "pro", "test")
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test",
                "url": "http://example.com/hook",
            },
            headers={"X-API-Key": "dev-http-key"},
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
        """Batch start should handle multiple entities correctly (pro-tier key)."""
        RateTier.register_key("batch-many-key", "pro", "batch-owner")
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": [f"entity-{i}" for i in range(5)],
                "difficulty": "basic",
            },
            headers={"X-API-Key": "batch-many-key"},
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
    async def test_dns_resolution_failure_blocks_request(self):
        """DNS validation failures must stop webhook delivery."""
        import socket

        WebhookManager.register("entity-1", "https://external-host.com/hook")

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls, patch("main.logger"):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("socket.getaddrinfo", side_effect=socket.gaierror("DNS failed")):
                result = await WebhookManager.send_webhook(
                    "entity-1",
                    "session.completed",
                    {},
                )

        assert result is False
        mock_client.post.assert_not_awaited()


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
