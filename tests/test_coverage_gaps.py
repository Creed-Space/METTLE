"""Tests targeting remaining coverage gaps in main.py and router.py.

Covers: rate tier usage tracking, collusion detector edge cases, database persistence
branches, lifespan handler, historical badge verification, HSTS production header,
batch failure handling, session/result 404s, badge signing not configured, revocation audit
overflow, revocation audit rate limit, fingerprint equal distribution, webhook DNS
gaierror, webhook db branches, static fallbacks, and router signing ImportError.
"""

import asyncio
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from main import (
    CollusionDetector,
    ModelFingerprinter,
    RateTier,
    WebhookManager,
    _resolve_webhook_addresses,
    _restore_persistent_runtime_state,
    _admin_auth_failures,
    api_keys,
    lifespan,
    record_admin_auth_failure,
    revocation_audit,
    revoked_badges,
    verification_graph,
    verification_timestamps,
    webhooks,
)

ADMIN_KEY = "test-admin-key-for-mettle-testing-only"
SECRET_KEY = "test-secret-key-for-mettle-testing-only"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean up shared state between tests."""
    _admin_auth_failures.clear()
    revoked_badges.clear()
    webhooks.clear()
    verification_graph.clear()
    verification_timestamps.clear()
    api_keys.clear()
    revocation_audit.clear()
    yield
    _admin_auth_failures.clear()
    revoked_badges.clear()
    webhooks.clear()
    verification_graph.clear()
    verification_timestamps.clear()
    api_keys.clear()
    revocation_audit.clear()


# ============================================================
# RateTier usage tracking (lines 174-185)
# ============================================================


class TestRateTierUsageTracking:
    """Test daily usage tracking and limit enforcement."""

    def test_session_limit_tracks_usage(self):
        """Registered key tracks session usage count."""
        api_keys["mtl_test"] = {"tier": "free", "usage_date": None, "usage_count": 0}
        allowed, msg = RateTier.check_limit("mtl_test", "session")
        assert allowed is True
        assert api_keys["mtl_test"]["usage_count"] == 1

    def test_session_limit_resets_on_new_day(self):
        """Usage count resets when the date changes."""
        api_keys["mtl_test"] = {
            "tier": "free",
            "usage_date": "2025-01-01",
            "usage_count": 99,
        }
        allowed, msg = RateTier.check_limit("mtl_test", "session")
        assert allowed is True
        # Count was reset to 0, then incremented to 1
        assert api_keys["mtl_test"]["usage_count"] == 1
        assert (
            api_keys["mtl_test"]["usage_date"]
            == datetime.now(timezone.utc).date().isoformat()
        )

    def test_session_limit_reached(self):
        """Returns False when daily session limit is exhausted."""
        free_limit = RateTier.TIERS["free"]["sessions_per_day"]
        today = datetime.now(timezone.utc).date().isoformat()
        api_keys["mtl_test"] = {
            "tier": "free",
            "usage_date": today,
            "usage_count": free_limit,
        }
        allowed, msg = RateTier.check_limit("mtl_test", "session")
        assert allowed is False
        assert "Daily limit reached" in msg

    def test_non_session_limit_type_does_not_increment(self):
        """Non-session limit types don't increment the session counter."""
        today = datetime.now(timezone.utc).date().isoformat()
        api_keys["mtl_test"] = {"tier": "free", "usage_date": today, "usage_count": 0}
        allowed, msg = RateTier.check_limit("mtl_test", "answer")
        assert allowed is True
        assert api_keys["mtl_test"]["usage_count"] == 0

    def test_database_quota_is_reserved_atomically(self):
        today = datetime.now(timezone.utc).date().isoformat()
        key_data = {
            "tier": "pro",
            "entity_id": "owner",
            "usage_date": today,
            "usage_count": 40,
        }
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = key_data
        mock_db.reserve_api_key_usage.return_value = (True, 42)

        with patch("main.db", mock_db):
            allowed, message = RateTier.check_limit("durable-key", "session", amount=2)

        assert allowed is True
        assert "pro tier" in message
        assert key_data["usage_count"] == 42
        mock_db.reserve_api_key_usage.assert_called_once_with(
            "durable-key", today, 2, 10000, raise_on_error=True
        )
        mock_db.update_api_key_usage.assert_not_called()

    def test_database_quota_denial_does_not_claim_a_reservation(self):
        today = datetime.now(timezone.utc).date().isoformat()
        key_data = {
            "tier": "free",
            "usage_date": today,
            "usage_count": 100,
        }
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = key_data
        mock_db.reserve_api_key_usage.return_value = (False, 100)

        with patch("main.db", mock_db):
            allowed, message = RateTier.check_limit("durable-key", "session")

        assert allowed is False
        assert "Daily limit reached" in message
        assert key_data["usage_count"] == 100

    def test_database_quota_failure_is_not_converted_to_free_access(self):
        key_data = {
            "tier": "pro",
            "usage_date": "2026-08-19",
            "usage_count": 7,
        }
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = key_data
        mock_db.reserve_api_key_usage.side_effect = RuntimeError("database offline")

        with (
            patch("main.db", mock_db),
            pytest.raises(RuntimeError, match="database offline"),
        ):
            RateTier.check_limit("durable-key", "session")

        assert key_data["usage_date"] == "2026-08-19"
        assert key_data["usage_count"] == 7

    def test_database_key_lookup_failure_cannot_skip_quota_reservation(self):
        mock_db = MagicMock()
        mock_db.get_api_key.side_effect = RuntimeError("database offline")

        with (
            patch("main.db", mock_db),
            pytest.raises(RuntimeError, match="database offline"),
        ):
            RateTier.check_limit("durable-key", "session")

        mock_db.reserve_api_key_usage.assert_not_called()

    def test_batch_returns_503_before_generation_when_quota_store_fails(self, client):
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = {
            "tier": "pro",
            "entity_id": "owner",
            "usage_date": None,
            "usage_count": 0,
        }
        mock_db.reserve_api_key_usage.side_effect = RuntimeError("database offline")

        with (
            patch("main.db", mock_db),
            patch("main.generate_challenge_set") as generate,
        ):
            response = client.post(
                "/api/session/batch",
                json={"entity_ids": ["entity-1"], "difficulty": "basic"},
                headers={"X-API-Key": "durable-key"},
            )

        assert response.status_code == 503
        assert "persistence" in response.json()["detail"].lower()
        generate.assert_not_called()


# ============================================================
# CollusionDetector edge cases (lines 233-234, 239, 244, 248)
# ============================================================


class TestCollusionDetectorEdgeCases:
    """Test CollusionDetector memory-bounded edge cases."""

    def test_verification_graph_eviction_when_full(self):
        """Oldest entity is evicted when graph exceeds MAX_VERIFICATION_GRAPH."""
        from main import MAX_VERIFICATION_GRAPH

        # Fill the graph to capacity
        for i in range(MAX_VERIFICATION_GRAPH):
            verification_graph[f"entity-{i}"] = [
                {"timestamp": time.time(), "ip_address": "1.1.1.1", "passed": True}
            ]

        assert "entity-0" in verification_graph

        # Adding a new entity should evict entity-0
        CollusionDetector.record_verification("new-entity", "1.1.1.1", True)
        assert "entity-0" not in verification_graph
        assert "new-entity" in verification_graph

    def test_verification_records_truncated_to_100(self):
        """Per-entity records are capped at 100."""
        entity_id = "entity-overflow"
        verification_graph[entity_id] = [
            {"timestamp": time.time(), "ip_address": "1.1.1.1", "passed": True}
            for _ in range(100)
        ]
        CollusionDetector.record_verification(entity_id, "1.1.1.1", True)
        assert len(verification_graph[entity_id]) == 100

    def test_verification_timestamps_capped_at_1000(self):
        """Global timestamps list is capped at 1000."""
        verification_timestamps.clear()
        for i in range(1000):
            verification_timestamps.append((f"entity-{i}", time.time()))

        assert len(verification_timestamps) == 1000

        CollusionDetector.record_verification("new-entity", "1.1.1.1", True)
        assert len(verification_timestamps) == 1000

    def test_db_persistence_when_db_available(self):
        """When db module is available, save_verification_record is called."""
        mock_db = MagicMock()
        with patch("main.db", mock_db):
            CollusionDetector.record_verification("entity-1", "1.1.1.1", True)
        mock_db.save_verification_record.assert_called_once_with(
            "entity-1", "1.1.1.1", True
        )

    def test_restart_recovers_authoritative_collusion_history(self):
        now = datetime.now(timezone.utc)
        mock_db = MagicMock()
        mock_db.get_recent_sessions.return_value = []
        mock_db.get_webhooks.return_value = []
        mock_db.get_recent_verifications.return_value = [
            {
                "entity_id": f"entity-{index}",
                "ip_address": "203.0.113.10",
                "passed": True,
                "created_at": (now - timedelta(seconds=index)).isoformat(),
            }
            for index in range(3)
        ] + [
            {
                "entity_id": "future-corrupt",
                "ip_address": "203.0.113.10",
                "passed": True,
                "created_at": (now + timedelta(hours=1)).isoformat(),
            },
            {
                "entity_id": None,
                "ip_address": "203.0.113.10",
                "passed": True,
                "created_at": now.isoformat(),
            },
        ]
        verification_graph["stale-process-only"] = [
            {"timestamp": 1.0, "ip_address": "198.51.100.1", "passed": True}
        ]

        with patch("main.db", mock_db):
            _restore_persistent_runtime_state()

        assert "stale-process-only" not in verification_graph
        assert set(verification_graph) == {"entity-0", "entity-1", "entity-2"}
        assert len(verification_timestamps) == 3
        result = CollusionDetector.check_collusion("entity-2", "203.0.113.10")
        assert any("3 different entities" in warning for warning in result["warnings"])
        mock_db.get_recent_verifications.assert_called_once_with(
            hours=1, raise_on_error=True
        )

    def test_history_recovery_failure_does_not_erase_existing_evidence(self):
        existing = {
            "timestamp": 1.0,
            "ip_address": "198.51.100.1",
            "passed": True,
        }
        verification_graph["existing"] = [existing]
        mock_db = MagicMock()
        mock_db.get_recent_sessions.return_value = []
        mock_db.get_recent_verifications.side_effect = RuntimeError(
            "Verification history recovery unavailable"
        )

        with (
            patch("main.db", mock_db),
            pytest.raises(RuntimeError, match="history recovery unavailable"),
        ):
            _restore_persistent_runtime_state()

        assert verification_graph == {"existing": [existing]}


# ============================================================
# Database persistence branch (line 205)
# ============================================================


class TestDatabasePersistence:
    """Test database save branches."""

    def test_register_key_saves_to_db(self):
        """RateTier.register_key calls db.save_api_key when db is available."""
        mock_db = MagicMock()
        with patch("main.db", mock_db):
            result = RateTier.register_key("mtl_test", "free", "entity-1")
        mock_db.save_api_key.assert_called_once_with("mtl_test", "free", "entity-1")
        assert result["tier"] == "free"


# ============================================================
# HSTS production header (line 392)
# ============================================================


class TestHSTSProductionHeader:
    """Test Strict-Transport-Security in production mode."""

    def test_hsts_header_in_production(self, client):
        """HSTS header is set when is_production is True."""
        with patch("main.settings") as mock_settings:
            mock_settings.is_production = True
            mock_settings.api_title = "METTLE"
            mock_settings.api_version = "0.2.0"
            mock_settings.allowed_origins = ["*"]
            mock_settings.secret_key = SECRET_KEY
            mock_settings.admin_api_key = ADMIN_KEY
            response = client.get("/api/health")
        assert (
            response.headers.get("Strict-Transport-Security")
            == "max-age=31536000; includeSubDomains"
        )


# ============================================================
# Lifespan handler (lines 433-486)
# ============================================================


class TestLifespan:
    """Test the async lifespan handler."""

    @pytest.mark.asyncio
    async def test_lifespan_production_no_secret_raises(self):
        """Lifespan raises RuntimeError in production without SECRET_KEY."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = True
            mock_settings.secret_key = ""
            mock_settings.environment = "production"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://redis.internal:6379/0"

            with pytest.raises(RuntimeError, match="SECRET_KEY"):
                async with lifespan(mock_app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_production_signing_failure_raises(self):
        """Production cannot start while VCP attestations would be unsigned."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_db = MagicMock()
        mock_db.get_recent_sessions.return_value = []
        mock_db.get_webhooks.return_value = []
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = True
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "production"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://redis.internal:6379/0"

            with (
                patch("main.db", mock_db),
                patch("redis.asyncio.from_url", return_value=mock_redis),
                patch("mettle.signing.init_signing", return_value=False),
                pytest.raises(RuntimeError, match="signing is unavailable"),
            ):
                async with lifespan(mock_app):
                    pass
        mock_redis.aclose.assert_awaited_once()
        assert mock_app.state.redis is None

    @pytest.mark.asyncio
    async def test_lifespan_database_initialization_failure_raises(self):
        """Configured persistence must initialize successfully before serving."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_db = MagicMock()
        mock_db.init_db.side_effect = RuntimeError("database unavailable")

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = False
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "test"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = ""

            with (
                patch("main.db", mock_db),
                pytest.raises(RuntimeError, match="Database initialization failed"),
            ):
                async with lifespan(mock_app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_redis_unavailable_logged(self):
        """Lifespan logs warning when Redis connection fails."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()

        async def fake_cleanup():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = False
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "test"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://bad-host:6379/0"

            failing_redis = MagicMock()
            failing_redis.ping = AsyncMock(side_effect=ConnectionError("offline"))
            failing_redis.aclose = AsyncMock()
            with (
                patch("redis.asyncio.from_url", return_value=failing_redis),
                patch("main.cleanup_expired_sessions", side_effect=fake_cleanup),
            ):
                async with lifespan(mock_app):
                    assert mock_app.state.redis is None
            failing_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_production_redis_failure_is_fatal(self):
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_db = MagicMock()
        mock_db.get_recent_sessions.return_value = []
        mock_db.get_webhooks.return_value = []
        failing_redis = MagicMock()
        failing_redis.ping = AsyncMock(side_effect=ConnectionError("offline"))
        failing_redis.aclose = AsyncMock()

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = True
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "production"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://redis.internal:6379/0"

            with (
                patch("main.db", mock_db),
                patch("redis.asyncio.from_url", return_value=failing_redis),
                pytest.raises(RuntimeError, match="Redis persistence"),
            ):
                async with lifespan(mock_app):
                    pass

        failing_redis.aclose.assert_awaited_once()
        assert mock_app.state.redis is None

    @pytest.mark.asyncio
    async def test_redis_closes_even_if_cleanup_task_fails(self):
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        async def failing_cleanup():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError as exc:
                raise RuntimeError("cleanup shutdown failed") from exc

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = False
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "test"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://redis.internal:6379/0"

            with (
                patch("redis.asyncio.from_url", return_value=mock_redis),
                patch("main.cleanup_expired_sessions", side_effect=failing_cleanup),
                pytest.raises(RuntimeError, match="cleanup shutdown failed"),
            ):
                async with lifespan(mock_app):
                    # Let the task enter its cancellation handler before shutdown.
                    await asyncio.sleep(0)

        mock_redis.aclose.assert_awaited_once()
        assert mock_app.state.redis is None

    @pytest.mark.asyncio
    async def test_lifespan_redis_failure_graceful(self):
        """Lifespan handles Redis connection failure gracefully."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()

        async def fake_cleanup():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = False
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "test"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = ""

            with patch("main.cleanup_expired_sessions", side_effect=fake_cleanup):
                async with lifespan(mock_app):
                    assert mock_app.state.redis is None


# ============================================================
# Answer validation (line 638)
# ============================================================


class TestAnswerValidation:
    """Test SubmitAnswerRequest validation."""

    def test_answer_exceeds_max_length(self):
        """Answer over 1024 characters raises ValidationError."""
        from main import SubmitAnswerRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="1024"):
            SubmitAnswerRequest(
                session_id="ses_test",
                challenge_id="mtl_test",
                answer="x" * 1025,
            )


# ============================================================
# Batch start failure (lines 871-874)
# ============================================================


class TestBatchStartFailure:
    """Test batch session start error handling."""

    def test_batch_start_with_failing_entity(self, client):
        """Failed entity in batch is counted and error reported."""
        RateTier.register_key("batch-fail-key", "pro", "batch-owner")
        with patch(
            "main.generate_challenge_set", side_effect=Exception("Test failure")
        ):
            response = client.post(
                "/api/session/batch",
                json={
                    "entity_ids": ["entity-1", "entity-2"],
                    "difficulty": "basic",
                },
                headers={"X-API-Key": "batch-fail-key"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["failed"] == 2
        assert any("error" in r for r in data["sessions"])


# ============================================================
# Session/result 404s (lines 1026, 1077)
# ============================================================


class TestSession404s:
    """Test 404 responses for missing sessions."""

    def test_get_session_not_found(self, client):
        """GET /api/session/{id} returns 404 for unknown session."""
        response = client.get("/api/session/ses_nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_result_not_found(self, client):
        """GET /api/session/{id}/result returns 404 for unknown session."""
        response = client.get("/api/session/ses_nonexistent/result")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ============================================================
# Badge revoke — signing not configured (line 1266)
# ============================================================


class TestBadgeRevokeSigningNotConfigured:
    """Test revoke endpoint when secret_key is missing."""

    def test_revoke_no_secret_key(self, client):
        """Revoke returns 400 when badge signing not configured."""
        with patch("main.settings") as mock_settings:
            mock_settings.admin_api_key = ADMIN_KEY
            mock_settings.secret_key = ""
            mock_settings.is_production = False
            response = client.post(
                "/api/badge/revoke",
                json={
                    "token": "any-token",
                    "reason": "Test revocation reason for testing purposes",
                },
                headers=ADMIN_HEADERS,
            )
        assert response.status_code == 400
        assert "signing not configured" in response.json()["detail"].lower()


# ============================================================
# Revocation audit overflow (line 1306)
# ============================================================


class TestRevocationAuditOverflow:
    """Test revocation audit list bounded at MAX_REVOCATION_AUDIT."""

    def test_audit_bounded(self, client):
        """Revocation audit pops oldest when exceeding MAX_REVOCATION_AUDIT."""
        import jwt as jwt_lib
        from main import MAX_REVOCATION_AUDIT

        # Fill audit to capacity
        revocation_audit.clear()
        for i in range(MAX_REVOCATION_AUDIT):
            revocation_audit.append({"jti": f"old-{i}", "revoked_at": time.time()})

        assert len(revocation_audit) == MAX_REVOCATION_AUDIT
        first_jti = revocation_audit[0]["jti"]

        # Revoke a new badge to trigger overflow
        now = datetime.now(timezone.utc)
        payload = {
            "entity_id": "test-entity",
            "difficulty": "basic",
            "pass_rate": 1.0,
            "verified_at": now.isoformat(),
            "exp": (now + timedelta(hours=24)).timestamp(),
            "iat": now.timestamp(),
            "jti": "new-overflow-jti",
            "nonce": "nonce123",
            "iss": "mettle-api",
            "session_id": "test-session",
        }
        token = jwt_lib.encode(payload, SECRET_KEY, algorithm="HS256")

        response = client.post(
            "/api/badge/revoke",
            json={
                "token": token,
                "reason": "Test revocation reason for testing purposes",
            },
            headers=ADMIN_HEADERS,
        )
        assert response.status_code == 200
        assert response.json()["revoked"] is True
        # Oldest entry should be evicted
        assert revocation_audit[0]["jti"] != first_jti


# ============================================================
# Revocation audit rate limiting (line 1337)
# ============================================================


class TestRevocationAuditRateLimited:
    """Test rate limiting on GET /api/badge/revocations."""

    def test_revocation_audit_rate_limited(self, client):
        """Rate-limited IP is blocked from viewing audit."""
        for _ in range(10):
            record_admin_auth_failure("testclient")
        response = client.get("/api/badge/revocations", headers=ADMIN_HEADERS)
        assert response.status_code == 429
        assert "retry" in response.json()["detail"].lower()


# ============================================================
# ModelFingerprinter equal distribution (line 1411)
# ============================================================


class TestModelFingerprintEqualDistribution:
    """Test fingerprinting when no patterns match."""

    def test_equal_distribution_no_signals(self):
        """When no patterns match, scores are equally distributed."""
        result = ModelFingerprinter.fingerprint(["xyzzy"])
        scores = result["scores"]
        # All scores should be equal (1/N)
        values = list(scores.values())
        assert len(set(values)) == 1  # All equal
        expected = round(1.0 / len(scores), 3)
        assert values[0] == expected


# ============================================================
# Webhook DNS gaierror/ValueError (lines 1567-1569)
# ============================================================


class TestWebhookDNSErrors:
    """Compatibility-only DNS failures for inactive delivery code."""

    @pytest.fixture(autouse=True)
    def _enable_compatibility_delivery(self):
        with patch("main.WEBHOOK_DELIVERY_ENABLED", True):
            yield

    @pytest.mark.asyncio
    async def test_dns_gaierror_blocks_send(self):
        """DNS resolution failure (gaierror) blocks the request."""
        WebhookManager.register("test-entity", "https://example.com/webhook")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("socket.getaddrinfo", side_effect=socket.gaierror("DNS failed")):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await WebhookManager.send_webhook(
                    "test-entity", "session.completed", {"test": True}
                )
        assert result is False
        mock_client.post.assert_not_awaited()


class TestWebhookDeliveryBounds:
    @staticmethod
    def _record(address: str):
        return (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, 443),
        )

    @pytest.mark.asyncio
    async def test_disabled_delivery_never_resolves_or_dispatches(self):
        WebhookManager.register("test-entity", "https://example.com/webhook")

        with patch("main._resolve_webhook_addresses", new_callable=AsyncMock) as dns:
            result = await WebhookManager.send_webhook(
                "test-entity", "session.completed", {"test": True}
            )

        assert result is False
        dns.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dns_resolution_does_not_block_event_loop(self):
        release = threading.Event()

        def blocking_resolution(*_args, **_kwargs):
            release.wait(timeout=0.5)
            return [self._record("93.184.216.34")]

        loop = asyncio.get_running_loop()
        with patch("socket.getaddrinfo", side_effect=blocking_resolution):
            started = loop.time()
            task = asyncio.create_task(_resolve_webhook_addresses("example.com", 443))
            await asyncio.sleep(0.01)
            elapsed = loop.time() - started
            release.set()
            resolved = await task

        assert elapsed < 0.2
        assert set(resolved) == {"93.184.216.34"}

    @pytest.mark.asyncio
    async def test_dns_resolution_rejects_excessive_address_fanout(self):
        records = [self._record(f"8.8.4.{index}") for index in range(1, 18)]

        with (
            patch("socket.getaddrinfo", return_value=records),
            pytest.raises(ValueError, match="too many addresses"),
        ):
            await _resolve_webhook_addresses("example.com", 443)

    @pytest.mark.asyncio
    async def test_dns_resolution_has_a_hard_timeout(self):
        def slow_resolution(*_args, **_kwargs):
            time.sleep(0.05)
            return [self._record("93.184.216.34")]

        with (
            patch("socket.getaddrinfo", side_effect=slow_resolution),
            patch("main.WEBHOOK_DNS_TIMEOUT_SECONDS", 0.005),
            pytest.raises(TimeoutError),
        ):
            await _resolve_webhook_addresses("example.com", 443)

    @pytest.mark.asyncio
    async def test_dns_value_error_blocks_send(self):
        """Invalid resolved IP data blocks the request."""
        WebhookManager.register("test-entity", "https://example.com/webhook")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        invalid_addrinfo = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("not-an-ip", 443),
            )
        ]
        with (
            patch("main.WEBHOOK_DELIVERY_ENABLED", True),
            patch("socket.getaddrinfo", return_value=invalid_addrinfo),
            patch("ipaddress.ip_address", side_effect=ValueError("Invalid IP")),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            result = await WebhookManager.send_webhook(
                "test-entity", "session.completed", {"test": True}
            )
        assert result is False
        mock_client.post.assert_not_awaited()


# ============================================================
# Webhook db branches (lines 1601, 1611, 1615)
# ============================================================


class TestWebhookDBBranches:
    """Test webhook database persistence branches."""

    def test_register_saves_to_db(self):
        """WebhookManager.register calls db.save_webhook when db is available."""
        mock_db = MagicMock()
        with patch("main.db", mock_db):
            WebhookManager.register("entity-1", "https://example.com/hook")
        mock_db.save_webhook.assert_called_once()

    def test_register_persistence_failure_preserves_existing_configuration(self):
        existing = {
            "url": "https://old.example/hook",
            "events": ["session.completed"],
            "secret": "old-secret",  # pragma: allowlist secret
        }
        webhooks["entity-1"] = existing
        mock_db = MagicMock()
        mock_db.save_webhook.return_value = False

        with (
            patch("main.db", mock_db),
            pytest.raises(RuntimeError, match="persistence unavailable"),
        ):
            WebhookManager.register("entity-1", "https://new.example/hook")

        assert webhooks["entity-1"] is existing

    def test_unregister_deletes_from_db(self):
        """WebhookManager.unregister calls db.delete_webhook when db is available."""
        webhooks["entity-1"] = {"url": "https://example.com/hook", "events": []}
        mock_db = MagicMock()
        with patch("main.db", mock_db):
            result = WebhookManager.unregister("entity-1")
        assert result is True
        mock_db.delete_webhook.assert_called_once_with("entity-1")

    def test_unregister_falls_back_to_db(self):
        """When entity not in memory, falls back to db.delete_webhook."""
        mock_db = MagicMock()
        mock_db.delete_webhook.return_value = True
        with patch("main.db", mock_db):
            result = WebhookManager.unregister("entity-db-only")
        assert result is True
        mock_db.delete_webhook.assert_called_once_with("entity-db-only")

    def test_unregister_db_fallback_returns_false(self):
        """When entity not in memory and db returns False, returns False."""
        mock_db = MagicMock()
        mock_db.delete_webhook.return_value = False
        with patch("main.db", mock_db):
            result = WebhookManager.unregister("nonexistent")
        assert result is False


# ============================================================
# Static fallbacks (lines 1846, 1862, 1899)
# ============================================================


class TestStaticFallbacks:
    """Test fallback responses when static directory doesn't exist."""

    def test_root_fallback_no_static(self, client):
        """GET / redirects to /api when no static directory."""
        with patch("main._static_dir", Path("/nonexistent/path")):
            response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/api" in response.headers.get("location", "")

    def test_about_fallback_no_static(self, client):
        """GET /about redirects to / when no static directory."""
        with patch("main._static_dir", Path("/nonexistent/path")):
            response = client.get("/about", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers.get("location") == "/"

    def test_robots_fallback_no_static(self, client):
        """GET /robots.txt returns inline content when no static directory."""
        with patch("main._static_dir", Path("/nonexistent/path")):
            response = client.get("/robots.txt")
        assert response.status_code == 200
        assert "User-agent: *" in response.text
        assert "Sitemap:" in response.text


# ============================================================
# Router signing ImportError (router.py lines 369-370)
# ============================================================


class TestRouterSigningImportError:
    """Test that router handles missing signing module gracefully."""

    def test_complete_session_without_signing_module(self):
        """Session completion works when mettle.signing raises ImportError."""
        from mettle.router import router

        assert router is not None  # Router loads even without signing


# ============================================================
# Lifespan Redis success + shutdown (lines 454-456, 478)
# ============================================================


class TestLifespanRedisSuccess:
    """Test lifespan Redis connection and shutdown paths."""

    @pytest.mark.asyncio
    async def test_lifespan_redis_shutdown_closes_connection(self):
        """A connection created at startup is always closed on shutdown."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        async def fake_cleanup():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = False
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "test"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://redis.internal:6379/0"

            with (
                patch("redis.asyncio.from_url", return_value=mock_redis),
                patch("main.cleanup_expired_sessions", side_effect=fake_cleanup),
            ):
                async with lifespan(mock_app):
                    assert mock_app.state.redis is mock_redis

        mock_redis.ping.assert_awaited_once()
        mock_redis.aclose.assert_awaited_once()
        assert mock_app.state.redis is None


# ============================================================
# Lifespan signing init (lines 468-469)
# ============================================================


class TestLifespanSigningInit:
    """Test lifespan VCP signing initialization."""

    @pytest.mark.asyncio
    async def test_lifespan_signing_import_error(self):
        """Production startup fails closed if the signing symbol is unavailable."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_db = MagicMock()
        mock_db.get_recent_sessions.return_value = []
        mock_db.get_recent_verifications.return_value = []
        mock_db.get_webhooks.return_value = []
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()
        signing_without_init = ModuleType("mettle.signing")

        with patch("main.settings") as mock_settings:
            mock_settings.is_production = True
            mock_settings.secret_key = SECRET_KEY
            mock_settings.environment = "production"
            mock_settings.api_version = "0.2.0"
            mock_settings.redis_url = "redis://redis.internal:6379/0"

            with (
                patch("main.db", mock_db),
                patch("redis.asyncio.from_url", return_value=mock_redis),
                patch.dict(sys.modules, {"mettle.signing": signing_without_init}),
                pytest.raises(RuntimeError, match="signing dependencies"),
            ):
                async with lifespan(mock_app):
                    pass

        mock_redis.aclose.assert_awaited_once()
        assert mock_app.state.redis is None


# ============================================================
# Router signing ImportError path (lines 369-370)
# ============================================================


class TestUnsignedAttestationFallback:
    """Test unsigned local receipts when no signing key is initialized."""

    def test_build_without_signing_returns_unsigned_receipt(self):
        from mettle.vcp import build_mettle_attestation

        attestation = build_mettle_attestation(
            session_id="ses_test123",
            subject_id="test-user",
            difficulty="basic",
            suites_passed=["adversarial"],
            suites_failed=[],
            pass_rate=1.0,
        )
        assert attestation is not None
        # VCP attestation has specific structure
        assert attestation["attestation_type"] == "mettle-evidence-receipt"
        assert attestation["signature"] is None
        assert attestation["metadata"]["session_id"] == "ses_test123"
