"""Tests for METTLE security features."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from main import (
    CollusionDetector,
    ModelFingerprinter,
    RateTier,
    WebhookManager,
    _admin_auth_failures,
    api_keys,
    app,
    limiter,
    revoked_badges,
    verification_graph,
    verification_timestamps,
    webhooks,
)

# Matches METTLE_ADMIN_API_KEY set in conftest.py
TEST_ADMIN_KEY = "test-admin-key-for-mettle-testing-only"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear all in-memory state before each test."""
    verification_graph.clear()
    verification_timestamps.clear()
    api_keys.clear()
    webhooks.clear()
    revoked_badges.clear()
    # SECURITY: admin auth failures is a module global shared across tests;
    # clearing it prevents cross-test 429s when negative tests send wrong keys.
    _admin_auth_failures.clear()
    # Reset slowapi limiter so per-minute caps (e.g. batch 5/min) don't leak between tests.
    limiter.reset()
    yield
    limiter.reset()


# === Collusion Detection Tests ===


class TestCollusionDetector:
    """Tests for CollusionDetector class."""

    def test_record_verification(self):
        """Test recording a verification."""
        CollusionDetector.record_verification("entity-1", "192.168.1.1", True)

        assert "entity-1" in verification_graph
        assert len(verification_graph["entity-1"]) == 1
        assert verification_graph["entity-1"][0]["ip_address"] == "192.168.1.1"
        assert verification_graph["entity-1"][0]["passed"] is True

    def test_record_verification_empty_entity(self):
        """Test that empty entity_id is ignored."""
        CollusionDetector.record_verification("", "192.168.1.1", True)
        CollusionDetector.record_verification(None, "192.168.1.1", True)

        assert len(verification_graph) == 0

    def test_check_collusion_clean(self):
        """Test collusion check with no suspicious activity."""
        result = CollusionDetector.check_collusion("entity-1", "192.168.1.1")

        assert result["risk_score"] == 0.0
        assert result["flagged"] is False
        assert len(result["warnings"]) == 0

    def test_check_collusion_ip_clustering(self):
        """Test detection of same IP verifying multiple entities."""
        # Same IP verifies 3 different entities
        for i in range(3):
            CollusionDetector.record_verification(f"entity-{i}", "192.168.1.1", True)

        result = CollusionDetector.check_collusion("entity-new", "192.168.1.1")

        assert result["risk_score"] >= 0.3
        assert any("verified" in w and "entities" in w for w in result["warnings"])

    def test_check_collusion_frequent_reverification(self):
        """Test detection of entity verified too frequently."""
        # Verify same entity 15 times
        for _ in range(15):
            CollusionDetector.record_verification("entity-1", "192.168.1.1", True)

        result = CollusionDetector.check_collusion("entity-1", "192.168.1.1")

        assert result["risk_score"] >= 0.2
        assert any("times in last hour" in w for w in result["warnings"])

    def test_get_stats(self):
        """Test getting collusion stats."""
        CollusionDetector.record_verification("entity-1", "192.168.1.1", True)
        CollusionDetector.record_verification("entity-2", "192.168.1.2", True)

        stats = CollusionDetector.get_stats()

        assert stats["tracked_entities"] == 2
        assert stats["recent_verifications"] == 2
        assert stats["unique_ips"] == 2


class TestCollusionEndpoints:
    """Tests for collusion detection API endpoints."""

    def test_get_collusion_stats_unauthenticated(self, client):
        """Operational collusion state requires administrator authentication."""
        response = client.get("/api/security/collusion")

        assert response.status_code == 401
        assert "stats" not in response.json()

    def test_get_collusion_stats_authenticated(self, client):
        """Test GET /api/security/collusion with admin key returns full info."""
        response = client.get(
            "/api/security/collusion",
            headers={"X-Admin-Key": "test-admin-key-for-mettle-testing-only"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "thresholds" in data

    def test_check_entity_collusion(self, client):
        """Test POST /api/security/collusion/check with admin key returns indicators."""
        response = client.post(
            "/api/security/collusion/check?entity_id=test-entity",
            headers={"X-Admin-Key": TEST_ADMIN_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "flagged" in data
        assert "warnings" in data

    def test_check_entity_collusion_no_admin_key(self, client):
        """SECURITY: collusion check without admin key returns 401 and no risk data."""
        response = client.post("/api/security/collusion/check?entity_id=any-entity")

        assert response.status_code == 401
        data = response.json()
        # Must NOT leak detector internals to unauthenticated callers
        assert "risk_score" not in data
        assert "warnings" not in data
        assert "flagged" not in data

    def test_check_entity_collusion_wrong_admin_key(self, client):
        """SECURITY: collusion check with invalid admin key returns 401."""
        response = client.post(
            "/api/security/collusion/check?entity_id=any-entity",
            headers={"X-Admin-Key": "wrong-key"},
        )

        assert response.status_code == 401
        assert "risk_score" not in response.json()


# === Model Fingerprinting Tests ===


class TestModelFingerprinter:
    """Tests for ModelFingerprinter class."""

    def test_fingerprint_empty_responses(self):
        """Test fingerprinting with no responses."""
        result = ModelFingerprinter.fingerprint([])

        assert "error" in result
        assert result["scores"] == {}

    def test_fingerprint_claude_patterns(self):
        """Test fingerprinting with Claude-like responses."""
        responses = [
            "I'd be happy to help with that request.",
            "I cannot provide harmful content.",
            "I should note that this is a complex topic.",
        ]

        result = ModelFingerprinter.fingerprint(responses)

        assert result["best_match"] == "claude"
        assert result["scores"]["claude"] > result["scores"]["gpt"]
        assert result["responses_analyzed"] == 3

    def test_fingerprint_gpt_patterns(self):
        """Test fingerprinting with GPT-like responses."""
        responses = [
            "Sure! I can help with that.",
            "Certainly! Here's what you need.",
            "I can help you with this task.",
        ]

        result = ModelFingerprinter.fingerprint(responses)

        # GPT patterns should score higher
        assert result["scores"]["gpt"] > 0

    def test_fingerprint_normalization(self):
        """Test that scores sum to approximately 1.0."""
        responses = ["Some generic response."]
        result = ModelFingerprinter.fingerprint(responses)

        total = sum(result["scores"].values())
        assert 0.99 <= total <= 1.01


class TestFingerprintEndpoint:
    """Tests for model fingerprinting API endpoint."""

    def test_fingerprint_endpoint(self, client):
        """Test POST /api/security/fingerprint."""
        api_key = "mtl_fingerprint-test-key-123456"  # pragma: allowlist secret
        RateTier.register_key(api_key, "pro", "fingerprint-test")
        response = client.post(
            "/api/security/fingerprint",
            json={"responses": ["I'd be happy to help.", "I cannot do that."]},
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert "best_match" in data
        assert "confidence" in data

    def test_fingerprint_endpoint_empty(self, client):
        """Test fingerprint endpoint with empty responses."""
        response = client.post(
            "/api/security/fingerprint",
            json={"responses": []},
        )

        # Should fail validation (min_length=1)
        assert response.status_code == 422


# === Rate Limiting Tiers Tests ===


class TestRateTier:
    """Tests for RateTier class."""

    def test_get_tier_no_key(self):
        """Test tier lookup with no API key."""
        tier = RateTier.get_tier(None)
        assert tier == "free"

    def test_get_tier_unknown_key(self):
        """Test tier lookup with unknown API key."""
        tier = RateTier.get_tier("unknown-key")
        assert tier == "free"

    def test_get_tier_registered_key(self):
        """Test tier lookup with registered API key."""
        RateTier.register_key("test-key", "pro", "entity-1")
        tier = RateTier.get_tier("test-key")
        assert tier == "pro"

    def test_get_limits(self):
        """Test getting tier limits."""
        limits = RateTier.get_limits("free")

        assert limits["sessions_per_day"] == 100
        assert limits["answers_per_minute"] == 60
        assert "basic" in limits["suites"]

    def test_get_limits_pro(self):
        """Test getting Pro tier limits."""
        limits = RateTier.get_limits("pro")

        assert limits["sessions_per_day"] == 10000
        assert "full" in limits["suites"]
        assert "webhooks" in limits["features"]

    def test_get_limits_enterprise(self):
        """Test getting Enterprise tier limits."""
        limits = RateTier.get_limits("enterprise")

        assert limits["sessions_per_day"] == -1  # Unlimited
        assert "all" in limits["features"]

    def test_register_key(self):
        """Test registering an API key."""
        result = RateTier.register_key("new-key", "pro", "entity-1")

        assert result["tier"] == "pro"
        assert result["entity_id"] == "entity-1"
        assert "new-key" in api_keys

    def test_register_key_invalid_tier(self):
        """Test registering with invalid tier."""
        with pytest.raises(ValueError, match="Invalid tier"):
            RateTier.register_key("bad-key", "invalid-tier")

    def test_revoke_key_removes_in_memory_authority(self):
        RateTier.register_key("revoke-key", "pro", "entity-1")

        result = RateTier.revoke_key("revoke-key")

        assert result is not None
        assert result["tier"] == "pro"
        assert RateTier.get_key_data("revoke-key") is None

    def test_durable_storage_overrides_stale_cached_authority(self):
        api_keys["revoked-elsewhere"] = {
            "tier": "enterprise",
            "entity_id": "entity-1",
        }
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = None

        with patch("main.db", mock_db):
            assert RateTier.get_key_data("revoked-elsewhere") is None

        assert "revoked-elsewhere" not in api_keys

    def test_durable_storage_refreshes_cached_authority(self):
        key_data = {"tier": "pro", "entity_id": "durable-entity"}
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = key_data

        with patch("main.db", mock_db):
            assert RateTier.get_key_data("durable-key") == key_data

        assert api_keys["durable-key"] == key_data

    def test_register_key_persistence_failure_removes_authority(self):
        mock_db = MagicMock()
        mock_db.save_api_key.return_value = False

        with patch("main.db", mock_db):
            with pytest.raises(RuntimeError, match="persistence unavailable"):
                RateTier.register_key("failed-key", "pro", "entity-1")

        assert "failed-key" not in api_keys

    def test_revoke_key_removes_durable_authority(self):
        key_data = {"tier": "pro", "entity_id": "durable-entity"}
        mock_db = MagicMock()
        mock_db.get_api_key.return_value = key_data
        mock_db.delete_api_key.return_value = True

        with patch("main.db", mock_db):
            assert RateTier.revoke_key("durable-key") == key_data

        mock_db.get_api_key.assert_called_once_with("durable-key", raise_on_error=True)
        mock_db.delete_api_key.assert_called_once_with(
            "durable-key", raise_on_error=True
        )

    def test_check_limit_free(self):
        """Test rate limit check for free tier."""
        allowed, message = RateTier.check_limit(None, "session")

        assert allowed is True
        assert "free tier" in message

    def test_check_limit_enterprise(self):
        """Test rate limit check for enterprise (unlimited)."""
        RateTier.register_key("ent-key", "enterprise")
        allowed, message = RateTier.check_limit("ent-key", "session")

        assert allowed is True
        assert "unlimited" in message


class TestRateTierEndpoints:
    """Tests for rate tier API endpoints."""

    def test_list_tiers(self, client):
        """Test GET /api/keys/tiers."""
        response = client.get("/api/keys/tiers")

        assert response.status_code == 200
        data = response.json()
        assert "tiers" in data
        assert "free" in data["tiers"]
        assert "pro" in data["tiers"]
        assert "enterprise" in data["tiers"]

    def test_revoke_api_key(self, client):
        RateTier.register_key(
            "mtl_key-to-revoke-123456789",  # pragma: allowlist secret
            "pro",
            "entity-1",
        )

        response = client.post(
            "/api/keys/revoke",
            json={
                "api_key": "mtl_key-to-revoke-123456789",  # pragma: allowlist secret
                "reason": "Staging security rotation",
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY},
        )

        assert response.status_code == 200
        assert response.json() == {
            "revoked": True,
            "tier": "pro",
            "entity_id": "entity-1",
        }
        assert "api_key" not in response.json()
        assert RateTier.get_key_data("mtl_key-to-revoke-123456789") is None

    def test_revoke_api_key_requires_admin(self, client):
        RateTier.register_key(
            "mtl_key-to-protect-1234567",  # pragma: allowlist secret
            "pro",
            "entity-1",
        )

        response = client.post(
            "/api/keys/revoke",
            json={
                "api_key": "mtl_key-to-protect-1234567",  # pragma: allowlist secret
                "reason": "Attempted unauthorized revocation",
            },
        )

        assert response.status_code == 401
        assert RateTier.get_tier("mtl_key-to-protect-1234567") == "pro"

    def test_revoke_api_key_persistence_failure_is_fail_closed(self, client):
        mock_db = MagicMock()
        mock_db.delete_api_key.side_effect = RuntimeError(
            "API key persistence unavailable"
        )
        api_keys["mtl_key-persistence-123456"] = {  # pragma: allowlist secret
            "tier": "pro",
            "entity_id": "entity-1",
        }

        with patch("main.db", mock_db):
            response = client.post(
                "/api/keys/revoke",
                json={
                    "api_key": "mtl_key-persistence-123456",  # pragma: allowlist secret
                    "reason": "Required security revocation",
                },
                headers={"X-Admin-Key": TEST_ADMIN_KEY},
            )

        assert response.status_code == 503
        assert RateTier.get_tier("mtl_key-persistence-123456") == "pro"

    def test_revoke_unknown_api_key_returns_not_found(self, client):
        response = client.post(
            "/api/keys/revoke",
            json={
                "api_key": "mtl_unknown-key-123456789",  # pragma: allowlist secret
                "reason": "Synthetic unknown key revocation",
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "API key not found"

    def test_register_api_key_persistence_failure_returns_unavailable(self, client):
        mock_db = MagicMock()
        mock_db.save_api_key.return_value = False

        with patch("main.db", mock_db):
            response = client.post(
                "/api/keys/register",
                json={"tier": "pro", "entity_id": "entity-1"},
                headers={"X-Admin-Key": TEST_ADMIN_KEY},
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "API key persistence unavailable"


# === Webhook Tests ===


class TestWebhookManager:
    """Tests for WebhookManager class."""

    def test_register_webhook(self):
        """Test registering a webhook."""
        config = WebhookManager.register(
            "entity-1",
            "https://example.com/webhook",
            ["session.completed"],
            "secret123",
        )

        assert config["url"] == "https://example.com/webhook"
        assert config["events"] == ["session.completed"]
        assert config["secret"] == "secret123"
        assert "entity-1" in webhooks

    def test_register_webhook_default_events(self):
        """Test registering webhook with default events."""
        config = WebhookManager.register("entity-2", "https://example.com/hook")

        assert config["events"] == WebhookManager.EVENTS

    def test_unregister_webhook(self):
        """Test unregistering a webhook."""
        WebhookManager.register("entity-1", "https://example.com/webhook")
        result = WebhookManager.unregister("entity-1")

        assert result is True
        assert "entity-1" not in webhooks

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent webhook."""
        result = WebhookManager.unregister("nonexistent")
        assert result is False

    def test_events_list(self):
        """Test that all expected events are defined."""
        events = WebhookManager.EVENTS

        assert "session.started" in events
        assert "session.completed" in events
        assert "badge.issued" in events
        assert "badge.revoked" in events


class TestWebhookEndpoints:
    """Tests for webhook API endpoints."""

    @staticmethod
    def _owner_key(entity_id: str) -> str:
        """Register a pro API key that owns ``entity_id`` and return it."""
        api_key = f"owner-key-{entity_id}"
        RateTier.register_key(api_key, "pro", entity_id)
        return api_key

    def test_register_webhook_endpoint(self, client):
        """Test POST /api/webhooks/register with an owning API key."""
        api_key = self._owner_key("test-entity")
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test-entity",
                "url": "https://example.com/webhook",
            },
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["registered"] is True
        assert data["entity_id"] == "test-entity"

    def test_register_webhook_with_events(self, client):
        """Test registering webhook with specific events."""
        api_key = self._owner_key("test-entity")
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test-entity",
                "url": "https://example.com/webhook",
                "events": ["session.completed", "badge.issued"],
            },
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["events"] == ["session.completed", "badge.issued"]

    def test_register_webhook_invalid_event(self, client):
        """Test registering webhook with invalid event (after passing auth)."""
        api_key = self._owner_key("test-entity")
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test-entity",
                "url": "https://example.com/webhook",
                "events": ["invalid.event"],
            },
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 400

    def test_register_webhook_no_api_key(self, client):
        """SECURITY: anonymous webhook registration is rejected with 401."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "victim-entity",
                "url": "https://example.com/webhook",
            },
        )

        assert response.status_code == 401
        assert "victim-entity" not in webhooks

    def test_register_webhook_unknown_api_key(self, client):
        """SECURITY: an unrecognized API key is rejected with 401."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "victim-entity",
                "url": "https://example.com/webhook",
            },
            headers={"X-API-Key": "not-a-real-key"},
        )

        assert response.status_code == 401
        assert "victim-entity" not in webhooks

    def test_register_webhook_entity_mismatch(self, client):
        """SECURITY: alice's key cannot register a webhook for bob (IDOR)."""
        alice_key = self._owner_key("entity-alice")
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "entity-bob",
                "url": "https://example.com/webhook",
            },
            headers={"X-API-Key": alice_key},
        )

        assert response.status_code == 403
        # The cross-entity webhook must not have been registered
        assert "entity-bob" not in webhooks

    def test_register_webhook_free_tier_is_forbidden(self, client):
        """SECURITY: entity ownership cannot bypass the webhook feature tier."""
        RateTier.register_key("free-webhook-key", "free", "free-entity")
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "free-entity",
                "url": "https://example.com/webhook",
            },
            headers={"X-API-Key": "free-webhook-key"},
        )
        assert response.status_code == 403
        assert "pro or enterprise" in response.json()["detail"]
        assert "free-entity" not in webhooks

    def test_unregister_webhook_endpoint(self, client):
        """Test DELETE /api/webhooks/{entity_id} with admin key."""
        # First register with the owning key
        api_key = self._owner_key("test-entity")
        client.post(
            "/api/webhooks/register",
            json={"entity_id": "test-entity", "url": "https://example.com/webhook"},
            headers={"X-API-Key": api_key},
        )

        # Then unregister with admin key
        response = client.delete(
            "/api/webhooks/test-entity",
            headers={"X-Admin-Key": TEST_ADMIN_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["unregistered"] is True

    def test_unregister_webhook_no_admin_key(self, client):
        """SECURITY: deleting a webhook without admin key returns 401."""
        WebhookManager.register("test-entity", "https://example.com/webhook")

        response = client.delete("/api/webhooks/test-entity")

        assert response.status_code == 401
        # Webhook must still exist after a rejected delete
        assert "test-entity" in webhooks

    def test_unregister_webhook_wrong_admin_key(self, client):
        """SECURITY: deleting a webhook with a wrong admin key returns 401."""
        WebhookManager.register("test-entity", "https://example.com/webhook")

        response = client.delete(
            "/api/webhooks/test-entity",
            headers={"X-Admin-Key": "wrong-key"},
        )

        assert response.status_code == 401
        assert "test-entity" in webhooks

    def test_unregister_nonexistent_webhook(self, client):
        """Test unregistering non-existent webhook (with admin key) returns 404."""
        response = client.delete(
            "/api/webhooks/nonexistent",
            headers={"X-Admin-Key": TEST_ADMIN_KEY},
        )
        assert response.status_code == 404

    def test_unregister_webhook_rate_limited_after_failures(self, client):
        """SECURITY: repeated wrong-key deletes trigger brute-force rate limiting."""
        WebhookManager.register("test-entity", "https://example.com/webhook")

        statuses = []
        # _ADMIN_AUTH_MAX_FAILURES is 5; the 6th wrong-key attempt should be 429
        for _ in range(7):
            resp = client.delete(
                "/api/webhooks/test-entity",
                headers={"X-Admin-Key": "wrong-key"},
            )
            statuses.append(resp.status_code)

        assert 429 in statuses

    def test_list_webhook_events(self, client):
        """Test GET /api/webhooks/events."""
        response = client.get("/api/webhooks/events")

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert len(data["events"]) == 4


# === Batch Verification Tests ===


class TestBatchVerification:
    """Tests for batch verification endpoint."""

    @staticmethod
    def _pro_key() -> str:
        """Register a pro-tier API key (has the batch feature) and return it."""
        api_key = "batch-pro-key"
        RateTier.register_key(api_key, "pro", "batch-owner")
        return api_key

    def test_batch_start_sessions(self, client):
        """Test POST /api/session/batch with a pro-tier key."""
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1", "entity-2", "entity-3"],
                "difficulty": "basic",
            },
            headers={"X-API-Key": self._pro_key()},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["failed"] == 0
        assert len(data["sessions"]) == 3

        # Each session should have required fields
        for session in data["sessions"]:
            assert "entity_id" in session
            assert "session_id" in session
            assert "challenge_id" in session

    def test_batch_start_single(self, client):
        """Test batch with single entity."""
        response = client.post(
            "/api/session/batch",
            json={"entity_ids": ["single-entity"]},
            headers={"X-API-Key": self._pro_key()},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_batch_empty_list(self, client):
        """Test batch with empty entity list."""
        response = client.post(
            "/api/session/batch",
            json={"entity_ids": []},
            headers={"X-API-Key": self._pro_key()},
        )

        # Should fail validation (min_length=1)
        assert response.status_code == 422

    def test_batch_full_difficulty(self, client):
        """Test batch with full difficulty."""
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1"],
                "difficulty": "full",
            },
            headers={"X-API-Key": self._pro_key()},
        )

        assert response.status_code == 200

    def test_batch_unauthorized(self, client):
        """SECURITY: batch start without an API key returns 401."""
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1"],
                "difficulty": "basic",
            },
        )

        assert response.status_code == 401
        assert "session_id" not in response.text

    def test_batch_free_tier(self, client):
        """SECURITY: a free-tier key cannot use batch (403)."""
        RateTier.register_key("free-batch-key", "free", "free-owner")
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1"],
                "difficulty": "basic",
            },
            headers={"X-API-Key": "free-batch-key"},
        )

        assert response.status_code == 403

    def test_batch_unknown_key_is_free_tier(self, client):
        """SECURITY: an unrecognized key defaults to free tier and is forbidden (403)."""
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1"],
                "difficulty": "basic",
            },
            headers={"X-API-Key": "totally-unknown-key"},
        )

        assert response.status_code == 403

    def test_batch_enterprise_tier(self, client):
        """An enterprise-tier key (features ['all']) can use batch (200)."""
        RateTier.register_key("ent-batch-key", "enterprise", "ent-owner")
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1"],
                "difficulty": "basic",
            },
            headers={"X-API-Key": "ent-batch-key"},
        )

        assert response.status_code == 200


# === Badge Revocation Tests ===


class TestBadgeRevocation:
    """Tests for badge revocation functionality."""

    def test_revoke_badge_no_admin_key(self, client):
        """Test revoking badge without admin key fails with 401."""
        response = client.post(
            "/api/badge/revoke",
            json={
                "token": "some-token",
                "reason": "Test revocation reason here",
            },
        )

        # SECURITY: Revocation requires admin authorization
        assert response.status_code == 401

    def test_list_revocations_unauthenticated(self, client):
        """Test GET /api/badge/revocations without admin key fails."""
        response = client.get("/api/badge/revocations")

        # SECURITY: Revocation audit requires admin authorization
        assert response.status_code == 401

    def test_list_revocations_authenticated(self, client):
        """Test GET /api/badge/revocations with admin key."""
        response = client.get(
            "/api/badge/revocations",
            headers={"X-Admin-Key": "test-admin-key-for-mettle-testing-only"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "revoked_count" in data
        assert "audit" in data

    def test_list_revocations_uses_durable_database_audit(self, client):
        mock_db = MagicMock()
        mock_db.count_revoked_badges.return_value = 2
        mock_db.get_revoked_badges.return_value = [
            {"jti": "durable-jti", "reason": "persisted"}
        ]

        with patch("main.db", mock_db):
            response = client.get(
                "/api/badge/revocations",
                headers={"X-Admin-Key": "test-admin-key-for-mettle-testing-only"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "revoked_count": 2,
            "audit": [{"jti": "durable-jti", "reason": "persisted"}],
        }

    def test_list_revocations_database_failure_returns_503(self, client):
        mock_db = MagicMock()
        mock_db.count_revoked_badges.side_effect = RuntimeError("database unavailable")

        with patch("main.db", mock_db):
            response = client.get(
                "/api/badge/revocations",
                headers={"X-Admin-Key": "test-admin-key-for-mettle-testing-only"},
            )

        assert response.status_code == 503
