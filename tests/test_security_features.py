"""Tests for METTLE v2 security features."""

import pytest
from fastapi.testclient import TestClient

from main import (
    CollusionDetector,
    ModelFingerprinter,
    RateTier,
    WebhookManager,
    api_keys,
    app,
    revoked_badges,
    verification_graph,
    verification_timestamps,
    webhooks,
)


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
    yield


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

    def test_get_collusion_stats(self, client):
        """Test GET /api/security/collusion."""
        response = client.get("/api/security/collusion")

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "thresholds" in data

    def test_check_entity_collusion(self, client):
        """Test POST /api/security/collusion/check."""
        response = client.post("/api/security/collusion/check?entity_id=test-entity")

        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "flagged" in data
        assert "warnings" in data


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
        response = client.post(
            "/api/security/fingerprint",
            json={"responses": ["I'd be happy to help.", "I cannot do that."]},
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

    def test_register_webhook_endpoint(self, client):
        """Test POST /api/webhooks/register."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test-entity",
                "url": "https://example.com/webhook",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["registered"] is True
        assert data["entity_id"] == "test-entity"

    def test_register_webhook_with_events(self, client):
        """Test registering webhook with specific events."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test-entity",
                "url": "https://example.com/webhook",
                "events": ["session.completed", "badge.issued"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["events"] == ["session.completed", "badge.issued"]

    def test_register_webhook_invalid_event(self, client):
        """Test registering webhook with invalid event."""
        response = client.post(
            "/api/webhooks/register",
            json={
                "entity_id": "test-entity",
                "url": "https://example.com/webhook",
                "events": ["invalid.event"],
            },
        )

        assert response.status_code == 400

    def test_unregister_webhook_endpoint(self, client):
        """Test DELETE /api/webhooks/{entity_id}."""
        # First register
        client.post(
            "/api/webhooks/register",
            json={"entity_id": "test-entity", "url": "https://example.com/webhook"},
        )

        # Then unregister
        response = client.delete("/api/webhooks/test-entity")

        assert response.status_code == 200
        data = response.json()
        assert data["unregistered"] is True

    def test_unregister_nonexistent_webhook(self, client):
        """Test unregistering non-existent webhook."""
        response = client.delete("/api/webhooks/nonexistent")
        assert response.status_code == 404

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

    def test_batch_start_sessions(self, client):
        """Test POST /api/session/batch."""
        response = client.post(
            "/api/session/batch",
            json={
                "entity_ids": ["entity-1", "entity-2", "entity-3"],
                "difficulty": "basic",
            },
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
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_batch_empty_list(self, client):
        """Test batch with empty entity list."""
        response = client.post(
            "/api/session/batch",
            json={"entity_ids": []},
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
        )

        assert response.status_code == 200


# === Badge Revocation Tests ===


class TestBadgeRevocation:
    """Tests for badge revocation functionality."""

    def test_revoke_badge_no_secret(self, client):
        """Test revoking badge when no secret key configured."""
        response = client.post(
            "/api/badge/revoke",
            json={
                "token": "some-token",
                "reason": "Test revocation reason here",
            },
        )

        # Should fail - no secret key in test env
        assert response.status_code == 400

    def test_list_revocations(self, client):
        """Test GET /api/badge/revocations."""
        response = client.get("/api/badge/revocations")

        assert response.status_code == 200
        data = response.json()
        assert "revoked_count" in data
        assert "audit" in data
