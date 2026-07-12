"""Regression tests for badge revocation durability.

Two defects these lock down:

1. **A revocation could be silently reversed.** ``revoke_badge`` recorded the JTI via
   ``add_with_limit``, which evicts the OLDEST entry once ``MAX_REVOKED_BADGES`` is
   reached. The evicted badge then passed ``verify_badge`` as VALID again. Eviction is
   never a safe policy for a revocation list, and the DoS rationale did not apply here
   anyway (revocation is admin-authenticated). The store is now pruned by *badge expiry*:
   a revoked JTI is forgotten only once the badge itself has expired, which
   ``verify_badge`` rejects regardless.

2. **A revocation was memory-only.** ``database.add_revoked_badge`` /
   ``database.is_badge_revoked`` already existed and were tested, but were never wired
   to these endpoints, so a revocation was lost on restart and invisible to other
   instances.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt as jwt_lib
import pytest
from main import (
    MAX_REVOKED_BADGES,
    _is_badge_revoked,
    _prune_expired_revocations,
    revocation_audit,
    revoked_badges,
)

SECRET_KEY = "test-secret-key-for-mettle-testing-only"
ADMIN_HEADERS = {"X-Admin-Key": "test-admin-key-for-mettle-testing-only"}
REASON = "Test revocation reason for testing purposes"


@pytest.fixture(autouse=True)
def _clean_revocation_state():
    revoked_badges.clear()
    revocation_audit.clear()
    yield
    revoked_badges.clear()
    revocation_audit.clear()


def _badge(jti: str, ttl_hours: int = 24) -> str:
    """Mint a signed badge JWT the endpoints will accept."""
    now = datetime.now(timezone.utc)
    return jwt_lib.encode(
        {
            "entity_id": "agent-under-test",
            "difficulty": "basic",
            "pass_rate": 1.0,
            "verified_at": now.isoformat(),
            "exp": (now + timedelta(hours=ttl_hours)).timestamp(),
            "iat": now.timestamp(),
            "jti": jti,
            "nonce": "n0nce",
            "iss": "mettle-api",
        },
        SECRET_KEY,
        algorithm="HS256",
    )


class TestRevocationIsNeverSilentlyReversed:
    """The revocation list must not evict a still-valid revocation."""

    def test_capacity_pressure_does_not_un_revoke_the_oldest_badge(self, client) -> None:
        victim_token = _badge("victim")

        resp = client.post(
            "/api/badge/revoke",
            json={"token": victim_token, "reason": REASON},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True

        # "victim" is now the OLDEST entry. Fill the store to capacity with live
        # (unexpired) revocations so the next insert hits the cap.
        far_future = time.time() + 86_400
        while len(revoked_badges) < MAX_REVOKED_BADGES:
            revoked_badges[f"live-{len(revoked_badges)}"] = far_future

        # One more revocation. Under the old add_with_limit() this deleted the oldest
        # key -- the victim -- making its badge verify as VALID again.
        resp2 = client.post(
            "/api/badge/revoke",
            json={"token": _badge("newly-revoked"), "reason": REASON},
            headers=ADMIN_HEADERS,
        )
        assert resp2.status_code == 200

        assert _is_badge_revoked("victim") is True
        assert _is_badge_revoked("newly-revoked") is True

        body = client.get(f"/api/badge/verify/{victim_token}").json()
        assert body["revoked"] is True
        assert body["valid"] is False


class TestExpiryBasedPruning:
    """Bounded, but never at the cost of un-revoking a live badge."""

    def test_prunes_only_badges_that_already_expired(self) -> None:
        now = time.time()
        revoked_badges["already-expired"] = now - 10  # the badge itself has expired
        revoked_badges["still-live"] = now + 3_600
        revoked_badges["no-exp-claim"] = float("inf")

        _prune_expired_revocations()

        assert "already-expired" not in revoked_badges
        assert "still-live" in revoked_badges
        assert "no-exp-claim" in revoked_badges


class TestRevocationIsDurable:
    """Revocations must outlive the process that issued them."""

    def test_revocation_is_persisted_to_the_database(self, client) -> None:
        fake_db = MagicMock()
        fake_db.is_badge_revoked_strict.return_value = False
        fake_db.add_revoked_badge.return_value = True

        with patch("main.db", fake_db):
            resp = client.post(
                "/api/badge/revoke",
                json={"token": _badge("db-jti"), "reason": REASON},
                headers=ADMIN_HEADERS,
            )

        assert resp.status_code == 200
        fake_db.add_revoked_badge.assert_called_once()
        assert fake_db.add_revoked_badge.call_args.args[0] == "db-jti"

    def test_revocation_survives_a_restart(self, client) -> None:
        """Fresh process (empty memory), but the DB remembers: still revoked."""
        token = _badge("persisted")
        fake_db = MagicMock()
        fake_db.is_badge_revoked_strict.return_value = True

        revoked_badges.clear()  # simulate a restart / a different instance

        with patch("main.db", fake_db):
            body = client.get(f"/api/badge/verify/{token}").json()

        assert body["revoked"] is True
        assert body["valid"] is False
        fake_db.is_badge_revoked_strict.assert_called_with("persisted")


class TestRevocationFailsClosed:
    """An unreadable revocation store must never let a badge through."""

    def test_verify_fails_closed_when_the_store_is_unavailable(self, client) -> None:
        """A DB outage must NOT verify a badge we cannot prove is unrevoked."""
        token = _badge("unknown-status")
        fake_db = MagicMock()
        fake_db.is_badge_revoked_strict.side_effect = RuntimeError("revocation DB down")

        revoked_badges.clear()  # not in the local cache: we must consult the store

        with patch("main.db", fake_db):
            body = client.get(f"/api/badge/verify/{token}").json()

        assert body["valid"] is False
        assert "unavailable" in body["error"].lower()

    def test_revoke_returns_503_when_the_store_is_unavailable(self, client) -> None:
        fake_db = MagicMock()
        fake_db.is_badge_revoked_strict.side_effect = RuntimeError("revocation DB down")

        with patch("main.db", fake_db):
            resp = client.post(
                "/api/badge/revoke",
                json={"token": _badge("cannot-check"), "reason": REASON},
                headers=ADMIN_HEADERS,
            )

        assert resp.status_code == 503
        assert "cannot-check" not in revoked_badges

    def test_failed_persistence_revokes_nothing_and_is_retryable(self, client) -> None:
        """If the durable write fails the request must fail and cache NOTHING.

        Otherwise the admin is told the badge is revoked while a restart (or another
        instance) still accepts it, and the retry short-circuits on "already revoked"
        without ever re-driving the write.
        """
        fake_db = MagicMock()
        fake_db.is_badge_revoked_strict.return_value = False
        fake_db.add_revoked_badge.return_value = False  # durable write fails

        with patch("main.db", fake_db):
            resp = client.post(
                "/api/badge/revoke",
                json={"token": _badge("not-persisted"), "reason": REASON},
                headers=ADMIN_HEADERS,
            )

        assert resp.status_code == 503
        # The memory cache must not have run ahead of the durable store.
        assert "not-persisted" not in revoked_badges
