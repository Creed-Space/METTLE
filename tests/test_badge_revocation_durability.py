"""Regression tests for badge revocation: durability, replica, and fail-safety.

The revocation store is an in-memory REPLICA of the durable DB set, refreshed in the
background. Properties locked down here:

1. **A revocation is never silently reversed.** The store is pruned by *badge expiry*,
   never by LRU eviction (which once dropped the oldest revoked JTI at capacity and let
   that badge verify as VALID again).
2. **Durable across restart.** A fresh process loads the revoked set from the DB
   (``refresh_revocation_replica``) before serving.
3. **Verify stays AVAILABLE during a DB outage** once the replica has loaded: no
   per-request DB read, so a signature+expiry+replica check keeps working while the DB is
   down, and replica-known revocations are still enforced.
4. **Fails CLOSED in exactly one case:** cold start where the replica has never loaded
   (zero revocation knowledge).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt as jwt_lib
import main
import pytest
from main import (
    MAX_REVOKED_BADGES,
    refresh_revocation_replica,
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
    # Reset the replica-loaded flag to its natural default (True when there is no DB).
    main._revocation_replica_loaded = main.db is None
    yield
    revoked_badges.clear()
    revocation_audit.clear()
    main._revocation_replica_loaded = main.db is None


def _badge(jti: str, ttl_hours: int = 24) -> str:
    """Mint a signed badge JWT the endpoints will accept."""
    now = datetime.now(UTC)
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

        # "victim" is now the OLDEST entry. Fill to capacity with live revocations.
        far_future = time.time() + 86_400
        while len(revoked_badges) < MAX_REVOKED_BADGES:
            revoked_badges[f"live-{len(revoked_badges)}"] = far_future

        # One more revocation. Under the old LRU eviction this deleted the oldest key
        # (the victim), making its badge verify as VALID again.
        resp2 = client.post(
            "/api/badge/revoke",
            json={"token": _badge("newly-revoked"), "reason": REASON},
            headers=ADMIN_HEADERS,
        )
        assert resp2.status_code == 200

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

        main._prune_expired_revocations()

        assert "already-expired" not in revoked_badges
        assert "still-live" in revoked_badges
        assert "no-exp-claim" in revoked_badges


class TestReplicaLoadDurability:
    """A revocation must outlive the process that issued it, via the DB replica."""

    def test_revocation_is_persisted_to_the_database(self, client) -> None:
        fake_db = MagicMock()
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
        # And it is immediately in the local replica.
        assert "db-jti" in revoked_badges

    def test_replica_loads_revoked_set_from_db_on_startup(self, client) -> None:
        """Fresh process (empty memory): loading the replica from the DB then rejects."""
        token = _badge("persisted")
        fake_db = MagicMock()
        fake_db.get_all_revoked_badges_strict.return_value = [
            {"jti": "persisted", "revoked_at": datetime.now(UTC).isoformat()}
        ]

        with patch("main.db", fake_db):
            revoked_badges.clear()
            main._revocation_replica_loaded = False
            assert refresh_revocation_replica() is True  # startup load
            assert "persisted" in revoked_badges
            body = client.get(f"/api/badge/verify/{token}").json()

        assert body["revoked"] is True
        assert body["valid"] is False

    def test_replica_load_never_shortens_an_exact_local_bound(self) -> None:
        """A DB-derived conservative bound must not overwrite a shorter exact one... and
        vice versa: only ever raise a bound, never lower it."""
        fake_db = MagicMock()
        # revoked_at now => conservative prune bound ~ now + badge_expiry (large).
        fake_db.get_all_revoked_badges_strict.return_value = [
            {"jti": "j", "revoked_at": datetime.now(UTC).isoformat()}
        ]
        with patch("main.db", fake_db):
            revoked_badges["j"] = float("inf")  # an exact local bound: keep forever
            refresh_revocation_replica()
            assert revoked_badges["j"] == float("inf")  # not shortened by the DB bound


class TestNaiveTimestampIsUtc:
    """The DB stores revoked_at as a NAIVE datetime; it must be read as UTC, not local TZ.

    Regression for the deployment-TZ bug: on a non-UTC instance, parsing the offset-less
    revoked_at with a bare .timestamp() shifted the prune bound by the UTC offset and could
    drop a revocation hours before the badge expired. The earlier fixtures missed this
    because they used aware (+00:00) timestamps, which parse correctly under every TZ.
    """

    def test_naive_iso_is_interpreted_as_utc(self) -> None:
        # Epoch start; a non-UTC .timestamp() on the naive value would be nonzero.
        assert main._parse_iso_ts("1970-01-01T00:00:00") == 0.0

    def test_naive_and_aware_parse_to_the_same_instant(self) -> None:
        assert main._parse_iso_ts("2026-07-12T09:30:00") == main._parse_iso_ts("2026-07-12T09:30:00+00:00")

    def test_replica_bound_from_naive_revoked_at_is_not_tz_shifted(self) -> None:
        """The prune bound for a DB-loaded revocation must be revoked_at(UTC) + TTL."""
        keep = main.settings.badge_expiry_seconds
        revoked_at_utc = datetime.now(UTC)
        naive_str = revoked_at_utc.replace(tzinfo=None).isoformat()  # what SQLAlchemy emits
        fake_db = MagicMock()
        fake_db.get_all_revoked_badges_strict.return_value = [{"jti": "j", "revoked_at": naive_str}]

        with patch("main.db", fake_db):
            revoked_badges.clear()
            main._revocation_replica_loaded = False
            refresh_revocation_replica()

        expected = revoked_at_utc.timestamp() + keep
        assert abs(revoked_badges["j"] - expected) < 2  # UTC-correct, not offset-shifted


class TestRevocationFailsClosedOnlyWhenBlind:
    """Fail closed when we have NO revocation knowledge; stay available otherwise."""

    def test_verify_fails_closed_when_replica_never_loaded(self, client) -> None:
        """Cold start + DB down: no revocation knowledge, so do not accept any badge."""
        fake_db = MagicMock()
        with patch("main.db", fake_db):
            revoked_badges.clear()
            main._revocation_replica_loaded = False
            body = client.get(f"/api/badge/verify/{_badge('unknown-status')}").json()

        assert body["valid"] is False
        assert "unavailable" in body["error"].lower()

    def test_verify_stays_available_during_outage_once_loaded(self, client) -> None:
        """The whole point: once loaded, a DB outage does NOT break verification."""
        fake_db = MagicMock()
        fake_db.get_all_revoked_badges_strict.side_effect = RuntimeError("DB down")

        with patch("main.db", fake_db):
            main._revocation_replica_loaded = True  # loaded earlier, before the outage
            revoked_badges.clear()
            revoked_badges["known-revoked"] = time.time() + 3_600

            # A refresh during the outage fails, but must not tear down the replica.
            assert refresh_revocation_replica() is False

            good = client.get(f"/api/badge/verify/{_badge('fresh-good')}").json()
            bad = client.get(f"/api/badge/verify/{_badge('known-revoked')}").json()

        assert good["valid"] is True  # available despite the DB being down
        assert bad["valid"] is False and bad["revoked"] is True  # revocation still enforced

    def test_revoke_returns_503_when_replica_never_loaded(self, client) -> None:
        fake_db = MagicMock()
        with patch("main.db", fake_db):
            main._revocation_replica_loaded = False
            resp = client.post(
                "/api/badge/revoke",
                json={"token": _badge("cannot-check"), "reason": REASON},
                headers=ADMIN_HEADERS,
            )
        assert resp.status_code == 503
        assert "cannot-check" not in revoked_badges

    def test_failed_persistence_revokes_nothing_and_is_retryable(self, client) -> None:
        """Durable write fails => request fails and NOTHING is cached (retry is clean)."""
        fake_db = MagicMock()
        fake_db.add_revoked_badge.return_value = False  # durable write fails

        with patch("main.db", fake_db):
            main._revocation_replica_loaded = True
            resp = client.post(
                "/api/badge/revoke",
                json={"token": _badge("not-persisted"), "reason": REASON},
                headers=ADMIN_HEADERS,
            )

        assert resp.status_code == 503
        assert "not-persisted" not in revoked_badges
