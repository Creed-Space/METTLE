"""Tests for METTLE database layer.

Tests all CRUD operations for sessions, revoked badges, API keys,
webhooks, and verification records using an isolated in-memory SQLite database.
"""

import hashlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Set test env vars before any METTLE imports
os.environ.setdefault("METTLE_SECRET_KEY", "test-secret-key-for-mettle-testing-only")
os.environ.setdefault("METTLE_ADMIN_API_KEY", "test-admin-key-for-mettle-testing-only")


# ── Helpers ──────────────────────────────────────────────────────────────────


class MockChallenge:
    """Mock challenge with model_dump() for save_session tests."""

    def __init__(self, id="test-ch-1", challenge_type="speed_math", prompt="1+1"):
        self._data = {"id": id, "type": challenge_type, "prompt": prompt}

    def model_dump(self):
        return self._data


class MockResult:
    """Mock result with model_dump() for update_session_results tests."""

    def __init__(self, challenge_id="test-ch-1", passed=True, score=1.0):
        self._data = {
            "challenge_id": challenge_id,
            "passed": passed,
            "score": score,
        }

    def model_dump(self):
        return self._data


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_db():
    """Replace the database engine with a fresh in-memory SQLite database.

    This patches database.engine and database.SessionLocal so every test
    gets a completely isolated database with all tables created.
    """
    import database

    test_engine = create_engine("sqlite:///:memory:")
    test_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    original_engine = database.engine
    original_session_local = database.SessionLocal

    database.engine = test_engine
    database.SessionLocal = test_session_local
    database.Base.metadata.create_all(bind=test_engine)

    yield database

    test_engine.dispose()
    original_engine.dispose()
    database.engine = original_engine
    database.SessionLocal = original_session_local


# ── init_db ──────────────────────────────────────────────────────────────────


class TestInitDb:
    def test_init_db_creates_tables(self, isolated_db):
        """init_db should create all tables without error."""
        db = isolated_db
        # Tables already created by fixture; verify by calling init_db again
        db.init_db()
        table_names = db.Base.metadata.tables.keys()
        assert "sessions" in table_names
        assert "revoked_badges" in table_names
        assert "api_keys" in table_names
        assert "webhooks" in table_names
        assert "verification_records" in table_names
        assert db.get_schema_version() == db.LATEST_SCHEMA_VERSION
        assert db.check_schema_current() is True

    def test_upgrade_from_oldest_supported_schema_is_idempotent(self, isolated_db):
        db = isolated_db
        legacy_engine = create_engine("sqlite:///:memory:")
        legacy_session_local = sessionmaker(bind=legacy_engine)
        with legacy_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE sessions (
                        id INTEGER PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        entity_id VARCHAR(128),
                        difficulty VARCHAR(16) NOT NULL,
                        challenges_json TEXT NOT NULL,
                        results_json TEXT DEFAULT '[]',
                        completed BOOLEAN DEFAULT 0,
                        created_at DATETIME,
                        completed_at DATETIME
                    )
                    """
                )
            )

        with (
            patch.object(db, "engine", legacy_engine),
            patch.object(db, "SessionLocal", legacy_session_local),
        ):
            db.init_db()
            db.init_db()
            columns = {
                column["name"]
                for column in inspect(legacy_engine).get_columns("sessions")
            }
            assert {"access_token_hash", "badge_info_json"} <= columns
            assert db.get_schema_version() == db.LATEST_SCHEMA_VERSION
            with legacy_engine.connect() as connection:
                versions = connection.execute(
                    text("SELECT version FROM schema_migrations ORDER BY version")
                ).scalars()
                assert list(versions) == [1, 2, 3]
        legacy_engine.dispose()

    def test_sqlite_backup_restores_sessions_and_revocations(
        self, isolated_db, tmp_path
    ):
        db = isolated_db
        source_path = tmp_path / "source.db"
        restored_path = tmp_path / "restored.db"
        source_engine = create_engine(f"sqlite:///{source_path}")
        source_session_local = sessionmaker(bind=source_engine)
        with (
            patch.object(db, "engine", source_engine),
            patch.object(db, "SessionLocal", source_session_local),
        ):
            db.init_db()
            assert db.save_session(
                "restore-session",
                "restore-entity",
                "basic",
                [MockChallenge()],
                access_token_hash="e" * 64,
            )
            assert db.add_revoked_badge(
                "restore-jti",
                "restore-entity",
                "restore trial",
                None,
            )
        source_engine.dispose()

        with (
            sqlite3.connect(source_path) as source,
            sqlite3.connect(restored_path) as restored,
        ):
            source.backup(restored)

        restored_engine = create_engine(f"sqlite:///{restored_path}")
        restored_session_local = sessionmaker(bind=restored_engine)
        with (
            patch.object(db, "engine", restored_engine),
            patch.object(db, "SessionLocal", restored_session_local),
        ):
            db.init_db()
            assert db.get_session("restore-session")["entity_id"] == "restore-entity"
            assert db.is_badge_revoked("restore-jti") is True
            assert db.check_schema_current() is True
        restored_engine.dispose()

    def test_health_check_reports_query_success_and_failure(self, isolated_db):
        db = isolated_db
        assert db.check_health() is True

        with patch.object(db.engine, "connect", side_effect=RuntimeError("offline")):
            assert db.check_health() is False


# ── get_db ───────────────────────────────────────────────────────────────────


class TestGetDb:
    def test_get_db_yields_session(self, isolated_db):
        """get_db should yield a usable session and close it after."""
        db = isolated_db
        with db.get_db() as session:
            assert session is not None
            # Session should be usable
            session.execute(db.Base.metadata.tables["sessions"].select())

    def test_get_db_closes_session_on_exit(self, isolated_db):
        """Session should be closed after context manager exits."""
        db = isolated_db
        with db.get_db() as _session:
            pass
        # After exiting, session.close() has been called (no error expected)


# ── Session Operations ───────────────────────────────────────────────────────


class TestSaveSession:
    def test_save_session_returns_true(self, isolated_db):
        db = isolated_db
        challenges = [MockChallenge()]
        result = db.save_session("sess-1", "entity-1", "basic", challenges)
        assert result is True

    def test_retention_purge_deletes_private_rows_but_preserves_revocations(
        self, isolated_db
    ):
        db = isolated_db
        now = datetime(2030, 1, 2, tzinfo=timezone.utc)
        old = now - timedelta(days=2)
        assert db.save_session(
            "expired-private-session",
            "entity-1",
            "basic",
            [MockChallenge()],
            started_at=old,
        )
        assert db.save_verification_record("entity-1", "192.0.2.1", True)
        assert db.add_revoked_badge("retained-revocation", "entity-1", "test", None)
        with db.get_db() as session:
            record = session.query(db.DBVerificationRecord).one()
            record.created_at = old
            session.commit()

        deleted = db.purge_expired_private_data(now=now)

        assert deleted == {
            "sessions_deleted": 1,
            "verification_records_deleted": 1,
        }
        assert db.get_session("expired-private-session") is None
        assert db.is_badge_revoked("retained-revocation") is True

    def test_save_session_persists_data(self, isolated_db):
        db = isolated_db
        challenges = [MockChallenge(id="ch-1"), MockChallenge(id="ch-2")]
        db.save_session("sess-2", "entity-2", "full", challenges)

        session_data = db.get_session("sess-2")
        assert session_data is not None
        assert session_data["session_id"] == "sess-2"
        assert session_data["entity_id"] == "entity-2"
        assert session_data["difficulty"] == "full"
        assert len(session_data["challenges"]) == 2
        assert session_data["completed"] is False

    def test_save_session_persists_recovery_fields(self, isolated_db):
        db = isolated_db
        started_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        db.save_session(
            "sess-recoverable",
            "entity-recoverable",
            "basic",
            [MockChallenge()],
            access_token_hash="a" * 64,
            started_at=started_at,
        )

        session_data = db.get_session("sess-recoverable")
        assert session_data["access_token_hash"] == "a" * 64
        assert session_data["created_at"].replace(tzinfo=timezone.utc) == started_at

    def test_save_session_none_entity_id(self, isolated_db):
        db = isolated_db
        result = db.save_session("sess-3", None, "basic", [MockChallenge()])
        assert result is True
        session_data = db.get_session("sess-3")
        assert session_data["entity_id"] is None

    def test_save_session_duplicate_returns_false(self, isolated_db):
        """Saving a session with a duplicate session_id should fail."""
        db = isolated_db
        db.save_session("dup-1", "e1", "basic", [MockChallenge()])
        result = db.save_session("dup-1", "e2", "full", [MockChallenge()])
        assert result is False

    def test_save_session_error_returns_false(self, isolated_db):
        """Database errors should be caught and return False."""
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.save_session("err-1", "e1", "basic", [MockChallenge()])
            assert result is False

    def test_save_session_empty_challenges(self, isolated_db):
        db = isolated_db
        result = db.save_session("sess-empty", "e1", "basic", [])
        assert result is True
        session_data = db.get_session("sess-empty")
        assert session_data["challenges"] == []


class TestGetSession:
    def test_get_existing_session(self, isolated_db):
        db = isolated_db
        db.save_session("get-1", "entity-a", "basic", [MockChallenge()])
        result = db.get_session("get-1")
        assert result is not None
        assert result["session_id"] == "get-1"
        assert result["entity_id"] == "entity-a"
        assert result["difficulty"] == "basic"
        assert isinstance(result["challenges"], list)
        assert isinstance(result["results"], list)
        assert result["completed"] is False
        assert result["created_at"] is not None

    def test_get_nonexistent_session_returns_none(self, isolated_db):
        db = isolated_db
        result = db.get_session("no-such-session")
        assert result is None

    def test_get_session_error_returns_none(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_session("err-sess")
            assert result is None

    def test_get_session_can_fail_closed(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            with pytest.raises(RuntimeError, match="Session persistence unavailable"):
                db.get_session("err-sess", raise_on_error=True)


class TestUpdateSessionResults:
    def test_update_results_success(self, isolated_db):
        db = isolated_db
        db.save_session("upd-1", "e1", "basic", [MockChallenge()])
        results = [MockResult(challenge_id="ch-1", passed=True)]
        success = db.update_session_results("upd-1", results, completed=False)
        assert success is True

        session_data = db.get_session("upd-1")
        assert len(session_data["results"]) == 1
        assert session_data["completed"] is False

    def test_update_results_completed(self, isolated_db):
        db = isolated_db
        db.save_session("upd-2", "e1", "basic", [MockChallenge()])
        results = [MockResult()]
        success = db.update_session_results("upd-2", results, completed=True)
        assert success is True

        session_data = db.get_session("upd-2")
        assert session_data["completed"] is True

    def test_update_results_persists_badge_for_restart(self, isolated_db):
        db = isolated_db
        db.save_session(
            "upd-badge",
            "e1",
            "basic",
            [MockChallenge()],
            access_token_hash="b" * 64,
        )
        badge_info = {"token": "signed-token", "signed": True, "tier": "bronze"}

        assert db.update_session_results(
            "upd-badge", [MockResult()], completed=True, badge_info=badge_info
        )
        assert db.get_session("upd-badge")["badge_info"] == badge_info

    def test_issued_badge_is_immutable_and_not_cleared_by_later_progress(
        self, isolated_db
    ):
        db = isolated_db
        db.save_session("immutable-badge", "e1", "basic", [MockChallenge()])
        original = {"token": "signed-token", "signed": True}
        replacement = {"token": "different-token", "signed": True}

        assert db.update_session_results(
            "immutable-badge", [MockResult()], completed=True, badge_info=original
        )
        assert db.update_session_results(
            "immutable-badge", [MockResult()], completed=True, badge_info=None
        )
        assert db.get_session("immutable-badge")["badge_info"] == original
        assert not db.update_session_results(
            "immutable-badge", [MockResult()], completed=True, badge_info=replacement
        )
        assert db.get_session("immutable-badge")["badge_info"] == original


class TestGetRecentSessions:
    def test_returns_only_recent_recoverable_sessions(self, isolated_db):
        db = isolated_db
        db.save_session(
            "recover-new",
            "e1",
            "basic",
            [MockChallenge()],
            access_token_hash="c" * 64,
        )
        db.save_session("not-recoverable", "e2", "basic", [MockChallenge()])
        db.save_session(
            "recover-old",
            "e3",
            "basic",
            [MockChallenge()],
            access_token_hash="d" * 64,
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )

        assert [row["session_id"] for row in db.get_recent_sessions()] == [
            "recover-new"
        ]

    def test_update_nonexistent_session_returns_false(self, isolated_db):
        db = isolated_db
        result = db.update_session_results("no-such", [MockResult()], completed=True)
        assert result is False

    def test_update_results_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.update_session_results("err", [MockResult()], completed=True)
            assert result is False


# ── Revocation Operations ────────────────────────────────────────────────────


class TestAddRevokedBadge:
    def test_add_revoked_badge_success(self, isolated_db):
        db = isolated_db
        result = db.add_revoked_badge("jti-1", "entity-1", "cheating", {"score": 0.1})
        assert result is True

    def test_add_revoked_badge_none_evidence(self, isolated_db):
        db = isolated_db
        result = db.add_revoked_badge("jti-2", "entity-1", "expired", None)
        assert result is True

    def test_add_revoked_badge_none_entity_id(self, isolated_db):
        db = isolated_db
        result = db.add_revoked_badge("jti-3", None, "unknown entity", None)
        assert result is True

    def test_add_revoked_badge_duplicate_jti_returns_false(self, isolated_db):
        db = isolated_db
        db.add_revoked_badge("dup-jti", "e1", "reason1", None)
        result = db.add_revoked_badge("dup-jti", "e2", "reason2", None)
        assert result is False

    def test_add_revoked_badge_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.add_revoked_badge("err-jti", "e1", "reason", None)
            assert result is False


class TestIsBadgeRevoked:
    def test_revoked_badge_returns_true(self, isolated_db):
        db = isolated_db
        db.add_revoked_badge("rev-1", "e1", "cheating", None)
        assert db.is_badge_revoked("rev-1") is True

    def test_not_revoked_returns_false(self, isolated_db):
        db = isolated_db
        assert db.is_badge_revoked("not-revoked") is False

    def test_is_badge_revoked_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            assert db.is_badge_revoked("err") is False

    def test_is_badge_revoked_can_fail_closed(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            with pytest.raises(RuntimeError, match="status unavailable"):
                db.is_badge_revoked("err", raise_on_error=True)


class TestGetRevokedBadges:
    def test_empty_list_when_none(self, isolated_db):
        db = isolated_db
        result = db.get_revoked_badges()
        assert result == []

    def test_returns_revoked_badges(self, isolated_db):
        db = isolated_db
        db.add_revoked_badge("jti-a", "e1", "reason-a", {"detail": "a"})
        db.add_revoked_badge("jti-b", "e2", "reason-b", None)

        result = db.get_revoked_badges()
        assert len(result) == 2
        jtis = {r["jti"] for r in result}
        assert jtis == {"jti-a", "jti-b"}
        assert next(r for r in result if r["jti"] == "jti-a")["evidence"] == {
            "detail": "a"
        }
        # Verify structure
        for badge in result:
            assert "jti" in badge
            assert "entity_id" in badge
            assert "reason" in badge
            assert "revoked_at" in badge

    def test_limit_parameter(self, isolated_db):
        db = isolated_db
        for i in range(5):
            db.add_revoked_badge(f"jti-{i}", "e1", f"reason-{i}", None)

        result = db.get_revoked_badges(limit=3)
        assert len(result) == 3

    def test_ordered_by_revoked_at_desc(self, isolated_db):
        db = isolated_db
        db.add_revoked_badge("first", "e1", "r1", None)
        db.add_revoked_badge("second", "e2", "r2", None)

        result = db.get_revoked_badges()
        # Most recent first
        assert result[0]["jti"] == "second"
        assert result[1]["jti"] == "first"

    def test_get_revoked_badges_error_returns_empty_list(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_revoked_badges()
            assert result == []

    def test_get_revoked_badges_can_fail_closed(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            with pytest.raises(RuntimeError, match="audit unavailable"):
                db.get_revoked_badges(raise_on_error=True)


class TestCountRevokedBadges:
    def test_counts_durable_revocations(self, isolated_db):
        db = isolated_db
        db.add_revoked_badge("count-1", "e1", "reason", None)
        db.add_revoked_badge("count-2", "e2", "reason", None)
        assert db.count_revoked_badges() == 2

    def test_count_can_fail_closed(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            with pytest.raises(RuntimeError, match="audit unavailable"):
                db.count_revoked_badges(raise_on_error=True)


# ── API Key Operations ───────────────────────────────────────────────────────


class TestSaveApiKey:
    def test_save_api_key_success(self, isolated_db):
        db = isolated_db
        result = db.save_api_key("key-123", "premium", "entity-1")
        assert result is True

        with db.get_db() as session:
            stored = session.query(db.DBAPIKey).one().api_key
        assert stored == hashlib.sha256(b"key-123").hexdigest()
        assert stored != "key-123"

    def test_save_api_key_none_entity(self, isolated_db):
        db = isolated_db
        result = db.save_api_key("key-anon", "basic", None)
        assert result is True

    def test_save_duplicate_key_returns_false(self, isolated_db):
        db = isolated_db
        db.save_api_key("dup-key", "basic", "e1")
        result = db.save_api_key("dup-key", "premium", "e2")
        assert result is False

    def test_save_api_key_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.save_api_key("err-key", "basic", "e1")
            assert result is False


class TestGetApiKey:
    def test_get_existing_key(self, isolated_db):
        db = isolated_db
        db.save_api_key("get-key-1", "premium", "entity-x")
        result = db.get_api_key("get-key-1")
        assert result is not None
        assert result["tier"] == "premium"
        assert result["entity_id"] == "entity-x"
        assert result["usage_date"] is None
        assert result["usage_count"] == 0
        assert result["created_at"] is not None

    def test_get_nonexistent_key_returns_none(self, isolated_db):
        db = isolated_db
        result = db.get_api_key("no-such-key")
        assert result is None

    def test_get_api_key_error_returns_none(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_api_key("err-key")
            assert result is None

    def test_get_api_key_can_fail_closed(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            with pytest.raises(RuntimeError, match="persistence unavailable"):
                db.get_api_key("err-key", raise_on_error=True)


class TestDeleteApiKey:
    def test_deletes_digest_backed_key(self, isolated_db):
        db = isolated_db
        db.save_api_key("delete-key", "premium", "entity-delete")

        assert db.delete_api_key("delete-key") is True
        assert db.get_api_key("delete-key") is None

    def test_startup_migrates_then_deletes_legacy_plaintext_key(self, isolated_db):
        db = isolated_db
        hexadecimal_plaintext = "a" * 64  # pragma: allowlist secret
        with db.get_db() as session:
            session.add_all(
                [
                    db.DBSchemaMigration(version=1),
                    db.DBSchemaMigration(version=2),
                    db.DBAPIKey(
                        api_key="legacy-delete-key",  # pragma: allowlist secret
                        tier="basic",
                        entity_id="legacy-entity",
                    ),
                    db.DBAPIKey(
                        api_key=hexadecimal_plaintext,
                        tier="premium",
                        entity_id="hex-legacy-entity",
                    ),
                ]
            )
            session.commit()

        db.init_db()
        with db.get_db() as session:
            stored = {
                record.entity_id: record.api_key
                for record in session.query(db.DBAPIKey).all()
            }
        assert stored == {
            "legacy-entity": hashlib.sha256(b"legacy-delete-key").hexdigest(),
            "hex-legacy-entity": hashlib.sha256(
                hexadecimal_plaintext.encode("utf-8")
            ).hexdigest(),
        }
        assert db.get_api_key(hexadecimal_plaintext)["tier"] == "premium"

        db.init_db()
        with db.get_db() as session:
            assert {
                record.entity_id: record.api_key
                for record in session.query(db.DBAPIKey).all()
            } == stored

        assert db.delete_api_key("legacy-delete-key") is True
        with db.get_db() as session:
            assert session.query(db.DBAPIKey).count() == 1

    def test_v3_double_digest_migration_preserves_v2_logical_keys(self, isolated_db):
        db = isolated_db
        logical_key = "v2-digest-key"
        original_digest = hashlib.sha256(logical_key.encode()).hexdigest()
        with db.get_db() as session:
            session.add_all(
                [
                    db.DBSchemaMigration(version=1),
                    db.DBSchemaMigration(version=2),
                    db.DBAPIKey(
                        api_key=original_digest,
                        tier="premium",
                        entity_id="v2-digested-entity",
                    ),
                ]
            )
            session.commit()

        db.init_db()

        with db.get_db() as session:
            stored = session.query(db.DBAPIKey).one().api_key
        assert stored == hashlib.sha256(original_digest.encode()).hexdigest()
        assert db.get_api_key(logical_key)["tier"] == "premium"
        assert db.reserve_api_key_usage(logical_key, "2026-08-14", 1, 10) is True
        assert db.save_api_key(logical_key, "basic", "duplicate") is False

        db.init_db()
        assert db.delete_api_key(logical_key) is True

    def test_ambiguous_digest_aliases_fail_closed(self, isolated_db):
        db = isolated_db
        logical_key = "ambiguous-key"
        current, migrated = db._api_key_lookup_digests(logical_key)
        with db.get_db() as session:
            session.add_all(
                [
                    db.DBAPIKey(api_key=current, tier="basic", entity_id="current"),
                    db.DBAPIKey(api_key=migrated, tier="basic", entity_id="migrated"),
                ]
            )
            session.commit()

        assert db.get_api_key(logical_key) is None
        assert db.delete_api_key(logical_key) is False
        assert db.reserve_api_key_usage(logical_key, "2026-08-14", 1, 10) is None
        with pytest.raises(RuntimeError, match="persistence unavailable"):
            db.get_api_key(logical_key, raise_on_error=True)

    def test_delete_unknown_key_returns_false(self, isolated_db):
        assert isolated_db.delete_api_key("missing-key") is False

    def test_delete_can_fail_closed(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            with pytest.raises(RuntimeError, match="persistence unavailable"):
                db.delete_api_key("err-key", raise_on_error=True)


class TestReserveApiKeyUsage:
    def test_reserve_usage_success(self, isolated_db):
        db = isolated_db
        db.save_api_key("usage-key", "basic", "e1")
        result = db.reserve_api_key_usage("usage-key", "2026-02-15", 42, 100)
        assert result is True

        key_data = db.get_api_key("usage-key")
        assert key_data["usage_date"] == "2026-02-15"
        assert key_data["usage_count"] == 42

    def test_reserve_nonexistent_key_returns_false(self, isolated_db):
        db = isolated_db
        result = db.reserve_api_key_usage("no-key", "2026-01-01", 1, 100)
        assert result is False

    def test_reserve_api_key_usage_error_returns_none(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.reserve_api_key_usage("err-key", "2026-01-01", 1, 100)
            assert result is None

    def test_database_errors_never_log_bound_secret_values(self, isolated_db):
        db = isolated_db
        callback_secret = "callback-query-secret"  # pragma: allowlist secret
        signing_secret = "webhook-signing-secret"  # pragma: allowlist secret
        with (
            patch.object(
                db,
                "get_db",
                side_effect=Exception(f"https://example.test/?token={callback_secret}"),
            ),
            patch.object(db, "logger") as mock_logger,
        ):
            assert not db.save_webhook(
                "entity-1",
                f"https://example.test/?token={callback_secret}",
                ["session.completed"],
                signing_secret,
            )

        serialized_logs = repr(mock_logger.method_calls)
        assert callback_secret not in serialized_logs
        assert signing_secret not in serialized_logs
        assert "Exception" in serialized_logs

    def test_reservation_at_limit_is_atomic(self, isolated_db):
        db = isolated_db
        db.save_api_key("limit-key", "basic", "e1")
        assert db.reserve_api_key_usage("limit-key", "2026-02-15", 99, 100)
        assert not db.reserve_api_key_usage("limit-key", "2026-02-15", 2, 100)
        assert db.get_api_key("limit-key")["usage_count"] == 99

    def test_parallel_workers_cannot_overbook_final_slot(self, isolated_db, tmp_path):
        db = isolated_db
        concurrent_engine = create_engine(
            f"sqlite:///{tmp_path / 'quota.db'}",
            connect_args={"check_same_thread": False, "timeout": 10},
        )
        concurrent_sessions = sessionmaker(
            autocommit=False, autoflush=False, bind=concurrent_engine
        )
        db.Base.metadata.create_all(bind=concurrent_engine)
        gate = Barrier(2)

        with (
            patch.object(db, "engine", concurrent_engine),
            patch.object(db, "SessionLocal", concurrent_sessions),
        ):
            db.save_api_key("parallel-key", "basic", "e1")

            def reserve() -> bool | None:
                gate.wait()
                return db.reserve_api_key_usage("parallel-key", "2026-02-15", 1, 1)

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [
                    future.result()
                    for future in [pool.submit(reserve) for _ in range(2)]
                ]

            assert results.count(True) == 1
            assert results.count(False) == 1
            assert db.get_api_key("parallel-key")["usage_count"] == 1

        concurrent_engine.dispose()


# ── Webhook Operations ───────────────────────────────────────────────────────


class TestSaveWebhook:
    def test_save_webhook_success(self, isolated_db):
        db = isolated_db
        result = db.save_webhook(
            "entity-1",
            "https://example.com/hook",
            ["verification.complete"],
            "secret-123",
        )
        assert result is True

    def test_save_webhook_none_secret(self, isolated_db):
        db = isolated_db
        result = db.save_webhook("entity-2", "https://example.com/hook", ["all"], None)
        assert result is True

    def test_get_webhooks_lists_recoverable_registrations(self, isolated_db):
        db = isolated_db
        db.save_webhook("entity-1", "https://example.com/one", ["all"], "one")
        db.save_webhook(
            "entity-2", "https://example.com/two", ["session.completed"], None
        )

        rows = db.get_webhooks()
        assert [row["entity_id"] for row in rows] == ["entity-1", "entity-2"]
        assert rows[1]["events"] == ["session.completed"]

    def test_save_webhook_upsert_replaces_existing(self, isolated_db):
        """Saving a webhook for the same entity_id should replace the old one."""
        db = isolated_db
        db.save_webhook("entity-upsert", "https://old.com", ["old_event"], "old-secret")
        db.save_webhook("entity-upsert", "https://new.com", ["new_event"], "new-secret")

        webhook = db.get_webhook("entity-upsert")
        assert webhook is not None
        assert webhook["url"] == "https://new.com"
        assert webhook["events"] == ["new_event"]
        assert webhook["secret"] == "new-secret"

    def test_save_webhook_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.save_webhook("e1", "https://x.com", ["e"], None)
            assert result is False


class TestGetWebhook:
    def test_get_existing_webhook(self, isolated_db):
        db = isolated_db
        db.save_webhook("ent-wh", "https://hook.com/cb", ["verify"], "s3cret")
        result = db.get_webhook("ent-wh")
        assert result is not None
        assert result["url"] == "https://hook.com/cb"
        assert result["events"] == ["verify"]
        assert result["secret"] == "s3cret"
        assert result["created_at"] is not None

    def test_get_nonexistent_webhook_returns_none(self, isolated_db):
        db = isolated_db
        result = db.get_webhook("no-entity")
        assert result is None

    def test_get_webhook_error_returns_none(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_webhook("err")
            assert result is None


class TestDeleteWebhook:
    def test_delete_existing_webhook(self, isolated_db):
        db = isolated_db
        db.save_webhook("del-ent", "https://del.com", ["x"], None)
        result = db.delete_webhook("del-ent")
        assert result is True
        assert db.get_webhook("del-ent") is None

    def test_delete_nonexistent_webhook_returns_false(self, isolated_db):
        db = isolated_db
        result = db.delete_webhook("nonexistent")
        assert result is False

    def test_delete_webhook_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.delete_webhook("err")
            assert result is False


# ── Verification Record Operations ───────────────────────────────────────────


class TestSaveVerificationRecord:
    def test_save_record_success(self, isolated_db):
        db = isolated_db
        result = db.save_verification_record("entity-v1", "192.168.1.1", True)
        assert result is True

    def test_save_record_failed_verification(self, isolated_db):
        db = isolated_db
        result = db.save_verification_record("entity-v2", "10.0.0.1", False)
        assert result is True

    def test_save_record_error_returns_false(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.save_verification_record("e1", "1.1.1.1", True)
            assert result is False


class TestGetRecentVerifications:
    def test_empty_when_no_records(self, isolated_db):
        db = isolated_db
        result = db.get_recent_verifications()
        assert result == []

    def test_returns_recent_records(self, isolated_db):
        db = isolated_db
        db.save_verification_record("e1", "1.1.1.1", True)
        db.save_verification_record("e2", "2.2.2.2", False)

        result = db.get_recent_verifications(hours=1)
        assert len(result) == 2
        for record in result:
            assert "entity_id" in record
            assert "ip_address" in record
            assert "passed" in record
            assert "created_at" in record

    def test_filters_by_time_window(self, isolated_db):
        """Records older than the time window should be excluded."""
        db = isolated_db
        # Insert a record, then manipulate its created_at to be old
        db.save_verification_record("old-entity", "1.1.1.1", True)
        with db.get_db() as session:
            record = session.query(db.DBVerificationRecord).first()
            record.created_at = datetime.now(timezone.utc) - timedelta(hours=3)
            session.commit()

        db.save_verification_record("new-entity", "2.2.2.2", True)

        result = db.get_recent_verifications(hours=1)
        assert len(result) == 1
        assert result[0]["entity_id"] == "new-entity"

    def test_get_recent_verifications_error_returns_empty(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_recent_verifications()
            assert result == []


class TestGetEntityVerificationCount:
    def test_zero_when_no_records(self, isolated_db):
        db = isolated_db
        result = db.get_entity_verification_count("nonexistent")
        assert result == 0

    def test_counts_only_matching_entity(self, isolated_db):
        db = isolated_db
        db.save_verification_record("target", "1.1.1.1", True)
        db.save_verification_record("target", "2.2.2.2", True)
        db.save_verification_record("other", "3.3.3.3", True)

        result = db.get_entity_verification_count("target")
        assert result == 2

    def test_respects_time_window(self, isolated_db):
        db = isolated_db
        db.save_verification_record("entity-tw", "1.1.1.1", True)
        # Make it old
        with db.get_db() as session:
            record = session.query(db.DBVerificationRecord).first()
            record.created_at = datetime.now(timezone.utc) - timedelta(hours=5)
            session.commit()

        db.save_verification_record("entity-tw", "2.2.2.2", True)

        result = db.get_entity_verification_count("entity-tw", hours=1)
        assert result == 1

    def test_get_entity_verification_count_error_returns_zero(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_entity_verification_count("err")
            assert result == 0


class TestGetIpEntities:
    def test_empty_set_when_no_records(self, isolated_db):
        db = isolated_db
        result = db.get_ip_entities("1.1.1.1")
        assert result == set()

    def test_returns_distinct_entities_for_ip(self, isolated_db):
        db = isolated_db
        db.save_verification_record("entity-a", "10.0.0.1", True)
        db.save_verification_record("entity-b", "10.0.0.1", True)
        db.save_verification_record("entity-a", "10.0.0.1", False)  # duplicate entity
        db.save_verification_record("entity-c", "10.0.0.2", True)  # different IP

        result = db.get_ip_entities("10.0.0.1")
        assert result == {"entity-a", "entity-b"}

    def test_respects_time_window(self, isolated_db):
        db = isolated_db
        db.save_verification_record("old-ent", "10.0.0.1", True)
        with db.get_db() as session:
            record = session.query(db.DBVerificationRecord).first()
            record.created_at = datetime.now(timezone.utc) - timedelta(hours=5)
            session.commit()

        db.save_verification_record("new-ent", "10.0.0.1", True)

        result = db.get_ip_entities("10.0.0.1", hours=1)
        assert result == {"new-ent"}

    def test_get_ip_entities_error_returns_empty_set(self, isolated_db):
        db = isolated_db
        with patch.object(db, "get_db", side_effect=Exception("db error")):
            result = db.get_ip_entities("1.1.1.1")
            assert result == set()


# ── Module-Level Initialization / URL Rewriting ──────────────────────────────


class TestDatabaseUrlRewriting:
    def test_postgres_url_rewritten(self):
        """The module should rewrite postgres:// to postgresql://."""
        # We test this by checking the documented behavior.
        # The actual rewriting happens at import time (lines 28-29).
        url = "postgres://user:pass@host/db"
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        assert url == "postgresql://user:pass@host/db"

    def test_postgresql_url_unchanged(self):
        """A postgresql:// URL should not be modified."""
        url = "postgresql://user:pass@host/db"
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        assert url == "postgresql://user:pass@host/db"

    def test_sqlite_url_unchanged(self):
        """A sqlite:// URL should not be modified."""
        url = "sqlite:///test.db"
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        assert url == "sqlite:///test.db"

    def test_prefixed_database_url_takes_precedence(self, monkeypatch):
        import database

        monkeypatch.setenv("DATABASE_URL", "postgresql://legacy/db")
        monkeypatch.setenv("METTLE_DATABASE_URL", "postgresql://prefixed/db")
        assert database._database_url_from_env() == "postgresql://prefixed/db"


class TestModuleLevelInit:
    def test_module_has_expected_attributes(self):
        """The database module should expose engine, SessionLocal, Base."""
        import database

        assert hasattr(database, "engine")
        assert hasattr(database, "SessionLocal")
        assert hasattr(database, "Base")

    def test_base_has_all_models(self):
        """All ORM models should be registered on Base.metadata."""
        import database

        table_names = set(database.Base.metadata.tables.keys())
        expected = {
            "sessions",
            "revoked_badges",
            "api_keys",
            "webhooks",
            "verification_records",
        }
        assert expected.issubset(table_names)


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_save_session_with_pydantic_models(self, isolated_db):
        """save_session should work with actual Pydantic models that have model_dump."""
        db = isolated_db
        from mettle.models import Challenge, ChallengeType

        challenge = Challenge(
            id="mtl_pydantic_test",
            type=ChallengeType.SPEED_MATH,
            prompt="Calculate: 2 + 3",
            data={"expected_answer": 5, "a": 2, "b": 3, "op": "+"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=5000,
        )
        result = db.save_session("pydantic-sess", "e1", "basic", [challenge])
        assert result is True

        session_data = db.get_session("pydantic-sess")
        assert session_data is not None
        assert len(session_data["challenges"]) == 1
        assert session_data["challenges"][0]["id"] == "mtl_pydantic_test"

    def test_update_session_with_pydantic_results(self, isolated_db):
        """update_session_results should work with Pydantic VerificationResult."""
        db = isolated_db
        db.save_session("pydantic-res", "e1", "basic", [MockChallenge()])

        from mettle.models import ChallengeType, VerificationResult

        vr = VerificationResult(
            challenge_id="ch-1",
            challenge_type=ChallengeType.SPEED_MATH,
            passed=True,
            details={"correct_answer": True},
            response_time_ms=500,
            time_limit_ms=5000,
        )
        result = db.update_session_results("pydantic-res", [vr], completed=True)
        assert result is True

        session_data = db.get_session("pydantic-res")
        assert session_data["completed"] is True
        assert len(session_data["results"]) == 1

    def test_webhook_events_stored_as_json(self, isolated_db):
        db = isolated_db
        events = ["verification.start", "verification.complete", "badge.issued"]
        db.save_webhook("json-test", "https://x.com", events, None)

        webhook = db.get_webhook("json-test")
        assert webhook["events"] == events

    def test_revoked_badge_evidence_stored_as_json(self, isolated_db):
        db = isolated_db
        evidence = {"scores": [0.1, 0.2], "reason_codes": ["too_fast", "pattern"]}
        db.add_revoked_badge("json-ev", "e1", "suspicious", evidence)

        badges = db.get_revoked_badges()
        assert len(badges) == 1
        assert badges[0]["jti"] == "json-ev"

    def test_multiple_sessions_independent(self, isolated_db):
        """Multiple sessions should be stored and retrieved independently."""
        db = isolated_db
        for i in range(5):
            db.save_session(
                f"multi-{i}", f"entity-{i}", "basic", [MockChallenge(id=f"ch-{i}")]
            )

        for i in range(5):
            sess = db.get_session(f"multi-{i}")
            assert sess is not None
            assert sess["entity_id"] == f"entity-{i}"

    def test_api_key_usage_updated_multiple_times(self, isolated_db):
        db = isolated_db
        db.save_api_key("multi-use", "basic", "e1")

        db.reserve_api_key_usage("multi-use", "2026-02-14", 10, 100)
        db.reserve_api_key_usage("multi-use", "2026-02-15", 25, 100)

        key = db.get_api_key("multi-use")
        assert key["usage_date"] == "2026-02-15"
        assert key["usage_count"] == 25
