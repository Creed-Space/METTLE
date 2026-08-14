"""METTLE database persistence for sessions and security records.

Production configuration requires PostgreSQL. SQLite remains available for
local development and tests.
"""

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    case,
    inspect,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _database_url_from_env() -> str:
    """Resolve the configured URL, preferring the documented METTLE prefix."""
    return (
        os.environ.get("METTLE_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///mettle.db"
    )


# Database configuration
DATABASE_URL = _database_url_from_env()

# Handle Render's postgres:// vs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def _build_engine(database_url: str):
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
    return create_engine(database_url, pool_pre_ping=True)


engine = _build_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger = logging.getLogger(__name__)


def configure_database(database_url: str) -> None:
    """Bind persistence to the exact URL validated by the application settings."""
    global DATABASE_URL, SessionLocal, engine
    normalized = database_url.replace("postgres://", "postgresql://", 1)
    if not normalized or "://" not in normalized:
        raise ValueError("Database URL is invalid")
    if normalized == DATABASE_URL:
        return
    replacement = _build_engine(normalized)
    previous = engine
    DATABASE_URL = normalized
    engine = replacement
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    previous.dispose()


def check_health() -> bool:
    """Return whether the configured database can execute a trivial query."""
    try:
        with engine.connect() as connection:
            return connection.execute(text("SELECT 1")).scalar_one() == 1
    except Exception as exc:
        logger.warning("Database health check failed: %s", type(exc).__name__)
        return False


def _model_json_dict(value: object) -> dict:
    """Return a JSON-ready model mapping while supporting lightweight test doubles."""
    try:
        return value.model_dump(mode="json")  # type: ignore[attr-defined]
    except TypeError:
        # Simple test doubles commonly expose model_dump() without Pydantic's
        # keyword arguments. Their return values are already JSON-compatible.
        return value.model_dump()  # type: ignore[attr-defined,no-any-return]


class Base(DeclarativeBase):
    """Base class for persisted METTLE records."""


# === Database Models ===


class DBSession(Base):
    """Verification session record."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    entity_id = Column(String(128), index=True)
    difficulty = Column(String(16), nullable=False)
    challenges_json = Column(Text, nullable=False)  # JSON array
    results_json = Column(Text, default="[]")  # JSON array
    access_token_hash = Column(String(64), nullable=True)
    badge_info_json = Column(Text, nullable=True)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class DBRevokedBadge(Base):
    """Revoked badge record."""

    __tablename__ = "revoked_badges"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(64), unique=True, index=True, nullable=False)
    entity_id = Column(String(128), index=True)
    reason = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)
    revoked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DBAPIKey(Base):
    """API key record."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String(64), unique=True, index=True, nullable=False)
    tier = Column(String(16), nullable=False)
    entity_id = Column(String(128), index=True)
    usage_date = Column(String(10), nullable=True)  # YYYY-MM-DD
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DBWebhook(Base):
    """Webhook registration."""

    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String(128), unique=True, index=True, nullable=False)
    url = Column(String(512), nullable=False)
    events_json = Column(Text, nullable=False)  # JSON array
    secret = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DBVerificationRecord(Base):
    """Verification record for collusion detection."""

    __tablename__ = "verification_records"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String(128), index=True, nullable=False)
    ip_address = Column(String(45), index=True, nullable=False)
    passed = Column(Boolean, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class DBSchemaMigration(Base):
    """Applied database schema version."""

    __tablename__ = "schema_migrations"

    version = Column(Integer, primary_key=True)
    applied_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# === Database Functions ===


LATEST_SCHEMA_VERSION = 3
_MIGRATION_LOCK_ID = int.from_bytes(
    hashlib.sha256(b"mettle-schema-migrations").digest()[:8], "big"
) & ((1 << 63) - 1)


def _upgrade_session_recovery_columns(connection) -> None:
    """Version 2: add restart-recovery and stable-badge session columns."""
    existing = {
        column["name"] for column in inspect(connection).get_columns("sessions")
    }
    additions = [
        (DBSession.access_token_hash.name, "VARCHAR(64)"),
        (DBSession.badge_info_json.name, "TEXT"),
    ]
    for column, column_type in additions:
        if column not in existing:
            connection.execute(
                text(f"ALTER TABLE sessions ADD COLUMN {column} {column_type}")
            )


def _upgrade_api_key_digests(connection) -> None:
    """Version 3: irreversibly hash every key created under the plaintext schema.

    Migration version, rather than value shape, distinguishes legacy plaintext
    from new digests. A two-phase rewrite avoids a transient unique-key conflict
    when one legacy value happens to equal another row's eventual digest.
    """
    records = list(connection.execute(select(DBAPIKey.id, DBAPIKey.api_key)).all())
    targets = {
        record.id: hashlib.sha256(record.api_key.encode("utf-8")).hexdigest()
        for record in records
    }
    if len(set(targets.values())) != len(targets):
        raise RuntimeError("API key digest collision during migration")

    reserved = {record.api_key for record in records} | set(targets.values())
    temporary: dict[int, str] = {}
    for record in records:
        counter = 0
        while True:
            candidate = hashlib.sha256(
                f"mettle-v3-temporary:{record.id}:{counter}".encode("utf-8")
            ).hexdigest()
            if candidate not in reserved:
                temporary[record.id] = candidate
                reserved.add(candidate)
                break
            counter += 1

    for record_id, digest in temporary.items():
        connection.execute(
            update(DBAPIKey).where(DBAPIKey.id == record_id).values(api_key=digest)
        )
    for record_id, digest in targets.items():
        connection.execute(
            update(DBAPIKey).where(DBAPIKey.id == record_id).values(api_key=digest)
        )


def init_db() -> None:
    """Create tables and apply every pending forward-only schema migration."""
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                {"lock_id": _MIGRATION_LOCK_ID},
            )
        # ``create_all`` establishes the version-1 baseline on a clean database.
        # It deliberately does not mutate an existing table, leaving upgrades to
        # the numbered migration below.
        Base.metadata.create_all(bind=connection)
        applied = set(
            connection.execute(select(DBSchemaMigration.version)).scalars().all()
        )
        migrations = {
            1: lambda _connection: None,
            2: _upgrade_session_recovery_columns,
            3: _upgrade_api_key_digests,
        }
        for version in range(1, LATEST_SCHEMA_VERSION + 1):
            if version in applied:
                continue
            migrations[version](connection)
            connection.execute(insert(DBSchemaMigration).values(version=version))


def get_schema_version() -> int:
    """Return the latest applied schema version, or zero before initialization."""
    try:
        if "schema_migrations" not in inspect(engine).get_table_names():
            return 0
        with engine.connect() as connection:
            versions = connection.execute(
                select(DBSchemaMigration.version).order_by(
                    DBSchemaMigration.version.desc()
                )
            ).scalars()
            return int(next(iter(versions), 0))
    except Exception as exc:
        logger.warning("Database schema version check failed: %s", type(exc).__name__)
        return 0


def check_schema_current() -> bool:
    """Return whether every migration required by this release is applied."""
    return get_schema_version() == LATEST_SCHEMA_VERSION


@contextmanager
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# === Session Operations ===


def save_session(
    session_id: str,
    entity_id: str | None,
    difficulty: str,
    challenges: list,
    access_token_hash: str | None = None,
    started_at: datetime | None = None,
) -> bool:
    """Save a new session to database."""
    try:
        with get_db() as db:
            db_session = DBSession(
                session_id=session_id,
                entity_id=entity_id,
                difficulty=difficulty,
                challenges_json=json.dumps(
                    [_model_json_dict(challenge) for challenge in challenges]
                ),
                access_token_hash=access_token_hash,
                created_at=started_at or datetime.now(timezone.utc),
            )
            db.add(db_session)
            db.commit()
            return True
    except Exception as exc:
        logger.error("Database save-session failed: %s", type(exc).__name__)
        return False


def get_session(session_id: str, *, raise_on_error: bool = False) -> dict | None:
    """Get session from database."""
    try:
        with get_db() as db:
            result = (
                db.query(DBSession).filter(DBSession.session_id == session_id).first()
            )
            if result:
                return {
                    "session_id": result.session_id,
                    "entity_id": result.entity_id,
                    "difficulty": result.difficulty,
                    "challenges": json.loads(result.challenges_json),
                    "results": json.loads(result.results_json),
                    "completed": result.completed,
                    "created_at": result.created_at,
                    "access_token_hash": result.access_token_hash,
                    "badge_info": json.loads(result.badge_info_json)
                    if result.badge_info_json
                    else None,
                }
            return None
    except Exception as exc:
        logger.error("Database fetch-session failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("Session persistence unavailable") from exc
        return None


def update_session_results(
    session_id: str,
    results: list,
    completed: bool = False,
    badge_info: dict | None = None,
) -> bool:
    """Update session results."""
    try:
        with get_db() as db:
            db_session = (
                db.query(DBSession).filter(DBSession.session_id == session_id).first()
            )
            if db_session:
                db_session.results_json = json.dumps(
                    [_model_json_dict(result) for result in results]
                )
                db_session.completed = bool(db_session.completed or completed)
                if db_session.badge_info_json:
                    existing_badge = json.loads(db_session.badge_info_json)
                    if badge_info is not None and existing_badge != badge_info:
                        raise RuntimeError(
                            "A session credential cannot be replaced after issuance"
                        )
                elif badge_info is not None:
                    db_session.badge_info_json = json.dumps(badge_info, default=str)
                if completed:
                    db_session.completed_at = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
    except Exception as exc:
        logger.error("Database update-session failed: %s", type(exc).__name__)
        return False


def get_recent_sessions(max_age_seconds: int = 1800, limit: int = 5000) -> list[dict]:
    """Load recent sessions that are eligible for process-restart recovery."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    try:
        with get_db() as db:
            rows = (
                db.query(DBSession)
                .filter(DBSession.created_at >= cutoff)
                .order_by(DBSession.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "session_id": row.session_id,
                    "entity_id": row.entity_id,
                    "difficulty": row.difficulty,
                    "challenges": json.loads(row.challenges_json),
                    "results": json.loads(row.results_json),
                    "completed": row.completed,
                    "created_at": row.created_at,
                    "access_token_hash": row.access_token_hash,
                    "badge_info": json.loads(row.badge_info_json)
                    if row.badge_info_json
                    else None,
                }
                for row in rows
                if row.access_token_hash
            ]
    except Exception as exc:
        logger.error("Database session-recovery failed: %s", type(exc).__name__)
        raise RuntimeError("Session recovery unavailable") from exc


def purge_expired_private_data(
    *,
    now: datetime | None = None,
    session_retention_seconds: int = 86400,
    verification_retention_seconds: int = 86400,
) -> dict[str, int]:
    """Delete expired private session and collusion records in one transaction.

    Revocations, API-key metadata, and webhook registrations are authority records
    with explicit lifecycle operations, so this timed purge intentionally does not
    remove them. Signed credentials are bearer artifacts and cannot be recalled by
    deleting issuer-side data.
    """
    if session_retention_seconds < 1800 or verification_retention_seconds < 3600:
        raise ValueError("Retention values are below the supported safety minimum")
    current = now or datetime.now(timezone.utc)
    session_cutoff = current - timedelta(seconds=session_retention_seconds)
    verification_cutoff = current - timedelta(seconds=verification_retention_seconds)
    with get_db() as db:
        sessions_deleted = (
            db.query(DBSession).filter(DBSession.created_at < session_cutoff).delete()
        )
        verifications_deleted = (
            db.query(DBVerificationRecord)
            .filter(DBVerificationRecord.created_at < verification_cutoff)
            .delete()
        )
        db.commit()
    return {
        "sessions_deleted": int(sessions_deleted),
        "verification_records_deleted": int(verifications_deleted),
    }


# === Revocation Operations ===


def add_revoked_badge(
    jti: str, entity_id: str | None, reason: str, evidence: dict | None
) -> bool:
    """Add a badge to the revocation list."""
    try:
        with get_db() as db:
            record = DBRevokedBadge(
                jti=jti,
                entity_id=entity_id,
                reason=reason,
                evidence_json=json.dumps(evidence) if evidence else None,
            )
            db.add(record)
            db.commit()
            return True
    except Exception as exc:
        logger.error("Database add-revocation failed: %s", type(exc).__name__)
        return False


def is_badge_revoked(jti: str, *, raise_on_error: bool = False) -> bool:
    """Check if a badge is revoked."""
    try:
        with get_db() as db:
            result = db.query(DBRevokedBadge).filter(DBRevokedBadge.jti == jti).first()
            return result is not None
    except Exception as exc:
        logger.error("Database check-revocation failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("Badge revocation status unavailable") from exc
        return False


def get_revoked_badges(limit: int = 100, *, raise_on_error: bool = False) -> list[dict]:
    """Get list of revoked badges."""
    try:
        with get_db() as db:
            results = (
                db.query(DBRevokedBadge)
                .order_by(DBRevokedBadge.revoked_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "jti": r.jti,
                    "entity_id": r.entity_id,
                    "reason": r.reason,
                    "evidence": json.loads(r.evidence_json)
                    if r.evidence_json
                    else None,
                    "revoked_at": r.revoked_at.isoformat() if r.revoked_at else None,
                }
                for r in results
            ]
    except Exception as exc:
        logger.error("Database list-revocations failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("Badge revocation audit unavailable") from exc
        return []


def count_revoked_badges(*, raise_on_error: bool = False) -> int:
    """Return the durable number of revoked badges."""
    try:
        with get_db() as db:
            return db.query(DBRevokedBadge).count()
    except Exception as exc:
        logger.error("Database count-revocations failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("Badge revocation audit unavailable") from exc
        return 0


# === API Key Operations ===


def _api_key_digest(api_key: str) -> str:
    """Return the irreversible database lookup value for an API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _api_key_lookup_digests(api_key: str) -> tuple[str, ...]:
    """Return current and migration-v3 double-digest lookup aliases."""
    current = _api_key_digest(api_key)
    migrated = _api_key_digest(current)
    return tuple(dict.fromkeys((current, migrated)))


def _find_api_key_record(db, api_key: str):
    """Resolve exactly one logical key across migration-safe digest aliases."""
    records = (
        db.query(DBAPIKey)
        .filter(DBAPIKey.api_key.in_(_api_key_lookup_digests(api_key)))
        .limit(2)
        .all()
    )
    if len(records) > 1:
        raise RuntimeError("Ambiguous API key digest aliases")
    return records[0] if records else None


def save_api_key(api_key: str, tier: str, entity_id: str | None) -> bool:
    """Save only an API key digest so a database read cannot recover the key."""
    try:
        with get_db() as db:
            if _find_api_key_record(db, api_key) is not None:
                return False
            record = DBAPIKey(
                api_key=_api_key_digest(api_key),
                tier=tier,
                entity_id=entity_id,
            )
            db.add(record)
            db.commit()
            return True
    except Exception as exc:
        logger.error("Database save-api-key failed: %s", type(exc).__name__)
        return False


def get_api_key(api_key: str, *, raise_on_error: bool = False) -> dict | None:
    """Get API key info."""
    try:
        with get_db() as db:
            result = _find_api_key_record(db, api_key)
            if result:
                return {
                    "tier": result.tier,
                    "entity_id": result.entity_id,
                    "usage_date": result.usage_date,
                    "usage_count": result.usage_count,
                    "created_at": result.created_at.isoformat()
                    if result.created_at
                    else None,
                }
            return None
    except Exception as exc:
        logger.error("Database fetch-api-key failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("API key persistence unavailable") from exc
        return None


def delete_api_key(api_key: str, *, raise_on_error: bool = False) -> bool:
    """Delete a digest-backed API key."""
    try:
        with get_db() as db:
            record = _find_api_key_record(db, api_key)
            if record is None:
                return False
            db.delete(record)
            db.commit()
            return True
    except Exception as exc:
        logger.error("Database delete-api-key failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("API key persistence unavailable") from exc
        return False


def reserve_api_key_usage(
    api_key: str,
    usage_date: str,
    amount: int,
    maximum: int,
) -> bool | None:
    """Atomically reserve daily API-key quota across every worker.

    ``True`` means reserved, ``False`` means missing or exhausted, and ``None``
    means the durable authority was unavailable.
    """
    if amount < 1 or maximum < 1:
        raise ValueError("Quota amount and maximum must be positive")
    try:
        with get_db() as db:
            record = _find_api_key_record(db, api_key)
            if record is None:
                return False
            next_count = case(
                (
                    DBAPIKey.usage_date == usage_date,
                    DBAPIKey.usage_count + amount,
                ),
                else_=amount,
            )
            result = db.execute(
                update(DBAPIKey)
                .where(
                    DBAPIKey.id == record.id,
                    next_count <= maximum,
                )
                .values(usage_date=usage_date, usage_count=next_count)
            )
            db.commit()
            return result.rowcount == 1
    except Exception as exc:
        logger.error("Database reserve-api-key-quota failed: %s", type(exc).__name__)
        return None


# === Webhook Operations ===


def save_webhook(
    entity_id: str, url: str, events: list[str], secret: str | None
) -> bool:
    """Save a webhook registration."""
    try:
        with get_db() as db:
            # Upsert - delete existing and create new
            db.query(DBWebhook).filter(DBWebhook.entity_id == entity_id).delete()
            record = DBWebhook(
                entity_id=entity_id,
                url=url,
                events_json=json.dumps(events),
                secret=secret,
            )
            db.add(record)
            db.commit()
            return True
    except Exception as exc:
        logger.error("Database save-webhook failed: %s", type(exc).__name__)
        return False


def get_webhook(entity_id: str) -> dict | None:
    """Get webhook for an entity."""
    try:
        with get_db() as db:
            result = (
                db.query(DBWebhook).filter(DBWebhook.entity_id == entity_id).first()
            )
            if result:
                return {
                    "url": result.url,
                    "events": json.loads(result.events_json),
                    "secret": result.secret,
                    "created_at": result.created_at.isoformat()
                    if result.created_at
                    else None,
                }
            return None
    except Exception as exc:
        logger.error("Database fetch-webhook failed: %s", type(exc).__name__)
        return None


def get_webhooks(limit: int = 1000, *, raise_on_error: bool = False) -> list[dict]:
    """Load persisted webhook registrations for restart recovery."""
    try:
        with get_db() as db:
            rows = db.query(DBWebhook).order_by(DBWebhook.created_at.asc()).limit(limit)
            return [
                {
                    "entity_id": row.entity_id,
                    "url": row.url,
                    "events": json.loads(row.events_json),
                    "secret": row.secret,
                    "created_at": row.created_at.isoformat()
                    if row.created_at
                    else None,
                }
                for row in rows
            ]
    except Exception as exc:
        logger.error("Database load-webhooks failed: %s", type(exc).__name__)
        if raise_on_error:
            raise RuntimeError("Webhook recovery unavailable") from exc
        return []


def delete_webhook(entity_id: str) -> bool:
    """Delete a webhook registration."""
    try:
        with get_db() as db:
            result = (
                db.query(DBWebhook).filter(DBWebhook.entity_id == entity_id).delete()
            )
            db.commit()
            return result > 0
    except Exception as exc:
        logger.error("Database delete-webhook failed: %s", type(exc).__name__)
        return False


# === Verification Record Operations ===


def save_verification_record(entity_id: str, ip_address: str, passed: bool) -> bool:
    """Save a verification record for collusion detection."""
    try:
        with get_db() as db:
            record = DBVerificationRecord(
                entity_id=entity_id,
                ip_address=ip_address,
                passed=passed,
            )
            db.add(record)
            db.commit()
            return True
    except Exception as exc:
        logger.error("Database save-verification failed: %s", type(exc).__name__)
        return False


def get_recent_verifications(hours: int = 1) -> list[dict]:
    """Get recent verification records."""
    try:
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_db() as db:
            results = (
                db.query(DBVerificationRecord)
                .filter(DBVerificationRecord.created_at >= cutoff)
                .order_by(DBVerificationRecord.created_at.desc())
                .limit(1000)
                .all()
            )
            return [
                {
                    "entity_id": r.entity_id,
                    "ip_address": r.ip_address,
                    "passed": r.passed,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ]
    except Exception as exc:
        logger.error("Database fetch-verifications failed: %s", type(exc).__name__)
        return []


def get_entity_verification_count(entity_id: str, hours: int = 1) -> int:
    """Get verification count for an entity in the last N hours."""
    try:
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_db() as db:
            return (
                db.query(DBVerificationRecord)
                .filter(DBVerificationRecord.entity_id == entity_id)
                .filter(DBVerificationRecord.created_at >= cutoff)
                .count()
            )
    except Exception as exc:
        logger.error("Database count-verifications failed: %s", type(exc).__name__)
        return 0


def get_ip_entities(ip_address: str, hours: int = 1) -> set[str]:
    """Get entities verified from an IP address."""
    try:
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_db() as db:
            results = (
                db.query(DBVerificationRecord.entity_id)
                .filter(DBVerificationRecord.ip_address == ip_address)
                .filter(DBVerificationRecord.created_at >= cutoff)
                .distinct()
                .all()
            )
            return {r.entity_id for r in results}
    except Exception as exc:
        logger.error("Database fetch-ip-entities failed: %s", type(exc).__name__)
        return set()
