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
    inspect,
    text,
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

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger = logging.getLogger(__name__)


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


# === Database Functions ===


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    # Existing deployments may have the original sessions table. Add the two
    # recovery columns without requiring a destructive table rebuild.
    existing = {column["name"] for column in inspect(engine).get_columns("sessions")}
    additions = [
        (DBSession.access_token_hash.name, "VARCHAR(64)"),
        (DBSession.badge_info_json.name, "TEXT"),
    ]
    with engine.begin() as connection:
        for column, column_type in additions:
            if column not in existing:
                connection.execute(
                    text(f"ALTER TABLE sessions ADD COLUMN {column} {column_type}")
                )


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
        logger.exception("Failed to save session '%s': %s", session_id, exc)
        return False


def get_session(session_id: str) -> dict | None:
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
        logger.exception("Failed to fetch session '%s': %s", session_id, exc)
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
                db_session.completed = completed
                db_session.badge_info_json = (
                    json.dumps(badge_info, default=str) if badge_info else None
                )
                if completed:
                    db_session.completed_at = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
    except Exception as exc:
        logger.exception("Failed to update session '%s' results: %s", session_id, exc)
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
        logger.exception("Failed to load recent sessions: %s", exc)
        raise RuntimeError("Session recovery unavailable") from exc


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
        logger.exception("Failed to add revoked badge '%s': %s", jti, exc)
        return False


def is_badge_revoked(jti: str, *, raise_on_error: bool = False) -> bool:
    """Check if a badge is revoked."""
    try:
        with get_db() as db:
            result = db.query(DBRevokedBadge).filter(DBRevokedBadge.jti == jti).first()
            return result is not None
    except Exception as exc:
        logger.exception("Failed to check revoked badge '%s': %s", jti, exc)
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
        logger.exception("Failed to list revoked badges: %s", exc)
        if raise_on_error:
            raise RuntimeError("Badge revocation audit unavailable") from exc
        return []


def count_revoked_badges(*, raise_on_error: bool = False) -> int:
    """Return the durable number of revoked badges."""
    try:
        with get_db() as db:
            return db.query(DBRevokedBadge).count()
    except Exception as exc:
        logger.exception("Failed to count revoked badges: %s", exc)
        if raise_on_error:
            raise RuntimeError("Badge revocation audit unavailable") from exc
        return 0


# === API Key Operations ===


def _api_key_digest(api_key: str) -> str:
    """Return the irreversible database lookup value for an API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def save_api_key(api_key: str, tier: str, entity_id: str | None) -> bool:
    """Save only an API key digest so a database read cannot recover the key."""
    try:
        with get_db() as db:
            record = DBAPIKey(
                api_key=_api_key_digest(api_key),
                tier=tier,
                entity_id=entity_id,
            )
            db.add(record)
            db.commit()
            return True
    except Exception as exc:
        logger.exception("Failed to save API key: %s", exc)
        return False


def get_api_key(api_key: str) -> dict | None:
    """Get API key info."""
    try:
        with get_db() as db:
            digest = _api_key_digest(api_key)
            result = db.query(DBAPIKey).filter(DBAPIKey.api_key == digest).first()
            if result is None:
                # Transparently migrate records written by older releases.
                result = db.query(DBAPIKey).filter(DBAPIKey.api_key == api_key).first()
                if result is not None:
                    result.api_key = digest
                    db.commit()
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
        logger.exception("Failed to fetch API key metadata: %s", exc)
        return None


def update_api_key_usage(api_key: str, usage_date: str, usage_count: int) -> bool:
    """Update API key usage."""
    try:
        with get_db() as db:
            record = (
                db.query(DBAPIKey)
                .filter(DBAPIKey.api_key == _api_key_digest(api_key))
                .first()
            )
            if record:
                record.usage_date = usage_date
                record.usage_count = usage_count
                db.commit()
                return True
            return False
    except Exception as exc:
        logger.exception("Failed to update API key usage: %s", exc)
        return False


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
        logger.exception("Failed to save webhook for entity '%s': %s", entity_id, exc)
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
        logger.exception("Failed to fetch webhook for entity '%s': %s", entity_id, exc)
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
        logger.exception("Failed to load webhook registrations: %s", exc)
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
        logger.exception("Failed to delete webhook for entity '%s': %s", entity_id, exc)
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
        logger.exception(
            "Failed to save verification record for entity '%s': %s", entity_id, exc
        )
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
        logger.exception("Failed to fetch recent verifications: %s", exc)
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
        logger.exception(
            "Failed to count verifications for entity '%s': %s", entity_id, exc
        )
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
        logger.exception("Failed to fetch entities for IP '%s': %s", ip_address, exc)
        return set()


# Initialize database on import
try:
    init_db()
except Exception as e:
    logger.exception("Database initialization failed; using in-memory mode: %s", e)
