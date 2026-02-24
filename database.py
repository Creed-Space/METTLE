"""
METTLE Database Layer

SQLite-based persistence for production deployments.
Falls back to in-memory storage if database unavailable.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///mettle.db")

# Handle Render's postgres:// vs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
logger = logging.getLogger(__name__)


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


# === Database Functions ===


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


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


def save_session(session_id: str, entity_id: str | None, difficulty: str, challenges: list) -> bool:
    """Save a new session to database."""
    try:
        with get_db() as db:
            db_session = DBSession(
                session_id=session_id,
                entity_id=entity_id,
                difficulty=difficulty,
                challenges_json=json.dumps([c.model_dump() for c in challenges], default=str),
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
            result = db.query(DBSession).filter(DBSession.session_id == session_id).first()
            if result:
                return {
                    "session_id": result.session_id,
                    "entity_id": result.entity_id,
                    "difficulty": result.difficulty,
                    "challenges": json.loads(result.challenges_json),
                    "results": json.loads(result.results_json),
                    "completed": result.completed,
                    "created_at": result.created_at,
                }
            return None
    except Exception as exc:
        logger.exception("Failed to fetch session '%s': %s", session_id, exc)
        return None


def update_session_results(session_id: str, results: list, completed: bool = False) -> bool:
    """Update session results."""
    try:
        with get_db() as db:
            db_session = db.query(DBSession).filter(DBSession.session_id == session_id).first()
            if db_session:
                db_session.results_json = json.dumps([r.model_dump() for r in results], default=str)
                db_session.completed = completed
                if completed:
                    db_session.completed_at = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
    except Exception as exc:
        logger.exception("Failed to update session '%s' results: %s", session_id, exc)
        return False


# === Revocation Operations ===


def add_revoked_badge(jti: str, entity_id: str | None, reason: str, evidence: dict | None) -> bool:
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


def is_badge_revoked(jti: str) -> bool:
    """Check if a badge is revoked."""
    try:
        with get_db() as db:
            result = db.query(DBRevokedBadge).filter(DBRevokedBadge.jti == jti).first()
            return result is not None
    except Exception as exc:
        logger.exception("Failed to check revoked badge '%s': %s", jti, exc)
        return False


def get_revoked_badges(limit: int = 100) -> list[dict]:
    """Get list of revoked badges."""
    try:
        with get_db() as db:
            results = db.query(DBRevokedBadge).order_by(DBRevokedBadge.revoked_at.desc()).limit(limit).all()
            return [
                {
                    "jti": r.jti,
                    "entity_id": r.entity_id,
                    "reason": r.reason,
                    "revoked_at": r.revoked_at.isoformat() if r.revoked_at else None,
                }
                for r in results
            ]
    except Exception as exc:
        logger.exception("Failed to list revoked badges: %s", exc)
        return []


# === API Key Operations ===


def save_api_key(api_key: str, tier: str, entity_id: str | None) -> bool:
    """Save an API key."""
    try:
        with get_db() as db:
            record = DBAPIKey(
                api_key=api_key,
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
            result = db.query(DBAPIKey).filter(DBAPIKey.api_key == api_key).first()
            if result:
                return {
                    "tier": result.tier,
                    "entity_id": result.entity_id,
                    "usage_date": result.usage_date,
                    "usage_count": result.usage_count,
                    "created_at": result.created_at.isoformat() if result.created_at else None,
                }
            return None
    except Exception as exc:
        logger.exception("Failed to fetch API key metadata: %s", exc)
        return None


def update_api_key_usage(api_key: str, usage_date: str, usage_count: int) -> bool:
    """Update API key usage."""
    try:
        with get_db() as db:
            record = db.query(DBAPIKey).filter(DBAPIKey.api_key == api_key).first()
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


def save_webhook(entity_id: str, url: str, events: list[str], secret: str | None) -> bool:
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
            result = db.query(DBWebhook).filter(DBWebhook.entity_id == entity_id).first()
            if result:
                return {
                    "url": result.url,
                    "events": json.loads(result.events_json),
                    "secret": result.secret,
                    "created_at": result.created_at.isoformat() if result.created_at else None,
                }
            return None
    except Exception as exc:
        logger.exception("Failed to fetch webhook for entity '%s': %s", entity_id, exc)
        return None


def delete_webhook(entity_id: str) -> bool:
    """Delete a webhook registration."""
    try:
        with get_db() as db:
            result = db.query(DBWebhook).filter(DBWebhook.entity_id == entity_id).delete()
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
        logger.exception("Failed to save verification record for entity '%s': %s", entity_id, exc)
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
        logger.exception("Failed to count verifications for entity '%s': %s", entity_id, exc)
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
