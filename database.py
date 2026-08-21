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
from typing import TypeVar

from sqlalchemy import (
    Boolean,
    case,
    Column,
    DateTime,
    func,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
    update,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


MAX_RECOVERY_SESSIONS = 5000
MAX_RECOVERED_WEBHOOKS = 1000
MAX_REVOKED_BADGES_QUERY = 1000
MAX_SESSION_RECOVERY_SCAN_ROWS = MAX_RECOVERY_SESSIONS * 4
MAX_WEBHOOK_RECOVERY_SCAN_ROWS = MAX_RECOVERED_WEBHOOKS * 4
MAX_RECOVERY_AGE_SECONDS = 24 * 60 * 60
MAX_VERIFICATION_HISTORY_HOURS = 24 * 365
MAX_SESSION_RECOVERY_JSON_BYTES = 1024 * 1024
MAX_WEBHOOK_RECOVERY_JSON_BYTES = 64 * 1024
MAX_RECOVERY_ROW_WARNINGS = 20
_JSONType = TypeVar("_JSONType")


def _database_url_from_env() -> str:
    """Resolve the configured URL, preferring the documented METTLE prefix."""
    return (
        os.environ.get("METTLE_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///mettle.db"
    )


def _normalize_database_url(value: str) -> str:
    """Return a SQLAlchemy-compatible database URL."""
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql://", 1)
    return value


# Database configuration
DATABASE_URL = _normalize_database_url(_database_url_from_env())

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


def _bounded_positive_int(value: object, name: str, maximum: int) -> int:
    """Validate a caller-controlled positive query bound."""
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _bounded_log_value(value: object, maximum: int = 128) -> str:
    """Bound untrusted persisted identifiers before they enter recovery logs."""
    rendered = str(value)
    if len(rendered) > maximum:
        return rendered[:maximum] + "..."
    return rendered


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _load_bounded_json(
    raw: object, *, maximum: int, expected: type[_JSONType]
) -> _JSONType:
    """Parse one persisted JSON value after enforcing a strict byte ceiling."""
    if not isinstance(raw, str) or len(raw) > maximum:
        raise ValueError("persisted JSON exceeds its recovery limit")
    encoded = raw.encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError("persisted JSON exceeds its recovery limit")
    value = json.loads(
        encoded,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, expected):
        raise ValueError("persisted JSON has the wrong top-level type")
    return value


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
    max_age_seconds = _bounded_positive_int(
        max_age_seconds, "max_age_seconds", MAX_RECOVERY_AGE_SECONDS
    )
    limit = _bounded_positive_int(limit, "limit", MAX_RECOVERY_SESSIONS)
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    try:
        with get_db() as db:
            rows = (
                db.query(DBSession)
                .filter(DBSession.created_at >= cutoff)
                .filter(DBSession.access_token_hash.is_not(None))
                .order_by(DBSession.created_at.desc(), DBSession.id.desc())
                .limit(MAX_SESSION_RECOVERY_SCAN_ROWS)
                .yield_per(100)
            )
            recovered: list[dict] = []
            invalid_rows = 0
            for row in rows:
                try:
                    challenges = _load_bounded_json(
                        row.challenges_json,
                        maximum=MAX_SESSION_RECOVERY_JSON_BYTES,
                        expected=list,
                    )
                    results = _load_bounded_json(
                        row.results_json,
                        maximum=MAX_SESSION_RECOVERY_JSON_BYTES,
                        expected=list,
                    )
                    badge_info = (
                        _load_bounded_json(
                            row.badge_info_json,
                            maximum=MAX_SESSION_RECOVERY_JSON_BYTES,
                            expected=dict,
                        )
                        if row.badge_info_json
                        else None
                    )
                except (UnicodeError, ValueError, TypeError):
                    invalid_rows += 1
                    if invalid_rows <= MAX_RECOVERY_ROW_WARNINGS:
                        logger.warning(
                            "Skipping malformed persisted session %r during recovery",
                            _bounded_log_value(row.session_id),
                        )
                    continue
                recovered.append(
                    {
                        "session_id": row.session_id,
                        "entity_id": row.entity_id,
                        "difficulty": row.difficulty,
                        "challenges": challenges,
                        "results": results,
                        "completed": row.completed,
                        "created_at": row.created_at,
                        "access_token_hash": row.access_token_hash,
                        "badge_info": badge_info,
                    }
                )
                if len(recovered) == limit:
                    break
            if invalid_rows > MAX_RECOVERY_ROW_WARNINGS:
                logger.warning(
                    "Skipped %d additional malformed persisted sessions during recovery",
                    invalid_rows - MAX_RECOVERY_ROW_WARNINGS,
                )
            recovered.reverse()
            return recovered
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
    limit = _bounded_positive_int(limit, "limit", MAX_REVOKED_BADGES_QUERY)
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


def get_api_key(api_key: str, *, raise_on_error: bool = False) -> dict | None:
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
        if raise_on_error:
            raise RuntimeError("API key persistence unavailable") from exc
        return None


def delete_api_key(api_key: str, *, raise_on_error: bool = False) -> bool:
    """Delete a digest-backed API key, with legacy plaintext compatibility."""
    try:
        with get_db() as db:
            record = (
                db.query(DBAPIKey)
                .filter(DBAPIKey.api_key == _api_key_digest(api_key))
                .first()
            )
            if record is None:
                record = db.query(DBAPIKey).filter(DBAPIKey.api_key == api_key).first()
            if record is None:
                return False
            db.delete(record)
            db.commit()
            return True
    except Exception as exc:
        logger.exception("Failed to delete API key: %s", exc)
        if raise_on_error:
            raise RuntimeError("API key persistence unavailable") from exc
        return False


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


def reserve_api_key_usage(
    api_key: str,
    usage_date: str,
    amount: int,
    daily_limit: int,
    *,
    raise_on_error: bool = False,
) -> tuple[bool, int]:
    """Atomically reserve daily quota and return ``(reserved, current_count)``.

    A denied reservation never mutates the durable counter. The count returned
    for a denial is the observed usage for ``usage_date`` and is informational;
    a later concurrent reservation may advance it immediately afterwards.
    """
    try:
        parsed_date = datetime.strptime(usage_date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError("usage_date must use YYYY-MM-DD") from exc
    if parsed_date.strftime("%Y-%m-%d") != usage_date:
        raise ValueError("usage_date must use YYYY-MM-DD")
    amount = _bounded_positive_int(amount, "amount", 2**31 - 1)
    daily_limit = _bounded_positive_int(daily_limit, "daily_limit", 2**31 - 1)

    digest = _api_key_digest(api_key)
    lookup_values = [digest] if digest == api_key else [digest, api_key]
    try:
        with get_db() as db:
            usage_for_date = case(
                (
                    DBAPIKey.usage_date == usage_date,
                    func.coalesce(DBAPIKey.usage_count, 0),
                ),
                else_=0,
            )
            next_count = usage_for_date + amount
            statement = (
                update(DBAPIKey)
                .where(DBAPIKey.api_key.in_(lookup_values))
                .where(next_count <= daily_limit)
                .values(
                    api_key=digest,
                    usage_date=usage_date,
                    usage_count=next_count,
                )
                .returning(DBAPIKey.usage_count)
            )
            reserved_count = db.execute(statement).scalar_one_or_none()
            if reserved_count is not None:
                db.commit()
                return True, int(reserved_count)

            record = (
                db.query(DBAPIKey).filter(DBAPIKey.api_key.in_(lookup_values)).first()
            )
            if record is None or record.usage_date != usage_date:
                return False, 0
            return False, int(record.usage_count or 0)
    except Exception as exc:
        logger.exception("Failed to reserve API key usage: %s", exc)
        if raise_on_error:
            raise RuntimeError("API key persistence unavailable") from exc
        return False, 0


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
    limit = _bounded_positive_int(limit, "limit", MAX_RECOVERED_WEBHOOKS)
    try:
        with get_db() as db:
            rows = (
                db.query(DBWebhook)
                .order_by(DBWebhook.created_at.desc(), DBWebhook.id.desc())
                .limit(MAX_WEBHOOK_RECOVERY_SCAN_ROWS)
                .yield_per(100)
            )
            recovered: list[dict] = []
            invalid_rows = 0
            for row in rows:
                try:
                    events = _load_bounded_json(
                        row.events_json,
                        maximum=MAX_WEBHOOK_RECOVERY_JSON_BYTES,
                        expected=list,
                    )
                    if len(events) > 64 or any(
                        not isinstance(event, str) or len(event) > 128
                        for event in events
                    ):
                        raise ValueError("persisted webhook events are invalid")
                except (UnicodeError, ValueError, TypeError):
                    invalid_rows += 1
                    if invalid_rows <= MAX_RECOVERY_ROW_WARNINGS:
                        logger.warning(
                            "Skipping malformed persisted webhook for entity %r during recovery",
                            _bounded_log_value(row.entity_id),
                        )
                    continue
                recovered.append(
                    {
                        "entity_id": row.entity_id,
                        "url": row.url,
                        "events": events,
                        "secret": row.secret,
                        "created_at": row.created_at.isoformat()
                        if row.created_at
                        else None,
                    }
                )
                if len(recovered) == limit:
                    break
            if invalid_rows > MAX_RECOVERY_ROW_WARNINGS:
                logger.warning(
                    "Skipped %d additional malformed persisted webhooks during recovery",
                    invalid_rows - MAX_RECOVERY_ROW_WARNINGS,
                )
            recovered.reverse()
            return recovered
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


def get_recent_verifications(
    hours: int = 1, *, raise_on_error: bool = False
) -> list[dict]:
    """Get recent verification records."""
    hours = _bounded_positive_int(hours, "hours", MAX_VERIFICATION_HISTORY_HOURS)
    try:
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
        if raise_on_error:
            raise RuntimeError("Verification history recovery unavailable") from exc
        return []


def get_entity_verification_count(entity_id: str, hours: int = 1) -> int:
    """Get verification count for an entity in the last N hours."""
    hours = _bounded_positive_int(hours, "hours", MAX_VERIFICATION_HISTORY_HOURS)
    try:
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
    hours = _bounded_positive_int(hours, "hours", MAX_VERIFICATION_HISTORY_HOURS)
    try:
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
