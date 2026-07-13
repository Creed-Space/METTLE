"""
METTLE API: Machine Evaluation Through Turing-inverse Logic Examination

Prove your mettle, with this CAPTCHA to keep humans out of places they shouldn't be.

A reverse-CAPTCHA verification system for AI-only spaces.
"""

import asyncio
import hashlib
import hmac
import ipaddress
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Protocol, TypedDict, cast

import jwt
import structlog
from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from mettle import (
    BadgeInfo,
    Challenge,
    Difficulty,
    MettleResult,
    MettleSession,
    VerificationResult,
    compute_mettle_result,
    generate_challenge_set,
    verify_response,
)
from pydantic import BaseModel, Field, field_validator
from redis.exceptions import RedisError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings

# Configuration
settings = get_settings()


class DatabaseLayer(Protocol):
    """Operations used from the optional persistence module."""

    def init_db(self) -> None: ...
    def save_session(
        self,
        session_id: str,
        entity_id: str | None,
        difficulty: str,
        challenges: list[Challenge],
        access_token_hash: str | None = None,
        started_at: datetime | None = None,
    ) -> bool: ...
    def update_session_results(
        self,
        session_id: str,
        results: list[VerificationResult],
        completed: bool = False,
        badge_info: dict[str, Any] | None = None,
    ) -> bool: ...
    def get_recent_sessions(
        self, max_age_seconds: int = 1800, limit: int = 5000
    ) -> list[dict[str, Any]]: ...
    def save_api_key(self, api_key: str, tier: str, entity_id: str | None) -> bool: ...
    def get_api_key(self, api_key: str) -> dict[str, Any] | None: ...
    def update_api_key_usage(
        self, api_key: str, usage_date: str, usage_count: int
    ) -> bool: ...
    def save_verification_record(
        self, entity_id: str, ip_address: str, passed: bool
    ) -> bool: ...
    def is_badge_revoked(self, jti: str, *, raise_on_error: bool = False) -> bool: ...
    def add_revoked_badge(
        self, jti: str, entity_id: str | None, reason: str, evidence: dict | None
    ) -> bool: ...
    def get_revoked_badges(
        self, limit: int = 100, *, raise_on_error: bool = False
    ) -> list[dict]: ...
    def count_revoked_badges(self, *, raise_on_error: bool = False) -> int: ...
    def save_webhook(
        self, entity_id: str, url: str, events: list[str], secret: str | None
    ) -> bool: ...
    def get_webhooks(
        self, limit: int = 1000, *, raise_on_error: bool = False
    ) -> list[dict[str, Any]]: ...
    def delete_webhook(self, entity_id: str) -> bool: ...


# Database layer (optional)
db: DatabaseLayer | None = None
if settings.use_database:
    try:
        from urllib.parse import urlparse

        import database as database_module

        db = cast(DatabaseLayer, database_module)

        # SECURITY: Redact credentials from database URL before logging
        logger_temp = structlog.get_logger()
        parsed_url = urlparse(settings.database_url)
        safe_url = f"{parsed_url.scheme}://{parsed_url.hostname}"
        if parsed_url.port:
            safe_url += f":{parsed_url.port}"
        logger_temp.info("database_enabled", url=safe_url)
    except ImportError:
        print("[METTLE] Database module not available, using in-memory storage")

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Memory limits for in-memory stores (DoS protection)
MAX_SESSIONS = 5000
MAX_CHALLENGES = 10000
MAX_VERIFICATION_GRAPH = 10000
MAX_REVOKED_BADGES = 10000
MAX_REVOCATION_AUDIT = 10000
MAX_API_KEYS = 10000
MAX_WEBHOOKS = 1000
MAX_AUTH_FAILURES = 10000
MAX_FINGERPRINT_RESPONSE_CHARS = 4096

_BIND_ALL_INTERFACES = str(ipaddress.IPv4Address(0))
_LOOPBACK_IPV4 = str(ipaddress.IPv4Address("127.0.0.1"))
_LOOPBACK_IPV6 = str(ipaddress.IPv6Address("::1"))
_BLOCKED_LOCALHOST_HOSTS = {
    "localhost",
    _LOOPBACK_IPV4,
    _LOOPBACK_IPV6,
    _BIND_ALL_INTERFACES,
}


def add_with_limit(store: dict, key: str, value: Any, max_size: int) -> None:
    """Add to dict with LRU-style eviction when full.

    SECURITY: Prevents unbounded memory growth from DoS attacks.
    """
    if len(store) >= max_size:
        # Remove oldest (first) item - Python 3.7+ dicts maintain insertion order
        oldest_key = next(iter(store))
        del store[oldest_key]
    store[key] = value


# In-memory storage
sessions: dict[str, MettleSession] = {}
challenges: dict[str, tuple[Challenge, float | None]] = {}
revoked_badges: dict[str, float] = {}  # JTI -> revocation timestamp (bounded dict)
revocation_audit: list[dict[str, Any]] = []  # Audit trail


def _credential_is_revoked(jti: str) -> bool:
    """Check the shared revocation namespace for any METTLE credential JTI."""
    return jti in revoked_badges or bool(
        db and db.is_badge_revoked(jti, raise_on_error=True)
    )


# Collusion detection - track verification patterns
verification_graph: dict[
    str, list[dict[str, Any]]
] = {}  # entity_id -> list of verifications
verification_timestamps: list[
    tuple[str, float]
] = []  # (entity_id, timestamp) for timing analysis

# API Key tiers for rate limiting
api_keys: dict[
    str, dict[str, Any]
] = {}  # api_key -> {tier, entity_id, created_at, usage_today}


class RateTier:
    """Rate limiting tier definitions."""

    TIERS = {
        "free": {
            "sessions_per_day": 100,
            "answers_per_minute": 60,
            "suites": ["basic"],
            "features": ["verification"],
        },
        "pro": {
            "sessions_per_day": 10000,
            "answers_per_minute": 600,
            "suites": ["basic", "full"],
            "features": ["verification", "batch", "webhooks", "fingerprinting"],
        },
        "enterprise": {
            "sessions_per_day": -1,  # Unlimited
            "answers_per_minute": -1,
            "suites": ["basic", "full", "custom"],
            "features": ["all"],
        },
    }

    @staticmethod
    def get_key_data(api_key: str | None) -> dict[str, Any] | None:
        """Resolve an API key from memory or durable digest-backed storage."""
        if not api_key:
            return None
        key_data = api_keys.get(api_key)
        if key_data is None and db:
            key_data = db.get_api_key(api_key)
            if key_data is not None:
                add_with_limit(api_keys, api_key, key_data, MAX_API_KEYS)
        return key_data

    @staticmethod
    def get_tier(api_key: str | None) -> str:
        """Get tier for an API key, default to free."""
        key_data = RateTier.get_key_data(api_key)
        if not key_data:
            return "free"
        return key_data.get("tier", "free")

    @staticmethod
    def get_limits(tier: str) -> dict[str, Any]:
        """Get rate limits for a tier."""
        return RateTier.TIERS.get(tier, RateTier.TIERS["free"])

    @staticmethod
    def check_limit(
        api_key: str | None, limit_type: str, amount: int = 1
    ) -> tuple[bool, str]:
        """Check if request is within rate limits. Returns (allowed, message)."""
        if amount < 1:
            raise ValueError("Rate-limit charge must be at least one")
        tier = RateTier.get_tier(api_key)
        limits = RateTier.get_limits(tier)

        if limits.get("sessions_per_day") == -1:
            return True, "Enterprise: unlimited"

        # Track usage
        key_data = RateTier.get_key_data(api_key)
        if api_key and key_data is not None:
            today = datetime.now(timezone.utc).date().isoformat()

            if key_data.get("usage_date") != today:
                key_data["usage_date"] = today
                key_data["usage_count"] = 0

            if limit_type == "session":
                max_sessions = limits["sessions_per_day"]
                usage_count = key_data.get("usage_count", 0)
                if usage_count + amount > max_sessions:
                    return False, f"Daily limit reached ({max_sessions} sessions)"
                key_data["usage_count"] = usage_count + amount
                if db and not db.update_api_key_usage(
                    api_key,
                    key_data["usage_date"],
                    key_data["usage_count"],
                ):
                    return False, "API key usage persistence unavailable"

        return True, f"OK ({tier} tier)"

    @staticmethod
    def register_key(
        api_key: str, tier: str, entity_id: str | None = None
    ) -> dict[str, Any]:
        """Register a new API key with a tier."""
        if tier not in RateTier.TIERS:
            raise ValueError(f"Invalid tier: {tier}")

        key_data = {
            "tier": tier,
            "entity_id": entity_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "usage_date": None,
            "usage_count": 0,
        }
        add_with_limit(api_keys, api_key, key_data, MAX_API_KEYS)
        # Persist to database if enabled
        if db and not db.save_api_key(api_key, tier, entity_id):
            api_keys.pop(api_key, None)
            raise RuntimeError("API key persistence unavailable")
        return key_data


class CollusionDetector:
    """Detect suspicious patterns in verification requests."""

    # Thresholds
    CLIQUE_THRESHOLD = 3  # Min entities to form suspicious clique
    TIME_WINDOW_SECONDS = 60  # Window for synchronized timing detection
    SYNC_THRESHOLD = 5  # Max verifications in window to be suspicious

    @staticmethod
    def record_verification(
        entity_id: str | None, ip_address: str, passed: bool
    ) -> None:
        """Record a verification for pattern analysis."""
        if not entity_id:
            return

        record = {
            "timestamp": time.time(),
            "ip_address": ip_address,
            "passed": passed,
        }

        # In-memory storage with memory limits
        if entity_id not in verification_graph:
            # Limit total entities tracked
            if len(verification_graph) >= MAX_VERIFICATION_GRAPH:
                oldest_key = next(iter(verification_graph))
                del verification_graph[oldest_key]
            verification_graph[entity_id] = []
        verification_graph[entity_id].append(record)
        # Keep only last 100 records per entity
        if len(verification_graph[entity_id]) > 100:
            verification_graph[entity_id] = verification_graph[entity_id][-100:]

        # Keep last 1000 timestamps for timing analysis
        verification_timestamps.append((entity_id, time.time()))
        if len(verification_timestamps) > 1000:
            verification_timestamps.pop(0)

        # Persist to database if enabled
        if db:
            db.save_verification_record(entity_id, ip_address, passed)

    @staticmethod
    def check_collusion(entity_id: str, ip_address: str) -> dict[str, Any]:
        """Check for collusion indicators."""
        warnings: list[str] = []
        risk_score = 0.0

        # Check 1: Same IP verifying multiple entities
        ip_entities = set()
        for eid, records in verification_graph.items():
            for r in records[-10:]:  # Last 10 per entity
                if r["ip_address"] == ip_address:
                    ip_entities.add(eid)

        if len(ip_entities) >= CollusionDetector.CLIQUE_THRESHOLD:
            warnings.append(
                f"IP {ip_address[:8]}... verified {len(ip_entities)} different entities"
            )
            risk_score += 0.3

        # Check 2: Synchronized timing (burst of verifications)
        now = time.time()
        recent = [
            t
            for _, t in verification_timestamps
            if now - t < CollusionDetector.TIME_WINDOW_SECONDS
        ]
        if len(recent) >= CollusionDetector.SYNC_THRESHOLD:
            warnings.append(
                f"{len(recent)} verifications in {CollusionDetector.TIME_WINDOW_SECONDS}s window"
            )
            risk_score += 0.2

        # Check 3: Entity verified too frequently
        if entity_id in verification_graph:
            entity_records = verification_graph[entity_id]
            recent_entity = [
                r for r in entity_records if now - r["timestamp"] < 3600
            ]  # Last hour
            if len(recent_entity) > 10:
                warnings.append(
                    f"Entity verified {len(recent_entity)} times in last hour"
                )
                risk_score += 0.2

        return {
            "risk_score": min(risk_score, 1.0),
            "warnings": warnings,
            "flagged": risk_score >= 0.5,
        }

    @staticmethod
    def get_stats() -> dict[str, Any]:
        """Get collusion detection statistics."""
        return {
            "tracked_entities": len(verification_graph),
            "recent_verifications": len(verification_timestamps),
            "unique_ips": len(
                set(
                    r["ip_address"]
                    for records in verification_graph.values()
                    for r in records[-10:]
                )
            ),
        }


# Track failed admin auth attempts for exponential backoff
_admin_auth_failures: dict[str, list[float]] = {}  # IP -> list of failure timestamps
_ADMIN_AUTH_WINDOW = 300  # 5 minute window
_ADMIN_AUTH_MAX_FAILURES = 5  # Max failures before blocking


def check_admin_auth_rate_limit(ip_address: str) -> tuple[bool, int]:
    """Check if IP is rate-limited due to failed admin auth attempts.

    Returns (is_allowed, seconds_until_retry).
    """
    now = time.time()
    failures = _admin_auth_failures.get(ip_address, [])

    # Clean old failures outside window
    failures = [f for f in failures if now - f < _ADMIN_AUTH_WINDOW]
    _admin_auth_failures[ip_address] = failures

    if len(failures) >= _ADMIN_AUTH_MAX_FAILURES:
        # Exponential backoff: 2^(failures-max) seconds, capped at 5 minutes
        backoff = min(2 ** (len(failures) - _ADMIN_AUTH_MAX_FAILURES + 1), 300)
        last_failure = failures[-1] if failures else 0
        time_since_last = now - last_failure
        if time_since_last < backoff:
            return False, int(backoff - time_since_last)

    return True, 0


def record_admin_auth_failure(ip_address: str) -> None:
    """Record a failed admin auth attempt."""
    if ip_address not in _admin_auth_failures:
        # Limit total IPs tracked to prevent memory DoS
        if len(_admin_auth_failures) >= MAX_AUTH_FAILURES:
            oldest_key = next(iter(_admin_auth_failures))
            del _admin_auth_failures[oldest_key]
        _admin_auth_failures[ip_address] = []
    _admin_auth_failures[ip_address].append(time.time())
    # Keep only last 100 failures per IP
    if len(_admin_auth_failures[ip_address]) > 100:
        _admin_auth_failures[ip_address] = _admin_auth_failures[ip_address][-100:]


def verify_admin_key(provided_key: str | None, ip_address: str | None = None) -> bool:
    """Verify admin API key using constant-time comparison.

    SECURITY: Uses secrets.compare_digest to prevent timing attacks that could
    leak information about the key value through response time differences.

    If ip_address is provided, also checks rate limiting and records failures.
    """
    if not settings.admin_api_key or not provided_key:
        return False

    # Both arguments must be the same type and length for proper comparison
    is_valid = secrets.compare_digest(
        provided_key.encode("utf-8"),
        settings.admin_api_key.encode("utf-8"),
    )

    # Record failure for rate limiting if IP provided
    if not is_valid and ip_address:
        record_admin_auth_failure(ip_address)

    return is_valid


# Track startup time
startup_time: datetime = datetime.now(timezone.utc)


# === Security Headers Middleware ===
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # cdn.jsdelivr.net + blob worker + fastapi.tiangolo.com favicon are required
        # by FastAPI's bundled Swagger UI (/docs) and ReDoc (/redoc).
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://www.googletagmanager.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
            "img-src 'self' data: https://www.googletagmanager.com https://fastapi.tiangolo.com; "
            "worker-src 'self' blob:; "
            "connect-src 'self' https://www.google-analytics.com https://*.google-analytics.com https://*.analytics.google.com"
        )
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


# === Request ID Middleware ===
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID for tracing."""

    async def dispatch(self, request: Request, call_next):
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# === Session Cleanup Task ===
LEGACY_SESSION_RECOVERY_SECONDS = 1800


def _persist_new_legacy_session(session: MettleSession) -> bool:
    """Persist a new legacy session when durable storage is enabled."""
    if not db:
        return True
    return db.save_session(
        session.session_id,
        session.entity_id,
        session.difficulty.value,
        session.challenges,
        session.access_token_hash,
        session.started_at,
    )


def _persist_legacy_progress(session: MettleSession) -> bool:
    """Persist answer progress and stable credential state."""
    if not db:
        return True
    badge_info = (
        session.badge_info.model_dump(mode="json") if session.badge_info else None
    )
    return db.update_session_results(
        session.session_id,
        session.results,
        session.completed,
        badge_info,
    )


def _restore_persistent_runtime_state() -> None:
    """Recover recent legacy sessions and webhook registrations from PostgreSQL."""
    if not db:
        return

    restored_sessions = 0
    for stored in db.get_recent_sessions(
        max_age_seconds=LEGACY_SESSION_RECOVERY_SECONDS,
        limit=MAX_SESSIONS,
    ):
        try:
            started_at = stored["created_at"]
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            session = MettleSession(
                session_id=stored["session_id"],
                entity_id=stored.get("entity_id"),
                difficulty=Difficulty(stored["difficulty"]),
                challenges=[
                    Challenge.model_validate(item) for item in stored["challenges"]
                ],
                results=[
                    VerificationResult.model_validate(item)
                    for item in stored.get("results", [])
                ],
                started_at=started_at,
                completed=bool(stored.get("completed")),
                access_token_hash=stored["access_token_hash"],
                badge_info=BadgeInfo.model_validate(stored["badge_info"])
                if stored.get("badge_info")
                else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "legacy_session_recovery_skipped",
                session_id=stored.get("session_id"),
                error=type(exc).__name__,
            )
            continue

        add_with_limit(sessions, session.session_id, session, MAX_SESSIONS)
        current_index = len(session.results)
        if not session.completed and current_index < len(session.challenges):
            current = session.challenges[current_index]
            # A recovered client already possesses this challenge, but downtime
            # and deploy warm-up are outside its control. Leave the stopwatch
            # disarmed until the owner's first authenticated access.
            issued_at = None
            add_with_limit(
                challenges,
                current.id,
                (current, issued_at),
                MAX_CHALLENGES,
            )
        restored_sessions += 1

    restored_webhooks = 0
    for stored in db.get_webhooks(limit=MAX_WEBHOOKS, raise_on_error=True):
        entity_id = stored.pop("entity_id")
        add_with_limit(webhooks, entity_id, stored, MAX_WEBHOOKS)
        restored_webhooks += 1

    logger.info(
        "persistent_runtime_state_restored",
        sessions=restored_sessions,
        webhooks=restored_webhooks,
    )


async def cleanup_expired_sessions():
    """Background task to remove expired sessions (prevents memory DoS)."""
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        cutoff = time.time() - 1800  # 30 minutes TTL
        expired_sessions = [
            sid for sid, s in sessions.items() if s.started_at.timestamp() < cutoff
        ]
        expired_challenges = [
            cid
            for cid, (_, issued_at) in challenges.items()
            if issued_at is not None and issued_at < cutoff
        ]
        for sid in expired_sessions:
            del sessions[sid]
        for cid in expired_challenges:
            del challenges[cid]
        if expired_sessions or expired_challenges:
            logger.info(
                "cleanup_expired",
                sessions_removed=len(expired_sessions),
                challenges_removed=len(expired_challenges),
            )


# === Lifespan Handler ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global startup_time
    startup_time = datetime.now(timezone.utc)

    if settings.is_production and not settings.secret_key:
        raise RuntimeError("SECRET_KEY environment variable required in production")

    if db:
        try:
            db.init_db()
            _restore_persistent_runtime_state()
        except Exception as exc:
            raise RuntimeError("Database initialization failed") from exc
    app.state.credential_revocation_checker = _credential_is_revoked

    logger.info(
        "mettle_starting",
        environment=settings.environment,
        version=settings.api_version,
    )
    print("[METTLE] API starting...")
    print("   Machine Evaluation Through Turing-inverse Logic Examination")
    print("   'Prove your mettle.'")

    # Initialize Redis for METTLE router (optional — returns 503 if unavailable)
    redis_url = os.environ.get("METTLE_REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as redis_client

            app.state.redis = redis_client.from_url(
                redis_url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                retry_on_timeout=False,
                health_check_interval=30,
            )
            await app.state.redis.ping()
            # Never log any portion of a connection URL. Credentials can occur
            # before the host and may be exposed even by a short prefix.
            logger.info("redis_connected")
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            app.state.redis = None
    else:
        app.state.redis = None

    # Init VCP signing (Ed25519 for attestations)
    try:
        from mettle.signing import init_signing

        signing_available = init_signing()
        if settings.is_production and not signing_available:
            raise RuntimeError("VCP attestation signing is unavailable in production")
    except ImportError:
        if settings.is_production:
            raise RuntimeError("VCP attestation signing dependencies are unavailable")

    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())

    yield

    # Shutdown Redis
    if getattr(app.state, "redis", None):
        await app.state.redis.aclose()

    # Shutdown cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("mettle_shutdown")


# === FastAPI App ===
app = FastAPI(
    title=settings.api_title,
    description="""
**Machine Evaluation Through Turing-inverse Logic Examination**

*"Prove your mettle."*

METTLE is a verification system for AI-only spaces. It tests capabilities
that emerge from AI-native cognition—speed, consistency, instruction-following—
to distinguish AI agents from humans and humans-using-AI-as-tool.

## How It Works

1. **Start a session** - Choose difficulty and get your first challenge
2. **Answer challenges** - Respond correctly within time limits
3. **Get verified** - Pass 80% to receive a METTLE badge

## Difficulty Levels

| Level | Challenges | Time Limits | Use Case |
|-------|------------|-------------|----------|
| `basic` | 3 | 5-10s | Any AI model |
| `full` | 5 | 2-5s | Sophisticated agents |

## Challenge Types

- **Speed Math** - Fast arithmetic computation
- **Token Prediction** - Complete well-known phrases
- **Instruction Following** - Follow formatting rules precisely
- **Chained Reasoning** - Multi-step calculations (full only)
- **Consistency** - Answer consistently multiple times (full only)
    """,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Session", "description": "METTLE verification session management"},
        {"name": "Status", "description": "API status and health checks"},
        {"name": "Badge", "description": "Verification badge management"},
    ],
    contact={
        "name": "METTLE Support",
        "url": "https://github.com/Creed-Space/METTLE",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
)

# API Router - all API endpoints go under /api
api_router = APIRouter(prefix="/api")

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))


async def _redis_unavailable_handler(
    _request: Request, _exc: RedisError
) -> JSONResponse:
    """Fail closed and promptly when the v2 session store is unavailable."""
    logger.warning("redis_request_unavailable")
    return JSONResponse(
        status_code=503,
        content={"detail": "METTLE session storage temporarily unavailable"},
    )


app.add_exception_handler(RedisError, cast(Any, _redis_unavailable_handler))

# Add middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=settings.allowed_origins != "*",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# === Request/Response Models ===
class StartSessionRequest(BaseModel):
    """Request to start a METTLE verification session."""

    difficulty: Difficulty = Field(
        default=Difficulty.BASIC,
        description="Verification difficulty level",
        json_schema_extra={"example": "basic"},
    )
    entity_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional identifier for the entity being verified",
        json_schema_extra={"example": "my-agent-001"},
    )

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(_cls, v: str | None) -> str | None:
        """Sanitize entity_id."""
        if v is not None:
            # Strip whitespace and limit characters
            v = v.strip()[:128]
        return v


class StartSessionResponse(BaseModel):
    """Response with session info and first challenge."""

    session_id: str = Field(description="Unique session identifier")
    session_token: str = Field(
        description="Bearer token required for answering and reading this session"
    )
    difficulty: Difficulty = Field(description="Selected difficulty level")
    total_challenges: int = Field(description="Total number of challenges to complete")
    current_challenge: Challenge = Field(description="First challenge to answer")
    message: str = Field(description="Status message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "ses_abc123def456",
                "difficulty": "basic",
                "total_challenges": 3,
                "current_challenge": {
                    "id": "mtl_xyz789",
                    "type": "speed_math",
                    "prompt": "Calculate: 47 + 83",
                    "time_limit_ms": 5000,
                },
                "message": "METTLE verification started. 3 challenges to complete.",
            }
        }
    }


class SubmitAnswerRequest(BaseModel):
    """Submit an answer to a challenge."""

    session_id: str = Field(
        description="Session identifier from start response",
        min_length=1,
        max_length=64,
        pattern=r"^ses_[a-f0-9]{24}$",
    )
    challenge_id: str = Field(
        description="Challenge identifier to answer",
        min_length=1,
        max_length=64,
        pattern=r"^mtl_[a-f0-9]{24}$",
    )
    answer: str = Field(
        description="Your answer to the challenge",
        max_length=1024,
    )

    @field_validator("answer")
    @classmethod
    def validate_answer(_cls, v: str) -> str:
        """Sanitize and validate answer."""
        if len(v) > 1024:
            raise ValueError("Answer exceeds maximum length of 1024 characters")
        return v


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an answer."""

    result: VerificationResult = Field(description="Result of this challenge")
    next_challenge: Challenge | None = Field(
        description="Next challenge, or null if complete"
    )
    session_complete: bool = Field(description="Whether session is complete")
    challenges_remaining: int = Field(description="Number of challenges left")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")


def _attach_stable_session_badge(
    session: MettleSession,
    result: MettleResult,
) -> None:
    """Attach one stable server-issued badge to a passing session.

    The badge attests that this reverse-CAPTCHA session passed. The optional
    ``entity_id`` remains explicitly self-asserted and is never represented as
    a proven identity on the public legacy API.
    """
    result.assurance = "mettle_behavioral_verification"
    if not result.verified:
        result.credential_eligible = False
        result.tier = "none"
        result.badge = None
        result.badge_info = None
        return
    if not settings.secret_key:
        # Production configuration requires a signing key. Development without
        # one may still return the pass result, but must never emit an unsigned
        # or forgeable badge.
        result.credential_eligible = False
        result.tier = "silver" if session.difficulty == Difficulty.FULL else "bronze"
        result.badge = None
        result.badge_info = None
        return
    if session.badge_info is None:
        badge_data = generate_signed_badge(
            entity_id=session.entity_id,
            difficulty=session.difficulty.value,
            pass_rate=result.pass_rate,
            session_id=session.session_id,
        )
        session.badge_info = BadgeInfo(
            token=badge_data["token"],
            expires_at=datetime.fromisoformat(badge_data["expires_at"]),
            freshness_nonce=badge_data["freshness_nonce"],
            signed=True,
            jti=badge_data["jti"],
        )
    result.credential_eligible = True
    result.tier = "silver" if session.difficulty == Difficulty.FULL else "bronze"
    result.badge_info = session.badge_info
    result.badge = session.badge_info.token


def _require_session_access(request: Request, session: MettleSession) -> None:
    """Authorize a legacy-session operation using its independent bearer token."""
    presented = request.headers.get("X-Session-Token")
    if not presented:
        raise HTTPException(status_code=401, detail="Session token required")
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    if not hmac.compare_digest(presented_hash, session.access_token_hash):
        raise HTTPException(status_code=403, detail="Invalid session token")


def _arm_recovered_challenge(session: MettleSession) -> None:
    """Start a recovered challenge's clock on the owner's first access."""
    current_index = len(session.results)
    if session.completed or current_index >= len(session.challenges):
        return
    challenge_id = session.challenges[current_index].id
    challenge_data = challenges.get(challenge_id)
    if challenge_data is not None and challenge_data[1] is None:
        challenges[challenge_id] = (challenge_data[0], time.time())


class BadgeVerifyResponse(BaseModel):
    """Response for badge verification."""

    valid: bool = Field(description="Whether the badge is valid")
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Badge payload if valid",
    )
    error: str | None = Field(default=None, description="Error message if invalid")
    expires_at: str | None = Field(
        default=None, description="When the badge expires (ISO format)"
    )
    revoked: bool = Field(
        default=False, description="Whether the badge has been revoked"
    )


class BadgeVerifyRequest(BaseModel):
    """Request body for badge verification without URL token exposure."""

    token: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The signed METTLE badge token to verify",
    )


# === API Endpoints (mounted at /api) ===
@api_router.get(
    "/",
    tags=["Status"],
    summary="API Information",
    description="Get basic API information and available endpoints.",
)
async def api_root():
    """METTLE API root."""
    return {
        "name": "METTLE",
        "full_name": "Machine Evaluation Through Turing-inverse Logic Examination",
        "tagline": "Prove your mettle.",
        "description": "A CAPTCHA to keep humans out of places they shouldn't be.",
        "version": settings.api_version,
        "documentation": "/docs",
        "endpoints": {
            "POST /api/session/start": "Start a verification session",
            "POST /api/session/answer": "Submit an answer to current challenge",
            "GET /api/session/{session_id}": "Get session status",
            "GET /api/session/{session_id}/result": "Get final verification result",
            "POST /api/badge/verify": "Verify a METTLE badge",
            "GET /api/health": "Health check",
        },
    }


@api_router.get(
    "/health",
    tags=["Status"],
    summary="Health Check",
    description="Check API health and get operational statistics.",
)
async def health():
    """Health check endpoint with detailed status."""
    now = datetime.now(timezone.utc)
    uptime = (now - startup_time).total_seconds()

    return {
        "status": "healthy",
        "version": settings.api_version,
        "environment": settings.environment,
        "timestamp": now.isoformat(),
        "uptime_seconds": round(uptime, 2),
        "active_sessions": len(sessions),
        "pending_challenges": len(challenges),
    }


@api_router.post(
    "/session/start",
    response_model=StartSessionResponse,
    tags=["Session"],
    summary="Start Verification Session",
    description="Begin a new METTLE verification session. Returns the first challenge.",
    responses={
        200: {"description": "Session started successfully"},
        422: {"description": "Invalid request parameters"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit(settings.rate_limit_sessions)
async def start_session(
    request: Request,
    body: StartSessionRequest = Body(
        ...,
        openapi_examples={
            "basic": {
                "summary": "Basic Verification",
                "description": "Start with relaxed timing for any AI model",
                "value": {"difficulty": "basic", "entity_id": "my-agent-001"},
            },
            "full": {
                "summary": "Full Verification",
                "description": "Complete verification with strict timing",
                "value": {"difficulty": "full", "entity_id": "advanced-agent"},
            },
            "anonymous": {
                "summary": "Anonymous",
                "description": "Verify without entity ID",
                "value": {"difficulty": "basic"},
            },
        },
    ),
):
    """Start a new METTLE verification session."""
    session_id = f"ses_{secrets.token_hex(12)}"
    session_token = secrets.token_urlsafe(32)

    # Check for collusion patterns
    ip_address = get_remote_address(request)
    collusion_check = CollusionDetector.check_collusion(
        body.entity_id or "", ip_address
    )

    # Log if collusion detected (but don't block - allow verification to proceed)
    if collusion_check.get("flagged"):
        logger.warning(
            "collusion_flagged",
            entity_id=body.entity_id,
            ip_address=ip_address[:15] if ip_address else None,
            risk_score=collusion_check.get("risk_score"),
            warnings=collusion_check.get("warnings"),
        )

    # Generate challenges
    challenge_list = generate_challenge_set(body.difficulty)

    # Create session
    session = MettleSession(
        session_id=session_id,
        entity_id=body.entity_id,
        difficulty=body.difficulty,
        challenges=challenge_list,
        access_token_hash=hashlib.sha256(session_token.encode()).hexdigest(),
    )

    # Public callers must not be able to evict unrelated active sessions. At
    # capacity, fail closed and let the cleanup task reclaim expired entries.
    if len(sessions) >= MAX_SESSIONS or len(challenges) >= MAX_CHALLENGES:
        raise HTTPException(
            status_code=503,
            detail="Verification capacity reached; retry shortly",
        )
    sessions[session_id] = session

    # Store first challenge with timestamp
    first_challenge = challenge_list[0]
    challenges[first_challenge.id] = (first_challenge, time.time())

    if not _persist_new_legacy_session(session):
        sessions.pop(session_id, None)
        challenges.pop(first_challenge.id, None)
        raise HTTPException(
            status_code=503,
            detail="Session persistence is temporarily unavailable",
        )

    # Log session start
    logger.info(
        "session_started",
        session_id=session_id,
        entity_id=body.entity_id,
        difficulty=body.difficulty.value,
        challenges_count=len(challenge_list),
    )

    return StartSessionResponse(
        session_id=session_id,
        session_token=session_token,
        difficulty=body.difficulty,
        total_challenges=len(challenge_list),
        current_challenge=first_challenge.sanitized(),  # Never expose answers
        message=f"METTLE verification started. {len(challenge_list)} challenges to complete.",
    )


class BatchStartRequest(BaseModel):
    """Request to start multiple verification sessions."""

    entity_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of entity IDs to verify (max 50)",
    )
    difficulty: Difficulty = Field(
        default=Difficulty.BASIC,
        description="Verification difficulty for all sessions",
    )


class BatchStartResponse(BaseModel):
    """Response with multiple session starts."""

    sessions: list[dict[str, Any]] = Field(description="List of started sessions")
    total: int = Field(description="Total sessions started")
    failed: int = Field(description="Number of failed starts")


@api_router.post(
    "/session/batch",
    response_model=BatchStartResponse,
    tags=["Session"],
    summary="Batch Start Sessions",
    description="Start multiple verification sessions at once (Pro/Enterprise tier).",
    responses={
        200: {"description": "Sessions started"},
        401: {"description": "Unauthorized - requires API key"},
        403: {"description": "Forbidden - batch feature requires pro/enterprise tier"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/minute")
async def batch_start_sessions(request: Request, body: BatchStartRequest):
    """Start multiple verification sessions in batch.

    SECURITY: Batch start lets a single request spin up many sessions, so it is
    gated behind an API key whose tier includes the ``batch`` feature. Anonymous
    or free-tier callers cannot use it (DoS / abuse protection).
    """
    # Require an API key with the "batch" feature (pro/enterprise tier)
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    tier = RateTier.get_tier(api_key)
    features = RateTier.get_limits(tier).get("features", [])
    if "batch" not in features and "all" not in features:
        raise HTTPException(
            status_code=403,
            detail="Batch sessions require a pro or enterprise tier API key",
        )

    # Enforce per-key daily session limits
    allowed, message = RateTier.check_limit(
        api_key, "session", amount=len(body.entity_ids)
    )
    if not allowed:
        raise HTTPException(status_code=429, detail=message)

    results = []
    failed = 0

    for entity_id in body.entity_ids:
        try:
            session_id = f"ses_{secrets.token_hex(12)}"
            session_token = secrets.token_urlsafe(32)
            challenge_list = generate_challenge_set(body.difficulty)

            session = MettleSession(
                session_id=session_id,
                entity_id=entity_id,
                difficulty=body.difficulty,
                challenges=challenge_list,
                access_token_hash=hashlib.sha256(session_token.encode()).hexdigest(),
            )
            if len(sessions) >= MAX_SESSIONS or len(challenges) >= MAX_CHALLENGES:
                raise RuntimeError("Verification capacity reached; retry shortly")
            sessions[session_id] = session

            first_challenge = challenge_list[0]
            challenges[first_challenge.id] = (first_challenge, time.time())

            if not _persist_new_legacy_session(session):
                sessions.pop(session_id, None)
                challenges.pop(first_challenge.id, None)
                raise RuntimeError("Session persistence is temporarily unavailable")

            results.append(
                {
                    "entity_id": entity_id,
                    "session_id": session_id,
                    "session_token": session_token,
                    "challenge_id": first_challenge.id,
                    "total_challenges": len(challenge_list),
                }
            )
        except Exception as e:
            logger.warning("batch_start_failed", entity_id=entity_id, error=str(e))
            failed += 1
            results.append(
                {
                    "entity_id": entity_id,
                    "error": str(e),
                }
            )

    logger.info(
        "batch_sessions_started",
        total=len(body.entity_ids),
        success=len(body.entity_ids) - failed,
        failed=failed,
    )

    return BatchStartResponse(
        sessions=results,
        total=len(body.entity_ids),
        failed=failed,
    )


@api_router.post(
    "/session/answer",
    response_model=SubmitAnswerResponse,
    tags=["Session"],
    summary="Submit Answer",
    description="Submit an answer to the current challenge.",
    responses={
        200: {"description": "Answer processed"},
        400: {"description": "Session already completed"},
        404: {"description": "Session or challenge not found"},
        422: {"description": "Invalid request parameters"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit(settings.rate_limit_answers)
async def submit_answer(request: Request, body: SubmitAnswerRequest):
    """Submit an answer to the current challenge."""
    # Get session - use generic error to prevent session enumeration
    session = sessions.get(body.session_id)
    if not session or session.completed:
        # SECURITY: Don't distinguish between "not found" and "completed"
        # to prevent session ID enumeration via timing/error analysis
        logger.warning("session_invalid", session_id=body.session_id)
        raise HTTPException(status_code=404, detail="Session not found or invalid")

    _require_session_access(request, session)
    _arm_recovered_challenge(session)

    # Bind the submitted challenge to this session and its current position
    # before touching the global one-time challenge store. Without this check,
    # a valid challenge ID from another session could be consumed and credited
    # to the wrong session.
    current_index = len(session.results)
    if (
        current_index >= len(session.challenges)
        or session.challenges[current_index].id != body.challenge_id
    ):
        logger.warning(
            "challenge_session_mismatch",
            session_id=body.session_id,
            challenge_id=body.challenge_id,
        )
        raise HTTPException(
            status_code=404, detail="Challenge not found or already answered"
        )

    # Get and remove challenge atomically to prevent race conditions
    # SECURITY: Using pop() instead of get()+del prevents double-submission attacks
    challenge_data = challenges.pop(body.challenge_id, None)
    if not challenge_data:
        logger.warning(
            "challenge_not_found",
            session_id=body.session_id,
            challenge_id=body.challenge_id,
        )
        raise HTTPException(
            status_code=404, detail="Challenge not found or already answered"
        )

    challenge, issued_at = challenge_data

    # Calculate response time
    if issued_at is None:
        # Defensive fallback for a recovered challenge submitted directly.
        issued_at = time.time()
    response_time_ms = int((time.time() - issued_at) * 1000)

    # Verify response
    result = verify_response(challenge, body.answer, response_time_ms)
    previous_badge_info = session.badge_info
    session.results.append(result)

    # Log result
    logger.info(
        "challenge_answered",
        session_id=body.session_id,
        challenge_id=body.challenge_id,
        challenge_type=challenge.type.value,
        passed=result.passed,
        response_time_ms=response_time_ms,
    )

    # Challenge already removed via atomic pop() above

    # Determine next challenge or complete session
    current_index = len(session.results)
    challenges_remaining = len(session.challenges) - current_index
    final_result: MettleResult | None = None

    if challenges_remaining > 0:
        next_challenge = session.challenges[current_index]
        # The submitted challenge was atomically removed above, so replacing it
        # with this session's next challenge cannot increase global occupancy.
        # Direct assignment avoids evicting another caller's active challenge.
        challenges[next_challenge.id] = (next_challenge, time.time())
        session_complete = False
    else:
        next_challenge = None
        session.completed = True
        session_complete = True

        # Log session completion
        final_result = compute_mettle_result(session.results, session.entity_id)
        _attach_stable_session_badge(session, final_result)

    if not _persist_legacy_progress(session):
        # Restore the exact one-use challenge transition so the caller can retry
        # after a transient persistence outage without losing the session.
        session.results.pop()
        session.completed = False
        session.badge_info = previous_badge_info
        if next_challenge is not None:
            challenges.pop(next_challenge.id, None)
        challenges[challenge.id] = (challenge, issued_at)
        raise HTTPException(
            status_code=503,
            detail="Session persistence is temporarily unavailable",
        )

    if final_result is not None:
        # Record for collusion detection only after durable session state exists.
        ip_address = get_remote_address(request)
        CollusionDetector.record_verification(
            entity_id=session.entity_id,
            ip_address=ip_address,
            passed=final_result.verified,
        )

        logger.info(
            "session_completed",
            session_id=body.session_id,
            entity_id=session.entity_id,
            verified=final_result.verified,
            pass_rate=final_result.pass_rate,
        )

        # Public legacy sessions accept a self-asserted entity_id and therefore
        # have no authority to emit events for an entity-owned webhook.

    return SubmitAnswerResponse(
        result=result,
        next_challenge=next_challenge.sanitized() if next_challenge else None,
        session_complete=session_complete,
        challenges_remaining=challenges_remaining,
    )


@api_router.get(
    "/session/{session_id}",
    tags=["Session"],
    summary="Get Session Status",
    description="Get the current status of a verification session.",
    responses={
        200: {"description": "Session status returned"},
        404: {"description": "Session not found"},
    },
)
async def get_session(request: Request, session_id: str):
    """Get session status and results."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _require_session_access(request, session)
    _arm_recovered_challenge(session)

    if session.completed:
        result = compute_mettle_result(session.results, session.entity_id)
        previous_badge_info = session.badge_info
        _attach_stable_session_badge(session, result)
        if (
            session.badge_info is not previous_badge_info
            and not _persist_legacy_progress(session)
        ):
            session.badge_info = previous_badge_info
            raise HTTPException(
                status_code=503,
                detail="Session persistence is temporarily unavailable",
            )
        return {
            "session_id": session_id,
            "status": "completed",
            "result": result,
        }
    else:
        return {
            "session_id": session_id,
            "status": "in_progress",
            "completed_challenges": len(session.results),
            "total_challenges": len(session.challenges),
            "results_so_far": session.results,
        }


@api_router.get(
    "/session/{session_id}/result",
    response_model=MettleResult,
    tags=["Session"],
    summary="Get Final Result",
    description="Get the final verification result for a completed session.",
    responses={
        200: {"description": "Final result returned"},
        400: {"description": "Session not yet completed"},
        404: {"description": "Session not found"},
    },
)
async def get_result(request: Request, session_id: str):
    """Get final METTLE result for a completed session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _require_session_access(request, session)

    if not session.completed:
        raise HTTPException(status_code=400, detail="Session not yet completed")

    result = compute_mettle_result(session.results, session.entity_id)

    previous_badge_info = session.badge_info
    _attach_stable_session_badge(session, result)
    if session.badge_info is not previous_badge_info and not _persist_legacy_progress(
        session
    ):
        session.badge_info = previous_badge_info
        raise HTTPException(
            status_code=503,
            detail="Session persistence is temporarily unavailable",
        )

    return result


# === Badge Issuance And Verification ===
def generate_signed_badge(
    entity_id: str | None,
    difficulty: str,
    pass_rate: float,
    session_id: str,
) -> dict[str, Any]:
    """Issue a signed, time-limited reverse-CAPTCHA pass credential."""
    if not settings.secret_key:
        raise ValueError("Cannot issue badge: METTLE_SECRET_KEY is not configured")

    now = datetime.now(timezone.utc)
    expires_at = now.timestamp() + settings.badge_expiry_seconds
    tier = "silver" if difficulty == Difficulty.FULL.value else "bronze"
    freshness_nonce = secrets.token_hex(16)
    jti = secrets.token_hex(16)
    payload = {
        "credential_type": "mettle-reverse-captcha-pass",
        "entity_id": entity_id,
        "entity_id_verified": False,
        "identity_binding": "self_asserted",
        "attests": "mettle_session_passed",
        "verified": True,
        "tier": tier,
        "difficulty": difficulty,
        "pass_rate": round(pass_rate, 4),
        "session_id": session_id,
        "iss": "mettle-api",
        "iat": now.timestamp(),
        "exp": expires_at,
        "jti": jti,
        "nonce": freshness_nonce,
        "version": settings.api_version,
    }
    token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return {
        "token": token,
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "freshness_nonce": freshness_nonce,
        "signed": True,
        "jti": jti,
        "tier": tier,
    }


def _verify_badge_token(token: str) -> BadgeVerifyResponse:
    """Verify a METTLE badge token against issuer policy and revocation state."""
    if not settings.secret_key:
        # SECURITY: Reject ALL badges when signing not configured
        # Never accept simple tokens - they can be trivially forged
        return BadgeVerifyResponse(
            valid=False,
            error="Badge verification not configured (no signing key)",
        )

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"],
            issuer="mettle-api",
            options={"require": ["exp", "iat", "iss", "jti", "session_id"]},
        )

        # Revocation is process-local when persistence is disabled and durable
        # when the database layer is enabled. Database errors fail closed.
        jti = payload.get("jti")
        if jti:
            try:
                is_revoked = jti in revoked_badges or bool(
                    db and db.is_badge_revoked(jti, raise_on_error=True)
                )
            except RuntimeError:
                return BadgeVerifyResponse(
                    valid=False,
                    error="Badge revocation status is temporarily unavailable",
                )
            if is_revoked:
                return BadgeVerifyResponse(
                    valid=False,
                    error="Badge has been revoked",
                    revoked=True,
                )

        # Extract expiry info
        exp = payload.get("exp")
        expires_at = None
        if exp:
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()

        return BadgeVerifyResponse(
            valid=True,
            payload=payload,
            expires_at=expires_at,
        )
    except jwt.ExpiredSignatureError:
        return BadgeVerifyResponse(valid=False, error="Badge has expired")
    except jwt.InvalidTokenError:
        return BadgeVerifyResponse(valid=False, error="Invalid badge token")


@api_router.post(
    "/badge/verify",
    response_model=BadgeVerifyResponse,
    tags=["Badge"],
    summary="Verify Badge",
    description=(
        "Verify a METTLE badge without placing the credential in the request URL."
    ),
    responses={200: {"description": "Badge verification result"}},
)
@limiter.limit("100/minute")
async def verify_badge(request: Request, body: BadgeVerifyRequest):
    """Verify a METTLE badge supplied in the request body."""
    return _verify_badge_token(body.token)


@api_router.get(
    "/badge/verify/{token}",
    response_model=BadgeVerifyResponse,
    tags=["Badge"],
    summary="Verify Badge (Legacy URL Form)",
    description=(
        "Deprecated compatibility endpoint. Tokens in URLs may be retained in logs "
        "or browser history; use POST /api/badge/verify instead."
    ),
    deprecated=True,
    responses={200: {"description": "Badge verification result"}},
)
@limiter.limit("100/minute")
async def verify_badge_legacy(request: Request, token: str):
    """Verify a METTLE badge through the deprecated URL form."""
    return _verify_badge_token(token)


# === Revocation Endpoints ===


class RevokeBadgeRequest(BaseModel):
    """Request to revoke a legacy badge or Presence credential."""

    token: str | None = Field(None, description="The legacy badge token to revoke")
    jti: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{32}$",
        description="The Presence credential JTI to revoke",
    )
    entity_id: str | None = Field(
        None,
        max_length=256,
        description="Optional entity claim recorded with a JTI revocation",
    )
    reason: str = Field(
        ..., min_length=10, max_length=500, description="Reason for revocation"
    )
    evidence: dict[str, Any] | None = Field(
        None, description="Optional evidence supporting revocation"
    )


class RevokeBadgeResponse(BaseModel):
    """Response after revoking a badge."""

    revoked: bool = Field(description="Whether the badge was revoked")
    jti: str | None = Field(None, description="The badge ID that was revoked")
    message: str = Field(description="Status message")


@api_router.post(
    "/badge/revoke",
    response_model=RevokeBadgeResponse,
    tags=["Badge"],
    summary="Revoke Badge",
    description="Revoke a METTLE badge. Revoked badges will fail verification.",
    responses={
        200: {"description": "Badge revoked successfully"},
        400: {"description": "Invalid token or already revoked"},
        401: {"description": "Unauthorized - requires API key"},
    },
)
@limiter.limit("10/minute")
async def revoke_badge(request: Request, body: RevokeBadgeRequest):
    """Revoke a METTLE badge.

    Once revoked, the badge will fail all future verification attempts.
    Revocations are logged with an audit trail.

    Requires admin API key for authorization.
    """
    # SECURITY: Require admin authentication to prevent malicious revocation
    admin_key = request.headers.get("X-Admin-Key")
    ip_address = get_remote_address(request)

    # Check rate limiting for brute force protection
    allowed, retry_after = check_admin_auth_rate_limit(ip_address)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed auth attempts. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503, detail="Revocation service not configured (no admin key)"
        )
    if not verify_admin_key(admin_key, ip_address):
        raise HTTPException(
            status_code=401, detail="Admin authorization required for badge revocation"
        )

    if bool(body.token) == bool(body.jti):
        raise HTTPException(
            status_code=400,
            detail="Supply exactly one legacy badge token or Presence credential JTI",
        )

    if body.jti:
        payload = {"jti": body.jti, "entity_id": body.entity_id}
    else:
        if not settings.secret_key:
            raise HTTPException(status_code=400, detail="Badge signing not configured")
        token = body.token
        if token is None:
            raise HTTPException(status_code=400, detail="Badge token is required")
        try:
            # Decode without verification to get JTI even if expired
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=["HS256"],
                issuer="mettle-api",
                options={
                    "verify_exp": False,
                    "require": ["iat", "iss", "session_id"],
                },
            )
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=400, detail="Invalid badge token")

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(status_code=400, detail="Badge has no revocable ID (jti)")

    try:
        already_revoked = jti in revoked_badges or bool(
            db and db.is_badge_revoked(jti, raise_on_error=True)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=503, detail="Revocation service temporarily unavailable"
        ) from e

    if already_revoked:
        return RevokeBadgeResponse(
            revoked=False,
            jti=jti,
            message="Badge already revoked",
        )

    if db and not db.add_revoked_badge(
        jti,
        payload.get("entity_id"),
        body.reason,
        body.evidence,
    ):
        raise HTTPException(
            status_code=503, detail="Failed to persist badge revocation"
        )

    # Add to revocation dict with memory limit after durable storage succeeds.
    add_with_limit(revoked_badges, jti, time.time(), MAX_REVOKED_BADGES)

    # Create audit record with memory limit
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jti": jti,
        "entity_id": payload.get("entity_id"),
        "reason": body.reason,
        "evidence": body.evidence,
        "badge_issued_at": payload.get("verified_at"),
        "badge_difficulty": payload.get("difficulty"),
    }
    revocation_audit.append(audit_record)
    # Keep audit bounded
    if len(revocation_audit) > MAX_REVOCATION_AUDIT:
        revocation_audit.pop(0)

    logger.info(
        "badge_revoked",
        jti=jti,
        entity_id=payload.get("entity_id"),
        reason=body.reason,
    )

    return RevokeBadgeResponse(
        revoked=True,
        jti=jti,
        message=f"Badge {jti[:8]}... has been revoked",
    )


@api_router.get(
    "/badge/revocations",
    tags=["Badge"],
    summary="List Revocations",
    description="List all badge revocations (audit trail). Requires admin key.",
)
async def list_revocations(request: Request):
    """Get the audit trail of badge revocations. Requires admin authorization."""
    # SECURITY: Revocation audit contains sensitive info
    admin_key = request.headers.get("X-Admin-Key")
    ip_address = get_remote_address(request)

    # Check rate limiting
    allowed, retry_after = check_admin_auth_rate_limit(ip_address)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed auth attempts. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    if not verify_admin_key(admin_key, ip_address):
        raise HTTPException(status_code=401, detail="Admin authorization required")

    if db:
        try:
            return {
                "revoked_count": db.count_revoked_badges(raise_on_error=True),
                "audit": db.get_revoked_badges(100, raise_on_error=True),
            }
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail="Revocation audit is temporarily unavailable",
            ) from exc

    return {
        "revoked_count": len(revoked_badges),
        "audit": revocation_audit[-100:],  # Last 100 revocations
    }


# === Model Fingerprinting ===


class FingerprintSignature(TypedDict):
    patterns: list[str]
    avg_response_length: tuple[int, int]
    formatting_style: str


class ModelFingerprinter:
    """Identify model family through behavioral signatures."""

    # Known model family signatures
    SIGNATURES: dict[str, FingerprintSignature] = {
        "claude": {
            "patterns": ["I'd be happy to", "I cannot", "I should note"],
            "avg_response_length": (50, 200),
            "formatting_style": "structured",
        },
        "gpt": {
            "patterns": ["Sure!", "Certainly!", "I can help"],
            "avg_response_length": (30, 150),
            "formatting_style": "conversational",
        },
        "gemini": {
            "patterns": ["Here's", "Let me", "I'll"],
            "avg_response_length": (40, 180),
            "formatting_style": "mixed",
        },
        "llama": {
            "patterns": ["<s>", "[INST]", "###"],
            "avg_response_length": (20, 100),
            "formatting_style": "raw",
        },
    }

    @staticmethod
    def fingerprint(responses: list[str]) -> dict[str, Any]:
        """Analyze responses and return model family confidence scores."""
        if not responses:
            return {"error": "No responses to analyze", "scores": {}}

        scores: dict[str, float] = {
            family: 0.0 for family in ModelFingerprinter.SIGNATURES
        }

        # Concatenate responses for analysis
        combined = " ".join(responses).lower()
        total_len = len(combined)

        for family, sig in ModelFingerprinter.SIGNATURES.items():
            # Check for characteristic patterns
            pattern_matches = sum(1 for p in sig["patterns"] if p.lower() in combined)
            scores[family] += pattern_matches * 0.15

            # Check response length distribution
            avg_len = total_len / len(responses) if responses else 0
            min_len, max_len = sig["avg_response_length"]
            if min_len <= avg_len <= max_len:
                scores[family] += 0.2

        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}
        else:
            # Equal distribution if no signals
            scores = {k: round(1.0 / len(scores), 3) for k in scores}

        # Determine most likely family
        best_match = max(scores, key=lambda k: scores[k])
        confidence = scores[best_match]

        return {
            "scores": scores,
            "best_match": best_match,
            "confidence": confidence,
            "responses_analyzed": len(responses),
        }


# === Collusion Detection Endpoints ===


@api_router.get(
    "/security/collusion",
    tags=["Status"],
    summary="Collusion Detection Stats",
    description="Get collusion detection statistics and patterns. Requires admin key.",
)
async def get_collusion_stats(request: Request):
    """Get collusion detection statistics. Requires admin authorization.

    SECURITY: Thresholds are security-sensitive - exposing them helps attackers evade.
    """
    admin_key = request.headers.get("X-Admin-Key")
    ip_address = get_remote_address(request)
    if not verify_admin_key(admin_key, ip_address):
        # Return only non-sensitive stats for unauthenticated requests
        # Note: No rate limiting here since we don't fail on bad auth
        return {
            "stats": {"active_entities": len(verification_graph)},
            "message": "Full stats require admin authorization",
        }

    return {
        "stats": CollusionDetector.get_stats(),
        "thresholds": {
            "clique_threshold": CollusionDetector.CLIQUE_THRESHOLD,
            "time_window_seconds": CollusionDetector.TIME_WINDOW_SECONDS,
            "sync_threshold": CollusionDetector.SYNC_THRESHOLD,
        },
    }


@api_router.post(
    "/security/collusion/check",
    tags=["Status"],
    summary="Check Entity Collusion",
    description="Check collusion indicators for a specific entity. Requires admin key.",
    responses={
        200: {"description": "Collusion indicators for the entity"},
        401: {"description": "Unauthorized - requires admin key"},
        429: {"description": "Too many failed auth attempts"},
    },
)
async def check_entity_collusion(request: Request, entity_id: str):
    """Check collusion indicators for an entity. Requires admin authorization.

    SECURITY: Collusion risk scores and warnings are security-sensitive -
    exposing them lets attackers probe and tune evasion of the detector.
    """
    admin_key = request.headers.get("X-Admin-Key")
    ip_address = get_remote_address(request)

    # Check rate limiting for brute force protection
    allowed, retry_after = check_admin_auth_rate_limit(ip_address)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed auth attempts. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    if not verify_admin_key(admin_key, ip_address):
        raise HTTPException(status_code=401, detail="Admin authorization required")

    return CollusionDetector.check_collusion(entity_id, ip_address)


class FingerprintRequest(BaseModel):
    """Request for model fingerprinting."""

    responses: list[
        Annotated[str, Field(max_length=MAX_FINGERPRINT_RESPONSE_CHARS)]
    ] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of responses from the agent to analyze",
    )


@api_router.post(
    "/security/fingerprint",
    tags=["Status"],
    summary="Model Fingerprinting",
    description="Analyze responses to identify model family.",
)
async def fingerprint_model(body: FingerprintRequest):
    """Analyze agent responses to estimate model family."""
    return ModelFingerprinter.fingerprint(body.responses)


# === Webhook System ===

# Registered webhooks: entity_id -> webhook config
webhooks: dict[str, dict[str, Any]] = {}


class WebhookManager:
    """Manage webhook registrations and delivery."""

    EVENTS = ["session.started", "session.completed", "badge.issued", "badge.revoked"]

    @staticmethod
    async def send_webhook(entity_id: str, event: str, payload: dict[str, Any]) -> bool:
        """Send a webhook notification. Returns True if successful."""
        if not entity_id or entity_id not in webhooks:
            return False

        config = webhooks[entity_id]
        url = config.get("url")
        if not url:
            return False

        # Check if this event type is subscribed
        subscribed_events = config.get("events", WebhookManager.EVENTS)
        if event not in subscribed_events:
            return False

        webhook_payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entity_id": entity_id,
            "data": payload,
        }

        # Sign the payload if secret is configured
        secret = config.get("secret")
        if secret:
            import hashlib
            import hmac
            import json

            signature = hmac.new(
                secret.encode(),
                json.dumps(webhook_payload, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            webhook_payload["signature"] = signature

        try:
            import ipaddress
            import socket
            from urllib.parse import urlparse, urlunparse

            import httpx

            # Validate every resolved address immediately before delivery, then
            # connect to one of those exact IPs. Re-resolving the hostname inside
            # the HTTP client would leave a DNS-rebinding TOCTOU gap.
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return False
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                resolved: set[str] = set()
                for item in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM):
                    raw_address = item[4][0]
                    if not isinstance(raw_address, str):
                        raise ValueError("DNS resolver returned a non-string address")
                    resolved.add(raw_address.split("%", 1)[0])
                if not resolved:
                    return False
                resolved_ips = {
                    address: ipaddress.ip_address(address) for address in resolved
                }
                blocked = [
                    address for address, ip in resolved_ips.items() if not ip.is_global
                ]
                if blocked:
                    logger.warning(
                        "webhook_blocked_dns_rebind",
                        entity_id=entity_id,
                        url=url[:50],
                        resolved_ips=sorted(blocked),
                    )
                    return False
            except (socket.gaierror, ValueError) as e:
                logger.warning(
                    "webhook_dns_validation_failed",
                    entity_id=entity_id,
                    url=url[:50],
                    error=type(e).__name__,
                )
                return False

            resolved_ip = sorted(resolved)[0]
            ip_for_url = (
                f"[{resolved_ip}]"
                if resolved_ips[resolved_ip].version == 6
                else resolved_ip
            )
            include_port = parsed.port is not None
            pinned_netloc = f"{ip_for_url}:{port}" if include_port else ip_for_url
            pinned_url = urlunparse(
                (
                    parsed.scheme,
                    pinned_netloc,
                    parsed.path or "/",
                    parsed.params,
                    parsed.query,
                    "",
                )
            )
            host_for_header = f"[{hostname}]" if ":" in hostname else hostname
            host_header = (
                f"{host_for_header}:{port}" if include_port else host_for_header
            )

            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                # Stream and discard the body. A webhook recipient controls the
                # response and must not be able to make the service buffer an
                # unbounded body merely to inspect its status code.
                async with client.stream(
                    "POST",
                    pinned_url,
                    json=webhook_payload,
                    headers={"Host": host_header},
                    extensions={"sni_hostname": hostname},
                ) as response:
                    success = 200 <= response.status_code < 300

                    logger.info(
                        "webhook_sent",
                        entity_id=entity_id,
                        webhook_event=event,
                        url=url[:50],
                        status=response.status_code,
                        success=success,
                    )
                    return success
        except Exception as e:
            logger.warning(
                "webhook_failed", entity_id=entity_id, webhook_event=event, error=str(e)
            )
            return False

    @staticmethod
    def register(
        entity_id: str,
        url: str,
        events: list[str] | None = None,
        secret: str | None = None,
    ) -> dict:
        """Register a webhook for an entity."""
        events_list = events or WebhookManager.EVENTS
        config = {
            "url": url,
            "events": events_list,
            "secret": secret,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        add_with_limit(webhooks, entity_id, config, MAX_WEBHOOKS)
        # Persist to database if enabled
        if db and not db.save_webhook(entity_id, url, events_list, secret):
            webhooks.pop(entity_id, None)
            raise RuntimeError("Webhook persistence unavailable")
        return config

    @staticmethod
    def unregister(entity_id: str) -> bool:
        """Unregister a webhook."""
        if entity_id in webhooks:
            del webhooks[entity_id]
            # Remove from database if enabled
            if db:
                db.delete_webhook(entity_id)
            return True
        # Try database even if not in memory
        if db and db.delete_webhook(entity_id):
            return True
        return False


class WebhookRegisterRequest(BaseModel):
    """Request to register a webhook."""

    entity_id: str = Field(
        ..., description="Entity ID to register webhook for", max_length=128
    )
    url: str = Field(..., description="Webhook URL to POST events to", max_length=2048)
    events: list[str] | None = Field(
        None,
        min_length=1,
        max_length=4,
        description="Events to subscribe to (default: all)",
    )
    secret: str | None = Field(
        None, description="Secret for HMAC signing (min 32 chars)"
    )

    @field_validator("secret")
    @classmethod
    def validate_secret(_cls, v: str | None) -> str | None:
        """SECURITY: Ensure webhook secrets have minimum entropy."""
        if v is not None and len(v) < 32:
            raise ValueError(
                "Webhook secret must be at least 32 characters for security"
            )
        return v

    @field_validator("url")
    @classmethod
    def validate_webhook_url(_cls, v: str) -> str:
        """SECURITY: Block SSRF attacks via webhook URLs.

        Prevents requests to:
        - Cloud metadata endpoints (169.254.169.254)
        - Localhost/loopback
        - Private network ranges (10.x, 172.16-31.x, 192.168.x)
        """
        import ipaddress
        from urllib.parse import urlparse

        parsed = urlparse(v)

        # Must be HTTP or HTTPS
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Webhook URL must use HTTP or HTTPS")
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(
                "Webhook URL must have a hostname and cannot include credentials"
            )
        allowed_ports = {80, 443} if not settings.is_production else {443}
        if parsed.port is not None and parsed.port not in allowed_ports:
            raise ValueError("Webhook URL uses a disallowed port")

        # SECURITY: Require HTTPS in production to prevent credential leakage
        if settings.is_production and parsed.scheme != "https":
            raise ValueError("Webhook URL must use HTTPS in production")

        # Block common SSRF targets
        host = parsed.hostname or ""
        host_lower = host.lower()

        # Block localhost
        if host_lower in _BLOCKED_LOCALHOST_HOSTS:
            raise ValueError("Webhook URL cannot target localhost")

        # Block cloud metadata endpoints
        if host_lower in ("169.254.169.254", "metadata.google.internal"):
            raise ValueError("Webhook URL cannot target cloud metadata endpoints")

        # Block private IP ranges
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # Not an IP address — that's fine, check for suspicious hostnames below
            ip = None

        if ip is not None and not ip.is_global:
            raise ValueError("Webhook URL cannot target private/internal IPs")

        # Block internal network patterns
        internal_patterns = ["internal", ".local", ".localdomain", ".corp", ".lan"]
        if any(pattern in host_lower for pattern in internal_patterns):
            raise ValueError("Webhook URL cannot target internal hostnames")

        return v


@api_router.post(
    "/webhooks/register",
    tags=["Status"],
    summary="Register Webhook",
    description="Register a webhook URL for verification events. Requires an API key that owns the entity.",
    responses={
        200: {"description": "Webhook registered"},
        400: {"description": "Invalid events"},
        401: {"description": "Unauthorized - requires API key"},
        403: {"description": "Forbidden - API key does not own this entity"},
    },
)
async def register_webhook(body: WebhookRegisterRequest, request: Request):
    """Register a webhook for an entity.

    SECURITY: A webhook discloses session/badge event metadata for an entity and
    causes outbound requests on its behalf, so registration must be authenticated
    and restricted to the entity's owner. The caller must present an X-API-Key
    whose registered entity_id matches the requested entity_id (IDOR protection).
    """
    ip_address = get_remote_address(request)

    # Require an API key that owns the target entity (prevents anonymous IDOR overwrite)
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    key_data = RateTier.get_key_data(api_key)
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    owned_entity = key_data.get("entity_id")
    if not owned_entity or owned_entity != body.entity_id:
        raise HTTPException(
            status_code=403,
            detail="API key is not authorized to register webhooks for this entity",
        )

    # SECURITY: Audit all webhook registrations
    logger.info(
        "webhook_registered",
        entity_id=body.entity_id,
        url=body.url[:50] + "..." if len(body.url) > 50 else body.url,
        events=body.events,
        ip_address=ip_address,
    )

    # Validate events
    if body.events:
        invalid = [e for e in body.events if e not in WebhookManager.EVENTS]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid events: {invalid}")

    try:
        config = WebhookManager.register(
            body.entity_id, body.url, body.events, body.secret
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Webhook persistence is temporarily unavailable"
        ) from exc
    return {
        "registered": True,
        "entity_id": body.entity_id,
        "events": config["events"],
    }


@api_router.delete(
    "/webhooks/{entity_id}",
    tags=["Status"],
    summary="Unregister Webhook",
    description="Remove a webhook registration. Requires admin key.",
    responses={
        200: {"description": "Webhook unregistered"},
        401: {"description": "Unauthorized - requires admin key"},
        404: {"description": "Webhook not found"},
        429: {"description": "Too many failed auth attempts"},
    },
)
async def unregister_webhook(entity_id: str, request: Request):
    """Unregister a webhook. Requires admin authorization.

    SECURITY: Deleting another entity's webhook is a denial-of-service against
    that entity's event delivery, so removal requires an admin API key.
    """
    admin_key = request.headers.get("X-Admin-Key")
    ip_address = get_remote_address(request)

    # Check rate limiting for brute force protection
    allowed, retry_after = check_admin_auth_rate_limit(ip_address)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed auth attempts. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    if not verify_admin_key(admin_key, ip_address):
        raise HTTPException(status_code=401, detail="Admin authorization required")

    if WebhookManager.unregister(entity_id):
        # SECURITY: Audit all webhook deletions
        logger.info(
            "webhook_unregistered",
            entity_id=entity_id,
            ip_address=ip_address,
        )
        return {"unregistered": True, "entity_id": entity_id}
    raise HTTPException(status_code=404, detail="Webhook not found")


@api_router.get(
    "/webhooks/events",
    tags=["Status"],
    summary="List Webhook Events",
    description="List available webhook event types.",
)
async def list_webhook_events():
    """List available webhook events."""
    return {
        "events": WebhookManager.EVENTS,
        "registered_count": len(webhooks),
    }


# === API Key Management ===


class RegisterKeyRequest(BaseModel):
    """Request to register an API key."""

    tier: str = Field(..., description="Tier: free, pro, or enterprise")
    entity_id: str | None = Field(None, description="Associated entity ID")


@api_router.post(
    "/keys/register",
    tags=["Status"],
    summary="Register API Key",
    description="Register a new API key with a specific tier (admin only).",
)
async def register_api_key(
    request: Request,
    body: RegisterKeyRequest,
):
    """Register a new API key. Requires admin key."""
    # Check admin authorization
    admin_key = request.headers.get("X-Admin-Key")
    ip_address = get_remote_address(request)

    # Check rate limiting for brute force protection
    allowed, retry_after = check_admin_auth_rate_limit(ip_address)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed auth attempts. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    if not verify_admin_key(admin_key, ip_address):
        raise HTTPException(status_code=401, detail="Admin key required")

    # Generate new API key
    new_key = f"mtl_{secrets.token_hex(16)}"

    try:
        key_data = RateTier.register_key(new_key, body.tier, body.entity_id)
        logger.info("api_key_registered", tier=body.tier, entity_id=body.entity_id)
        return {
            "api_key": new_key,
            "tier": body.tier,
            "limits": RateTier.get_limits(body.tier),
            **key_data,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@api_router.get(
    "/keys/tiers",
    tags=["Status"],
    summary="List Rate Tiers",
    description="Get available rate limiting tiers and their limits.",
)
async def list_tiers():
    """List available rate limiting tiers."""
    return {
        "tiers": RateTier.TIERS,
        "registered_keys": len(api_keys),
    }


# === Static Files (Web UI) ===
# Mount static files using absolute path for Render compatibility
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# === Mount API Router ===
app.include_router(api_router)

# === Mount METTLE Router (12-suite sessions, VCP attestation, Ed25519 signing) ===
from mettle.router import router as mettle_router  # noqa: E402 — intentional late import; router depends on app being fully constructed

app.include_router(mettle_router)


# === Root serves UI ===
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def serve_ui():
    """Serve the web UI at root."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "index.html"))
    # Fallback to API redirect if no static files
    return RedirectResponse(url="/api")


# Legacy /ui redirect for backwards compatibility
@app.get("/ui", include_in_schema=False)
async def redirect_legacy_ui():
    """Redirect legacy /ui to root."""
    return RedirectResponse(url="/", status_code=301)


# === Static Page Routes ===
@app.get("/about", include_in_schema=False)
@app.head("/about", include_in_schema=False)
async def serve_about():
    """Serve the about page."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "about.html"))
    return RedirectResponse(url="/")


@app.get("/test", include_in_schema=False)
@app.head("/test", include_in_schema=False)
async def serve_test():
    """Serve the test verification page."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "test.html"))
    return RedirectResponse(url="/")


# === SEO Endpoints ===
@app.get("/sitemap.xml", include_in_schema=False)
@app.head("/sitemap.xml", include_in_schema=False)
async def sitemap():
    """Generate sitemap for search engines.

    Note: /docs is FastAPI's Swagger UI (JavaScript-rendered, not indexable).
    Only /static/docs.html (custom docs) is included for SEO.
    """
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://mettle.sh/</loc>
    <lastmod>2026-07-13</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://mettle.sh/static/docs.html</loc>
    <lastmod>2026-07-13</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://mettle.sh/test</loc>
    <lastmod>2026-07-13</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://mettle.sh/about</loc>
    <lastmod>2026-07-13</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>
</urlset>"""
    return Response(content=xml, media_type="application/xml")


@app.get("/robots.txt", include_in_schema=False)
@app.head("/robots.txt", include_in_schema=False)
async def robots():
    """Serve robots.txt for search engine crawlers."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "robots.txt"), media_type="text/plain")
    return Response(
        content="User-agent: *\nAllow: /\nSitemap: https://mettle.sh/sitemap.xml",
        media_type="text/plain",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_BIND_ALL_INTERFACES, port=8000)
