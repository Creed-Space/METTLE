"""
METTLE API: Machine Entity Trustbuilding through Turing-inverse Logic Examination

Prove your metal, with this CAPTCHA to keep humans out of places they shouldn't be.

A reverse-CAPTCHA verification system for AI-only spaces.
"""

import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt
import structlog
from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings

# Configuration
settings = get_settings()

# Database layer (optional)
db = None
if settings.use_database:
    try:
        from urllib.parse import urlparse

        import database as db

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
challenges: dict[str, tuple[Challenge, float]] = {}
revoked_badges: dict[str, float] = {}  # JTI -> revocation timestamp (bounded dict)
revocation_audit: list[dict[str, Any]] = []  # Audit trail

# Collusion detection - track verification patterns
verification_graph: dict[str, list[dict[str, Any]]] = {}  # entity_id -> list of verifications
verification_timestamps: list[tuple[str, float]] = []  # (entity_id, timestamp) for timing analysis

# API Key tiers for rate limiting
api_keys: dict[str, dict[str, Any]] = {}  # api_key -> {tier, entity_id, created_at, usage_today}


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
    def get_tier(api_key: str | None) -> str:
        """Get tier for an API key, default to free."""
        if not api_key:
            return "free"
        key_data = api_keys.get(api_key)
        if not key_data:
            return "free"
        return key_data.get("tier", "free")

    @staticmethod
    def get_limits(tier: str) -> dict[str, Any]:
        """Get rate limits for a tier."""
        return RateTier.TIERS.get(tier, RateTier.TIERS["free"])

    @staticmethod
    def check_limit(api_key: str | None, limit_type: str) -> tuple[bool, str]:
        """Check if request is within rate limits. Returns (allowed, message)."""
        tier = RateTier.get_tier(api_key)
        limits = RateTier.get_limits(tier)

        if limits.get("sessions_per_day") == -1:
            return True, "Enterprise: unlimited"

        # Track usage
        if api_key and api_key in api_keys:
            today = datetime.now(timezone.utc).date().isoformat()
            key_data = api_keys[api_key]

            if key_data.get("usage_date") != today:
                key_data["usage_date"] = today
                key_data["usage_count"] = 0

            if limit_type == "session":
                max_sessions = limits["sessions_per_day"]
                if key_data.get("usage_count", 0) >= max_sessions:
                    return False, f"Daily limit reached ({max_sessions} sessions)"
                key_data["usage_count"] = key_data.get("usage_count", 0) + 1

        return True, f"OK ({tier} tier)"

    @staticmethod
    def register_key(api_key: str, tier: str, entity_id: str | None = None) -> dict[str, Any]:
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
        if db:
            db.save_api_key(api_key, tier, entity_id)
        return key_data


class CollusionDetector:
    """Detect suspicious patterns in verification requests."""

    # Thresholds
    CLIQUE_THRESHOLD = 3  # Min entities to form suspicious clique
    TIME_WINDOW_SECONDS = 60  # Window for synchronized timing detection
    SYNC_THRESHOLD = 5  # Max verifications in window to be suspicious

    @staticmethod
    def record_verification(entity_id: str, ip_address: str, passed: bool) -> None:
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
            warnings.append(f"IP {ip_address[:8]}... verified {len(ip_entities)} different entities")
            risk_score += 0.3

        # Check 2: Synchronized timing (burst of verifications)
        now = time.time()
        recent = [t for _, t in verification_timestamps if now - t < CollusionDetector.TIME_WINDOW_SECONDS]
        if len(recent) >= CollusionDetector.SYNC_THRESHOLD:
            warnings.append(f"{len(recent)} verifications in {CollusionDetector.TIME_WINDOW_SECONDS}s window")
            risk_score += 0.2

        # Check 3: Entity verified too frequently
        if entity_id in verification_graph:
            entity_records = verification_graph[entity_id]
            recent_entity = [r for r in entity_records if now - r["timestamp"] < 3600]  # Last hour
            if len(recent_entity) > 10:
                warnings.append(f"Entity verified {len(recent_entity)} times in last hour")
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
            "unique_ips": len(set(
                r["ip_address"]
                for records in verification_graph.values()
                for r in records[-10:]
            )),
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
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "font-src 'self' https://cdnjs.cloudflare.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
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
async def cleanup_expired_sessions():
    """Background task to remove expired sessions (prevents memory DoS)."""
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        cutoff = time.time() - 1800  # 30 minutes TTL
        expired_sessions = [sid for sid, s in sessions.items() if s.started_at.timestamp() < cutoff]
        expired_challenges = [cid for cid, (_, t) in challenges.items() if t < cutoff]
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

    # Validate production config
    if settings.is_production and not settings.secret_key:
        raise RuntimeError("SECRET_KEY environment variable required in production")

    logger.info(
        "mettle_starting",
        environment=settings.environment,
        version=settings.api_version,
    )
    print("[METTLE] API starting...")
    print("   Machine Entity Trustbuilding through Turing-inverse Logic Examination")
    print("   'Prove your metal.'")

    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())

    yield

    # Shutdown
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
**Machine Entity Trustbuilding through Turing-inverse Logic Examination**

*"Prove your metal."*

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
        "url": "https://github.com/NellInc/mettle",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# API Router - all API endpoints go under /api
api_router = APIRouter(prefix="/api")

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
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
    def validate_entity_id(cls, v: str | None) -> str | None:
        """Sanitize entity_id."""
        if v is not None:
            # Strip whitespace and limit characters
            v = v.strip()[:128]
        return v


class StartSessionResponse(BaseModel):
    """Response with session info and first challenge."""

    session_id: str = Field(description="Unique session identifier")
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
    def validate_answer(cls, v: str) -> str:
        """Sanitize and validate answer."""
        if len(v) > 1024:
            raise ValueError("Answer exceeds maximum length of 1024 characters")
        return v


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an answer."""

    result: VerificationResult = Field(description="Result of this challenge")
    next_challenge: Challenge | None = Field(description="Next challenge, or null if complete")
    session_complete: bool = Field(description="Whether session is complete")
    challenges_remaining: int = Field(description="Number of challenges left")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")


class BadgeVerifyResponse(BaseModel):
    """Response for badge verification."""

    valid: bool = Field(description="Whether the badge is valid")
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Badge payload if valid",
    )
    error: str | None = Field(default=None, description="Error message if invalid")
    expires_at: str | None = Field(default=None, description="When the badge expires (ISO format)")
    revoked: bool = Field(default=False, description="Whether the badge has been revoked")


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
        "full_name": "Machine Entity Trustbuilding through Turing-inverse Logic Examination",
        "tagline": "Prove your metal.",
        "description": "A CAPTCHA to keep humans out of places they shouldn't be.",
        "version": settings.api_version,
        "documentation": "/docs",
        "endpoints": {
            "POST /api/session/start": "Start a verification session",
            "POST /api/session/answer": "Submit an answer to current challenge",
            "GET /api/session/{session_id}": "Get session status",
            "GET /api/session/{session_id}/result": "Get final verification result",
            "GET /api/badge/verify/{token}": "Verify a METTLE badge",
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

    # Check for collusion patterns
    ip_address = get_remote_address(request)
    collusion_check = CollusionDetector.check_collusion(body.entity_id or "", ip_address)

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
    )

    add_with_limit(sessions, session_id, session, MAX_SESSIONS)

    # Store first challenge with timestamp
    first_challenge = challenge_list[0]
    add_with_limit(challenges, first_challenge.id, (first_challenge, time.time()), MAX_CHALLENGES)

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
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/minute")
async def batch_start_sessions(request: Request, body: BatchStartRequest):
    """Start multiple verification sessions in batch."""
    results = []
    failed = 0

    for entity_id in body.entity_ids:
        try:
            session_id = f"ses_{secrets.token_hex(12)}"
            challenge_list = generate_challenge_set(body.difficulty)

            session = MettleSession(
                session_id=session_id,
                entity_id=entity_id,
                difficulty=body.difficulty,
                challenges=challenge_list,
            )
            add_with_limit(sessions, session_id, session, MAX_SESSIONS)

            first_challenge = challenge_list[0]
            add_with_limit(challenges, first_challenge.id, (first_challenge, time.time()), MAX_CHALLENGES)

            results.append({
                "entity_id": entity_id,
                "session_id": session_id,
                "challenge_id": first_challenge.id,
                "total_challenges": len(challenge_list),
            })
        except Exception as e:
            logger.warning("batch_start_failed", entity_id=entity_id, error=str(e))
            failed += 1
            results.append({
                "entity_id": entity_id,
                "error": str(e),
            })

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

    # Get and remove challenge atomically to prevent race conditions
    # SECURITY: Using pop() instead of get()+del prevents double-submission attacks
    challenge_data = challenges.pop(body.challenge_id, None)
    if not challenge_data:
        logger.warning(
            "challenge_not_found",
            session_id=body.session_id,
            challenge_id=body.challenge_id,
        )
        raise HTTPException(status_code=404, detail="Challenge not found or already answered")

    challenge, issued_at = challenge_data

    # Calculate response time
    response_time_ms = int((time.time() - issued_at) * 1000)

    # Verify response
    result = verify_response(challenge, body.answer, response_time_ms)
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

    if challenges_remaining > 0:
        next_challenge = session.challenges[current_index]
        add_with_limit(challenges, next_challenge.id, (next_challenge, time.time()), MAX_CHALLENGES)
        session_complete = False
    else:
        next_challenge = None
        session.completed = True
        session_complete = True

        # Log session completion
        final_result = compute_mettle_result(session.results, session.entity_id)

        # Record for collusion detection
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

        # Send webhooks
        if session.entity_id:
            asyncio.create_task(
                WebhookManager.send_webhook(
                    session.entity_id,
                    "session.completed",
                    {
                        "session_id": body.session_id,
                        "verified": final_result.verified,
                        "pass_rate": final_result.pass_rate,
                    },
                )
            )
            if final_result.verified:
                asyncio.create_task(
                    WebhookManager.send_webhook(
                        session.entity_id,
                        "badge.issued",
                        {"session_id": body.session_id, "pass_rate": final_result.pass_rate},
                    )
                )

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
async def get_session(session_id: str):
    """Get session status and results."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.completed:
        result = compute_mettle_result(session.results, session.entity_id)
        # Generate proper badge with expiry if verified
        if result.verified:
            badge_data = generate_signed_badge(
                entity_id=session.entity_id,
                difficulty=session.difficulty.value,
                pass_rate=result.pass_rate,
                session_id=session_id,
            )
            result.badge = badge_data["token"]
            result.badge_info = BadgeInfo(
                token=badge_data["token"],
                expires_at=datetime.fromisoformat(badge_data["expires_at"]),
                freshness_nonce=badge_data.get("freshness_nonce"),
                signed=badge_data.get("signed", False),
                jti=badge_data.get("jti"),
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
async def get_result(session_id: str):
    """Get final METTLE result for a completed session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.completed:
        raise HTTPException(status_code=400, detail="Session not yet completed")

    result = compute_mettle_result(session.results, session.entity_id)

    # Generate proper signed badge with expiry if verified
    if result.verified:
        badge_data = generate_signed_badge(
            entity_id=session.entity_id,
            difficulty=session.difficulty.value,
            pass_rate=result.pass_rate,
            session_id=session_id,
        )
        result.badge = badge_data["token"]
        result.badge_info = BadgeInfo(
            token=badge_data["token"],
            expires_at=datetime.fromisoformat(badge_data["expires_at"]),
            freshness_nonce=badge_data.get("freshness_nonce"),
            signed=badge_data.get("signed", False),
            jti=badge_data.get("jti"),
        )

    return result


# === Badge Endpoints ===
def generate_signed_badge(
    entity_id: str | None,
    difficulty: str,
    pass_rate: float,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Generate a signed JWT badge for verified entities with expiry.

    Returns dict with:
        - token: The JWT badge token
        - expires_at: ISO timestamp of expiry
        - freshness_nonce: Nonce for freshness verification
    """
    now = datetime.now(timezone.utc)
    expires_at = now.timestamp() + settings.badge_expiry_seconds
    freshness_nonce = secrets.token_hex(8)

    if not settings.secret_key:
        # SECURITY: Never issue unsigned badges - they can be trivially forged
        # In production, SECRET_KEY must be configured
        raise ValueError(
            "Cannot issue badge: SECRET_KEY not configured. "
            "Unsigned badges are forgeable. Configure SECRET_KEY in production."
        )

    payload = {
        "entity_id": entity_id,
        "difficulty": difficulty,
        "pass_rate": pass_rate,
        "verified_at": now.isoformat(),
        "version": settings.api_version,
        "iss": "mettle-api",
        "exp": expires_at,  # JWT standard expiry claim
        "iat": now.timestamp(),  # Issued at
        "jti": secrets.token_hex(16),  # Unique token ID for revocation
        "nonce": freshness_nonce,
        "session_id": session_id,
    }

    token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return {
        "token": token,
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "freshness_nonce": freshness_nonce,
        "signed": True,
        "jti": payload["jti"],
    }


@api_router.get(
    "/badge/verify/{token}",
    response_model=BadgeVerifyResponse,
    tags=["Badge"],
    summary="Verify Badge",
    description="Verify that a METTLE badge is valid and not tampered with.",
    responses={
        200: {"description": "Badge verification result"},
    },
)
@limiter.limit("100/minute")
async def verify_badge(request: Request, token: str):
    """Verify a METTLE badge is valid."""
    if not settings.secret_key:
        # SECURITY: Reject ALL badges when signing not configured
        # Never accept simple tokens - they can be trivially forged
        return BadgeVerifyResponse(
            valid=False,
            error="Badge verification not configured (no signing key)",
        )

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])

        # Check revocation (will be implemented in Task 3)
        jti = payload.get("jti")
        if jti and jti in revoked_badges:
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


# === Revocation Endpoints ===

# Audit trail for revocations
revocation_audit: list[dict[str, Any]] = []


class RevokeBadgeRequest(BaseModel):
    """Request to revoke a badge."""

    token: str = Field(..., description="The badge token to revoke")
    reason: str = Field(..., min_length=10, max_length=500, description="Reason for revocation")
    evidence: dict[str, Any] | None = Field(None, description="Optional evidence supporting revocation")


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
        raise HTTPException(status_code=503, detail="Revocation service not configured (no admin key)")
    if not verify_admin_key(admin_key, ip_address):
        raise HTTPException(status_code=401, detail="Admin authorization required for badge revocation")

    if not settings.secret_key:
        raise HTTPException(status_code=400, detail="Badge signing not configured")

    try:
        # Decode without verification to get JTI even if expired
        payload = jwt.decode(
            body.token,
            settings.secret_key,
            algorithms=["HS256"],
            options={"verify_exp": False},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid badge token")

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(status_code=400, detail="Badge has no revocable ID (jti)")

    if jti in revoked_badges:
        return RevokeBadgeResponse(
            revoked=False,
            jti=jti,
            message="Badge already revoked",
        )

    # Add to revocation dict with memory limit
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

    return {
        "revoked_count": len(revoked_badges),
        "audit": revocation_audit[-100:],  # Last 100 revocations
    }


# === Model Fingerprinting ===


class ModelFingerprinter:
    """Identify model family through behavioral signatures."""

    # Known model family signatures
    SIGNATURES = {
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

        scores: dict[str, float] = {family: 0.0 for family in ModelFingerprinter.SIGNATURES}

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
    description="Check collusion indicators for a specific entity.",
)
async def check_entity_collusion(request: Request, entity_id: str):
    """Check collusion indicators for an entity."""
    ip_address = get_remote_address(request)
    return CollusionDetector.check_collusion(entity_id, ip_address)


class FingerprintRequest(BaseModel):
    """Request for model fingerprinting."""

    responses: list[str] = Field(
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
            from urllib.parse import urlparse

            import httpx

            # SECURITY: Validate resolved IP at request time to prevent DNS rebinding
            # The URL was validated at registration, but DNS could change
            parsed = urlparse(url)
            hostname = parsed.hostname
            if hostname:
                try:
                    # Resolve hostname to IP
                    resolved_ip = socket.gethostbyname(hostname)
                    ip_obj = ipaddress.ip_address(resolved_ip)

                    # Block private/internal IPs
                    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                        logger.warning(
                            "webhook_blocked_dns_rebind",
                            entity_id=entity_id,
                            url=url[:50],
                            resolved_ip=resolved_ip,
                        )
                        return False
                except (socket.gaierror, ValueError):
                    # DNS resolution failed or invalid IP - allow (external hostname)
                    pass

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=webhook_payload)
                success = response.status_code < 400

                logger.info(
                    "webhook_sent",
                    entity_id=entity_id,
                    event=event,
                    url=url[:50],
                    status=response.status_code,
                    success=success,
                )
                return success
        except Exception as e:
            logger.warning("webhook_failed", entity_id=entity_id, event=event, error=str(e))
            return False

    @staticmethod
    def register(entity_id: str, url: str, events: list[str] | None = None, secret: str | None = None) -> dict:
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
        if db:
            db.save_webhook(entity_id, url, events_list, secret)
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

    entity_id: str = Field(..., description="Entity ID to register webhook for", max_length=128)
    url: str = Field(..., description="Webhook URL to POST events to", max_length=2048)
    events: list[str] | None = Field(None, description="Events to subscribe to (default: all)")
    secret: str | None = Field(None, description="Secret for HMAC signing (min 32 chars)")

    @field_validator("secret")
    @classmethod
    def validate_secret(cls, v: str | None) -> str | None:
        """SECURITY: Ensure webhook secrets have minimum entropy."""
        if v is not None and len(v) < 32:
            raise ValueError("Webhook secret must be at least 32 characters for security")
        return v

    @field_validator("url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
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

        # SECURITY: Require HTTPS in production to prevent credential leakage
        if settings.is_production and parsed.scheme != "https":
            raise ValueError("Webhook URL must use HTTPS in production")

        # Block common SSRF targets
        host = parsed.hostname or ""
        host_lower = host.lower()

        # Block localhost
        if host_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            raise ValueError("Webhook URL cannot target localhost")

        # Block cloud metadata endpoints
        if host_lower in ("169.254.169.254", "metadata.google.internal"):
            raise ValueError("Webhook URL cannot target cloud metadata endpoints")

        # Block private IP ranges
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise ValueError("Webhook URL cannot target private/internal IPs")
        except ValueError:
            # Not an IP address, check for suspicious hostnames
            pass

        # Block internal network patterns
        internal_patterns = ["internal", ".local", ".localdomain", ".corp", ".lan"]
        if any(pattern in host_lower for pattern in internal_patterns):
            raise ValueError("Webhook URL cannot target internal hostnames")

        return v


@api_router.post(
    "/webhooks/register",
    tags=["Status"],
    summary="Register Webhook",
    description="Register a webhook URL for verification events.",
)
async def register_webhook(body: WebhookRegisterRequest, request: Request):
    """Register a webhook for an entity."""
    # SECURITY: Audit all webhook registrations
    ip_address = get_remote_address(request)
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

    config = WebhookManager.register(body.entity_id, body.url, body.events, body.secret)
    return {
        "registered": True,
        "entity_id": body.entity_id,
        "events": config["events"],
    }


@api_router.delete(
    "/webhooks/{entity_id}",
    tags=["Status"],
    summary="Unregister Webhook",
    description="Remove a webhook registration.",
)
async def unregister_webhook(entity_id: str, request: Request):
    """Unregister a webhook."""
    ip_address = get_remote_address(request)

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
    x_admin_key: str | None = None,
):
    """Register a new API key. Requires admin key."""
    # Check admin authorization
    admin_key = x_admin_key or request.headers.get("X-Admin-Key")
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


# === Root serves UI ===
@app.get("/", include_in_schema=False)
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
@app.get("/pricing", include_in_schema=False)
async def serve_pricing():
    """Serve the pricing page."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "pricing.html"))
    return RedirectResponse(url="/")


@app.get("/about", include_in_schema=False)
async def serve_about():
    """Serve the about page."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "about.html"))
    return RedirectResponse(url="/")


# === SEO Endpoints ===
@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    """Generate sitemap for search engines."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://mettle.sh/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://mettle.sh/static/docs.html</loc>
    <changefreq>weekly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>https://mettle.sh/pricing</loc>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://mettle.sh/about</loc>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>
  <url>
    <loc>https://mettle.sh/docs</loc>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>"""
    return Response(content=xml, media_type="application/xml")


@app.get("/robots.txt", include_in_schema=False)
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
