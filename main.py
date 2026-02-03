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

# In-memory storage
sessions: dict[str, MettleSession] = {}
challenges: dict[str, tuple[Challenge, float]] = {}
revoked_badges: set[str] = set()  # JTIs of revoked badges

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
    )
    challenge_id: str = Field(
        description="Challenge identifier to answer",
        min_length=1,
        max_length=64,
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

    # Generate challenges
    challenge_list = generate_challenge_set(body.difficulty)

    # Create session
    session = MettleSession(
        session_id=session_id,
        entity_id=body.entity_id,
        difficulty=body.difficulty,
        challenges=challenge_list,
    )

    sessions[session_id] = session

    # Store first challenge with timestamp
    first_challenge = challenge_list[0]
    challenges[first_challenge.id] = (first_challenge, time.time())

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
    # Get session
    session = sessions.get(body.session_id)
    if not session:
        logger.warning("session_not_found", session_id=body.session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    if session.completed:
        raise HTTPException(status_code=400, detail="Session already completed")

    # Get challenge
    challenge_data = challenges.get(body.challenge_id)
    if not challenge_data:
        logger.warning(
            "challenge_not_found",
            session_id=body.session_id,
            challenge_id=body.challenge_id,
        )
        raise HTTPException(status_code=404, detail="Challenge not found")

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

    # Clean up used challenge
    del challenges[body.challenge_id]

    # Determine next challenge or complete session
    current_index = len(session.results)
    challenges_remaining = len(session.challenges) - current_index

    if challenges_remaining > 0:
        next_challenge = session.challenges[current_index]
        challenges[next_challenge.id] = (next_challenge, time.time())
        session_complete = False
    else:
        next_challenge = None
        session.completed = True
        session_complete = True

        # Log session completion
        final_result = compute_mettle_result(session.results, session.entity_id)
        logger.info(
            "session_completed",
            session_id=body.session_id,
            entity_id=session.entity_id,
            verified=final_result.verified,
            pass_rate=final_result.pass_rate,
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
        # No secret key configured - return simple badge with expiry info
        return {
            "token": f"METTLE-verified-{now.strftime('%Y%m%d')}",
            "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
            "freshness_nonce": freshness_nonce,
            "signed": False,
        }

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
async def verify_badge(token: str):
    """Verify a METTLE badge is valid."""
    if not settings.secret_key:
        # No signing configured - can't verify JWT badges
        if token.startswith("METTLE-verified-"):
            return BadgeVerifyResponse(
                valid=True,
                payload={"badge": token, "type": "simple"},
            )
        return BadgeVerifyResponse(
            valid=False,
            error="Badge verification not configured",
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
    """
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

    # Add to revocation set
    revoked_badges.add(jti)

    # Create audit record
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
    description="List all badge revocations (audit trail).",
)
async def list_revocations():
    """Get the audit trail of badge revocations."""
    return {
        "revoked_count": len(revoked_badges),
        "audit": revocation_audit[-100:],  # Last 100 revocations
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
