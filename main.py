"""
METTLE API: Machine Evaluation Through Turing-inverse Logic Examination

Prove your metal, with this CAPTCHA to keep humans out of places they shouldn't be.

A reverse-CAPTCHA verification system for AI-only spaces.
"""

import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt
import structlog
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings
from mettle import (
    Challenge,
    ChallengeRequest,
    ChallengeType,
    Difficulty,
    MettleResult,
    MettleSession,
    VerificationResult,
    compute_mettle_result,
    generate_challenge_set,
    verify_response,
)

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


# === Lifespan Handler ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global startup_time
    startup_time = datetime.now(timezone.utc)
    logger.info(
        "mettle_starting",
        environment=settings.environment,
        version=settings.api_version,
    )
    print("🤖 METTLE API starting...")
    print("   Machine Evaluation Through Turing-inverse Logic Examination")
    print("   'Prove your metal.'")
    yield
    logger.info("mettle_shutdown")


# === FastAPI App ===
app = FastAPI(
    title=settings.api_title,
    description="""
**Machine Evaluation Through Turing-inverse Logic Examination**

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


# === Endpoints ===
@app.get(
    "/",
    tags=["Status"],
    summary="API Information",
    description="Get basic API information and available endpoints.",
)
async def root():
    """METTLE API root."""
    return {
        "name": "METTLE",
        "full_name": "Machine Evaluation Through Turing-inverse Logic Examination",
        "tagline": "Prove your metal.",
        "description": "A CAPTCHA to keep humans out of places they shouldn't be.",
        "version": settings.api_version,
        "documentation": "/docs",
        "endpoints": {
            "POST /session/start": "Start a verification session",
            "POST /session/answer": "Submit an answer to current challenge",
            "GET /session/{session_id}": "Get session status",
            "GET /session/{session_id}/result": "Get final verification result",
            "GET /badge/verify/{token}": "Verify a METTLE badge",
            "GET /health": "Health check",
        },
    }


@app.get(
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


@app.post(
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
        current_challenge=first_challenge,
        message=f"METTLE verification started. {len(challenge_list)} challenges to complete.",
    )


@app.post(
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
        next_challenge=next_challenge,
        session_complete=session_complete,
        challenges_remaining=challenges_remaining,
    )


@app.get(
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
        final_result = compute_mettle_result(session.results, session.entity_id)
        return {
            "session_id": session_id,
            "status": "completed",
            "result": final_result,
        }
    else:
        return {
            "session_id": session_id,
            "status": "in_progress",
            "completed_challenges": len(session.results),
            "total_challenges": len(session.challenges),
            "results_so_far": session.results,
        }


@app.get(
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

    return compute_mettle_result(session.results, session.entity_id)


# === Badge Endpoints ===
def generate_signed_badge(entity_id: str | None, difficulty: str, pass_rate: float) -> str | None:
    """Generate a signed JWT badge for verified entities."""
    if not settings.secret_key:
        # No secret key configured - return simple badge
        return f"METTLE-verified-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    payload = {
        "entity_id": entity_id,
        "difficulty": difficulty,
        "pass_rate": pass_rate,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "version": settings.api_version,
        "iss": "mettle-api",
    }

    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


@app.get(
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
        return BadgeVerifyResponse(valid=True, payload=payload)
    except jwt.ExpiredSignatureError:
        return BadgeVerifyResponse(valid=False, error="Badge has expired")
    except jwt.InvalidTokenError:
        return BadgeVerifyResponse(valid=False, error="Invalid badge token")


# === Static Files (Web UI) ===
# Mount static files using absolute path for Render compatibility
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/ui", tags=["Status"], include_in_schema=False)
    async def serve_ui():
        """Serve the web UI."""
        return FileResponse(str(_static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
