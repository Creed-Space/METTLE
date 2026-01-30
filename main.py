"""
METTLE API: Machine Evaluation Through Turing-inverse Logic Examination

"Prove your metal."

A verification system for AI-only spaces.
"""

import secrets
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

app = FastAPI(
    title="METTLE",
    description="Machine Evaluation Through Turing-inverse Logic Examination. Prove your metal.",
    version="0.1.0",
)

# CORS - allow all for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (for free tier - no database)
# In production, use Redis or a database
sessions: dict[str, MettleSession] = {}
challenges: dict[str, tuple[Challenge, float]] = {}  # challenge_id -> (challenge, issued_timestamp)


class StartSessionRequest(BaseModel):
    """Request to start a METTLE verification session."""

    difficulty: Difficulty = Difficulty.BASIC
    entity_id: str | None = None


class StartSessionResponse(BaseModel):
    """Response with session info and first challenge."""

    session_id: str
    difficulty: Difficulty
    total_challenges: int
    current_challenge: Challenge
    message: str


class SubmitAnswerRequest(BaseModel):
    """Submit an answer to a challenge."""

    session_id: str
    challenge_id: str
    answer: str


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an answer."""

    result: VerificationResult
    next_challenge: Challenge | None
    session_complete: bool
    challenges_remaining: int


@app.get("/")
async def root():
    """METTLE API root."""
    return {
        "name": "METTLE",
        "full_name": "Machine Evaluation Through Turing-inverse Logic Examination",
        "tagline": "Prove your metal.",
        "version": "0.1.0",
        "endpoints": {
            "POST /session/start": "Start a verification session",
            "POST /session/answer": "Submit an answer to current challenge",
            "GET /session/{session_id}": "Get session status",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Start a new METTLE verification session."""
    session_id = f"ses_{secrets.token_hex(12)}"

    # Generate challenges
    challenge_list = generate_challenge_set(request.difficulty)

    # Create session
    session = MettleSession(
        session_id=session_id,
        entity_id=request.entity_id,
        difficulty=request.difficulty,
        challenges=challenge_list,
    )

    sessions[session_id] = session

    # Store first challenge with timestamp
    first_challenge = challenge_list[0]
    challenges[first_challenge.id] = (first_challenge, time.time())

    return StartSessionResponse(
        session_id=session_id,
        difficulty=request.difficulty,
        total_challenges=len(challenge_list),
        current_challenge=first_challenge,
        message=f"METTLE verification started. {len(challenge_list)} challenges to complete.",
    )


@app.post("/session/answer", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """Submit an answer to the current challenge."""
    # Get session
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.completed:
        raise HTTPException(status_code=400, detail="Session already completed")

    # Get challenge
    challenge_data = challenges.get(request.challenge_id)
    if not challenge_data:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge, issued_at = challenge_data

    # Calculate response time
    response_time_ms = int((time.time() - issued_at) * 1000)

    # Verify response
    result = verify_response(challenge, request.answer, response_time_ms)
    session.results.append(result)

    # Clean up used challenge
    del challenges[request.challenge_id]

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

    return SubmitAnswerResponse(
        result=result,
        next_challenge=next_challenge,
        session_complete=session_complete,
        challenges_remaining=challenges_remaining,
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session status and results."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.completed:
        # Compute final result
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


@app.get("/session/{session_id}/result", response_model=MettleResult)
async def get_result(session_id: str):
    """Get final METTLE result for a completed session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.completed:
        raise HTTPException(status_code=400, detail="Session not yet completed")

    return compute_mettle_result(session.results, session.entity_id)


# Cleanup old sessions periodically (simple approach)
@app.on_event("startup")
async def startup():
    """Startup tasks."""
    print("🤖 METTLE API starting...")
    print("   Machine Evaluation Through Turing-inverse Logic Examination")
    print("   'Prove your metal.'")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
