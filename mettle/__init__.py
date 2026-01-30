"""METTLE: Machine Evaluation Through Turing-inverse Logic Examination."""

from .challenger import generate_challenge, generate_challenge_set
from .models import (
    Challenge,
    ChallengeRequest,
    ChallengeResponse,
    ChallengeType,
    Difficulty,
    MettleResult,
    MettleSession,
    VerificationResult,
)
from .verifier import compute_mettle_result, verify_response

__all__ = [
    "Challenge",
    "ChallengeRequest",
    "ChallengeResponse",
    "ChallengeType",
    "Difficulty",
    "MettleResult",
    "MettleSession",
    "VerificationResult",
    "generate_challenge",
    "generate_challenge_set",
    "verify_response",
    "compute_mettle_result",
]

__version__ = "0.1.0"
