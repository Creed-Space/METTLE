"""METTLE: Machine Evaluation Through Turing-inverse Logic Examination."""

from .challenger import generate_challenge, generate_challenge_set
from .models import (
    BadgeInfo,
    Challenge,
    ChallengeRequest,
    ChallengeResponse,
    ChallengeType,
    Difficulty,
    MettleResult,
    MettleSession,
    VerificationResult,
)
from .vcp import (
    VCPTokenClaim,
    build_credential_status_receipt,
    build_mettle_attestation,
    compute_tier,
    format_csm1_line,
    parse_csm1_token,
    verify_credential_status_receipt,
    verify_mettle_credential_with_status,
)
from .verifier import compute_mettle_result, verify_response

__all__ = [
    "BadgeInfo",
    "Challenge",
    "ChallengeRequest",
    "ChallengeResponse",
    "ChallengeType",
    "Difficulty",
    "MettleResult",
    "MettleSession",
    "VCPTokenClaim",
    "VerificationResult",
    "build_credential_status_receipt",
    "build_mettle_attestation",
    "compute_mettle_result",
    "compute_tier",
    "format_csm1_line",
    "generate_challenge",
    "generate_challenge_set",
    "parse_csm1_token",
    "verify_credential_status_receipt",
    "verify_mettle_credential_with_status",
    "verify_response",
]

__version__ = "0.4.1"
