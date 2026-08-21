"""VCP (Value Context Protocol) integration for METTLE.

Provides CSM-1 token parsing, attestation building, tier computation,
and compact CSM-1 line formatting for VCP-METTLE integration.

Zero dependency on the Rewind/VCP codebase - operates purely on string formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

ASSURANCE_PROFILE = "mettle_behavioral_verification"
ALLOWED_DIFFICULTIES = frozenset({"basic", "full", "easy", "standard", "hard"})
MAX_ATTESTATION_LIFETIME = timedelta(hours=1)
MAX_ISSUANCE_CLOCK_SKEW = timedelta(minutes=5)
PROTOCOL_ACTION_PATTERN = re.compile(
    r"(?:suite|round):[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
)

# Suite numbers mapped to names for tier computation
SUITE_ORDER: dict[str, int] = {
    "adversarial": 1,
    "native": 2,
    "self-reference": 3,
    "social": 4,
    "inverse-turing": 5,
    "anti-thrall": 6,
    "agency": 7,
    "counter-coaching": 8,
    "intent-provenance": 9,
    "novel-reasoning": 10,
    "governance": 11,  # Suite 11: Governance verification
    "llm-dynamic": 12,  # Suite 12: LLM-dynamic (supplemental, not in tier ranges)
}

# A tier is earned only when every suite in its contiguous range passed. This
# prevents callers from cherry-picking a single easy or LLM-judged suite and
# presenting it as a stronger credential.
TIER_RANGES: dict[str, tuple[int, int]] = {
    "bronze": (1, 5),
    "silver": (1, 7),
    "gold": (1, 9),
    "platinum": (1, 11),
}


@dataclass
class VCPTokenClaim:
    """Structured representation of a parsed CSM-1 VCP token."""

    version: str
    profile_id: str
    constitution_id: str | None = None
    constitution_version: str | None = None
    persona: str | None = None
    adherence: int | None = None
    goal: str | None = None
    extra_lines: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def constitution_ref(self) -> str | None:
        """Full constitution reference (id@version)."""
        if not self.constitution_id:
            return None
        if self.constitution_version:
            return f"{self.constitution_id}@{self.constitution_version}"
        return self.constitution_id


def parse_csm1_token(token: str) -> VCPTokenClaim:
    """Parse a CSM-1 format VCP token into a structured dataclass.

    CSM-1 format example:
        VCP:3.1:agent-42
        C:professional.safe.balanced@2.0.0
        P:advisor:4
        G:assist:expert:analytical
        X:none
        F:none
        S:none
        R:none

    Args:
        token: Raw CSM-1 token string.

    Returns:
        VCPTokenClaim with parsed fields.

    Raises:
        ValueError: If token is malformed or missing required VCP header.
    """
    if not token or not isinstance(token, str):
        raise ValueError("VCP token must be a non-empty string")

    lines = [line.strip() for line in token.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("VCP token is empty")

    # Parse header line: VCP:<version>:<profile_id>
    header = lines[0]
    header_match = re.match(r"^VCP:([^:]+):(.+)$", header)
    if not header_match:
        raise ValueError(
            f"Invalid VCP header line: {header!r}. Expected format: VCP:<version>:<profile_id>"
        )

    version = header_match.group(1)
    profile_id = header_match.group(2)

    claim = VCPTokenClaim(
        version=version,
        profile_id=profile_id,
        raw=token.strip(),
    )

    # Parse remaining lines
    for line in lines[1:]:
        if ":" not in line:
            continue

        prefix, _, value = line.partition(":")
        prefix = prefix.upper()

        if prefix == "C":
            # Constitution line: C:id@version
            if "@" in value:
                cid, cver = value.rsplit("@", 1)
                claim.constitution_id = cid
                claim.constitution_version = cver
            else:
                claim.constitution_id = value
        elif prefix == "P":
            # Persona line: P:role:adherence
            parts = value.split(":")
            claim.persona = parts[0] if parts else value
            if len(parts) >= 2:
                try:
                    claim.adherence = int(parts[1])
                except ValueError:
                    pass
        elif prefix == "G":
            claim.goal = value
        elif prefix == "MT":
            # METTLE attestation line - store in extra
            claim.extra_lines["MT"] = value
        else:
            claim.extra_lines[prefix] = value

    return claim


def compute_tier(
    suites_passed: list[str],
) -> str:
    """Compute the highest contiguous METTLE challenge tier earned."""
    passed_numbers = {SUITE_ORDER[s] for s in suites_passed if s in SUITE_ORDER}
    for tier in ("platinum", "gold", "silver", "bronze"):
        lo, hi = TIER_RANGES[tier]
        if set(range(lo, hi + 1)) <= passed_numbers:
            return tier
    return "none"


def _validate_suite_partition(
    suites_passed: Any, suites_failed: Any
) -> tuple[list[str], list[str]]:
    """Validate one unambiguous partition of known METTLE suite names."""
    if not isinstance(suites_passed, list) or not isinstance(suites_failed, list):
        raise ValueError("Suite results must be lists")
    for suites in (suites_passed, suites_failed):
        if any(
            not isinstance(suite, str) or suite not in SUITE_ORDER for suite in suites
        ):
            raise ValueError("Suite results must contain only known suite names")
        if len(set(suites)) != len(suites):
            raise ValueError("Suite results must not contain duplicates")
    if set(suites_passed) & set(suites_failed):
        raise ValueError("A suite cannot be both passed and failed")
    return suites_passed, suites_failed


def _coherent_pass_rate(
    suites_passed: list[str], suites_failed: list[str], pass_rate: Any
) -> float:
    if (
        isinstance(pass_rate, bool)
        or not isinstance(pass_rate, (int, float))
        or not math.isfinite(pass_rate)
        or not 0.0 <= pass_rate <= 1.0
    ):
        raise ValueError("pass_rate must be a finite number between 0.0 and 1.0")
    total = len(suites_passed) + len(suites_failed)
    expected = len(suites_passed) / total if total else 0.0
    normalized = round(float(pass_rate), 4)
    if normalized != round(expected, 4):
        raise ValueError("pass_rate does not match the suite results")
    return normalized


def _bounded_protocol_text(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value


def _utc_protocol_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("Attestation timestamp must be a non-empty string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Attestation timestamps must use UTC")
    return parsed.astimezone(timezone.utc)


def build_mettle_attestation(
    session_id: str,
    difficulty: str,
    suites_passed: list[str],
    suites_failed: list[str],
    pass_rate: float,
    subject_id: str,
    entity_id: str | None = None,
    key_id: str = "mettle-vcp-v1",
    presence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a server-issued VCP-compatible METTLE result.

    Args:
        session_id: METTLE session ID.
        difficulty: Session difficulty level.
        suites_passed: List of suite names that passed.
        suites_failed: List of suite names that failed.
        pass_rate: Overall pass rate (0.0-1.0).
        key_id: Identifier for the server-owned Ed25519 issuer key.

    Returns:
        VCP-compatible attestation dict.
    """
    session_id = _bounded_protocol_text(session_id, "session_id")
    subject_id = _bounded_protocol_text(subject_id, "subject_id")
    if entity_id is not None:
        entity_id = _bounded_protocol_text(entity_id, "entity_id")
    difficulty = _bounded_protocol_text(difficulty, "difficulty", maximum=32)
    if difficulty not in ALLOWED_DIFFICULTIES:
        raise ValueError("difficulty is not a supported METTLE profile")
    key_id = _bounded_protocol_text(key_id, "key_id", maximum=128)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", key_id) is None:
        raise ValueError("key_id is invalid")
    suites_passed, suites_failed = _validate_suite_partition(
        suites_passed, suites_failed
    )
    pass_rate = _coherent_pass_rate(suites_passed, suites_failed, pass_rate)

    tier = compute_tier(suites_passed)
    credential_eligible = tier != "none"
    reviewed = datetime.now(tz=timezone.utc)
    reviewed_at = reviewed.isoformat()
    expires_at = (reviewed + timedelta(hours=1)).isoformat()

    metadata: dict[str, Any] = {
        "mettle_version": "2.0",
        "session_id": session_id,
        "subject_id": subject_id,
        "entity_id": entity_id,
        "tier": tier,
        "verified": credential_eligible,
        "assurance": ASSURANCE_PROFILE,
        "credential_eligible": credential_eligible,
        "suites_passed": sorted(suites_passed),
        "suites_failed": sorted(suites_failed),
        "difficulty": difficulty,
        "pass_rate": round(pass_rate, 4),
    }
    if presence is not None:
        from mettle.presence import validate_credential_presence

        validate_credential_presence(presence)
        timing_submissions = []
        for submission in presence.get("submissions", []):
            receipt = {
                "sequence": submission["sequence"],
                "action": submission["action"],
                "response_time_ms": submission["response_time_ms"],
                "transcript_hash": submission["transcript_hash"],
            }
            if presence.get("continuity_protocol") is not None:
                receipt["challenge_family"] = submission["challenge_family"]
                receipt["challenge_id"] = submission["challenge_id"]
            timing_submissions.append(receipt)
        completed_at_ms = (
            presence["submissions"][-1]["accepted_at_unix_ms"]
            if presence.get("submissions")
            else presence["started_at_unix_ms"]
        )
        metadata["jti"] = presence["credential_jti"]
        metadata["audience"] = presence["audience"]
        metadata["proof_of_possession"] = {
            "protocol": presence["protocol"],
            "public_key_pem": presence["public_key_pem"],
            "key_fingerprint": presence["key_fingerprint"],
            "transcript_hash": presence["transcript_hash"],
            "sequence": presence["sequence"],
            "server_timing": {
                "total_elapsed_ms": max(
                    0, completed_at_ms - presence["started_at_unix_ms"]
                ),
                "submissions": timing_submissions,
            },
        }
        if presence.get("continuity_protocol") is not None:
            metadata["proof_of_possession"]["continuity"] = {
                "protocol": presence["continuity_protocol"],
                "challenge_count": len(timing_submissions),
                "transcript_bound": True,
                "max_response_time_ms": max(
                    (item["response_time_ms"] for item in timing_submissions),
                    default=0,
                ),
            }

    # Hash the metadata for content integrity
    content_bytes = _canonical_bytes(metadata)
    content_hash = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"

    attestation: dict[str, Any] = {
        "auditor": "mettle.creed.space",
        "auditor_key_id": key_id,
        "attestation_type": (
            (
                "mettle-presence-credential"
                if presence is not None
                else "mettle-verification-credential"
            )
            if credential_eligible
            else "mettle-evidence-receipt"
        ),
        "reviewed_at": reviewed_at,
        "expires_at": expires_at,
        "content_hash": content_hash,
        "metadata": metadata,
        "credential_issued": credential_eligible,
    }

    # Only a qualifying contiguous suite battery reaches the signer. The
    # signing key is owned by the server and is never supplied by the caller.
    signature = None
    if credential_eligible:
        try:
            from mettle.signing import is_available, sign_attestation

            if is_available():
                signature = f"ed25519:{sign_attestation(_canonical_bytes(attestation))}"
        except (ImportError, RuntimeError):
            logger.warning("METTLE credential signing unavailable", exc_info=True)
    if credential_eligible and signature is None:
        attestation["attestation_type"] = "mettle-verification-evidence"
        attestation["credential_issued"] = False
    attestation["signature"] = signature

    return attestation


def verify_mettle_attestation(
    attestation: Any,
    public_key_pem: Any,
    *,
    expected_subject_id: str | None = None,
    expected_entity_id: str | None = None,
    expected_key_id: str | None = None,
    expected_assurance: str = ASSURANCE_PROFILE,
    expected_difficulty: str | None = None,
) -> bool:
    """Verify a credential envelope, its semantics, and its current validity.

    Verification is total over untrusted values: malformed inputs return ``False``.
    Optional expectations bind the signed result to the relying party's context.
    """
    try:
        if not isinstance(attestation, dict) or not isinstance(public_key_pem, str):
            return False
        attestation_type = attestation.get("attestation_type")
        if (
            attestation_type
            not in {"mettle-verification-credential", "mettle-presence-credential"}
            or attestation.get("credential_issued") is not True
            or attestation.get("auditor") != "mettle.creed.space"
        ):
            return False
        key_id = attestation.get("auditor_key_id")
        if (
            not isinstance(key_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", key_id) is None
            or (expected_key_id is not None and key_id != expected_key_id)
        ):
            return False

        metadata = attestation.get("metadata")
        if not isinstance(metadata, dict):
            return False
        suites_passed, suites_failed = _validate_suite_partition(
            metadata.get("suites_passed"), metadata.get("suites_failed")
        )
        tier = metadata.get("tier")
        session_id = metadata.get("session_id")
        subject_id = metadata.get("subject_id")
        entity_id = metadata.get("entity_id")
        difficulty = metadata.get("difficulty")
        if (
            metadata.get("mettle_version") != "2.0"
            or metadata.get("verified") is not True
            or metadata.get("credential_eligible") is not True
            or metadata.get("assurance") != expected_assurance
            or tier not in TIER_RANGES
            or compute_tier(suites_passed) != tier
            or not isinstance(session_id, str)
            or not session_id
            or len(session_id) > 256
            or not isinstance(subject_id, str)
            or not subject_id
            or len(subject_id) > 256
            or (
                entity_id is not None
                and (
                    not isinstance(entity_id, str)
                    or not entity_id
                    or len(entity_id) > 256
                )
            )
            or difficulty not in ALLOWED_DIFFICULTIES
            or (expected_subject_id is not None and subject_id != expected_subject_id)
            or (expected_entity_id is not None and entity_id != expected_entity_id)
            or (expected_difficulty is not None and difficulty != expected_difficulty)
        ):
            return False
        _coherent_pass_rate(suites_passed, suites_failed, metadata.get("pass_rate"))

        proof = metadata.get("proof_of_possession")
        if (attestation_type == "mettle-presence-credential") != isinstance(
            proof, dict
        ):
            return False
        if attestation_type == "mettle-presence-credential":
            from mettle.presence import key_fingerprint

            assert isinstance(proof, dict)
            timing = proof.get("server_timing")
            continuity = proof.get("continuity")
            sequence = proof.get("sequence")
            timing_submissions = (
                timing.get("submissions") if isinstance(timing, dict) else None
            )
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence <= 0
                or not isinstance(timing, dict)
                or isinstance(timing.get("total_elapsed_ms"), bool)
                or not isinstance(timing.get("total_elapsed_ms"), int)
                or timing["total_elapsed_ms"] < 0
                or not isinstance(timing_submissions, list)
                or len(timing_submissions) != sequence
            ):
                return False
            challenge_ids: set[str] = set()
            response_times: list[int] = []
            for expected_sequence, submission in enumerate(timing_submissions, start=1):
                if not isinstance(submission, dict):
                    return False
                action = submission.get("action")
                response_time = submission.get("response_time_ms")
                if (
                    isinstance(submission.get("sequence"), bool)
                    or submission.get("sequence") != expected_sequence
                    or not isinstance(action, str)
                    or PROTOCOL_ACTION_PATTERN.fullmatch(action) is None
                    or isinstance(response_time, bool)
                    or not isinstance(response_time, int)
                    or response_time < 0
                    or not isinstance(submission.get("transcript_hash"), str)
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}", submission["transcript_hash"]
                    )
                    is None
                ):
                    return False
                response_times.append(response_time)
                if continuity is not None:
                    challenge_id = submission.get("challenge_id")
                    if (
                        submission.get("challenge_family") != "mettle-continuity-v1"
                        or not isinstance(challenge_id, str)
                        or re.fullmatch(r"[0-9a-f]{32}", challenge_id) is None
                        or challenge_id in challenge_ids
                    ):
                        return False
                    challenge_ids.add(challenge_id)
            if timing_submissions[-1].get("transcript_hash") != proof.get(
                "transcript_hash"
            ) or timing["total_elapsed_ms"] != sum(response_times):
                return False
            if continuity is not None and (
                not isinstance(continuity, dict)
                or continuity.get("protocol") != "mettle-continuity-v1"
                or isinstance(continuity.get("challenge_count"), bool)
                or continuity.get("challenge_count") != sequence
                or continuity.get("transcript_bound") is not True
                or isinstance(continuity.get("max_response_time_ms"), bool)
                or continuity.get("max_response_time_ms") != max(response_times)
                or len(challenge_ids) != sequence
            ):
                return False
            if (
                not isinstance(metadata.get("jti"), str)
                or re.fullmatch(r"[0-9a-f]{32}", metadata["jti"]) is None
                or not isinstance(metadata.get("audience"), str)
                or not metadata["audience"]
                or len(metadata["audience"]) > 256
                or proof.get("protocol") != "mettle-presence-v1"
                or not isinstance(proof.get("public_key_pem"), str)
                or len(proof["public_key_pem"]) > 4096
                or key_fingerprint(proof["public_key_pem"])
                != proof.get("key_fingerprint")
                or not isinstance(proof.get("transcript_hash"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", proof["transcript_hash"])
                is None
            ):
                return False

        expected_hash = (
            f"sha256:{hashlib.sha256(_canonical_bytes(metadata)).hexdigest()}"
        )
        if attestation.get("content_hash") != expected_hash:
            return False
        signature = attestation.get("signature")
        if not isinstance(signature, str) or not signature.startswith("ed25519:"):
            return False

        reviewed_at = _utc_protocol_datetime(attestation.get("reviewed_at"))
        expires_at = _utc_protocol_datetime(attestation.get("expires_at"))
        now = datetime.now(timezone.utc)
        lifetime = expires_at - reviewed_at
        if (
            reviewed_at > now + MAX_ISSUANCE_CLOCK_SKEW
            or lifetime <= timedelta(0)
            or lifetime > MAX_ATTESTATION_LIFETIME
            or expires_at <= now
        ):
            return False

        unsigned = dict(attestation)
        unsigned.pop("signature", None)
        from mettle.signing import verify_signature

        return verify_signature(
            public_key_pem,
            _canonical_bytes(unsigned),
            signature.removeprefix("ed25519:"),
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return False


def format_csm1_line(tier: str, session_id: str, timestamp: str | None = None) -> str:
    """Produce a compact CSM-1 METTLE result reference.

    Format: MT:<tier>:<session_id_short>:<iso_timestamp>

    Args:
        tier: METTLE result tier.
        session_id: Full session ID (will be truncated for compact form).
        timestamp: ISO timestamp. Defaults to now.

    Returns:
        CSM-1 line string.
    """
    if tier not in {*TIER_RANGES, "none"}:
        raise ValueError(f"Unknown METTLE tier: {tier}")
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    # Use first 12 chars of session_id for compact form
    short_id = session_id[:12] if len(session_id) > 12 else session_id

    return f"MT:{tier}:{short_id}:{timestamp}"


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    """Convert dict to canonical bytes for hashing/signing.

    Uses sorted JSON keys for deterministic output.
    """
    if not isinstance(data, dict):
        raise ValueError("Canonical protocol JSON must be an object")
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
