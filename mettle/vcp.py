"""VCP (Value Context Protocol) integration for METTLE.

Provides CSM-1 token parsing, attestation building, tier computation,
and compact CSM-1 line formatting for VCP-METTLE integration.

Zero dependency on the Rewind/VCP codebase - operates purely on string formats.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from mettle.protocol import (
    CREDENTIAL_CLOCK_SKEW_SECONDS,
    CREDENTIAL_SCHEMA_VERSION,
    SUITE_POLICY_VERSION,
    credential_time_window_valid,
    credential_versions_supported,
)

logger = logging.getLogger(__name__)

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
    reviewed_at: datetime | None = None,
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
    if not session_id or not subject_id:
        raise ValueError("session_id and authenticated subject_id are required")
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError("pass_rate must be between 0.0 and 1.0")
    if set(suites_passed) & set(suites_failed):
        raise ValueError("A suite cannot be both passed and failed")

    tier = compute_tier(suites_passed)
    credential_eligible = tier != "none"
    reviewed = reviewed_at or datetime.now(tz=timezone.utc)
    if reviewed.tzinfo is None:
        raise ValueError("reviewed_at must be timezone-aware")
    reviewed_at_iso = reviewed.isoformat()
    expires_at = (reviewed + timedelta(hours=1)).isoformat()

    metadata: dict[str, Any] = {
        "mettle_version": "2.0",
        "credential_schema_version": CREDENTIAL_SCHEMA_VERSION,
        "suite_policy_version": SUITE_POLICY_VERSION,
        "session_id": session_id,
        "subject_id": subject_id,
        "entity_id": entity_id,
        "tier": tier,
        "verified": credential_eligible,
        "assurance": "mettle_behavioral_verification",
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
        "reviewed_at": reviewed_at_iso,
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
    attestation: dict[str, Any],
    public_key_pem: str,
    *,
    now: datetime | None = None,
    clock_skew_seconds: int = CREDENTIAL_CLOCK_SKEW_SECONDS,
) -> bool:
    """Verify a METTLE credential envelope and its current validity."""
    if (
        attestation.get("attestation_type")
        not in {"mettle-verification-credential", "mettle-presence-credential"}
        or attestation.get("credential_issued") is not True
    ):
        return False
    metadata = attestation.get("metadata")
    if not isinstance(metadata, dict):
        return False
    if not credential_versions_supported(metadata):
        return False
    tier = metadata.get("tier")
    suites_passed = metadata.get("suites_passed")
    suites_failed = metadata.get("suites_failed")
    if not isinstance(suites_passed, list) or not isinstance(suites_failed, list):
        return False
    if set(suites_passed) & set(suites_failed):
        return False
    if (
        not metadata.get("credential_eligible")
        or tier not in TIER_RANGES
        or compute_tier(suites_passed) != tier
        or not metadata.get("session_id")
        or not metadata.get("subject_id")
    ):
        return False
    if attestation.get("attestation_type") == "mettle-presence-credential":
        proof = metadata.get("proof_of_possession")
        if not isinstance(proof, dict):
            return False
        try:
            from mettle.presence import key_fingerprint

            timing = proof.get("server_timing")
            continuity = proof.get("continuity")
            timing_submissions = (
                timing.get("submissions") if isinstance(timing, dict) else None
            )
            timing_valid = (
                isinstance(timing, dict)
                and isinstance(timing.get("total_elapsed_ms"), int)
                and timing["total_elapsed_ms"] >= 0
                and isinstance(timing_submissions, list)
                and len(timing_submissions) == proof.get("sequence")
            )
            if timing_valid and isinstance(timing_submissions, list):
                for expected_sequence, submission in enumerate(
                    timing_submissions, start=1
                ):
                    if not (
                        isinstance(submission, dict)
                        and submission.get("sequence") == expected_sequence
                        and isinstance(submission.get("action"), str)
                        and submission["action"].startswith(("suite:", "round:"))
                        and isinstance(submission.get("response_time_ms"), int)
                        and submission["response_time_ms"] >= 0
                        and re.fullmatch(
                            r"sha256:[0-9a-f]{64}",
                            str(submission.get("transcript_hash", "")),
                        )
                        is not None
                    ):
                        timing_valid = False
                        break
                    if continuity is not None and not (
                        submission.get("challenge_family") == "mettle-continuity-v1"
                        and re.fullmatch(
                            r"[0-9a-f]{32}",
                            str(submission.get("challenge_id", "")),
                        )
                        is not None
                    ):
                        timing_valid = False
                        break
                if timing_submissions and (
                    timing_submissions[-1].get("transcript_hash")
                    != proof.get("transcript_hash")
                ):
                    timing_valid = False
            continuity_valid = continuity is None or (
                isinstance(continuity, dict)
                and continuity.get("protocol") == "mettle-continuity-v1"
                and continuity.get("challenge_count") == proof.get("sequence")
                and continuity.get("transcript_bound") is True
                and isinstance(continuity.get("max_response_time_ms"), int)
                and continuity["max_response_time_ms"] >= 0
                and isinstance(timing_submissions, list)
                and len(
                    {
                        submission.get("challenge_id")
                        for submission in timing_submissions
                        if isinstance(submission, dict)
                    }
                )
                == proof.get("sequence")
            )
            valid_presence = (
                isinstance(metadata.get("jti"), str)
                and re.fullmatch(r"[0-9a-f]{32}", metadata["jti"]) is not None
                and isinstance(metadata.get("audience"), str)
                and bool(metadata["audience"])
                and proof.get("protocol") == "mettle-presence-v1"
                and isinstance(proof.get("public_key_pem"), str)
                and key_fingerprint(proof["public_key_pem"])
                == proof.get("key_fingerprint")
                and isinstance(proof.get("transcript_hash"), str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", proof["transcript_hash"])
                is not None
                and isinstance(proof.get("sequence"), int)
                and proof["sequence"] > 0
                and timing_valid
                and continuity_valid
            )
        except (TypeError, ValueError):
            return False
        if not valid_presence:
            return False
    expected_hash = f"sha256:{hashlib.sha256(_canonical_bytes(metadata)).hexdigest()}"
    if attestation.get("content_hash") != expected_hash:
        return False
    signature = attestation.get("signature")
    if not isinstance(signature, str) or not signature.startswith("ed25519:"):
        return False
    try:
        reviewed_at = datetime.fromisoformat(str(attestation["reviewed_at"]))
        expires_at = datetime.fromisoformat(str(attestation["expires_at"]))
        if not credential_time_window_valid(
            reviewed_at=reviewed_at,
            expires_at=expires_at,
            now=now,
            clock_skew_seconds=clock_skew_seconds,
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False

    unsigned = dict(attestation)
    unsigned.pop("signature", None)
    from mettle.signing import verify_signature

    return verify_signature(
        public_key_pem,
        _canonical_bytes(unsigned),
        signature.removeprefix("ed25519:"),
    )


def verify_mettle_attestation_with_keyring(
    attestation: dict[str, Any],
    keyring: dict[str, str],
    *,
    now: datetime | None = None,
    clock_skew_seconds: int = CREDENTIAL_CLOCK_SKEW_SECONDS,
) -> bool:
    """Select the signed key ID from a public overlap keyring and verify."""
    key_id = attestation.get("auditor_key_id")
    if not isinstance(key_id, str):
        return False
    public_key_pem = keyring.get(key_id)
    if not public_key_pem:
        return False
    return verify_mettle_attestation(
        attestation,
        public_key_pem,
        now=now,
        clock_skew_seconds=clock_skew_seconds,
    )


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

    Schema 1.0 uses sorted UTF-8 JSON and normalizes integral floats to JSON
    integers so JavaScript, Python, and Rust consumers reproduce the same bytes.
    Historical unversioned envelopes retain the original ASCII-escaped encoding.
    """
    import json

    metadata_value = data.get("metadata")
    metadata: dict[str, Any] = (
        metadata_value if isinstance(metadata_value, dict) else data
    )
    schema_version = metadata.get("credential_schema_version")

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    versioned = schema_version == CREDENTIAL_SCHEMA_VERSION
    payload = normalize(data) if versioned else data
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=not versioned,
    ).encode("utf-8")
