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
from datetime import datetime, timezone
from typing import Any, Callable

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

# Tier definitions: tier name -> required suite numbers
# All suites in the range must pass for the tier to apply.
# Platinum now requires Suite 11 (governance verification) in addition to 1-10.
TIER_RANGES: dict[str, tuple[int, int]] = {
    "bronze": (1, 5),
    "silver": (1, 7),
    "gold": (1, 9),
    "platinum": (1, 11),  # Updated: requires governance suite
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
        raise ValueError(f"Invalid VCP header line: {header!r}. Expected format: VCP:<version>:<profile_id>")

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


def compute_tier(suites_passed: list[str]) -> str:
    """Compute METTLE verification tier from suites passed.

    Tier assignment is suite-based, not percentage-based:
        Bronze:   suites 1-5 all pass
        Silver:   suites 1-7 all pass
        Gold:     suites 1-9 all pass
        Platinum: suites 1-11 all pass (includes governance)

    Any suite failure below the tier's range drops the tier.
    E.g., pass suites 1-9 but fail suite 6 -> Bronze (not Silver).

    Args:
        suites_passed: List of suite names that passed.

    Returns:
        Tier string: "platinum", "gold", "silver", "bronze", or "none".
    """
    passed_numbers = {SUITE_ORDER[s] for s in suites_passed if s in SUITE_ORDER}

    # Check tiers from highest to lowest
    for tier in ("platinum", "gold", "silver", "bronze"):
        lo, hi = TIER_RANGES[tier]
        required = set(range(lo, hi + 1))
        if required <= passed_numbers:
            return tier

    return "none"


def build_mettle_attestation(
    session_id: str,
    difficulty: str,
    suites_passed: list[str],
    suites_failed: list[str],
    pass_rate: float,
    sign_fn: Callable[[bytes], str] | None = None,
    key_id: str = "mettle-vcp-v1",
) -> dict[str, Any]:
    """Build a VCP-compatible attestation dict from METTLE results.

    Args:
        session_id: METTLE session ID.
        difficulty: Session difficulty level.
        suites_passed: List of suite names that passed.
        suites_failed: List of suite names that failed.
        pass_rate: Overall pass rate (0.0-1.0).
        sign_fn: Optional Ed25519 signing function (bytes -> base64 sig).
        key_id: Key ID for the signing key.

    Returns:
        VCP-compatible attestation dict.
    """
    tier = compute_tier(suites_passed)
    reviewed_at = datetime.now(tz=timezone.utc).isoformat()

    metadata = {
        "mettle_version": "2.0",
        "session_id": session_id,
        "tier": tier,
        "suites_passed": sorted(suites_passed),
        "suites_failed": sorted(suites_failed),
        "difficulty": difficulty,
        "pass_rate": round(pass_rate, 4),
    }

    # Content-addressable digest of the metadata. Kept for integrity/dedup, but note this is
    # NOT what the signature covers -- see attestation_signing_bytes.
    content_bytes = _canonical_bytes(metadata)
    content_hash = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"

    attestation: dict[str, Any] = {
        "auditor": "mettle.creed.space",
        "auditor_key_id": key_id,
        "attestation_type": "mettle-verification",
        "signature_scheme": SIGNATURE_SCHEME,
        "reviewed_at": reviewed_at,
        "content_hash": content_hash,
        "metadata": metadata,
    }

    if sign_fn is None:
        attestation["signature"] = None
        return attestation

    try:
        signature = sign_fn(attestation_signing_bytes(attestation))
    except Exception as e:
        # SECURITY: fail CLOSED. This used to swallow the error and return the attestation with
        # signature=None -- an unsigned credential handed out as though it were valid. Refusing to
        # issue is the only safe outcome; an unsigned attestation is forgeable.
        logger.exception("Failed to sign VCP attestation")
        raise RuntimeError(
            "Cannot issue a VCP attestation: signing failed. An unsigned attestation is forgeable."
        ) from e

    attestation["signature"] = f"ed25519:{signature}"
    return attestation


def format_csm1_line(tier: str, session_id: str, timestamp: str | None = None) -> str:
    """Produce a compact CSM-1 METTLE attestation line.

    Format: MT:<tier>:<session_id_short>:<iso_timestamp>

    Args:
        tier: METTLE tier (bronze/silver/gold/platinum).
        session_id: Full session ID (will be truncated for compact form).
        timestamp: ISO timestamp. Defaults to now.

    Returns:
        CSM-1 line string.
    """
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    # Use first 12 chars of session_id for compact form
    short_id = session_id[:12] if len(session_id) > 12 else session_id

    return f"MT:{tier}:{short_id}:{timestamp}"


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    """Convert dict to canonical bytes for hashing/signing.

    Uses sorted JSON keys for deterministic output.
    """
    import json

    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


# Marks which bytes the signature covers. Carried INSIDE the signed envelope, so it cannot be
# downgraded by an attacker to trick a verifier into checking the old, weaker metadata-only bytes.
SIGNATURE_SCHEME = "mettle-envelope-v1"


def attestation_signing_bytes(attestation: dict[str, Any]) -> bytes:
    """The exact bytes an attestation signature covers: the whole envelope minus ``signature``.

    SECURITY -- the previous scheme signed only ``metadata``. That left ``reviewed_at``,
    ``auditor``, ``auditor_key_id`` and ``attestation_type`` OUTSIDE the signature, so a genuine
    old attestation could be re-dated and would still verify: freshness tampering on a credential
    whose entire purpose is to be checked for freshness.

    Both the issuer and any verifier MUST derive the signed bytes through this function. Do not
    reconstruct them by hand -- that is how the two sides drift apart and a check silently
    degrades into checking nothing.
    """
    return _canonical_bytes({k: v for k, v in attestation.items() if k != "signature"})


def verify_attestation(attestation: dict[str, Any], public_key_pem: str) -> bool:
    """Verify an attestation's envelope signature against a published Ed25519 public key.

    Fails closed on anything unexpected: a missing/None signature, an unknown signature scheme,
    or a malformed signature all return False. An unsigned attestation is NOT valid.
    """
    from mettle.signing import verify_signature

    signature = attestation.get("signature")
    if not signature or not isinstance(signature, str):
        return False
    if attestation.get("signature_scheme") != SIGNATURE_SCHEME:
        # Refuse unknown/legacy schemes rather than guessing which bytes were signed.
        return False
    if not signature.startswith("ed25519:"):
        return False

    return verify_signature(
        public_key_pem,
        attestation_signing_bytes(attestation),
        signature.removeprefix("ed25519:"),
    )
