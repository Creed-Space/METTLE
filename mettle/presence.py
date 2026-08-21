"""Key-bound liveness and presentation primitives for METTLE credentials."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_public_key,
)
from mettle.continuity import (
    CONTINUITY_PROTOCOL,
    consume_continuity_evidence,
    new_continuity_state,
)

PRESENCE_PROTOCOL = "mettle-presence-v1"
HASH_PREFIX = "sha256:"
PRESENCE_STATE_RECEIPT_PURPOSE = "mettle-presence-state"
PRESENCE_ACTION_PATTERN = re.compile(
    r"(?:suite|round):[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
)


def canonical_bytes(value: dict[str, Any]) -> bytes:
    """Serialize one protocol object deterministically."""
    if not isinstance(value, dict):
        raise ValueError("Canonical Presence JSON must be an object")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def answer_hash(answers: dict[str, Any]) -> str:
    """Hash the exact bounded JSON answer object submitted to METTLE."""
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(answers)).hexdigest()


def _load_public_key(public_key_pem: str) -> Ed25519PublicKey:
    try:
        public_key = load_pem_public_key(public_key_pem.encode("ascii"))
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Presence public key must be valid PEM") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("Presence public key must use Ed25519")
    return public_key


def validate_public_key(public_key_pem: str) -> str:
    """Validate and normalize an Ed25519 SubjectPublicKeyInfo PEM value."""
    public_key = _load_public_key(public_key_pem)
    return public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")


def key_fingerprint(public_key_pem: str) -> str:
    """Return the SHA-256 fingerprint of the canonical public-key DER."""
    public_key = _load_public_key(public_key_pem)
    der = public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    return HASH_PREFIX + hashlib.sha256(der).hexdigest()


def new_session_presence(
    *, session_id: str, public_key_pem: str, audience: str
) -> dict[str, Any]:
    """Create private persisted state for a key-bound METTLE session."""
    normalized_key = validate_public_key(public_key_pem)
    fingerprint = key_fingerprint(normalized_key)
    nonce = secrets.token_urlsafe(32)
    genesis = canonical_bytes(
        {
            "protocol": PRESENCE_PROTOCOL,
            "session_id": session_id,
            "audience": audience,
            "key_fingerprint": fingerprint,
            "nonce": nonce,
        }
    )
    now_ms = int(time.time() * 1000)
    state = {
        "protocol": PRESENCE_PROTOCOL,
        "public_key_pem": normalized_key,
        "key_fingerprint": fingerprint,
        "audience": audience,
        "credential_jti": secrets.token_hex(16),
        "nonce": nonce,
        "transcript_hash": HASH_PREFIX + hashlib.sha256(genesis).hexdigest(),
        "sequence": 0,
        "started_at_unix_ms": now_ms,
        "nonce_issued_at_unix_ms": now_ms,
        "submissions": [],
    }
    state.update(new_continuity_state())
    return state


def public_session_presence(
    presence: dict[str, Any] | None, *, completed: bool = False
) -> dict[str, Any] | None:
    """Project private session presence state into its client-safe form."""
    if not presence:
        return None
    return {
        "protocol": presence["protocol"],
        "key_fingerprint": presence["key_fingerprint"],
        "audience": presence["audience"],
        "nonce": None if completed else presence["nonce"],
        "transcript_hash": presence["transcript_hash"],
        "sequence": presence["sequence"],
        "action": None if completed else presence.get("current_action"),
        "completed": completed,
        "continuity_protocol": presence.get("continuity_protocol"),
    }


def presence_state_signing_bytes(*, session_id: str, presence: dict[str, Any]) -> bytes:
    """Build the exact issuer-signed message for one public Presence state."""
    if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
        raise ValueError("Presence state session ID is invalid")
    if not isinstance(presence, dict):
        raise ValueError("Presence state must be an object")
    state = {key: value for key, value in presence.items() if key != "issuer_receipt"}
    return canonical_bytes(
        {
            "protocol": PRESENCE_PROTOCOL,
            "purpose": PRESENCE_STATE_RECEIPT_PURPOSE,
            "session_id": session_id,
            "state": state,
        }
    )


def issuer_signed_session_presence(
    presence: dict[str, Any] | None,
    *,
    session_id: str,
    completed: bool = False,
) -> dict[str, Any] | None:
    """Project and authenticate public Presence state with the issuer key."""
    state = public_session_presence(presence, completed=completed)
    if state is None:
        return None
    from mettle.signing import get_public_key_info, sign_attestation

    key_info = get_public_key_info()
    key_id = key_info.get("key_id")
    if key_info.get("available") is not True or not isinstance(key_id, str):
        raise RuntimeError("Presence state signing is unavailable")
    state["issuer_receipt"] = {
        "key_id": key_id,
        "algorithm": "Ed25519",
        "signature": sign_attestation(
            presence_state_signing_bytes(session_id=session_id, presence=state)
        ),
    }
    return state


def validate_credential_presence(presence: dict[str, Any]) -> None:
    """Reject corrupt internal state before the issuer signs a credential."""
    if not isinstance(presence, dict):
        raise ValueError("Presence credential state must be an object")
    required_strings = {
        "public_key_pem",
        "key_fingerprint",
        "audience",
        "credential_jti",
        "transcript_hash",
    }
    if presence.get("protocol") != PRESENCE_PROTOCOL or any(
        not isinstance(presence.get(field), str) or not presence[field]
        for field in required_strings
    ):
        raise ValueError("Presence credential state is incomplete")
    if key_fingerprint(presence["public_key_pem"]) != presence["key_fingerprint"]:
        raise ValueError("Presence credential key fingerprint is inconsistent")
    if re.fullmatch(r"[0-9a-f]{32}", presence["credential_jti"]) is None:
        raise ValueError("Presence credential JTI is invalid")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", presence["transcript_hash"]) is None:
        raise ValueError("Presence transcript hash is invalid")
    sequence = presence.get("sequence")
    submissions = presence.get("submissions")
    continuity_protocol = presence.get("continuity_protocol")
    if continuity_protocol not in {None, CONTINUITY_PROTOCOL}:
        raise ValueError("Presence credential continuity protocol is invalid")
    started_at = presence.get("started_at_unix_ms")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
        or not isinstance(submissions, list)
        or len(submissions) != sequence
        or isinstance(started_at, bool)
        or not isinstance(started_at, int)
        or started_at < 0
    ):
        raise ValueError("Presence credential sequence is invalid")
    challenge_ids: set[str] = set()
    previous_accepted_at = started_at
    for expected_sequence, submission in enumerate(submissions, start=1):
        response_time = (
            submission.get("response_time_ms") if isinstance(submission, dict) else None
        )
        accepted_at = (
            submission.get("accepted_at_unix_ms")
            if isinstance(submission, dict)
            else None
        )
        if not (
            isinstance(submission, dict)
            and not isinstance(submission.get("sequence"), bool)
            and submission.get("sequence") == expected_sequence
            and isinstance(submission.get("action"), str)
            and PRESENCE_ACTION_PATTERN.fullmatch(submission["action"]) is not None
            and not isinstance(response_time, bool)
            and isinstance(response_time, int)
            and response_time >= 0
            and not isinstance(accepted_at, bool)
            and isinstance(accepted_at, int)
            and accepted_at >= previous_accepted_at
            and response_time == accepted_at - previous_accepted_at
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(submission.get("transcript_hash", "")),
            )
            is not None
        ):
            raise ValueError("Presence credential submission history is invalid")
        if continuity_protocol is not None:
            challenge_id = submission.get("challenge_id")
            if (
                submission.get("challenge_family") != CONTINUITY_PROTOCOL
                or not isinstance(challenge_id, str)
                or re.fullmatch(r"[0-9a-f]{32}", challenge_id) is None
                or challenge_id in challenge_ids
            ):
                raise ValueError("Presence credential continuity history is invalid")
            challenge_ids.add(challenge_id)
        previous_accepted_at = accepted_at
    if submissions[-1]["transcript_hash"] != presence["transcript_hash"]:
        raise ValueError("Presence credential transcript commitment is inconsistent")


def submission_signing_bytes(
    *,
    session_id: str,
    action: str,
    nonce: str,
    previous_transcript_hash: str,
    payload_hash: str,
) -> bytes:
    """Build the exact message a session holder signs for one submission."""
    return canonical_bytes(
        {
            "protocol": PRESENCE_PROTOCOL,
            "purpose": "mettle-session-submission",
            "session_id": session_id,
            "action": action,
            "nonce": nonce,
            "previous_transcript_hash": previous_transcript_hash,
            "payload_hash": payload_hash,
        }
    )


def _decode_signature(signature: str) -> bytes:
    try:
        decoded = base64.b64decode(signature, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise ValueError("Presence signature must be valid base64") from exc
    if len(decoded) != 64:
        raise ValueError("Presence signature must be an Ed25519 signature")
    return decoded


def transcript_hash_after_submission(
    *, previous_transcript_hash: str, message: bytes, signature: str
) -> str:
    """Compute the next transcript commitment for a signed submission."""
    if re.fullmatch(r"sha256:[0-9a-f]{64}", previous_transcript_hash) is None:
        raise ValueError("Previous Presence transcript hash is invalid")
    transcript_material = (
        previous_transcript_hash.encode("ascii")
        + b"\x00"
        + message
        + b"\x00"
        + _decode_signature(signature)
    )
    return HASH_PREFIX + hashlib.sha256(transcript_material).hexdigest()


def verify_submission_proof(
    *,
    presence: dict[str, Any] | None,
    proof: dict[str, Any] | None,
    session_id: str,
    action: str,
    answers: dict[str, Any],
) -> bytes | None:
    """Verify a proof against current nonce, transcript, action, and answers."""
    if presence is None:
        if proof is not None:
            raise ValueError("Presence proof supplied for a legacy session")
        return None
    if proof is None:
        raise ValueError("Presence proof is required for this session")
    if action != presence.get("current_action"):
        raise ValueError("Presence action is not currently issued")
    if proof.get("nonce") != presence.get("nonce"):
        raise ValueError("Presence nonce is invalid or has already been used")
    if proof.get("previous_transcript_hash") != presence.get("transcript_hash"):
        raise ValueError("Presence transcript does not match current session state")

    message = submission_signing_bytes(
        session_id=session_id,
        action=action,
        nonce=proof["nonce"],
        previous_transcript_hash=proof["previous_transcript_hash"],
        payload_hash=answer_hash(answers),
    )
    public_key = _load_public_key(presence["public_key_pem"])
    try:
        public_key.verify(_decode_signature(proof["signature"]), message)
    except InvalidSignature as exc:
        raise ValueError("Presence signature is invalid") from exc
    return message


def advance_session_presence(
    *, presence: dict[str, Any], message: bytes, signature: str, action: str
) -> None:
    """Commit a verified submission and rotate the single-use nonce."""
    presence["transcript_hash"] = transcript_hash_after_submission(
        previous_transcript_hash=presence["transcript_hash"],
        message=message,
        signature=signature,
    )
    issued_at_ms = int(presence["nonce_issued_at_unix_ms"])
    accepted_at_ms = max(issued_at_ms, int(time.time() * 1000))
    response_time_ms = accepted_at_ms - issued_at_ms
    next_sequence = int(presence.get("sequence", 0)) + 1
    receipt = {
        "sequence": next_sequence,
        "action": action,
        "response_time_ms": response_time_ms,
        "accepted_at_unix_ms": accepted_at_ms,
        "transcript_hash": presence["transcript_hash"],
    }
    receipt.update(consume_continuity_evidence(presence))
    presence.setdefault("submissions", []).append(receipt)
    presence["nonce"] = secrets.token_urlsafe(32)
    presence["sequence"] = next_sequence
    presence["nonce_issued_at_unix_ms"] = accepted_at_ms


def presentation_signing_bytes(
    *,
    challenge_id: str,
    nonce: str,
    audience: str,
    credential_jti: str,
    expires_at: str,
) -> bytes:
    """Build the exact verifier challenge a credential holder signs."""
    try:
        if not isinstance(expires_at, str):
            raise ValueError("expiry is not a string")
        parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if parsed_expiry.tzinfo is None:
            raise ValueError("expiry is timezone-naive")
        normalized_expiry = (
            parsed_expiry.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Presentation challenge expiry is invalid") from exc
    return canonical_bytes(
        {
            "protocol": PRESENCE_PROTOCOL,
            "purpose": "mettle-credential-presentation",
            "challenge_id": challenge_id,
            "nonce": nonce,
            "audience": audience,
            "credential_jti": credential_jti,
            "expires_at": normalized_expiry,
        }
    )


def verify_holder_signature(
    *,
    public_key_pem: str,
    signature: str,
    challenge: dict[str, Any],
) -> None:
    """Verify that the bound holder signed one live presentation challenge."""
    try:
        raw_expiry = challenge["expires_at"]
        if not isinstance(raw_expiry, str):
            raise ValueError("expiry is not a string")
        expires_at = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Presentation challenge expiry is invalid") from exc
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        raise ValueError("Presentation challenge has expired")
    message = presentation_signing_bytes(
        challenge_id=challenge["challenge_id"],
        nonce=challenge["nonce"],
        audience=challenge["audience"],
        credential_jti=challenge["credential_jti"],
        expires_at=challenge["expires_at"],
    )
    try:
        _load_public_key(public_key_pem).verify(_decode_signature(signature), message)
    except InvalidSignature as exc:
        raise ValueError("Holder signature is invalid") from exc
