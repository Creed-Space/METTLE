"""Policy-enforcing holder SDK for distributed METTLE Presence clients.

The holder is intentionally narrower than a generic signing service. It binds
an injected signer to configured issuers and audiences, authorizes bounded
session state, enforces monotonic transcript progression, and registers only
issuer-verified credentials before signing presentation challenges.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Protocol
from urllib.parse import urlparse

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mettle.presence import (
    PRESENCE_PROTOCOL,
    key_fingerprint,
    presentation_signing_bytes,
    submission_signing_bytes,
    transcript_hash_after_submission,
    validate_public_key,
)
from mettle.vcp import verify_mettle_attestation


HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
ACTION_PATTERN = re.compile(r"(?:suite|round):[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
JTI_PATTERN = re.compile(r"[0-9a-f]{32}")


class HolderPolicyError(ValueError):
    """A signing request violated the holder's local security policy."""


class HolderSigner(Protocol):
    """Minimal interface implemented by in-memory, Keychain, HSM, or KMS signers."""

    @property
    def public_key_pem(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


class EphemeralEd25519Signer:
    """In-memory signer for tests and ephemeral clients, never persistent storage."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None) -> None:
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key_pem = (
            self._private_key.public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            .decode("ascii")
        )

    @property
    def public_key_pem(self) -> str:
        return self._public_key_pem

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(message)


@dataclass(frozen=True)
class HolderPolicy:
    """Static authorization policy for one holder service instance."""

    issuer_public_keys: dict[str, str]
    allowed_audiences: frozenset[str]
    max_active_sessions: int = 4
    max_actions_per_session: int = 16
    max_presentations_per_credential: int = 32
    max_presentation_ttl_seconds: int = 600


@dataclass
class _PendingSubmission:
    message: bytes
    signature: str
    action: str


@dataclass
class _Session:
    issuer: str
    session_id: str
    audience: str
    nonce: str | None
    transcript_hash: str
    sequence: int
    action: str | None
    completed: bool = False
    pending: _PendingSubmission | None = None
    committed_actions: set[str] = field(default_factory=set)


@dataclass
class _Credential:
    issuer: str
    session_id: str
    credential_jti: str
    audience: str
    transcript_hash: str
    sequence: int
    presentations: dict[str, tuple[bytes, str]] = field(default_factory=dict)


def _normalize_issuer(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise HolderPolicyError("Issuer must be a non-empty bounded URL")
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.query or parsed.fragment or not parsed.netloc:
        raise HolderPolicyError("Issuer URL must not contain a query or fragment")
    if parsed.scheme == "https":
        return normalized
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return normalized
    raise HolderPolicyError("Issuer must use HTTPS unless it is loopback")


def _bounded_text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise HolderPolicyError(f"{name} must be a non-empty bounded string")
    return value


def _hash(value: Any, name: str) -> str:
    text = _bounded_text(value, name, maximum=71)
    if HASH_PATTERN.fullmatch(text) is None:
        raise HolderPolicyError(f"{name} must be a SHA-256 protocol hash")
    return text


def _nonce(value: Any) -> str:
    text = _bounded_text(value, "nonce")
    if len(text) < 32:
        raise HolderPolicyError("Nonce must contain at least 32 characters")
    return text


def _action(value: Any) -> str:
    text = _bounded_text(value, "action", maximum=134)
    if ACTION_PATTERN.fullmatch(text) is None:
        raise HolderPolicyError("Action is not a supported Presence action")
    return text


class PresenceHolder:
    """Stateful signing boundary for autonomous distributed Presence clients."""

    def __init__(self, signer: HolderSigner, policy: HolderPolicy) -> None:
        if policy.max_active_sessions < 1 or policy.max_active_sessions > 1024:
            raise HolderPolicyError("Active-session budget must be between 1 and 1024")
        if policy.max_actions_per_session < 1 or policy.max_actions_per_session > 1024:
            raise HolderPolicyError(
                "Per-session action budget must be between 1 and 1024"
            )
        if (
            policy.max_presentations_per_credential < 1
            or policy.max_presentations_per_credential > 10000
        ):
            raise HolderPolicyError("Presentation budget must be between 1 and 10000")
        if (
            policy.max_presentation_ttl_seconds < 1
            or policy.max_presentation_ttl_seconds > 3600
        ):
            raise HolderPolicyError(
                "Presentation TTL must be between 1 and 3600 seconds"
            )
        if not policy.issuer_public_keys:
            raise HolderPolicyError("At least one trusted issuer key is required")
        if not policy.allowed_audiences:
            raise HolderPolicyError("At least one audience must be allowed")

        self._signer = signer
        self._public_key_pem = validate_public_key(signer.public_key_pem)
        self._key_fingerprint = key_fingerprint(self._public_key_pem)
        self._issuer_keys = {
            _normalize_issuer(issuer): validate_public_key(public_key)
            for issuer, public_key in policy.issuer_public_keys.items()
        }
        self._allowed_audiences = frozenset(
            _bounded_text(audience, "audience") for audience in policy.allowed_audiences
        )
        self._policy = policy
        self._sessions: dict[str, _Session] = {}
        self._credentials: dict[str, _Credential] = {}
        self._presentation_ids: dict[str, tuple[bytes, str]] = {}
        self._lock = RLock()

    @property
    def public_key_pem(self) -> str:
        return self._public_key_pem

    @property
    def key_fingerprint(self) -> str:
        return self._key_fingerprint

    def authorize_session(
        self, *, issuer: str, session_id: str, presence: dict[str, Any]
    ) -> None:
        """Authorize exactly one server-created initial Presence state."""
        normalized_issuer = _normalize_issuer(issuer)
        session_id = _bounded_text(session_id, "session_id")
        if normalized_issuer not in self._issuer_keys:
            raise HolderPolicyError("Issuer is not trusted by this holder")
        if not isinstance(presence, dict):
            raise HolderPolicyError("Presence state must be an object")
        audience = _bounded_text(presence.get("audience"), "audience")
        if audience not in self._allowed_audiences:
            raise HolderPolicyError("Audience is not allowed by this holder")
        if presence.get("protocol") != PRESENCE_PROTOCOL:
            raise HolderPolicyError("Presence protocol is unsupported")
        if presence.get("key_fingerprint") != self._key_fingerprint:
            raise HolderPolicyError("Presence session is bound to a different key")
        if presence.get("sequence") != 0 or presence.get("completed") is not False:
            raise HolderPolicyError(
                "Only an uncompleted initial Presence state is allowed"
            )
        nonce = _nonce(presence.get("nonce"))
        transcript_hash = _hash(presence.get("transcript_hash"), "transcript_hash")
        action = _action(presence.get("action"))

        with self._lock:
            if session_id in self._sessions:
                raise HolderPolicyError("Session has already been authorized")
            active_count = sum(
                not session.completed for session in self._sessions.values()
            )
            if active_count >= self._policy.max_active_sessions:
                raise HolderPolicyError("Active-session budget is exhausted")
            self._sessions[session_id] = _Session(
                issuer=normalized_issuer,
                session_id=session_id,
                audience=audience,
                nonce=nonce,
                transcript_hash=transcript_hash,
                sequence=0,
                action=action,
            )

    def sign_submission(
        self,
        *,
        session_id: str,
        action: str,
        nonce: str,
        previous_transcript_hash: str,
        payload_hash: str,
    ) -> str:
        """Sign only the next action in one authorized monotonic session."""
        session_id = _bounded_text(session_id, "session_id")
        action = _action(action)
        nonce = _nonce(nonce)
        previous_transcript_hash = _hash(
            previous_transcript_hash, "previous_transcript_hash"
        )
        payload_hash = _hash(payload_hash, "payload_hash")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HolderPolicyError("Session is not authorized")
            if session.completed:
                raise HolderPolicyError("Session is already complete")
            if session.sequence >= self._policy.max_actions_per_session:
                raise HolderPolicyError("Per-session action budget is exhausted")
            if action != session.action:
                raise HolderPolicyError("Action does not match holder session state")
            if action in session.committed_actions:
                raise HolderPolicyError("Action has already been committed")
            if nonce != session.nonce:
                raise HolderPolicyError("Nonce does not match holder session state")
            if previous_transcript_hash != session.transcript_hash:
                raise HolderPolicyError(
                    "Transcript does not match holder session state"
                )
            message = submission_signing_bytes(
                session_id=session_id,
                action=action,
                nonce=nonce,
                previous_transcript_hash=previous_transcript_hash,
                payload_hash=payload_hash,
            )
            if session.pending is not None:
                if session.pending.message == message:
                    return session.pending.signature
                raise HolderPolicyError(
                    "A different submission is already pending for this session"
                )
            signature_bytes = self._signer.sign(message)
            if not isinstance(signature_bytes, bytes) or len(signature_bytes) != 64:
                raise HolderPolicyError("Signer did not return an Ed25519 signature")
            signature = base64.b64encode(signature_bytes).decode("ascii")
            session.pending = _PendingSubmission(
                message=message,
                signature=signature,
                action=action,
            )
            return signature

    def commit_submission(self, *, session_id: str, presence: dict[str, Any]) -> None:
        """Advance only when returned Presence state matches the signed transition."""
        session_id = _bounded_text(session_id, "session_id")
        if not isinstance(presence, dict):
            raise HolderPolicyError("Presence state must be an object")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HolderPolicyError("Session is not authorized")
            pending = session.pending
            if pending is None:
                raise HolderPolicyError("Session has no pending submission")
            if presence.get("protocol") != PRESENCE_PROTOCOL:
                raise HolderPolicyError("Presence protocol is unsupported")
            if presence.get("key_fingerprint") != self._key_fingerprint:
                raise HolderPolicyError("Returned state is bound to a different key")
            if presence.get("audience") != session.audience:
                raise HolderPolicyError("Returned state changed the audience")
            expected_sequence = session.sequence + 1
            if presence.get("sequence") != expected_sequence:
                raise HolderPolicyError("Returned state did not advance exactly once")
            returned_hash = _hash(presence.get("transcript_hash"), "transcript_hash")
            expected_hash = transcript_hash_after_submission(
                previous_transcript_hash=session.transcript_hash,
                message=pending.message,
                signature=pending.signature,
            )
            if returned_hash != expected_hash:
                raise HolderPolicyError(
                    "Returned state has an invalid transcript transition"
                )
            completed = presence.get("completed") is True
            if completed:
                if (
                    presence.get("nonce") is not None
                    or presence.get("action") is not None
                ):
                    raise HolderPolicyError(
                        "Completed state must not issue another action"
                    )
                next_nonce = None
                next_action = None
            else:
                if expected_sequence >= self._policy.max_actions_per_session:
                    raise HolderPolicyError("Returned state exceeds the action budget")
                next_nonce = _nonce(presence.get("nonce"))
                if next_nonce == session.nonce:
                    raise HolderPolicyError("Returned state did not rotate its nonce")
                next_action = _action(presence.get("action"))
                if (
                    next_action in session.committed_actions
                    or next_action == pending.action
                ):
                    raise HolderPolicyError(
                        "Returned state repeated a committed action"
                    )
            session.committed_actions.add(pending.action)
            session.nonce = next_nonce
            session.transcript_hash = returned_hash
            session.sequence = expected_sequence
            session.action = next_action
            session.completed = completed
            session.pending = None

    def register_credential(self, *, issuer: str, attestation: dict[str, Any]) -> str:
        """Register an issuer-verified credential matching a completed session."""
        normalized_issuer = _normalize_issuer(issuer)
        issuer_key = self._issuer_keys.get(normalized_issuer)
        if issuer_key is None:
            raise HolderPolicyError("Issuer is not trusted by this holder")
        if not isinstance(attestation, dict) or not verify_mettle_attestation(
            attestation, issuer_key
        ):
            raise HolderPolicyError("Credential issuer signature or policy is invalid")
        metadata = attestation.get("metadata")
        if not isinstance(metadata, dict):
            raise HolderPolicyError("Credential metadata is missing")
        proof = metadata.get("proof_of_possession")
        if not isinstance(proof, dict):
            raise HolderPolicyError("Credential is not bound to a holder key")
        session_id = _bounded_text(metadata.get("session_id"), "session_id")
        credential_jti = _bounded_text(metadata.get("jti"), "credential_jti")
        if JTI_PATTERN.fullmatch(credential_jti) is None:
            raise HolderPolicyError("Credential JTI is invalid")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.completed:
                raise HolderPolicyError(
                    "Credential session is not complete in holder state"
                )
            if session.issuer != normalized_issuer:
                raise HolderPolicyError("Credential issuer does not match the session")
            if metadata.get("audience") != session.audience:
                raise HolderPolicyError(
                    "Credential audience does not match the session"
                )
            if proof.get("public_key_pem") != self._public_key_pem:
                raise HolderPolicyError("Credential is bound to a different public key")
            if proof.get("key_fingerprint") != self._key_fingerprint:
                raise HolderPolicyError("Credential key fingerprint does not match")
            if proof.get("transcript_hash") != session.transcript_hash:
                raise HolderPolicyError(
                    "Credential transcript does not match holder state"
                )
            if proof.get("sequence") != session.sequence:
                raise HolderPolicyError(
                    "Credential sequence does not match holder state"
                )
            existing = self._credentials.get(credential_jti)
            if existing is not None and existing.session_id != session_id:
                raise HolderPolicyError(
                    "Credential JTI is already bound to another session"
                )
            self._credentials.setdefault(
                credential_jti,
                _Credential(
                    issuer=normalized_issuer,
                    session_id=session_id,
                    credential_jti=credential_jti,
                    audience=session.audience,
                    transcript_hash=session.transcript_hash,
                    sequence=session.sequence,
                ),
            )
            return credential_jti

    def sign_presentation(
        self,
        *,
        challenge_id: str,
        nonce: str,
        audience: str,
        credential_jti: str,
        expires_at: str,
    ) -> str:
        """Sign one bounded challenge for an issuer-verified registered credential."""
        challenge_id = _bounded_text(challenge_id, "challenge_id")
        nonce = _nonce(nonce)
        audience = _bounded_text(audience, "audience")
        credential_jti = _bounded_text(credential_jti, "credential_jti")
        expires_at = _bounded_text(expires_at, "expires_at")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HolderPolicyError("Presentation expiry is invalid") from exc
        now = datetime.now(timezone.utc)
        if expiry.tzinfo is None or expiry <= now:
            raise HolderPolicyError("Presentation challenge has expired")
        if (
            expiry.astimezone(timezone.utc) - now
        ).total_seconds() > self._policy.max_presentation_ttl_seconds:
            raise HolderPolicyError("Presentation challenge lifetime exceeds policy")
        with self._lock:
            credential = self._credentials.get(credential_jti)
            if credential is None:
                raise HolderPolicyError("Credential is not registered with this holder")
            if (
                audience != credential.audience
                or audience not in self._allowed_audiences
            ):
                raise HolderPolicyError("Presentation audience is not allowed")
            message = presentation_signing_bytes(
                challenge_id=challenge_id,
                nonce=nonce,
                audience=audience,
                credential_jti=credential_jti,
                expires_at=expires_at,
            )
            previous = self._presentation_ids.get(challenge_id)
            if previous is not None:
                if previous[0] == message:
                    return previous[1]
                raise HolderPolicyError(
                    "Presentation challenge ID was reused inconsistently"
                )
            if (
                len(credential.presentations)
                >= self._policy.max_presentations_per_credential
            ):
                raise HolderPolicyError("Credential presentation budget is exhausted")
            signature_bytes = self._signer.sign(message)
            if not isinstance(signature_bytes, bytes) or len(signature_bytes) != 64:
                raise HolderPolicyError("Signer did not return an Ed25519 signature")
            signature = base64.b64encode(signature_bytes).decode("ascii")
            credential.presentations[challenge_id] = (message, signature)
            self._presentation_ids[challenge_id] = (message, signature)
            return signature

    def status(self) -> dict[str, Any]:
        """Return non-secret policy and lifecycle counters for operations."""
        with self._lock:
            return {
                "key_fingerprint": self._key_fingerprint,
                "trusted_issuers": sorted(self._issuer_keys),
                "allowed_audiences": sorted(self._allowed_audiences),
                "sessions": len(self._sessions),
                "active_sessions": sum(
                    not session.completed for session in self._sessions.values()
                ),
                "credentials": len(self._credentials),
            }
