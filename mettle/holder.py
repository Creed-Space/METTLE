"""Policy-enforcing holder SDK for distributed METTLE Presence clients.

The holder is intentionally narrower than a generic signing service. It binds
an injected signer to configured issuers and audiences, authorizes bounded
session state, enforces monotonic transcript progression, and registers only
issuer-verified credentials before signing presentation challenges.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import stat
import subprocess  # nosec B404
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

from mettle.presence import (
    PRESENCE_PROTOCOL,
    key_fingerprint,
    presence_state_signing_bytes,
    presentation_signing_bytes,
    submission_signing_bytes,
    transcript_hash_after_submission,
    validate_public_key,
)
from mettle.signing import verify_signature
from mettle.vcp import verify_mettle_attestation


HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
ACTION_PATTERN = re.compile(r"(?:suite|round):[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
JTI_PATTERN = re.compile(r"[0-9a-f]{32}")
ISSUER_KEY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
HOLDER_STATE_SCHEMA = "mettle-holder-state-v1"


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
    issuer_public_keyrings: dict[str, dict[str, str]] = field(default_factory=dict)
    max_active_sessions: int = 4
    max_actions_per_session: int = 16
    max_presentations_per_credential: int = 32
    max_presentation_ttl_seconds: int = 600
    max_session_records: int = 4096
    max_credentials: int = 4096
    max_presentation_records: int = 100000


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


def _issuer_key_id(value: Any, name: str) -> str:
    text = _bounded_text(value, name, maximum=128)
    if ISSUER_KEY_ID_PATTERN.fullmatch(text) is None:
        raise HolderPolicyError(f"{name} is invalid")
    return text


def _timeout(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or value > 60
    ):
        raise HolderPolicyError(f"{name} must be greater than 0 and at most 60 seconds")
    return float(value)


def _state_bytes(value: Any, name: str, *, maximum: int = 1048576) -> bytes:
    text = _bounded_text(value, name, maximum=((maximum + 2) // 3) * 4)
    try:
        decoded = base64.b64decode(text, validate=True)
    except ValueError:
        raise HolderPolicyError(f"{name} is invalid") from None
    if not decoded or len(decoded) > maximum:
        raise HolderPolicyError(f"{name} is empty or oversized")
    return decoded


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


class MacOSKeychainEd25519Signer:
    """Load an Ed25519 key from macOS Keychain without plaintext key files.

    The PEM is retrieved once through the fixed ``security`` executable, parsed
    into a cryptography key object, and never retained as an attribute. The key
    remains process-readable memory, so deployments requiring non-exportable
    keys should provide an HSM or KMS implementation of ``HolderSigner``.
    """

    def __init__(
        self,
        *,
        service: str,
        account: str,
        security_binary: str = "/usr/bin/security",
        timeout_seconds: float = 5.0,
    ) -> None:
        pem = _read_macos_keychain_value(
            service=service,
            account=account,
            security_binary=security_binary,
            timeout_seconds=timeout_seconds,
            value_name="private key",
        )
        if not isinstance(pem, bytes) or not pem or len(pem) > 32768:
            raise HolderPolicyError("Keychain private key is empty or oversized")
        candidates = [pem]
        encoded = pem.strip()
        if (
            len(encoded) % 2 == 0
            and re.fullmatch(rb"[0-9a-fA-F]+", encoded) is not None
        ):
            candidates.append(bytes.fromhex(encoded.decode("ascii")))
        private_key: Any = None
        for candidate in candidates:
            try:
                private_key = load_pem_private_key(candidate, password=None)
                break
            except (TypeError, ValueError):
                continue
        if private_key is None:
            raise HolderPolicyError("Keychain item is not a valid private key")
        if not isinstance(private_key, Ed25519PrivateKey):
            raise HolderPolicyError("Keychain private key must use Ed25519")
        self._private_key = private_key
        self._public_key_pem = (
            private_key.public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            .decode("ascii")
        )

    @property
    def public_key_pem(self) -> str:
        return self._public_key_pem

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(message)


def _read_macos_keychain_value(
    *,
    service: str,
    account: str,
    security_binary: str,
    timeout_seconds: float,
    value_name: str,
) -> bytes:
    service = _bounded_text(service, "Keychain service")
    account = _bounded_text(account, "Keychain account")
    if "\x00" in service or "\x00" in account:
        raise HolderPolicyError("Keychain identifiers must not contain NUL")
    binary = Path(security_binary)
    if not binary.is_absolute():
        raise HolderPolicyError("Keychain security executable must be absolute")
    timeout_seconds = _timeout(timeout_seconds, "Keychain timeout")
    command = [
        str(binary),
        "find-generic-password",
        "-s",
        service,
        "-a",
        account,
        "-w",
    ]
    try:
        completed = subprocess.run(  # nosec B603
            command,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        raise HolderPolicyError(f"Keychain {value_name} lookup failed") from None
    if completed.returncode != 0:
        raise HolderPolicyError(f"Keychain {value_name} lookup failed")
    return completed.stdout


class MacOSKeychainSecretProvider:
    """Fetch a short-lived service token from Keychain only when it is needed."""

    def __init__(
        self,
        *,
        service: str,
        account: str,
        security_binary: str = "/usr/bin/security",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._service = _bounded_text(service, "Keychain service")
        self._account = _bounded_text(account, "Keychain account")
        self._security_binary = str(Path(security_binary))
        self._timeout_seconds = _timeout(timeout_seconds, "Keychain timeout")
        if not Path(self._security_binary).is_absolute():
            raise HolderPolicyError("Keychain security executable must be absolute")

    def __call__(self) -> str:
        raw = _read_macos_keychain_value(
            service=self._service,
            account=self._account,
            security_binary=self._security_binary,
            timeout_seconds=self._timeout_seconds,
            value_name="secret",
        )
        if not raw or len(raw) > 8192:
            raise HolderPolicyError("Keychain secret is empty or oversized")
        try:
            secret = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise HolderPolicyError("Keychain secret is not UTF-8") from None
        if not secret or len(secret) > 4096 or "\r" in secret or "\n" in secret:
            raise HolderPolicyError("Keychain secret is invalid")
        return secret


class FileSecretProvider:
    """Read a bounded secret from an owner-only regular file on every use."""

    def __init__(self, path: str, *, maximum_bytes: int = 8192) -> None:
        candidate = Path(_bounded_text(path, "secret file path", maximum=4096))
        if not candidate.is_absolute():
            raise HolderPolicyError("Secret file path must be absolute")
        if maximum_bytes < 1 or maximum_bytes > 1048576:
            raise HolderPolicyError("Secret file size limit is invalid")
        self._path = candidate
        self._maximum_bytes = maximum_bytes

    def __call__(self) -> str:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._path, flags)
        except OSError:
            raise HolderPolicyError("Secret file lookup failed") from None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
                or metadata.st_size < 1
                or metadata.st_size > self._maximum_bytes
            ):
                raise HolderPolicyError("Secret file permissions or size are invalid")
            raw = os.read(descriptor, self._maximum_bytes + 1)
        finally:
            os.close(descriptor)
        if not raw or len(raw) > self._maximum_bytes:
            raise HolderPolicyError("Secret file is empty or oversized")
        try:
            secret = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise HolderPolicyError("Secret file is not UTF-8") from None
        if not secret or "\r" in secret or "\n" in secret:
            raise HolderPolicyError("Secret file value is invalid")
        return secret


class VaultTransitEd25519Signer:
    """Sign through a non-exportable HashiCorp Vault Transit Ed25519 key.

    Only the public key and the token provider are retained locally. Vault's
    signature is verified against the pinned public key before it is returned.
    """

    _SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
    _SIGNATURE_PATTERN = re.compile(r"vault:v([1-9][0-9]*):([A-Za-z0-9+/]+={0,2})")

    def __init__(
        self,
        *,
        base_url: str,
        mount_path: str,
        key_name: str,
        public_key_pem: str,
        token_provider: Callable[[], str],
        key_version: int | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        base_url = _bounded_text(base_url, "Vault base URL", maximum=2048).rstrip("/")
        parsed = urlparse(base_url)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or not parsed.netloc
            or (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback))
        ):
            raise HolderPolicyError(
                "Vault base URL must be an HTTPS origin unless it is loopback"
            )
        for value, name in ((mount_path, "Vault mount"), (key_name, "Vault key")):
            if (
                not isinstance(value, str)
                or self._SEGMENT_PATTERN.fullmatch(value) is None
            ):
                raise HolderPolicyError(f"{name} name is invalid")
        if not callable(token_provider):
            raise HolderPolicyError("Vault token provider must be callable")
        if key_version is not None and (
            isinstance(key_version, bool)
            or not isinstance(key_version, int)
            or key_version < 1
            or key_version > 2147483647
        ):
            raise HolderPolicyError("Vault key version is invalid")
        timeout_seconds = _timeout(timeout_seconds, "Vault timeout")
        self._public_key_pem = validate_public_key(public_key_pem)
        self._token_provider = token_provider
        self._key_version = key_version
        self._timeout_seconds = timeout_seconds
        self._sign_url = f"{base_url}/v1/{mount_path}/sign/{key_name}"

    @property
    def public_key_pem(self) -> str:
        return self._public_key_pem

    def sign(self, message: bytes) -> bytes:
        if not isinstance(message, bytes) or not message or len(message) > 1048576:
            raise HolderPolicyError("Vault signing message is empty or oversized")
        try:
            token = self._token_provider()
        except Exception:
            raise HolderPolicyError(
                "Vault authentication token lookup failed"
            ) from None
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 4096
            or "\r" in token
            or "\n" in token
        ):
            raise HolderPolicyError("Vault authentication token is invalid")
        try:
            request_body: dict[str, str | int] = {
                "input": base64.b64encode(message).decode("ascii")
            }
            if self._key_version is not None:
                request_body["key_version"] = self._key_version
            with httpx.stream(
                "POST",
                self._sign_url,
                headers={"X-Vault-Token": token},
                json=request_body,
                timeout=self._timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            ) as response:
                if response.status_code != 200:
                    raise HolderPolicyError("Vault signing request failed")
                chunks: list[bytes] = []
                response_size = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    response_size += len(chunk)
                    if response_size > 32768:
                        raise HolderPolicyError("Vault signing request failed")
                    chunks.append(chunk)
                response_content = b"".join(chunks)
        except httpx.HTTPError:
            raise HolderPolicyError("Vault signing request failed") from None
        try:
            payload = json.loads(response_content)
            encoded_signature = payload["data"]["signature"]
        except (KeyError, TypeError, ValueError):
            raise HolderPolicyError("Vault signing response is invalid") from None
        if not isinstance(encoded_signature, str):
            raise HolderPolicyError("Vault signing response is invalid")
        matched = self._SIGNATURE_PATTERN.fullmatch(encoded_signature)
        if matched is None:
            raise HolderPolicyError("Vault signing response is invalid")
        response_version = int(matched.group(1))
        if self._key_version is not None and response_version != self._key_version:
            raise HolderPolicyError(
                "Vault signing response used an unexpected key version"
            )
        signature_b64 = matched.group(2)
        try:
            signature = base64.b64decode(signature_b64, validate=True)
        except ValueError:
            raise HolderPolicyError("Vault signing response is invalid") from None
        if len(signature) != 64 or not verify_signature(
            self._public_key_pem, message, signature_b64
        ):
            raise HolderPolicyError("Vault returned an invalid signature")
        return signature


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
        if policy.max_session_records < 1 or policy.max_session_records > 100000:
            raise HolderPolicyError(
                "Session record budget must be between 1 and 100000"
            )
        if policy.max_credentials < 1 or policy.max_credentials > 100000:
            raise HolderPolicyError("Credential budget must be between 1 and 100000")
        if (
            policy.max_presentation_records < 1
            or policy.max_presentation_records > 1000000
        ):
            raise HolderPolicyError(
                "Presentation record budget must be between 1 and 1000000"
            )
        if not policy.issuer_public_keys and not policy.issuer_public_keyrings:
            raise HolderPolicyError("At least one trusted issuer key is required")
        if not policy.allowed_audiences:
            raise HolderPolicyError("At least one audience must be allowed")

        self._signer = signer
        self._public_key_pem = validate_public_key(signer.public_key_pem)
        self._key_fingerprint = key_fingerprint(self._public_key_pem)
        self._issuer_keys: dict[str, dict[str, str]] = {}
        for issuer, public_key in policy.issuer_public_keys.items():
            normalized_issuer = _normalize_issuer(issuer)
            if normalized_issuer in self._issuer_keys:
                raise HolderPolicyError("Issuer is configured more than once")
            self._issuer_keys[normalized_issuer] = {
                "*": validate_public_key(public_key)
            }
        for issuer, keyring in policy.issuer_public_keyrings.items():
            normalized_issuer = _normalize_issuer(issuer)
            if normalized_issuer in self._issuer_keys:
                raise HolderPolicyError(
                    "Issuer cannot use both a legacy key and a keyed trust ring"
                )
            if not isinstance(keyring, dict) or not keyring:
                raise HolderPolicyError("Issuer trust ring must not be empty")
            normalized_ring: dict[str, str] = {}
            for key_id, public_key in keyring.items():
                key_id = _issuer_key_id(key_id, "issuer key_id")
                normalized_ring[key_id] = validate_public_key(public_key)
            self._issuer_keys[normalized_issuer] = normalized_ring
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

    @staticmethod
    def _verify_presence_state_receipt(
        *, issuer_keys: dict[str, str], session_id: str, presence: dict[str, Any]
    ) -> str:
        receipt = presence.get("issuer_receipt")
        if not isinstance(receipt, dict):
            raise HolderPolicyError("Presence state issuer receipt is required")
        if receipt.get("algorithm") != "Ed25519":
            raise HolderPolicyError("Presence state receipt algorithm is unsupported")
        key_id = _issuer_key_id(receipt.get("key_id"), "receipt key_id")
        issuer_key = issuer_keys.get(key_id) or issuer_keys.get("*")
        if issuer_key is None:
            raise HolderPolicyError("Presence state issuer key is not trusted")
        signature = _bounded_text(
            receipt.get("signature"), "receipt signature", maximum=128
        )
        try:
            message = presence_state_signing_bytes(
                session_id=session_id, presence=presence
            )
        except (TypeError, ValueError) as exc:
            raise HolderPolicyError(
                "Presence state receipt payload is invalid"
            ) from exc
        if not verify_signature(issuer_key, message, signature):
            raise HolderPolicyError("Presence state issuer receipt is invalid")
        return key_id

    def authorize_session(
        self, *, issuer: str, session_id: str, presence: dict[str, Any]
    ) -> None:
        """Authorize exactly one server-created initial Presence state."""
        normalized_issuer = _normalize_issuer(issuer)
        session_id = _bounded_text(session_id, "session_id")
        issuer_keys = self._issuer_keys.get(normalized_issuer)
        if issuer_keys is None:
            raise HolderPolicyError("Issuer is not trusted by this holder")
        if not isinstance(presence, dict):
            raise HolderPolicyError("Presence state must be an object")
        self._verify_presence_state_receipt(
            issuer_keys=issuer_keys,
            session_id=session_id,
            presence=presence,
        )
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
            if len(self._sessions) >= self._policy.max_session_records:
                raise HolderPolicyError("Session record budget is exhausted")
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
            self._verify_presence_state_receipt(
                issuer_keys=self._issuer_keys[session.issuer],
                session_id=session_id,
                presence=presence,
            )
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
        issuer_keys = self._issuer_keys.get(normalized_issuer)
        if issuer_keys is None:
            raise HolderPolicyError("Issuer is not trusted by this holder")
        if not isinstance(attestation, dict):
            raise HolderPolicyError("Credential issuer signature or policy is invalid")
        key_id = _issuer_key_id(attestation.get("auditor_key_id"), "auditor_key_id")
        issuer_key = issuer_keys.get(key_id) or issuer_keys.get("*")
        if issuer_key is None:
            raise HolderPolicyError("Credential issuer key is not trusted")
        if not verify_mettle_attestation(attestation, issuer_key):
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
            if (
                existing is None
                and len(self._credentials) >= self._policy.max_credentials
            ):
                raise HolderPolicyError("Credential budget is exhausted")
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
            if len(self._presentation_ids) >= self._policy.max_presentation_records:
                raise HolderPolicyError("Presentation record budget is exhausted")
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
                "presentations": len(self._presentation_ids),
            }

    def export_state(self) -> dict[str, Any]:
        """Export public protocol state for authenticated durable storage.

        The snapshot contains no private key, Vault token, or service credential.
        Callers must authenticate the serialized snapshot before storing it.
        """
        with self._lock:
            sessions = []
            for session_id in sorted(self._sessions):
                session = self._sessions[session_id]
                pending = None
                if session.pending is not None:
                    pending = {
                        "message": base64.b64encode(session.pending.message).decode(
                            "ascii"
                        ),
                        "signature": session.pending.signature,
                        "action": session.pending.action,
                    }
                sessions.append(
                    {
                        "issuer": session.issuer,
                        "session_id": session.session_id,
                        "audience": session.audience,
                        "nonce": session.nonce,
                        "transcript_hash": session.transcript_hash,
                        "sequence": session.sequence,
                        "action": session.action,
                        "completed": session.completed,
                        "pending": pending,
                        "committed_actions": sorted(session.committed_actions),
                    }
                )

            credentials = []
            for credential_jti in sorted(self._credentials):
                credential = self._credentials[credential_jti]
                presentations = [
                    {
                        "challenge_id": challenge_id,
                        "message": base64.b64encode(message).decode("ascii"),
                        "signature": signature,
                    }
                    for challenge_id, (message, signature) in sorted(
                        credential.presentations.items()
                    )
                ]
                credentials.append(
                    {
                        "issuer": credential.issuer,
                        "session_id": credential.session_id,
                        "credential_jti": credential.credential_jti,
                        "audience": credential.audience,
                        "transcript_hash": credential.transcript_hash,
                        "sequence": credential.sequence,
                        "presentations": presentations,
                    }
                )
            return {
                "schema": HOLDER_STATE_SCHEMA,
                "key_fingerprint": self._key_fingerprint,
                "sessions": sessions,
                "credentials": credentials,
            }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore one authenticated snapshot into an empty holder instance."""
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != HOLDER_STATE_SCHEMA
        ):
            raise HolderPolicyError("Holder state schema is invalid")
        if snapshot.get("key_fingerprint") != self._key_fingerprint:
            raise HolderPolicyError("Holder state belongs to a different signing key")
        raw_sessions = snapshot.get("sessions")
        raw_credentials = snapshot.get("credentials")
        if not isinstance(raw_sessions, list) or not isinstance(raw_credentials, list):
            raise HolderPolicyError("Holder state collections are invalid")
        if len(raw_sessions) > self._policy.max_session_records:
            raise HolderPolicyError("Holder state exceeds the session record budget")
        if len(raw_credentials) > self._policy.max_credentials:
            raise HolderPolicyError("Holder state exceeds the credential budget")

        sessions: dict[str, _Session] = {}
        credentials: dict[str, _Credential] = {}
        presentation_ids: dict[str, tuple[bytes, str]] = {}
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                raise HolderPolicyError("Holder session state is invalid")
            issuer = _normalize_issuer(
                _bounded_text(raw.get("issuer"), "issuer", maximum=512)
            )
            if issuer not in self._issuer_keys:
                raise HolderPolicyError("Holder state contains an untrusted issuer")
            session_id = _bounded_text(raw.get("session_id"), "session_id")
            if session_id in sessions:
                raise HolderPolicyError("Holder state repeats a session")
            audience = _bounded_text(raw.get("audience"), "audience")
            if audience not in self._allowed_audiences:
                raise HolderPolicyError("Holder state contains a disallowed audience")
            sequence = raw.get("sequence")
            completed = raw.get("completed")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or sequence > self._policy.max_actions_per_session
                or not isinstance(completed, bool)
            ):
                raise HolderPolicyError("Holder session lifecycle is invalid")
            if not completed and sequence >= self._policy.max_actions_per_session:
                raise HolderPolicyError(
                    "Active holder session exceeds its action budget"
                )
            nonce = raw.get("nonce")
            action = raw.get("action")
            if completed:
                if (
                    nonce is not None
                    or action is not None
                    or raw.get("pending") is not None
                ):
                    raise HolderPolicyError("Completed holder session state is invalid")
            else:
                nonce = _nonce(nonce)
                action = _action(action)
            transcript_hash = _hash(raw.get("transcript_hash"), "transcript_hash")
            raw_actions = raw.get("committed_actions")
            if not isinstance(raw_actions, list) or len(raw_actions) != sequence:
                raise HolderPolicyError("Committed holder actions are invalid")
            committed_actions = {_action(value) for value in raw_actions}
            if len(committed_actions) != len(raw_actions):
                raise HolderPolicyError("Committed holder actions are duplicated")
            if action is not None and action in committed_actions:
                raise HolderPolicyError("Current holder action was already committed")
            pending = None
            raw_pending = raw.get("pending")
            if raw_pending is not None:
                if completed or not isinstance(raw_pending, dict):
                    raise HolderPolicyError("Pending holder submission is invalid")
                pending_nonce = _nonce(nonce)
                pending_action = _action(raw_pending.get("action"))
                if pending_action != action:
                    raise HolderPolicyError("Pending holder action is invalid")
                pending_message = _state_bytes(
                    raw_pending.get("message"), "pending message"
                )
                pending_signature = _bounded_text(
                    raw_pending.get("signature"), "pending signature", maximum=128
                )
                if not verify_signature(
                    self._public_key_pem, pending_message, pending_signature
                ):
                    raise HolderPolicyError("Pending holder signature is invalid")
                try:
                    pending_payload = json.loads(pending_message)
                    if (
                        not isinstance(pending_payload, dict)
                        or pending_payload.get("protocol") != PRESENCE_PROTOCOL
                        or pending_payload.get("purpose") != "mettle-session-submission"
                        or pending_payload.get("session_id") != session_id
                        or pending_payload.get("action") != action
                        or pending_payload.get("nonce") != pending_nonce
                        or pending_payload.get("previous_transcript_hash")
                        != transcript_hash
                        or submission_signing_bytes(
                            session_id=session_id,
                            action=action,
                            nonce=pending_nonce,
                            previous_transcript_hash=transcript_hash,
                            payload_hash=_hash(
                                pending_payload.get("payload_hash"), "payload_hash"
                            ),
                        )
                        != pending_message
                    ):
                        raise HolderPolicyError("Pending holder message is invalid")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    raise HolderPolicyError(
                        "Pending holder message is invalid"
                    ) from None
                pending = _PendingSubmission(
                    message=pending_message,
                    signature=pending_signature,
                    action=pending_action,
                )
            sessions[session_id] = _Session(
                issuer=issuer,
                session_id=session_id,
                audience=audience,
                nonce=nonce,
                transcript_hash=transcript_hash,
                sequence=sequence,
                action=action,
                completed=completed,
                pending=pending,
                committed_actions=committed_actions,
            )

        for raw in raw_credentials:
            if not isinstance(raw, dict):
                raise HolderPolicyError("Holder credential state is invalid")
            credential_jti = _bounded_text(raw.get("credential_jti"), "credential_jti")
            if JTI_PATTERN.fullmatch(credential_jti) is None:
                raise HolderPolicyError("Holder credential JTI is invalid")
            if credential_jti in credentials:
                raise HolderPolicyError("Holder state repeats a credential")
            session_id = _bounded_text(raw.get("session_id"), "session_id")
            session = sessions.get(session_id)
            if session is None or not session.completed:
                raise HolderPolicyError("Holder credential session is invalid")
            issuer = _normalize_issuer(
                _bounded_text(raw.get("issuer"), "issuer", maximum=512)
            )
            audience = _bounded_text(raw.get("audience"), "audience")
            transcript_hash = _hash(raw.get("transcript_hash"), "transcript_hash")
            sequence = raw.get("sequence")
            if (
                issuer != session.issuer
                or audience != session.audience
                or transcript_hash != session.transcript_hash
                or sequence != session.sequence
            ):
                raise HolderPolicyError("Holder credential binding is invalid")
            raw_presentations = raw.get("presentations")
            if (
                not isinstance(raw_presentations, list)
                or len(raw_presentations)
                > self._policy.max_presentations_per_credential
            ):
                raise HolderPolicyError("Holder presentation state is invalid")
            presentations: dict[str, tuple[bytes, str]] = {}
            for raw_presentation in raw_presentations:
                if not isinstance(raw_presentation, dict):
                    raise HolderPolicyError("Holder presentation state is invalid")
                challenge_id = _bounded_text(
                    raw_presentation.get("challenge_id"), "challenge_id"
                )
                if challenge_id in presentation_ids:
                    raise HolderPolicyError("Holder state repeats a presentation")
                message = _state_bytes(
                    raw_presentation.get("message"), "presentation message"
                )
                signature = _bounded_text(
                    raw_presentation.get("signature"),
                    "presentation signature",
                    maximum=128,
                )
                if not verify_signature(self._public_key_pem, message, signature):
                    raise HolderPolicyError("Holder presentation signature is invalid")
                try:
                    presentation_payload = json.loads(message)
                    if (
                        not isinstance(presentation_payload, dict)
                        or presentation_payload.get("protocol") != PRESENCE_PROTOCOL
                        or presentation_payload.get("purpose")
                        != "mettle-credential-presentation"
                        or presentation_payload.get("challenge_id") != challenge_id
                        or presentation_payload.get("audience") != audience
                        or presentation_payload.get("credential_jti") != credential_jti
                        or presentation_signing_bytes(
                            challenge_id=challenge_id,
                            nonce=_nonce(presentation_payload.get("nonce")),
                            audience=audience,
                            credential_jti=credential_jti,
                            expires_at=_bounded_text(
                                presentation_payload.get("expires_at"), "expires_at"
                            ),
                        )
                        != message
                    ):
                        raise HolderPolicyError(
                            "Holder presentation message is invalid"
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    raise HolderPolicyError(
                        "Holder presentation message is invalid"
                    ) from None
                presentations[challenge_id] = (message, signature)
                presentation_ids[challenge_id] = (message, signature)
            credentials[credential_jti] = _Credential(
                issuer=issuer,
                session_id=session_id,
                credential_jti=credential_jti,
                audience=audience,
                transcript_hash=transcript_hash,
                sequence=sequence,
                presentations=presentations,
            )
        if len(presentation_ids) > self._policy.max_presentation_records:
            raise HolderPolicyError(
                "Holder state exceeds the presentation record budget"
            )

        with self._lock:
            if self._sessions or self._credentials or self._presentation_ids:
                raise HolderPolicyError("Holder state can only be restored once")
            self._sessions = sessions
            self._credentials = credentials
            self._presentation_ids = presentation_ids
