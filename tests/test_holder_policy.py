"""Security tests for the autonomous Presence holder policy boundary."""

from __future__ import annotations

import base64
import copy
import subprocess
import time
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_public_key,
)

import mettle.signing as issuer_signing
from mettle.holder import (
    EphemeralEd25519Signer,
    HolderPolicy,
    HolderPolicyError,
    MacOSKeychainEd25519Signer,
    PresenceHolder,
)
from mettle.presence import (
    presence_state_signing_bytes,
    presentation_signing_bytes,
    submission_signing_bytes,
    transcript_hash_after_submission,
)
from mettle.vcp import build_mettle_attestation


ISSUER = "https://mettle.example"
AUDIENCE = "service.example"
BRONZE_SUITES = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
]


@pytest.fixture()
def issuer_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("METTLE_DEV_MODE", "true")
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._initialized = False
    assert issuer_signing.init_signing() is True
    public_key_pem = issuer_signing.get_public_key_pem()
    assert isinstance(public_key_pem, str)
    yield public_key_pem
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._initialized = False


def _presence(
    holder: PresenceHolder,
    *,
    session_id: str = "session-1",
    nonce: str | None = "n" * 32,
    transcript_hash: str = "sha256:" + "a" * 64,
    sequence: int = 0,
    action: str | None = "suite:adversarial",
    completed: bool = False,
    audience: str = AUDIENCE,
    signed: bool = True,
) -> dict[str, object]:
    state: dict[str, object] = {
        "protocol": "mettle-presence-v1",
        "key_fingerprint": holder.key_fingerprint,
        "audience": audience,
        "nonce": nonce,
        "transcript_hash": transcript_hash,
        "sequence": sequence,
        "action": action,
        "completed": completed,
    }
    if signed:
        _sign_presence_state(state, session_id=session_id)
    return state


def _sign_presence_state(state: dict[str, object], *, session_id: str) -> None:
    state["issuer_receipt"] = {
        "key_id": "mettle-vcp-v1",
        "algorithm": "Ed25519",
        "signature": issuer_signing.sign_attestation(
            presence_state_signing_bytes(
                session_id=session_id,
                presence=state,
            )
        ),
    }


def _holder(
    issuer_public_key_pem: str,
    *,
    max_active_sessions: int = 1,
    max_actions: int = 2,
    max_presentations: int = 2,
) -> tuple[PresenceHolder, EphemeralEd25519Signer]:
    signer = EphemeralEd25519Signer()
    return (
        PresenceHolder(
            signer,
            HolderPolicy(
                issuer_public_keys={ISSUER: issuer_public_key_pem},
                allowed_audiences=frozenset({AUDIENCE}),
                max_active_sessions=max_active_sessions,
                max_actions_per_session=max_actions,
                max_presentations_per_credential=max_presentations,
            ),
        ),
        signer,
    )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"max_active_sessions": 0}, "Active-session budget"),
        ({"max_actions_per_session": 0}, "Per-session action budget"),
        ({"max_presentations_per_credential": 0}, "Presentation budget"),
        ({"max_presentation_ttl_seconds": 0}, "Presentation TTL"),
        ({"issuer_public_keys": {}}, "trusted issuer"),
        ({"allowed_audiences": frozenset()}, "audience"),
        (
            {"issuer_public_keys": {"http://evil.example": "issuer-key"}},
            "use HTTPS",
        ),
        (
            {"issuer_public_keys": {f"{ISSUER}?query=1": "issuer-key"}},
            "query or fragment",
        ),
    ],
)
def test_holder_validates_static_policy(
    issuer_key: str, updates: dict[str, Any], error: str
) -> None:
    values: dict[str, Any] = {
        "issuer_public_keys": {ISSUER: issuer_key},
        "allowed_audiences": frozenset({AUDIENCE}),
    }
    values.update(updates)
    if "issuer_public_keys" in updates and updates["issuer_public_keys"]:
        values["issuer_public_keys"] = {
            next(iter(updates["issuer_public_keys"])): issuer_key
        }
    with pytest.raises(HolderPolicyError, match=error):
        PresenceHolder(EphemeralEd25519Signer(), HolderPolicy(**values))


def test_macos_keychain_signer_loads_without_shell_or_plaintext_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        Encoding.PEM,
        PrivateFormat.PKCS8,
        NoEncryption(),
    )
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=private_pem.hex().encode("ascii") + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr("mettle.holder.subprocess.run", fake_run)
    signer = MacOSKeychainEd25519Signer(
        service="mettle-holder",
        account="presence-key",
    )
    message = b"holder-keychain-test"
    signature = signer.sign(message)
    public_key = load_pem_public_key(signer.public_key_pem.encode("ascii"))
    assert isinstance(public_key, Ed25519PublicKey)
    public_key.verify(signature, message)
    assert observed["command"] == [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        "mettle-holder",
        "-a",
        "presence-key",
        "-w",
    ]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "timeout": 5.0,
    }


def test_macos_keychain_signer_hides_lookup_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = b"private key lookup detail must not escape"
    monkeypatch.setattr(
        "mettle.holder.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 44, stdout=b"", stderr=diagnostic
        ),
    )
    with pytest.raises(HolderPolicyError, match="lookup failed") as error:
        MacOSKeychainEd25519Signer(service="mettle-holder", account="missing")
    assert diagnostic.decode("ascii") not in str(error.value)


def test_holder_rejects_malformed_session_and_presentation_inputs(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key)
    with pytest.raises(HolderPolicyError, match="object"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="session",
            presence=cast(Any, None),
        )
    invalid_states: list[tuple[dict[str, object], str]] = [
        ({"protocol": "other"}, "protocol"),
        ({"key_fingerprint": "other"}, "different key"),
        ({"completed": True}, "initial"),
        ({"nonce": "short"}, "at least 32"),
        ({"transcript_hash": "bad"}, "SHA-256"),
        ({"action": "unsupported"}, "supported Presence action"),
    ]
    for changes, error in invalid_states:
        session_id = f"session-{error}"
        presence = _presence(holder, session_id=session_id, signed=False)
        presence.update(changes)
        _sign_presence_state(presence, session_id=session_id)
        with pytest.raises(HolderPolicyError, match=error):
            holder.authorize_session(
                issuer=ISSUER,
                session_id=session_id,
                presence=presence,
            )
    with pytest.raises(HolderPolicyError, match="not authorized"):
        holder.sign_submission(
            session_id="missing",
            action="suite:adversarial",
            nonce="n" * 32,
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "b" * 64,
        )
    for expires_at, error in [
        ("not-a-date", "expiry is invalid"),
        ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "expired"),
        ((datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), "exceeds"),
    ]:
        with pytest.raises(HolderPolicyError, match=error):
            holder.sign_presentation(
                challenge_id="challenge",
                nonce="p" * 32,
                audience=AUDIENCE,
                credential_jti="c" * 32,
                expires_at=expires_at,
            )


def test_holder_rejects_unsigned_and_tampered_issuer_state(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key, max_active_sessions=2)
    with pytest.raises(HolderPolicyError, match="issuer receipt is required"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="fabricated-session",
            presence=_presence(
                holder,
                session_id="fabricated-session",
                signed=False,
            ),
        )
    tampered_initial = _presence(holder, session_id="tampered-session")
    tampered_initial["action"] = "suite:native"
    with pytest.raises(HolderPolicyError, match="issuer receipt is invalid"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="tampered-session",
            presence=tampered_initial,
        )

    session_id = "transition-session"
    holder.authorize_session(
        issuer=ISSUER,
        session_id=session_id,
        presence=_presence(holder, session_id=session_id),
    )
    payload_hash = "sha256:" + "b" * 64
    signature = holder.sign_submission(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=payload_hash,
    )
    message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=payload_hash,
    )
    next_state = _presence(
        holder,
        session_id=session_id,
        nonce="o" * 32,
        transcript_hash=transcript_hash_after_submission(
            previous_transcript_hash="sha256:" + "a" * 64,
            message=message,
            signature=signature,
        ),
        sequence=1,
        action="suite:native",
    )
    next_state["nonce"] = "p" * 32
    with pytest.raises(HolderPolicyError, match="issuer receipt is invalid"):
        holder.commit_submission(session_id=session_id, presence=next_state)


def test_holder_rejects_untrusted_issuer_audience_and_session_farming(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key)
    with pytest.raises(HolderPolicyError, match="not trusted"):
        holder.authorize_session(
            issuer="https://evil.example",
            session_id="session-evil",
            presence=_presence(holder, session_id="session-evil"),
        )
    with pytest.raises(HolderPolicyError, match="Audience"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="session-wrong-audience",
            presence=_presence(
                holder,
                session_id="session-wrong-audience",
                audience="other.example",
            ),
        )
    holder.authorize_session(
        issuer=ISSUER, session_id="session-1", presence=_presence(holder)
    )
    with pytest.raises(HolderPolicyError, match="already"):
        holder.authorize_session(
            issuer=ISSUER, session_id="session-1", presence=_presence(holder)
        )
    with pytest.raises(HolderPolicyError, match="budget"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="session-2",
            presence=_presence(holder, session_id="session-2"),
        )


def test_holder_enforces_pending_payload_and_monotonic_transcript(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key)
    holder.authorize_session(
        issuer=ISSUER, session_id="session-1", presence=_presence(holder)
    )
    with pytest.raises(HolderPolicyError, match="Action"):
        holder.sign_submission(
            session_id="session-1",
            action="suite:native",
            nonce="n" * 32,
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "b" * 64,
        )
    signature = holder.sign_submission(
        session_id="session-1",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    assert signature == holder.sign_submission(
        session_id="session-1",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    with pytest.raises(HolderPolicyError, match="different submission"):
        holder.sign_submission(
            session_id="session-1",
            action="suite:adversarial",
            nonce="n" * 32,
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "c" * 64,
        )
    message = submission_signing_bytes(
        session_id="session-1",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    next_hash = transcript_hash_after_submission(
        previous_transcript_hash="sha256:" + "a" * 64,
        message=message,
        signature=signature,
    )
    with pytest.raises(HolderPolicyError, match="advance exactly once"):
        holder.commit_submission(
            session_id="session-1",
            presence=_presence(holder, transcript_hash=next_hash, sequence=0),
        )
    with pytest.raises(HolderPolicyError, match="invalid transcript"):
        holder.commit_submission(
            session_id="session-1",
            presence=_presence(
                holder,
                nonce="o" * 32,
                transcript_hash="sha256:" + "d" * 64,
                sequence=1,
                action="suite:native",
            ),
        )
    holder.commit_submission(
        session_id="session-1",
        presence=_presence(
            holder,
            nonce="o" * 32,
            transcript_hash=next_hash,
            sequence=1,
            action="suite:native",
        ),
    )
    with pytest.raises(HolderPolicyError, match="Nonce"):
        holder.sign_submission(
            session_id="session-1",
            action="suite:native",
            nonce="n" * 32,
            previous_transcript_hash=next_hash,
            payload_hash="sha256:" + "e" * 64,
        )


def test_holder_registers_only_matching_signed_credential_and_bounds_presentation(
    issuer_key: str,
) -> None:
    holder, signer = _holder(issuer_key, max_actions=1, max_presentations=1)
    session_id = "session-credential"
    initial_hash = "sha256:" + "a" * 64
    holder.authorize_session(
        issuer=ISSUER,
        session_id=session_id,
        presence=_presence(
            holder,
            session_id=session_id,
            transcript_hash=initial_hash,
        ),
    )
    payload_hash = "sha256:" + "b" * 64
    signature = holder.sign_submission(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash=initial_hash,
        payload_hash=payload_hash,
    )
    message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash=initial_hash,
        payload_hash=payload_hash,
    )
    final_hash = transcript_hash_after_submission(
        previous_transcript_hash=initial_hash,
        message=message,
        signature=signature,
    )
    holder.commit_submission(
        session_id=session_id,
        presence=_presence(
            holder,
            session_id=session_id,
            nonce=None,
            transcript_hash=final_hash,
            sequence=1,
            action=None,
            completed=True,
        ),
    )
    now_ms = int(time.time() * 1000)
    internal_presence = {
        "protocol": "mettle-presence-v1",
        "public_key_pem": holder.public_key_pem,
        "key_fingerprint": holder.key_fingerprint,
        "audience": AUDIENCE,
        "credential_jti": "c" * 32,
        "transcript_hash": final_hash,
        "sequence": 1,
        "started_at_unix_ms": now_ms - 100,
        "submissions": [
            {
                "sequence": 1,
                "action": "suite:adversarial",
                "response_time_ms": 100,
                "accepted_at_unix_ms": now_ms,
                "transcript_hash": final_hash,
            }
        ],
    }
    attestation = build_mettle_attestation(
        session_id=session_id,
        difficulty="standard",
        suites_passed=BRONZE_SUITES,
        suites_failed=[],
        pass_rate=1.0,
        subject_id="holder-policy-test",
        presence=internal_presence,
    )
    tampered = copy.deepcopy(attestation)
    tampered["metadata"]["tier"] = "platinum"
    with pytest.raises(HolderPolicyError, match="signature or policy"):
        holder.register_credential(issuer=ISSUER, attestation=tampered)
    assert (
        holder.register_credential(issuer=ISSUER, attestation=attestation) == "c" * 32
    )

    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    presentation_signature = holder.sign_presentation(
        challenge_id="challenge-1",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti="c" * 32,
        expires_at=expires_at,
    )
    assert presentation_signature == holder.sign_presentation(
        challenge_id="challenge-1",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti="c" * 32,
        expires_at=expires_at,
    )
    presentation_message = presentation_signing_bytes(
        challenge_id="challenge-1",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti="c" * 32,
        expires_at=expires_at,
    )
    public_key = load_pem_public_key(signer.public_key_pem.encode("ascii"))
    assert isinstance(public_key, Ed25519PublicKey)
    public_key.verify(base64.b64decode(presentation_signature), presentation_message)
    with pytest.raises(HolderPolicyError, match="reused inconsistently"):
        holder.sign_presentation(
            challenge_id="challenge-1",
            nonce="q" * 32,
            audience=AUDIENCE,
            credential_jti="c" * 32,
            expires_at=expires_at,
        )
    with pytest.raises(HolderPolicyError, match="budget"):
        holder.sign_presentation(
            challenge_id="challenge-2",
            nonce="q" * 32,
            audience=AUDIENCE,
            credential_jti="c" * 32,
            expires_at=expires_at,
        )
