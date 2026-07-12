"""Proof-of-liveness for operator commitments, and envelope integrity for attestations.

These pin the two security properties fixed in this change:

1. An operator commitment must prove the operator is live NOW. It used to sign a static string
   ("I accept accountability for agent {entity_id}"), which made it a pure bearer artifact:
   capture one signature and replay it verbatim, on a new session, forever. The replay test
   below is the one that cannot pass by accident.

2. An attestation signature must cover the WHOLE envelope. It used to cover only `metadata`,
   leaving `reviewed_at` outside the signature -- so a genuine old attestation could be re-dated
   and still verify. On a credential whose entire point is freshness, that is fatal.
"""

from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from mettle.signing import operator_commitment_message
from mettle.vcp import (
    SIGNATURE_SCHEME,
    attestation_signing_bytes,
    build_mettle_attestation,
    verify_attestation,
)

import database


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the store at a scratch SQLite file so tests never touch a real DB."""
    engine = database.create_engine(
        f"sqlite:///{tmp_path/'t.db'}", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(
        database, "SessionLocal", database.sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    database.Base.metadata.create_all(bind=engine)


def _keypair():
    priv = Ed25519PrivateKey.generate()
    pem = priv.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pem


# --- Proof of liveness ---------------------------------------------------------------------


def test_nonce_is_single_use_replay_is_rejected():
    """THE replay test. Consuming a nonce twice must fail the second time."""
    ch = database.create_operator_challenge("agent-1")

    # First use succeeds.
    database.consume_operator_challenge_strict(ch["nonce"], "agent-1")

    # Replay of the very same nonce -- exactly what a captured commitment would do.
    with pytest.raises(ValueError, match="already been used"):
        database.consume_operator_challenge_strict(ch["nonce"], "agent-1")


def test_nonce_is_bound_to_entity_id():
    """A nonce issued for one agent cannot be used to vouch for another."""
    ch = database.create_operator_challenge("agent-1")

    with pytest.raises(ValueError, match="different entity_id"):
        database.consume_operator_challenge_strict(ch["nonce"], "agent-2")


def test_expired_nonce_is_rejected():
    ch = database.create_operator_challenge("agent-1", ttl_seconds=-1)  # already expired

    with pytest.raises(ValueError, match="expired"):
        database.consume_operator_challenge_strict(ch["nonce"], "agent-1")


def test_unknown_nonce_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        database.consume_operator_challenge_strict("never-issued", "agent-1")


def test_store_failure_fails_closed_not_open():
    """If the store cannot be read, we must NOT silently treat the nonce as fine."""
    broken = database.create_engine("sqlite:///:memory:")  # no tables created
    original = database.engine
    database.engine = broken
    database.SessionLocal = database.sessionmaker(bind=broken)
    try:
        with pytest.raises(database.OperatorChallengeStoreUnavailable):
            database.consume_operator_challenge_strict("n", "agent-1")
    finally:
        database.engine = original


def test_operator_signature_verifies_over_the_nonce_bound_message():
    priv, pem = _keypair()
    ch = database.create_operator_challenge("agent-1")
    expires_at = ch["expires_at"].isoformat()

    msg = operator_commitment_message(ch["nonce"], "agent-1", expires_at)
    sig = priv.sign(msg)

    # The verifier rebuilds the identical message through the same helper.
    priv.public_key().verify(sig, operator_commitment_message(ch["nonce"], "agent-1", expires_at))
    assert pem.startswith("-----BEGIN PUBLIC KEY-----")


def test_old_static_message_no_longer_validates():
    """The pre-fix message must not be accepted any more."""
    priv, _ = _keypair()
    stale = b"I accept accountability for agent agent-1"
    sig = priv.sign(stale)

    fresh = operator_commitment_message("nonce-abc", "agent-1", "2026-01-01T00:00:00+00:00")
    assert stale != fresh
    with pytest.raises(Exception):
        priv.public_key().verify(sig, fresh)


def test_commitment_message_is_domain_separated():
    """The issuer key also signs VCP attestations; message types must not collide."""
    msg = operator_commitment_message("n", "e", "2026-01-01T00:00:00+00:00")
    assert msg.startswith(b"METTLE-OPERATOR-COMMITMENT-v1|")


# --- End-to-end: the path a real operator actually walks ------------------------------------


def _commitment_for(priv, pem, nonce, entity_id, expires_at):
    import base64

    sig = priv.sign(operator_commitment_message(nonce, entity_id, expires_at))
    return {
        "operator_pseudonym": "anon-42",
        "operator_public_key": pem,
        "challenge_nonce": nonce,
        "signed_commitment": base64.b64encode(sig).decode(),
        "contact_method": "email_hash",
        "contact_hash": "sha256:abc",
    }


def test_end_to_end_challenge_sign_consume_then_attest():
    """challenge -> sign -> consume at creation -> attestation verifies at result.

    This drives the whole contract, not just the units: the unit tests would all still pass if
    the two halves disagreed about the message bytes.
    """
    from mettle.router import _build_operator_attestation, _verify_and_consume_operator_commitment

    priv, pem = _keypair()
    ch = database.create_operator_challenge("agent-xyz")

    commitment = _commitment_for(priv, pem, ch["nonce"], "agent-xyz", ch["expires_at"].isoformat())

    # Session creation: consumes the nonce once and binds the expiry into the stored commitment.
    bound = _verify_and_consume_operator_commitment(commitment, "agent-xyz")
    assert bound["challenge_expires_at"] == ch["expires_at"].isoformat()

    # Result time: rebuilds and re-verifies the signature. Idempotent -- must NOT burn the nonce,
    # because callers may fetch a result repeatedly.
    for _ in range(3):
        att = _build_operator_attestation(bound, "agent-xyz")
        assert att is not None
        assert att.operator_pseudonym == "anon-42"


def test_end_to_end_replayed_commitment_is_rejected_at_session_creation():
    """Capture a valid commitment, replay it on a SECOND session. This is the whole point."""
    from mettle.router import _verify_and_consume_operator_commitment

    priv, pem = _keypair()
    ch = database.create_operator_challenge("agent-xyz")
    commitment = _commitment_for(priv, pem, ch["nonce"], "agent-xyz", ch["expires_at"].isoformat())

    _verify_and_consume_operator_commitment(commitment, "agent-xyz")  # session 1: fine

    with pytest.raises(ValueError, match="already been used"):
        _verify_and_consume_operator_commitment(commitment, "agent-xyz")  # session 2: REJECTED


def test_commitment_without_a_nonce_is_rejected():
    from mettle.router import _verify_and_consume_operator_commitment

    priv, pem = _keypair()
    bad = _commitment_for(priv, pem, "n", "agent-xyz", "2026-01-01T00:00:00+00:00")
    del bad["challenge_nonce"]

    with pytest.raises(ValueError, match="requires a challenge_nonce"):
        _verify_and_consume_operator_commitment(bad, "agent-xyz")


def test_attestation_refuses_a_commitment_missing_its_challenge_binding():
    """A commitment with no challenge binding must not silently produce an attestation."""
    from mettle.router import _build_operator_attestation

    priv, pem = _keypair()
    unbound = _commitment_for(priv, pem, "n", "agent-xyz", "2026-01-01T00:00:00+00:00")
    unbound.pop("challenge_nonce")

    assert _build_operator_attestation(unbound, "agent-xyz") is None


# --- Attestation envelope integrity --------------------------------------------------------


def _signed_attestation(priv):
    import base64

    return build_mettle_attestation(
        session_id="s1",
        difficulty="standard",
        suites_passed=["a"],
        suites_failed=[],
        pass_rate=1.0,
        sign_fn=lambda data: base64.b64encode(priv.sign(data)).decode(),
    )


def test_attestation_signature_covers_reviewed_at():
    """Re-dating a genuine attestation must invalidate it. This was the freshness hole."""
    priv, pem = _keypair()
    att = _signed_attestation(priv)

    assert verify_attestation(att, pem) is True

    tampered = dict(att)
    tampered["reviewed_at"] = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()

    assert verify_attestation(tampered, pem) is False


def test_attestation_signature_covers_auditor_identity():
    priv, pem = _keypair()
    att = _signed_attestation(priv)

    for field in ("auditor", "auditor_key_id", "attestation_type"):
        tampered = dict(att)
        tampered[field] = "evil"
        assert verify_attestation(tampered, pem) is False, f"{field} is outside the signature"


def test_attestation_signature_covers_metadata():
    priv, pem = _keypair()
    att = _signed_attestation(priv)

    tampered = dict(att)
    tampered["metadata"] = {**att["metadata"], "tier": "platinum"}
    assert verify_attestation(tampered, pem) is False


def test_unsigned_attestation_does_not_verify():
    _, pem = _keypair()
    att = build_mettle_attestation(
        session_id="s1", difficulty="standard", suites_passed=[], suites_failed=[], pass_rate=0.0
    )
    assert att["signature"] is None
    assert verify_attestation(att, pem) is False


def test_scheme_downgrade_is_refused():
    """An attacker must not be able to force the verifier back to the metadata-only bytes."""
    priv, pem = _keypair()
    att = _signed_attestation(priv)

    downgraded = dict(att)
    downgraded["signature_scheme"] = "mettle-metadata-only"
    assert verify_attestation(downgraded, pem) is False


def test_signing_bytes_exclude_only_the_signature():
    priv, _ = _keypair()
    att = _signed_attestation(priv)

    signed_view = attestation_signing_bytes(att)
    assert b"signature_scheme" in signed_view
    assert b"reviewed_at" in signed_view
    assert att["signature"].encode() not in signed_view
    assert att["signature_scheme"] == SIGNATURE_SCHEME
