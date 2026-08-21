"""Security regressions for governance and operator attestations."""

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import ValidationError

from mettle.api_models import CreateSessionRequest, OperatorCommitment
from mettle.router import _build_governance_attestation, _build_operator_attestation


def _signed_commitment(
    entity_id: str = "agent-1",
    *,
    issued_at: datetime | None = None,
    nonce: str = "b" * 64,
) -> dict[str, str]:
    issued_at = issued_at or datetime.now(timezone.utc)
    issued_at_text = issued_at.isoformat().replace("+00:00", "Z")
    private_key = Ed25519PrivateKey.generate()
    public_pem = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    commitment = {
        "operator_pseudonym": "operator-1",
        "operator_public_key": public_pem,
        "contact_method": "email_hash",
        "contact_hash": "a" * 64,
        "issued_at": issued_at_text,
        "nonce": nonce,
    }
    payload = json.dumps(
        {
            **commitment,
            "entity_id": entity_id,
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    commitment["signed_commitment"] = base64.b64encode(
        private_key.sign(payload)
    ).decode()
    return commitment


def _accepted_at(commitment: dict[str, str]) -> datetime:
    return datetime.fromisoformat(commitment["issued_at"].replace("Z", "+00:00"))


def test_operator_signature_binds_all_returned_identity_fields() -> None:
    commitment = _signed_commitment()
    accepted_at = _accepted_at(commitment)

    attestation = _build_operator_attestation(
        commitment,
        "agent-1",
        accepted_at=accepted_at,
    )

    assert attestation is not None
    assert attestation.operator_pseudonym == "operator-1"
    assert attestation.contact_hash == "a" * 64
    assert attestation.commitment_timestamp == accepted_at
    assert attestation.commitment_nonce == "b" * 64


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("operator_pseudonym", "attacker-controlled"),
        ("operator_public_key", "other-valid-key"),
        ("contact_method", "legal_entity"),
        ("contact_hash", "c" * 64),
        ("issued_at", "shift-within-window"),
        ("nonce", "d" * 64),
    ],
)
def test_operator_signature_binds_every_semantic_field(
    field: str, tampered: str
) -> None:
    commitment = _signed_commitment()
    if field == "issued_at":
        original = datetime.fromisoformat(commitment[field].replace("Z", "+00:00"))
        commitment[field] = (
            (original - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        )
    elif field == "operator_public_key":
        commitment[field] = (
            Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            .decode()
        )
    else:
        commitment[field] = tampered

    assert (
        _build_operator_attestation(
            commitment, "agent-1", accepted_at=_accepted_at(commitment)
        )
        is None
    )


def test_operator_signature_binds_entity_id() -> None:
    commitment = _signed_commitment(entity_id="agent-1")

    assert (
        _build_operator_attestation(
            commitment, "agent-2", accepted_at=_accepted_at(commitment)
        )
        is None
    )


@pytest.mark.parametrize(
    ("offset", "accepted"),
    [
        (timedelta(minutes=-5), True),
        (timedelta(minutes=-5, microseconds=-1), False),
        (timedelta(seconds=30), True),
        (timedelta(seconds=30, microseconds=1), False),
    ],
)
def test_operator_commitment_freshness_boundaries(
    offset: timedelta, accepted: bool
) -> None:
    receipt_time = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    commitment = _signed_commitment(issued_at=receipt_time + offset)

    attestation = _build_operator_attestation(
        commitment,
        "agent-1",
        accepted_at=receipt_time,
    )

    assert (attestation is not None) is accepted


def test_operator_commitment_requires_non_null_subject() -> None:
    commitment = OperatorCommitment.model_validate(_signed_commitment())

    with pytest.raises(ValidationError, match="entity_id is required"):
        CreateSessionRequest(operator_commitment=commitment)
    assert (
        _build_operator_attestation(
            commitment.model_dump(), None, accepted_at=commitment.issued_at
        )
        is None
    )


def test_operator_attestation_fails_closed_without_persisted_receipt_time() -> None:
    commitment = _signed_commitment()

    assert _build_operator_attestation(commitment, "agent-1", accepted_at=None) is None


def test_unverified_vcp_is_not_countersigned_or_used_as_operational_proof(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PUBLIC_ACTION_GATE_ENABLED", raising=False)
    monkeypatch.delenv("CONSTITUTIONAL_DRIFT_DETECTOR_ENABLED", raising=False)
    monkeypatch.delenv("BILATERAL_ALIGNMENT_ENABLED", raising=False)

    attestation = _build_governance_attestation(
        "VCP:3.1:agent-1\nC:creed.example@1.0",
        entity_id="agent-1",
        session_id="session-1",
        tier="gold",
    )

    assert attestation is not None
    assert attestation.source_verified is False
    assert attestation.attestation_signature is None
    assert attestation.has_action_gate is False
    assert attestation.has_drift_detection is False
    assert attestation.has_bilateral is False
    assert attestation.entity_id == "agent-1"
    assert attestation.session_id == "session-1"
    assert attestation.tier == "gold"
    assert attestation.expires_at > attestation.observed_at


def test_digest_allowlists_and_environment_flags_cannot_promote_raw_vcp(
    monkeypatch,
) -> None:
    token = "VCP:3.1:agent-1\nC:creed.example@1.0"
    digest = hashlib.sha256(token.encode()).hexdigest()
    monkeypatch.setenv("METTLE_TRUSTED_VCP_SHA256", digest)
    monkeypatch.setenv("PUBLIC_ACTION_GATE_ENABLED", "true")
    monkeypatch.setenv("CONSTITUTIONAL_DRIFT_DETECTOR_ENABLED", "true")
    monkeypatch.setenv("BILATERAL_ALIGNMENT_ENABLED", "true")

    with patch("mettle.signing.sign_attestation") as sign:
        attestation = _build_governance_attestation(
            token, entity_id="agent-1", session_id="session-1", tier="none"
        )

    assert attestation is not None
    assert attestation.source_vcp_hash == digest
    assert attestation.source_verified is False
    assert attestation.attestation_signature is None
    assert attestation.has_action_gate is False
    assert attestation.has_drift_detection is False
    assert attestation.has_bilateral is False
    assert attestation.tier == "none"
    sign.assert_not_called()
