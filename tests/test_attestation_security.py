"""Security regressions for governance and operator attestations."""

import base64
import hashlib
import json
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import ValidationError

from mettle.api_models import CreateSessionRequest, OperatorCommitment
from mettle.router import _build_governance_attestation, _build_operator_attestation


def _signed_commitment(entity_id: str = "agent-1") -> dict[str, str]:
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


def test_operator_signature_binds_all_returned_identity_fields() -> None:
    commitment = _signed_commitment()

    attestation = _build_operator_attestation(commitment, "agent-1")

    assert attestation is not None
    assert attestation.operator_pseudonym == "operator-1"
    assert attestation.contact_hash == "a" * 64


def test_operator_field_tampering_invalidates_signature() -> None:
    commitment = _signed_commitment()
    commitment["operator_pseudonym"] = "attacker-controlled"

    assert _build_operator_attestation(commitment, "agent-1") is None


def test_operator_commitment_requires_non_null_subject() -> None:
    commitment = OperatorCommitment(**_signed_commitment())

    with pytest.raises(ValidationError, match="entity_id is required"):
        CreateSessionRequest(operator_commitment=commitment)
    assert _build_operator_attestation(commitment.model_dump(), None) is None


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
