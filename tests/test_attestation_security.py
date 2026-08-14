"""Security regressions for governance and operator attestations."""

import hashlib
from unittest.mock import patch

from mettle.router import _build_governance_attestation


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
