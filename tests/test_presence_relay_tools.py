"""Tests for three-party Presence relay trial instrumentation."""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_public_key,
)

from mettle.presence import (
    key_fingerprint,
    presence_state_signing_bytes,
    submission_signing_bytes,
    transcript_hash_after_submission,
)
from scripts.testing.presence_relay_workers import (
    HolderWorkerClient,
    SolverWorkerClient,
    WorkerProtocolError,
    worker_environment,
)
from scripts.testing.run_presence_relay_trials import (
    Cohort,
    REQUIRED_HOLDER_POLICY_ATTACKS,
    REQUIRED_PROTOCOL_REJECTIONS,
    build_report,
    evaluate_separation,
    parse_cohort,
)


def _passing_attacks() -> dict[str, object]:
    return {
        "passed": True,
        "holder_policy_attacks": {
            name: {"rejected": True} for name in REQUIRED_HOLDER_POLICY_ATTACKS
        },
        **{name: {"rejected": True} for name in REQUIRED_PROTOCOL_REJECTIONS},
        "valid_holder_service_presentation": {"accepted": True},
        "process_boundary": {
            "holder_private_key_crossed_ipc": False,
            "solver_received_holder_key": False,
            "workers_inherited_mettle_credentials": False,
        },
    }


def _add_issuer_receipt(
    issuer_key: Ed25519PrivateKey,
    *,
    session_id: str,
    presence: dict[str, object],
) -> None:
    presence["issuer_receipt"] = {
        "key_id": "test-issuer",
        "algorithm": "Ed25519",
        "signature": base64.b64encode(
            issuer_key.sign(
                presence_state_signing_bytes(
                    session_id=session_id,
                    presence=presence,
                )
            )
        ).decode("ascii"),
    }


def test_worker_environment_excludes_parent_mettle_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METTLE_API_KEY", "must-not-cross-process-boundary")
    environment = worker_environment()
    assert "METTLE_API_KEY" not in environment
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_holder_and_solver_are_isolated_and_holder_signature_verifies() -> None:
    with HolderWorkerClient() as holder, SolverWorkerClient() as solver:
        assert holder.pid != os.getpid()
        assert solver.pid not in {os.getpid(), holder.pid}
        assert "PRIVATE" not in holder.public_key_pem

        challenge = {
            "challenges": {
                "dynamic_math": {"problem": "( 2 x 3 ) + 4"},
                "chained_reasoning": {"seed": 2, "operations": ["double"]},
                "time_locked_secret": {
                    "secret_to_remember": "public-fixture"  # pragma: allowlist secret
                },
            }
        }
        answers, solve_ms, roundtrip_ms = solver.solve("adversarial", challenge)
        assert answers["dynamic_math"]["computed"] == 10
        assert solve_ms >= 0
        assert roundtrip_ms >= solve_ms

        nonce = "n" * 32
        previous_hash = "sha256:" + "a" * 64
        payload_hash = "sha256:" + "b" * 64
        with pytest.raises(WorkerProtocolError, match="not configured"):
            holder.sign_submission(
                session_id="session-1",
                action="suite:adversarial",
                nonce=nonce,
                previous_transcript_hash=previous_hash,
                payload_hash=payload_hash,
            )

        issuer_key = Ed25519PrivateKey.generate()
        issuer_public_key_pem = (
            issuer_key.public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            .decode("ascii")
        )
        holder.configure(
            issuer="https://mettle.example",
            issuer_public_key_pem=issuer_public_key_pem,
            allowed_audiences=["service.example"],
        )
        initial_presence: dict[str, object] = {
            "protocol": "mettle-presence-v1",
            "key_fingerprint": key_fingerprint(holder.public_key_pem),
            "audience": "service.example",
            "nonce": nonce,
            "transcript_hash": previous_hash,
            "sequence": 0,
            "action": "suite:adversarial",
            "completed": False,
        }
        _add_issuer_receipt(
            issuer_key,
            session_id="session-1",
            presence=initial_presence,
        )
        holder.authorize_session(
            issuer="https://mettle.example",
            session_id="session-1",
            presence=initial_presence,
        )
        message = submission_signing_bytes(
            session_id="session-1",
            action="suite:adversarial",
            nonce=nonce,
            previous_transcript_hash=previous_hash,
            payload_hash=payload_hash,
        )
        signature, holder_ms = holder.sign_submission(
            session_id="session-1",
            action="suite:adversarial",
            nonce=nonce,
            previous_transcript_hash=previous_hash,
            payload_hash=payload_hash,
        )
        public_key = load_pem_public_key(holder.public_key_pem.encode("ascii"))
        assert isinstance(public_key, Ed25519PublicKey)
        public_key.verify(base64.b64decode(signature), message)
        assert holder_ms >= 0
        completed_presence: dict[str, object] = {
            "protocol": "mettle-presence-v1",
            "key_fingerprint": key_fingerprint(holder.public_key_pem),
            "audience": "service.example",
            "nonce": None,
            "transcript_hash": transcript_hash_after_submission(
                previous_transcript_hash=previous_hash,
                message=message,
                signature=signature,
            ),
            "sequence": 1,
            "action": None,
            "completed": True,
        }
        _add_issuer_receipt(
            issuer_key,
            session_id="session-1",
            presence=completed_presence,
        )
        holder.commit_submission(
            session_id="session-1",
            presence=completed_presence,
        )
        assert holder.status()["active_sessions"] == 0
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        with pytest.raises(WorkerProtocolError, match="not registered"):
            holder.sign_presentation(
                challenge_id="challenge-1",
                nonce="p" * 32,
                audience="service.example",
                credential_jti="c" * 32,
                expires_at=expires_at,
            )
        with pytest.raises(WorkerProtocolError, match="unsupported holder action"):
            holder.request("export_private_key")


def test_separation_requires_measured_and_sufficient_human_receipts() -> None:
    descriptive = evaluate_separation(
        [20] * 300,
        [1000] * 29,
        measured_human=False,
    )
    assert descriptive["observed_rate_criteria_met"] is True
    assert descriptive["sample_sufficient"] is True
    assert descriptive["optional_human_calibration_eligible"] is False

    eligible = evaluate_separation(
        [20] * 300,
        [1000] * 29,
        measured_human=True,
    )
    assert eligible["optional_human_calibration_eligible"] is True

    overlap = evaluate_separation([20, 40], [30, 50], measured_human=True)
    assert overlap["observed_overlap"] is True
    assert overlap["best_observed_threshold"] is None


def test_relay_report_keeps_synthetic_and_product_boundaries_explicit() -> None:
    cohorts = [
        Cohort("direct", "direct", 0, 1),
        Cohort("relay", "process-relay", 0, 1),
        Cohort("paced", "synthetic-human", 1000, 1),
    ]
    samples = [
        {"cohort": "direct", "server_response_time_ms": 20},
        {"cohort": "relay", "server_response_time_ms": 25},
        {"cohort": "paced", "server_response_time_ms": 1020},
    ]
    report = build_report(
        base_url="https://mettle.example",
        cohorts=cohorts,
        samples=samples,
        attacks=_passing_attacks(),
    )
    assert report["summaries"]["paced"]["synthetic_delay"] is True
    assert report["separation"]["paced"]["observed_rate_criteria_met"] is True
    assert report["separation"]["paced"]["optional_human_calibration_eligible"] is False
    assert report["decision"]["threshold_enforcement_authorized"] is False
    assert report["decision"]["timing_used_for_authorization"] is False
    assert report["decision"]["measured_human_cohort_status"] == "not_required"
    assert report["decision"]["human_testing_required"] is False
    assert report["decision"]["status"] == "automated_security_controls_passed"
    assert parse_cohort("relay:process-relay:0:2") == Cohort(
        "relay", "process-relay", 0, 2
    )


def test_relay_report_does_not_claim_validation_when_attack_evidence_fails() -> None:
    report = build_report(
        base_url="https://mettle.example",
        cohorts=[Cohort("direct", "direct", 0, 1)],
        samples=[{"cohort": "direct", "server_response_time_ms": 20}],
        attacks={},
    )
    assert report["decision"]["status"] == "automated_attack_evidence_incomplete"
    assert report["decision"]["authorization_controls_validated"] is False
    assert "not validated" in report["decision"]["reason"]
    assert report["decision"]["timing_used_for_authorization"] is False


def test_timing_overlap_never_authorizes_threshold_enforcement() -> None:
    cohorts = [
        Cohort("direct", "direct", 0, 60),
        Cohort("human", "manual-human", 0, 6),
    ]
    samples = [
        *({"cohort": "direct", "server_response_time_ms": 20} for _ in range(299)),
        {"cohort": "direct", "server_response_time_ms": 1000},
        *({"cohort": "human", "server_response_time_ms": 1000} for _ in range(29)),
    ]
    report = build_report(
        base_url="https://mettle.example",
        cohorts=cohorts,
        samples=samples,
        attacks=_passing_attacks(),
    )
    assert report["separation"]["human"]["sample_sufficient"] is True
    assert report["separation"]["human"]["observed_overlap"] is True
    assert report["decision"]["status"] == "automated_security_controls_passed"
    assert report["decision"]["threshold_enforcement_authorized"] is False


def test_partial_attack_matrix_cannot_produce_a_passing_decision() -> None:
    attacks = _passing_attacks()
    holder_attacks = attacks["holder_policy_attacks"]
    assert isinstance(holder_attacks, dict)
    holder_attacks.pop("pending_submission_fork")
    report = build_report(
        base_url="https://mettle.example",
        cohorts=[Cohort("direct", "direct", 0, 1)],
        samples=[{"cohort": "direct", "server_response_time_ms": 20}],
        attacks=attacks,
    )
    assert report["decision"]["status"] == "automated_attack_evidence_incomplete"
    assert report["decision"]["authorization_controls_validated"] is False
