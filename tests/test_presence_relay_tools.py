"""Tests for three-party Presence relay trial instrumentation."""

from __future__ import annotations

import base64
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from mettle.presence import submission_signing_bytes
from scripts.testing.presence_relay_workers import (
    HolderWorkerClient,
    SolverWorkerClient,
    WorkerProtocolError,
    worker_environment,
)
from scripts.testing.run_presence_relay_trials import (
    Cohort,
    build_report,
    evaluate_separation,
    parse_cohort,
)


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

        message = submission_signing_bytes(
            session_id="session-1",
            action="suite:adversarial",
            nonce="nonce-1",
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "b" * 64,
        )
        signature, holder_ms = holder.sign_submission(
            session_id="session-1",
            action="suite:adversarial",
            nonce="nonce-1",
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "b" * 64,
        )
        public_key = load_pem_public_key(holder.public_key_pem.encode("ascii"))
        assert isinstance(public_key, Ed25519PublicKey)
        public_key.verify(base64.b64decode(signature), message)
        assert holder_ms >= 0
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
    assert descriptive["product_decision_eligible"] is False

    eligible = evaluate_separation(
        [20] * 300,
        [1000] * 29,
        measured_human=True,
    )
    assert eligible["product_decision_eligible"] is True

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
        attacks={"passed": True},
    )
    assert report["summaries"]["paced"]["synthetic_delay"] is True
    assert report["separation"]["paced"]["observed_rate_criteria_met"] is True
    assert report["separation"]["paced"]["product_decision_eligible"] is False
    assert report["decision"]["threshold_enforcement_authorized"] is False
    assert report["decision"]["measured_human_cohort_status"].startswith("not_run")
    assert parse_cohort("relay:process-relay:0:2") == Cohort(
        "relay", "process-relay", 0, 2
    )


def test_sufficient_human_cohort_that_overlaps_fails_the_gate() -> None:
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
        attacks=None,
    )
    assert report["separation"]["human"]["sample_sufficient"] is True
    assert report["separation"]["human"]["observed_overlap"] is True
    assert report["decision"]["status"] == "criteria_not_met"
