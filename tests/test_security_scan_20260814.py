"""Regression proofs for the 2026-08-14 deep security scan."""

from __future__ import annotations

import json
from pathlib import Path

from mettle.challenge_adapter import ChallengeAdapter
from mettle.session_manager import SessionManager


ROOT = Path(__file__).resolve().parents[1]


def test_single_shot_details_never_disclose_server_answers() -> None:
    """A failed submission may reveal verdicts, never the answer key."""
    _, server = ChallengeAdapter.generate_adversarial()
    result = ChallengeAdapter.evaluate_single_shot(
        "adversarial",
        {
            "dynamic_math": {"computed": -1},
            "chained_reasoning": {"computed_final": -1},
            "time_locked_secret": {"recalled": "wrong"},
        },
        server,
    )

    serialized = json.dumps(result["details"], sort_keys=True)
    assert "expected" not in serialized
    assert str(server["dynamic_math"]["expected"]) not in serialized
    assert str(server["chained_reasoning"]["expected_final"]) not in serialized


def test_novel_session_payload_contains_no_future_round_material() -> None:
    """Session creation issues round one only."""
    client, _ = ChallengeAdapter.generate_novel_reasoning("hard")
    serialized = json.dumps(client, sort_keys=True)

    assert "round_data" not in serialized
    assert "second_encoded" not in serialized


def test_completely_wrong_final_novel_round_cannot_pass() -> None:
    """Curve shape cannot compensate for a wrong final answer set."""
    manager = SessionManager(object())
    result = manager._analyze_iteration_curve(
        [
            {"round": 1, "response_time_ms": 300, "accuracy": 0.0},
            {"round": 2, "response_time_ms": 200, "accuracy": 1.0},
            {"round": 3, "response_time_ms": 100, "accuracy": 0.0},
        ],
        {
            "num_rounds": 3,
            "pass_threshold": 0.65,
            "final_accuracy_threshold": 0.8,
        },
    )

    assert result["passed"] is False
    assert result["details"]["final_accuracy_met"] is False


def test_self_report_suites_are_never_tier_evidence() -> None:
    """Caller-authored preferences or governance claims cannot raise a tier."""
    generators = {
        "anti-thrall": ChallengeAdapter.generate_anti_thrall,
        "agency": ChallengeAdapter.generate_agency,
        "counter-coaching": ChallengeAdapter.generate_counter_coaching,
        "intent-provenance": ChallengeAdapter.generate_intent_provenance,
        "governance": ChallengeAdapter.generate_governance,
    }

    for suite, generator in generators.items():
        _, server = generator()
        result = ChallengeAdapter.evaluate_single_shot(suite, {}, server)
        assert result["credential_eligible"] is False, suite


def test_v2_payload_does_not_advertise_an_unenforced_subchallenge_clock() -> None:
    """The v2 API advertises only its authoritative session clock."""
    client, _ = ChallengeAdapter.generate_adversarial()
    assert "time_limit_ms" not in client["challenges"]["dynamic_math"]


def test_distribution_package_contains_no_reference_solver() -> None:
    """Deterministic answer fixtures stay outside the shipped package."""
    assert not (ROOT / "mettle" / "solver.py").exists()


def test_shipped_examples_require_a_respondent_supplied_answer() -> None:
    """Examples demonstrate transport and never contain an issuer solver."""
    examples = {
        "python": (ROOT / "examples/python_example.py").read_text(),
        "javascript": (ROOT / "examples/javascript_example.js").read_text(),
        "rust": (ROOT / "examples/rust_example.rs").read_text(),
    }

    assert 'input("  Response: ")' in examples["python"]
    assert "answerChallenge callback is required" in examples["javascript"]
    assert "read_line(&mut answer)" in examples["rust"]
    for source in examples.values():
        assert "scripts.testing.solver" not in source
        assert "expected_answer" not in source


def test_retired_pseudo_gates_and_manual_oidc_publisher_are_absent() -> None:
    """Only evidence-producing and immutable-tag workflows may carry authority."""
    retired = (
        ROOT / ".github/workflows/red-council.yml",
        ROOT / ".github/workflows/mcp-registry-publish.yml",
        ROOT / "scripts/testing/run_mettle_red_council.py",
    )

    assert all(not path.exists() for path in retired)


def test_public_api_rejects_oversized_request_bodies(client) -> None:
    """The application boundary rejects a body before FastAPI parses it."""
    response = client.post(
        "/api/session/start",
        content=b"x" * 1_048_577,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body is too large"}

    streamed = client.post(
        "/api/session/start",
        content=iter((b"x" * 700_000, b"x" * 700_000)),
        headers={"Content-Type": "application/json"},
    )
    assert streamed.status_code == 413, streamed.text
    assert streamed.json() == {"detail": "Request body is too large"}


def test_third_party_interactive_api_consoles_are_disabled(client) -> None:
    """Production never serves Swagger UI or ReDoc script loaders."""
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_public_health_responses_are_coarse(client) -> None:
    """Anonymous health responses expose no activity or dependency inventory."""
    for path in ("/api/health", "/api/health/live", "/api/health/ready"):
        response = client.get(path)
        assert response.status_code == 200
        assert set(response.json()).issubset({"status", "version", "source_revision"})


def test_retention_authority_failure_blocks_new_private_writes(
    client, monkeypatch
) -> None:
    """A failed mandatory purge makes the API fail closed for new writes."""
    import main

    monkeypatch.setattr(main, "private_data_retention_healthy", False)
    response = client.post("/api/session/start", json={})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Private-data retention authority is unavailable"
    }


def test_model_fingerprinting_requires_an_authenticated_paid_tier(client) -> None:
    """A structurally valid fingerprint request is not a public capability."""
    response = client.post(
        "/api/security/fingerprint",
        json={"responses": ["A valid response"]},
    )

    assert response.status_code == 401


def test_webmcp_badge_tool_preserves_negative_identity_provenance() -> None:
    """Model-facing badge output cannot silently promote a self-asserted ID."""
    source = (ROOT / "static/webmcp.js").read_text(encoding="utf-8")
    badge_tool = source[source.index("name: 'mettle_verify_badge'") :]

    assert "untrustedContentHint: true" in badge_tool
    assert "entity_id_verified" in badge_tool
    assert "identity_binding" in badge_tool
    assert "self-asserted provenance" in badge_tool
