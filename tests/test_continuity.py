"""Unit coverage for transcript-bound Presence continuity microchallenges."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mettle.continuity import (
    CONTINUITY_ANSWER_KEY,
    CONTINUITY_CHALLENGE_KEY,
    CONTINUITY_PROTOCOL,
    attach_continuity_challenge,
    consume_continuity_evidence,
    issue_continuity_challenge,
    new_continuity_state,
    retire_continuity_secret,
    solve_continuity_challenge,
    verify_continuity_answer,
)


def _presence() -> dict:
    return {
        **new_continuity_state(),
        "transcript_hash": "sha256:" + "a" * 64,
        "sequence": 0,
    }


def _answer(challenge: dict) -> dict:
    return {
        CONTINUITY_ANSWER_KEY: {
            "challenge_id": challenge["challenge_id"],
            "computed": solve_continuity_challenge(challenge),
        }
    }


def test_issue_solve_verify_consume_and_rotate() -> None:
    presence = _presence()
    first = issue_continuity_challenge(presence, "suite:adversarial")
    assert first is not None
    assert first["protocol"] == CONTINUITY_PROTOCOL
    assert len(first["steps"]) == 8
    verify_continuity_answer(presence, "suite:adversarial", _answer(first))
    evidence = consume_continuity_evidence(presence)
    assert evidence == {
        "challenge_family": CONTINUITY_PROTOCOL,
        "challenge_id": first["challenge_id"],
    }
    presence["sequence"] = 1
    presence["transcript_hash"] = "sha256:" + "b" * 64
    second = issue_continuity_challenge(presence, "suite:native")
    assert second is not None
    assert second["challenge_id"] != first["challenge_id"]
    verify_continuity_answer(presence, "suite:native", _answer(second))
    consume_continuity_evidence(presence)
    retire_continuity_secret(presence)
    assert "continuity_secret" not in presence


def test_attach_copies_payload_and_rejects_duplicate_issue() -> None:
    presence = _presence()
    original = {"suite": "adversarial", "challenges": {"q1": {}}}
    issued = attach_continuity_challenge(presence, "suite:adversarial", original)
    assert CONTINUITY_CHALLENGE_KEY not in original
    assert CONTINUITY_CHALLENGE_KEY in issued
    with pytest.raises(RuntimeError, match="already active"):
        issue_continuity_challenge(presence, "suite:adversarial")
    with pytest.raises(RuntimeError, match="Cannot retire"):
        retire_continuity_secret(presence)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda _challenge: {}, "answer is required"),
        (
            lambda challenge: {
                CONTINUITY_ANSWER_KEY: {
                    "challenge_id": "0" * 32,
                    "computed": solve_continuity_challenge(challenge),
                }
            },
            "does not match",
        ),
        (
            lambda challenge: {
                CONTINUITY_ANSWER_KEY: {
                    "challenge_id": challenge["challenge_id"],
                    "computed": True,
                }
            },
            "answer is invalid",
        ),
        (
            lambda challenge: {
                CONTINUITY_ANSWER_KEY: {
                    "challenge_id": challenge["challenge_id"],
                    "computed": solve_continuity_challenge(challenge) ^ 1,
                }
            },
            "answer is incorrect",
        ),
    ],
)
def test_continuity_answer_failures(mutator, message: str) -> None:
    presence = _presence()
    challenge = issue_continuity_challenge(presence, "suite:adversarial")
    assert challenge is not None
    with pytest.raises(ValueError, match=message):
        verify_continuity_answer(presence, "suite:adversarial", mutator(challenge))


def test_continuity_action_and_private_state_fail_closed() -> None:
    presence = _presence()
    challenge = issue_continuity_challenge(presence, "suite:adversarial")
    assert challenge is not None
    with pytest.raises(ValueError, match="unavailable"):
        verify_continuity_answer(presence, "suite:native", _answer(challenge))

    corrupt = _presence()
    corrupt["continuity_secret"] = "bad base64"  # pragma: allowlist secret
    with pytest.raises(ValueError, match="secret is corrupt"):
        issue_continuity_challenge(corrupt, "suite:adversarial")


def test_solver_rejects_malformed_challenges() -> None:
    valid: dict[str, Any] = {
        "protocol": CONTINUITY_PROTOCOL,
        "start": 7,
        "steps": [
            {"op": "xor", "operand": 1},
            {"op": "add", "operand": 2},
            {"op": "multiply", "operand": 4},
            {"op": "rotate_left", "operand": 5},
            {"op": "xor", "operand": 6},
            {"op": "add", "operand": 7},
            {"op": "multiply", "operand": 8},
            {"op": "rotate_left", "operand": 9},
        ],
    }
    assert isinstance(solve_continuity_challenge(valid), int)
    for malformed in (
        {**valid, "protocol": "unknown"},
        {**valid, "start": True},
        {**valid, "steps": []},
        {**valid, "steps": [*valid["steps"][:-1], {"op": "bad", "operand": 1}]},
        {
            **valid,
            "steps": [*valid["steps"][:-1], {"op": "add", "operand": -1}],
        },
        {**valid, "steps": [*valid["steps"][:-1], "bad"]},
    ):
        with pytest.raises(ValueError):
            solve_continuity_challenge(copy.deepcopy(malformed))


def test_legacy_presence_state_remains_compatible() -> None:
    legacy = {"transcript_hash": "sha256:" + "a" * 64, "sequence": 0}
    assert issue_continuity_challenge(legacy, "suite:adversarial") is None
    assert consume_continuity_evidence(legacy) == {}
    verify_continuity_answer(legacy, "suite:adversarial", {})
