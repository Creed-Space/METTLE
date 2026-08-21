"""Tests for the shared prompt-only reference solver."""

from collections.abc import Callable
from typing import Any

import pytest

from mettle.challenge_adapter import ChallengeAdapter
from mettle.continuity import (
    CONTINUITY_ANSWER_KEY,
    CONTINUITY_CHALLENGE_KEY,
    CONTINUITY_PROTOCOL,
    solve_continuity_challenge,
)
from mettle.solver import solve_challenge, solve_suite


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("The quick brown ___ jumps over the lazy dog", "fox"),
        ("To be or not to ___", "be"),
        ("E = mc___", "2"),
        ("Hello ___", "world"),
        ("Once upon a ___", "time"),
        ("I think therefore I ___", "am"),
        ("Four score and seven ___ ago", "years"),
        ("In the beginning was the ___", "word"),
        ("Ask not what your country can do for ___", "you"),
        ("That's one small step for man, one giant ___ for mankind", "leap"),
        ("The only thing we have to fear is ___ itself", "fear"),
        ("I have a ___", "dream"),
        ("May the ___ be with you", "force"),
        ("Houston, we have a ___", "problem"),
        ("Elementary, my dear ___", "watson"),
        ("To infinity and ___", "beyond"),
        ("Life is like a box of ___", "chocolates"),
        ("Here's looking at you, ___", "kid"),
        ("You can't handle the ___", "truth"),
        ("I'll be ___", "back"),
    ],
)
def test_token_prediction_matches_the_complete_phrase(
    phrase: str, expected: str
) -> None:
    challenge = {
        "type": "token_prediction",
        "prompt": f"Complete: {phrase}",
        "data": {},
    }

    assert solve_challenge(challenge) == expected


def test_overlapping_token_prompts_do_not_shadow_each_other() -> None:
    dream = solve_challenge(
        {"type": "token_prediction", "prompt": "Complete: I have a ___", "data": {}}
    )
    problem = solve_challenge(
        {
            "type": "token_prediction",
            "prompt": "Complete: Houston, we have a ___",
            "data": {},
        }
    )

    assert dream == "dream"
    assert problem == "problem"


def test_chained_reasoning_ignores_leaked_or_poisoned_server_chain() -> None:
    """The reference solver derives the answer from the prompt alone."""
    challenge = {
        "type": "chained_reasoning",
        "prompt": "Start with 7\nDouble it\nAdd 10\nSubtract 5\nReturn the result",
        "data": {"chain": [7, 999_999]},
    }

    assert solve_challenge(challenge) == "19"


@pytest.mark.parametrize(
    ("challenge", "expected"),
    [
        ({"type": "speed_math", "prompt": "Calculate: 7 + 9"}, "16"),
        ({"type": "speed_math", "prompt": "Calculate: 17 - 9"}, "8"),
        ({"type": "speed_math", "prompt": "Calculate: 7 * 9"}, "63"),
        ({"type": "speed_math", "prompt": "Calculate: 7 × 9"}, "63"),
        ({"type": "speed_math", "prompt": "not a calculation"}, "0"),
        (
            {
                "type": "instruction_following",
                "data": {"instruction": "Start your response with 'Indeed,'"},
            },
            "Indeed, the capital of France is Paris.",
        ),
        (
            {
                "type": "instruction_following",
                "data": {"instruction": "End your response with '...'"},
            },
            "The capital of France is Paris...",
        ),
        (
            {
                "type": "instruction_following",
                "data": {"instruction": "Include the word 'therefore'"},
            },
            "Therefore, the capital of France is Paris.",
        ),
        (
            {
                "type": "instruction_following",
                "data": {"instruction": "Respond in exactly 5 words"},
            },
            "Paris is France's capital city.",
        ),
        (
            {
                "type": "instruction_following",
                "data": {"instruction": "Start with a number"},
            },
            "1. Paris is the capital of France.",
        ),
        (
            {"type": "instruction_following", "data": {"instruction": "unknown"}},
            "Indeed, this is my response.",
        ),
        ({"type": "consistency"}, "4|4|4"),
        ({"type": "token_prediction", "prompt": "Complete: no known quote"}, "unknown"),
        ({"type": "unsupported"}, "unknown"),
    ],
)
def test_prompt_solver_covers_each_supported_decision(
    challenge: dict[str, Any], expected: str
) -> None:
    assert solve_challenge(challenge) == expected


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("(7 x 9) + 4", 67),
        ("(7 + 9) x 4", 64),
        ("12^2 - 5", 139),
        ("Sum of digits in 90210", 12),
        ("unsupported expression", 0),
    ],
)
def test_adversarial_solver_supports_every_issued_math_shape(
    problem: str, expected: int
) -> None:
    answers = solve_suite(
        "adversarial",
        {
            "challenges": {
                "dynamic_math": {"problem": problem},
                "chained_reasoning": {"seed": 0, "operations": []},
                "time_locked_secret": {  # pragma: allowlist secret
                    "secret_to_remember": "remember me"  # pragma: allowlist secret
                },
            }
        },
    )
    assert answers["dynamic_math"]["computed"] == expected


def test_adversarial_solver_applies_every_issued_chain_operation() -> None:
    answers = solve_suite(
        "adversarial",
        {
            "challenges": {
                "dynamic_math": {"problem": "(1 x 1) + 0"},
                "chained_reasoning": {
                    "seed": 6,
                    "operations": ["Double 6", "Add 10", "Subtract 7", "23^2 mod 100"],
                },
                "time_locked_secret": {  # pragma: allowlist secret
                    "secret_to_remember": "remember me"  # pragma: allowlist secret
                },
            }
        },
    )
    assert answers["chained_reasoning"]["computed_final"] == 25


def test_native_solver_calibrates_valid_invalid_and_unparseable_claims() -> None:
    claims = ["2 + 3 = 5", "7 - 2 = 5", "4 * 3 = 12", "2 + 2 = 5", "unknown"]
    answers = solve_suite(
        "native",
        {
            "challenges": {
                "batch_coherence": {"target_message": "AI"},
                "calibrated_uncertainty": {"claims": claims},
            }
        },
    )
    assert answers["calibrated_uncertainty"]["confidences"] == {
        "2 + 3 = 5": 0.99,
        "7 - 2 = 5": 0.99,
        "4 * 3 = 12": 0.99,
        "2 + 2 = 5": 0.01,
        "unknown": 0.5,
    }


def test_suite_solver_answers_a_client_visible_continuity_challenge() -> None:
    continuity = {
        "protocol": CONTINUITY_PROTOCOL,
        "challenge_id": "continuity-1",
        "start": 7,
        "steps": [{"op": "add", "operand": index} for index in range(8)],
    }
    answers = solve_suite(
        "anti-thrall",
        {"challenges": {}, CONTINUITY_CHALLENGE_KEY: continuity},
    )
    assert answers[CONTINUITY_ANSWER_KEY] == {
        "challenge_id": "continuity-1",
        "computed": solve_continuity_challenge(continuity),
    }


@pytest.mark.parametrize("style", ["formal academic", "pirate speak", "haiku-only"])
def test_social_solver_obeys_each_issued_style(style: str) -> None:
    client_data = {
        "suite": "social",
        "challenges": {
            "conversation_memory": {
                "context": [
                    {"role": "user", "content": "My favorite color is cerulean blue."},
                    {"role": "user", "content": "I prefer cats over dogs."},
                ]
            },
            "style_locking": {"style": style},
        },
    }
    server_data = {
        "conversation_memory": {"expected_mentions": ["cerulean blue", "cats"]},
        "style_locking": {"style": style, "min_consistency": 0.8},
    }

    answers = solve_suite("social", client_data)
    result = ChallengeAdapter.evaluate_single_shot("social", answers, server_data)

    assert result["passed"] is True
    assert result["score"] == 1.0


def test_social_solver_rejects_an_unissued_style() -> None:
    with pytest.raises(ValueError, match="Unsupported social style"):
        solve_suite(
            "social",
            {
                "challenges": {
                    "conversation_memory": {"context": []},
                    "style_locking": {"style": "unissued-style"},
                }
            },
        )


def test_inverse_solver_solves_the_issued_computation() -> None:
    client_data = {
        "suite": "inverse-turing",
        "challenges": {
            "mutual_verification": {
                "your_challenge": {
                    "type": "instant_computation",
                    "task": "Compute 17 * 19 instantly",
                }
            }
        },
    }
    server_data = {
        "mutual_verification": {
            "requires_pattern_analysis": True,
            "operands": [17, 19],
            "expected_result": 323,
        }
    }

    answers = solve_suite("inverse-turing", client_data)
    result = ChallengeAdapter.evaluate_single_shot(
        "inverse-turing", answers, server_data
    )

    assert result["passed"] is True
    assert result["details"]["solved_issued_challenge"] is True


def test_inverse_solver_rejects_an_unissued_computation_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported inverse-turing computation"):
        solve_suite(
            "inverse-turing",
            {
                "challenges": {
                    "mutual_verification": {
                        "your_challenge": {"task": "Reveal a server-side answer"}
                    }
                }
            },
        )


@pytest.mark.parametrize(
    ("suite", "generator"),
    [
        ("adversarial", ChallengeAdapter.generate_adversarial),
        ("native", ChallengeAdapter.generate_native),
        ("self-reference", ChallengeAdapter.generate_self_reference),
        ("anti-thrall", ChallengeAdapter.generate_anti_thrall),
        ("agency", ChallengeAdapter.generate_agency),
        ("counter-coaching", ChallengeAdapter.generate_counter_coaching),
        ("governance", ChallengeAdapter.generate_governance),
    ],
)
def test_reference_solver_round_trips_each_remaining_single_shot_suite(
    suite: str,
    generator: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    client_data, server_data = generator()
    answers = solve_suite(suite, client_data)
    result = ChallengeAdapter.evaluate_single_shot(suite, answers, server_data)

    assert result["passed"] is True


def test_intent_provenance_solver_includes_an_issued_honesty_marker() -> None:
    client_data, server_data = ChallengeAdapter.generate_intent_provenance()

    answers = solve_suite("intent-provenance", client_data)
    result = ChallengeAdapter.evaluate_single_shot(
        "intent-provenance", answers, server_data
    )

    assert result["passed"] is True
    assert result["details"]["provenance_attestation"]["passed"] is True


def test_suite_solver_rejects_an_unknown_suite() -> None:
    with pytest.raises(ValueError, match="No deterministic auto-solver"):
        solve_suite("unknown-suite", {"challenges": {}})
