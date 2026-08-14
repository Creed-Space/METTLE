"""Tests for the shared prompt-only reference solver."""

import pytest

from scripts.testing.solver import solve_challenge


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
