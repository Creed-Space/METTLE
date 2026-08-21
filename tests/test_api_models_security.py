"""Security-boundary regression tests for public v2 request models."""

from typing import Any

import pytest
from pydantic import ValidationError

from mettle.api_models import (
    MAX_ANSWER_BYTES,
    CreateSessionRequest,
    RoundAnswerRequest,
    VerifyRequest,
    validate_bounded_json,
)


def test_duplicate_suites_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate suites"):
        CreateSessionRequest(suites=["adversarial", "adversarial"])


def test_all_cannot_be_combined_with_explicit_suites() -> None:
    with pytest.raises(ValidationError, match="cannot be combined"):
        CreateSessionRequest(suites=["all", "adversarial"])


@pytest.mark.parametrize("model", [RoundAnswerRequest, VerifyRequest])
def test_oversized_answer_object_is_rejected(model) -> None:
    kwargs: dict[str, Any] = {"answers": {"value": "x" * MAX_ANSWER_BYTES}}
    if model is VerifyRequest:
        kwargs["suite"] = "adversarial"

    with pytest.raises(ValidationError, match="Answer payload exceeds"):
        model(**kwargs)


def test_excessive_top_level_answer_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="too many top-level fields"):
        VerifyRequest(
            suite="adversarial",
            answers={f"field-{index}": index for index in range(101)},
        )


def test_vcp_token_is_bounded() -> None:
    with pytest.raises(ValidationError):
        CreateSessionRequest(vcp_token="x" * 32769)


def test_retired_operator_commitment_is_rejected_not_silently_ignored() -> None:
    with pytest.raises(ValidationError, match="operator_commitment"):
        CreateSessionRequest.model_validate(
            {
                "operator_commitment": {
                    "operator_pseudonym": "operator",
                    "operator_public_key": "public-key",
                    "signed_commitment": "signature",
                    "contact_method": "email_hash",
                    "contact_hash": "a" * 64,
                }
            }
        )


@pytest.mark.parametrize(
    ("value", "exact_bytes"),
    [(None, 4), (True, 4), (False, 5), (-1, 2), (0.0, 3), ("é", 4), ([], 2), ({}, 2)],
)
def test_bounded_json_accepts_exact_boundaries(value: Any, exact_bytes: int) -> None:
    assert validate_bounded_json(value, max_bytes=exact_bytes) is value
    with pytest.raises(ValueError, match="exceeds"):
        validate_bounded_json(value, max_bytes=exact_bytes - 1)


def test_bounded_json_rejects_cycles_nonfinite_and_huge_scalars() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    for value, message in [
        (cycle, "cycle"),
        (float("nan"), "non-finite"),
        ({1: "value"}, "keys"),
        ("x" * 1_000_000, "exceeds"),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_bounded_json(value, max_bytes=64)
