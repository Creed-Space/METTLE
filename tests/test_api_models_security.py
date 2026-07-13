"""Security-boundary regression tests for public v2 request models."""

from typing import Any

import pytest
from pydantic import ValidationError

from mettle.api_models import (
    MAX_ANSWER_BYTES,
    CreateSessionRequest,
    OperatorCommitment,
    RoundAnswerRequest,
    VerifyRequest,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operator_pseudonym", "x" * 257),
        ("operator_public_key", "x" * 8193),
        ("signed_commitment", "x" * 1025),
        ("contact_method", "x" * 65),
        ("contact_hash", "x" * 129),
    ],
)
def test_operator_commitment_fields_are_bounded(field: str, value: str) -> None:
    data = {
        "operator_pseudonym": "operator",
        "operator_public_key": "public-key",
        "signed_commitment": "signature",
        "contact_method": "email_hash",
        "contact_hash": "a" * 64,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        OperatorCommitment(**data)
