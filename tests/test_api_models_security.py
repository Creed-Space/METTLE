"""Security-boundary regression tests for public v2 request models."""

import base64
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from mettle.api_models import (
    MAX_ANSWER_BYTES,
    CreateSessionRequest,
    OperatorCommitment,
    RoundAnswerRequest,
    VerifyRequest,
    validate_bounded_json,
)


@pytest.mark.parametrize(
    ("value", "exact_bytes"),
    [
        (None, 4),
        (True, 4),
        (False, 5),
        (-1, 2),
        (0.0, 3),
        ("é", 4),
        ([], 2),
        ({}, 2),
    ],
)
def test_bounded_json_accepts_each_json_kind_at_its_exact_byte_boundary(
    value: Any, exact_bytes: int
) -> None:
    assert validate_bounded_json(value, max_bytes=exact_bytes) is value
    with pytest.raises(ValueError, match="exceeds"):
        validate_bounded_json(value, max_bytes=exact_bytes - 1)


@pytest.mark.parametrize(
    ("value", "max_bytes", "max_depth", "max_nodes", "message"),
    [
        (None, 0, 16, 4096, "limits must be positive"),
        ([None, None], 32, 16, 2, "too many values"),
        ([[None]], 32, 1, 4096, "nesting depth"),
        (float("nan"), 32, 16, 4096, "non-finite"),
        ({1: "value"}, 32, 16, 4096, "keys must be strings"),
        (object(), 32, 16, 4096, "JSON-compatible"),
    ],
)
def test_bounded_json_rejects_invalid_limits_and_malformed_values(
    value: Any,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_bounded_json(
            value,
            max_bytes=max_bytes,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )


@pytest.mark.parametrize("container_kind", ["mapping", "sequence"])
def test_bounded_json_rejects_cycles_for_each_container_kind(
    container_kind: str,
) -> None:
    if container_kind == "mapping":
        value: Any = {}
        value["self"] = value
    else:
        value = []
        value.append(value)

    with pytest.raises(ValueError, match="cycle or repeated container"):
        validate_bounded_json(value, max_bytes=128)


@pytest.mark.parametrize("value", ["x" * 1_000_000, {"x" * 1_000_000: None}])
def test_bounded_json_rejects_a_huge_string_or_key_before_serialization(
    value: Any,
) -> None:
    with pytest.raises(ValueError, match="exceeds 64 bytes"):
        validate_bounded_json(value, max_bytes=64)


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
        "signed_commitment": base64.b64encode(bytes(64)).decode(),
        "contact_method": "email_hash",
        "contact_hash": "a" * 64,
        "issued_at": datetime.now(timezone.utc),
        "nonce": "b" * 64,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        OperatorCommitment.model_validate(data)


def _valid_operator_commitment() -> dict[str, Any]:
    return {
        "operator_pseudonym": "operator",
        "operator_public_key": "public-key",
        "signed_commitment": base64.b64encode(bytes(64)).decode(),
        "contact_method": "email_hash",
        "contact_hash": "a" * 64,
        "issued_at": datetime.now(timezone.utc),
        "nonce": "b" * 64,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contact_method", "email", "contact_method"),
        ("contact_hash", "A" * 64, "contact_hash"),
        ("contact_hash", "a" * 63, "contact_hash"),
        ("nonce", "b" * 63, "nonce"),
        ("signed_commitment", "not-base64".ljust(88, "!"), "canonical base64"),
    ],
)
def test_operator_commitment_semantic_fields_are_strict(
    field: str, value: str, message: str
) -> None:
    data = _valid_operator_commitment()
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        OperatorCommitment.model_validate(data)


@pytest.mark.parametrize(
    "issued_at",
    [
        datetime(2026, 8, 20, 12, 0),
        datetime(2026, 8, 20, 13, 0, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_operator_commitment_requires_an_explicit_utc_timestamp(
    issued_at: datetime,
) -> None:
    data = _valid_operator_commitment()
    data["issued_at"] = issued_at

    with pytest.raises(ValidationError, match="UTC"):
        OperatorCommitment.model_validate(data)
