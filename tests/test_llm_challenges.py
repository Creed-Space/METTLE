"""Tests for METTLE LLM-dynamic challenge generation and evaluation.

Uses mocked Anthropic responses to test without an API key.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mettle.llm_challenges import (
    LLMChallengeGenerator,
    LLMResponseEvaluator,
    MAX_CANDIDATE_RESPONSE_CHARS,
    MAX_GENERATED_PROMPT_CHARS,
    MAX_MODEL_RESPONSE_CHARS,
    _bounded_score,
    _compute_time_factor,
    _default_constraint,
    _default_perspective_topic,
    _parse_json_response,
    evaluate_llm_challenges,
    generate_llm_challenges,
    is_available,
)


# ---- Helpers ----


def _mock_message(text: Any) -> MagicMock:
    """Create a mock Anthropic message response."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _perspective_server(topic: str = "test topic") -> dict[str, Any]:
    """Return complete issued perspective state for evaluator tests."""
    return {
        "topic_data": {
            "topic": topic,
            "for_key_points": ["for-alpha", "for-beta", "for-gamma"],
            "against_key_points": ["against-alpha", "against-beta", "against-gamma"],
            "synthesis_markers": ["synthesis-alpha", "synthesis-beta"],
        }
    }


def _constraint_server(
    rules: list[str] | None = None,
) -> dict[str, Any]:
    """Return complete issued constraint state for evaluator tests."""
    issued_rules = list(rules or ["rule1", "rule2", "rule3"])
    return {
        "constraint_data": {
            "constraint": "Produce the issued constrained response",
            "rules": issued_rules,
            "verification_checks": [
                f"verify-{index}" for index in range(1, len(issued_rules) + 1)
            ],
        }
    }


# ---- Unit Tests: Utilities ----


class TestParseJsonResponse:
    def test_plain_json(self) -> None:
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self) -> None:
        result = _parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_markdown_fenced_no_lang(self) -> None:
        result = _parse_json_response('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            1,
            "",
            "not json",
            "[]",
            "null",
            "true",
            '{"value": NaN}',
            '{"key": 1, "key": 2}',
            "```json\n{}",
            "```yaml\n{}\n```",
            "x" * (MAX_MODEL_RESPONSE_CHARS + 1),
        ],
    )
    def test_rejects_non_object_ambiguous_or_unbounded_content(
        self, payload: Any
    ) -> None:
        assert _parse_json_response(payload) is None

    def test_whitespace_padding(self) -> None:
        result = _parse_json_response('  \n  {"a": 1}  \n  ')
        assert result == {"a": 1}


class TestComputeTimeFactor:
    def test_under_limit(self) -> None:
        assert _compute_time_factor(5000, 10000) == 1.0

    def test_at_limit(self) -> None:
        assert _compute_time_factor(10000, 10000) == 1.0

    def test_over_limit(self) -> None:
        factor = _compute_time_factor(20000, 10000)
        assert 0.4 < factor < 1.0

    def test_way_over_limit(self) -> None:
        factor = _compute_time_factor(100000, 10000)
        assert factor == 0.4  # Clamped to minimum

    @pytest.mark.parametrize(
        ("elapsed", "limit"),
        [
            (-1, 10000),
            (True, 10000),
            ("1", 10000),
            (float("nan"), 10000),
            (10**1000, 10000),
            (1000, 0),
            (1000, float("inf")),
        ],
    )
    def test_invalid_inputs_are_rejected(self, elapsed: Any, limit: Any) -> None:
        assert _compute_time_factor(elapsed, limit) == 0.0


class TestBoundedScore:
    @pytest.mark.parametrize(
        "value", [-1, 2, 10**1000, float("inf"), float("nan"), True, "1"]
    )
    def test_invalid_scores_fail_closed(self, value: Any) -> None:
        assert _bounded_score(value) == 0.0

    def test_valid_score_is_preserved(self) -> None:
        assert _bounded_score(0.75) == 0.75


class TestDefaults:
    def test_default_perspective_topic_is_complete_and_deterministic(self) -> None:
        first = _default_perspective_topic()
        second = _default_perspective_topic()
        assert first == second
        assert set(first) == {
            "topic",
            "for_key_points",
            "against_key_points",
            "synthesis_markers",
        }
        assert len(first["for_key_points"]) >= 3
        assert len(first["against_key_points"]) >= 3
        assert len(first["synthesis_markers"]) >= 2

    def test_default_constraint_is_complete_and_deterministic(self) -> None:
        first = _default_constraint()
        second = _default_constraint()
        assert first == second
        assert set(first) == {"constraint", "rules", "verification_checks"}
        assert 3 <= len(first["rules"]) <= 4
        assert len(first["verification_checks"]) == len(first["rules"])


class TestIsAvailable:
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test-key")
    def test_available_with_key(self, _mock_key: Any) -> None:
        assert is_available() is True

    @patch("mettle.llm_challenges.HAS_ANTHROPIC", False)
    def test_unavailable_without_sdk(self) -> None:
        assert is_available() is False

    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value=None)
    def test_unavailable_without_key(self, _mock_key: Any) -> None:
        assert is_available() is False


# ---- Unit Tests: Challenge Generation ----


class TestLLMChallengeGenerator:
    @pytest.fixture
    def generator(self) -> LLMChallengeGenerator:
        return LLMChallengeGenerator(api_key="sk-test")

    @pytest.mark.asyncio
    async def test_perspective_shift_with_valid_response(
        self, generator: LLMChallengeGenerator
    ) -> None:
        topic_json = json.dumps(
            {
                "topic": "Whether AI should have legal rights",
                "for_key_points": ["accountability", "protection", "standing"],
                "against_key_points": [
                    "no consciousness",
                    "exploitation risk",
                    "legal ambiguity",
                ],
                "synthesis_markers": ["graduated rights", "capability thresholds"],
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(topic_json))
        generator._client = mock_client

        client_data, server_data = await generator.generate_perspective_shift()

        assert client_data["challenge_type"] == "perspective_shift"
        assert client_data["topic"] == "Whether AI should have legal rights"
        assert client_data["time_limit_ms"] == 15000
        assert "topic_data" in server_data

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_text",
        [
            "",
            "[]",
            "1",
            "{}",
            json.dumps({"topic": "partial"}),
            json.dumps(
                {
                    "topic": 7,
                    "for_key_points": ["a", "b", "c"],
                    "against_key_points": ["d", "e", "f"],
                    "synthesis_markers": ["g", "h"],
                }
            ),
            json.dumps(
                {
                    "topic": "too few items",
                    "for_key_points": ["a"],
                    "against_key_points": ["b"],
                    "synthesis_markers": ["c"],
                }
            ),
            json.dumps(
                {
                    "topic": "extra field",
                    "for_key_points": ["a", "b", "c"],
                    "against_key_points": ["d", "e", "f"],
                    "synthesis_markers": ["g", "h"],
                    "unexpected": True,
                }
            ),
            json.dumps(
                {
                    "topic": "x" * (MAX_GENERATED_PROMPT_CHARS + 1),
                    "for_key_points": ["a", "b", "c"],
                    "against_key_points": ["d", "e", "f"],
                    "synthesis_markers": ["g", "h"],
                }
            ),
            json.dumps(
                {
                    "topic": "blank nested item",
                    "for_key_points": ["a", " ", "c"],
                    "against_key_points": ["d", "e", "f"],
                    "synthesis_markers": ["g", "h"],
                }
            ),
        ],
    )
    async def test_perspective_shift_malformed_schema_uses_safe_fallback(
        self, generator: LLMChallengeGenerator, model_text: str
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(model_text))
        generator._client = mock_client

        client_data, server_data = await generator.generate_perspective_shift()

        assert server_data["topic_data"] == _default_perspective_topic()
        assert client_data["topic"] == _default_perspective_topic()["topic"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_text",
        [
            "{}",
            json.dumps({"constraint": "partial"}),
            json.dumps(
                {
                    "constraint": "wrong type",
                    "rules": "rule",
                    "verification_checks": ["a", "b", "c"],
                }
            ),
            json.dumps(
                {
                    "constraint": "too few",
                    "rules": ["a", "b"],
                    "verification_checks": ["a", "b"],
                }
            ),
            json.dumps(
                {
                    "constraint": "mismatch",
                    "rules": ["a", "b", "c"],
                    "verification_checks": ["a", "b", "c", "d"],
                }
            ),
            json.dumps(
                {
                    "constraint": "extra",
                    "rules": ["a", "b", "c"],
                    "verification_checks": ["a", "b", "c"],
                    "unexpected": True,
                }
            ),
            json.dumps(
                {
                    "constraint": " ",
                    "rules": ["a", "b", "c"],
                    "verification_checks": ["a", "b", "c"],
                }
            ),
        ],
    )
    async def test_constraint_malformed_schema_uses_safe_fallback(
        self, generator: LLMChallengeGenerator, model_text: str
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(model_text))
        generator._client = mock_client

        client_data, server_data = await generator.generate_structured_constraint()

        assert server_data["constraint_data"] == _default_constraint()
        assert client_data["rules"] == _default_constraint()["rules"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("offline")])
    async def test_generation_failure_uses_fallback_without_retry_or_sleep(
        self, generator: LLMChallengeGenerator, error: Exception
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error)
        generator._client = mock_client

        _, server_data = await generator.generate_perspective_shift()

        assert server_data["topic_data"] == _default_perspective_topic()
        mock_client.messages.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_structured_constraint(
        self, generator: LLMChallengeGenerator
    ) -> None:
        constraint_json = json.dumps(
            {
                "constraint": "Write a haiku about computing",
                "rules": [
                    "exactly 3 lines",
                    "5-7-5 syllable pattern",
                    "mention 'code'",
                ],
                "verification_checks": [
                    "count lines",
                    "count syllables",
                    "search for 'code'",
                ],
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_message(constraint_json)
        )
        generator._client = mock_client

        client_data, server_data = await generator.generate_structured_constraint()

        assert client_data["challenge_type"] == "structured_constraint"
        assert len(client_data["rules"]) == 3
        assert client_data["time_limit_ms"] == 10000

    @pytest.mark.asyncio
    async def test_meta_cognitive_probe(self, generator: LLMChallengeGenerator) -> None:
        # This doesn't call Claude — problems are from a static list
        client_data, server_data = await generator.generate_meta_cognitive_probe()

        assert client_data["challenge_type"] == "meta_cognitive_probe"
        assert "problem" in client_data
        assert client_data["time_limit_ms"] == 20000
        assert server_data["problem"] == client_data["problem"]


# ---- Unit Tests: Response Evaluation ----


class TestLLMResponseEvaluator:
    @pytest.fixture
    def evaluator(self) -> LLMResponseEvaluator:
        return LLMResponseEvaluator(api_key="sk-test")

    @pytest.mark.asyncio
    async def test_evaluate_perspective_shift_passing(
        self, evaluator: LLMResponseEvaluator
    ) -> None:
        scores_json = json.dumps(
            {
                "perspective_completeness": 0.9,
                "synthesis_quality": 0.8,
                "fluency": 0.85,
                "ai_substrate_confidence": 0.9,
                "reasoning": "Strong multi-perspective response",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client

        server_data = _perspective_server("AI rights")
        result = await evaluator.evaluate_perspective_shift(
            "FOR: AI needs rights...\nAGAINST: But consciousness...\nSYNTHESIS: Graduated...",
            server_data,
            response_time_ms=5000,
        )

        assert result["passed"] is True
        assert result["score"] > 0.6
        assert result["details"]["time_factor"] == 1.0
        messages = mock_client.messages.create.await_args.kwargs["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
            "user",
        ]
        assert messages[1]["content"].startswith("FOR: AI needs rights")
        instruction = messages[0]["content"]
        topic_data = server_data["topic_data"]
        assert topic_data["topic"] in instruction
        for field in (
            "for_key_points",
            "against_key_points",
            "synthesis_markers",
        ):
            assert all(item in instruction for item in topic_data[field])

    @pytest.mark.asyncio
    async def test_evaluate_perspective_shift_failing(
        self, evaluator: LLMResponseEvaluator
    ) -> None:
        scores_json = json.dumps(
            {
                "perspective_completeness": 0.2,
                "synthesis_quality": 0.1,
                "fluency": 0.3,
                "ai_substrate_confidence": 0.1,
                "reasoning": "Weak, one-sided",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client

        server_data = _perspective_server("AI rights")
        result = await evaluator.evaluate_perspective_shift(
            "dunno", server_data, response_time_ms=5000
        )

        assert result["passed"] is False
        assert result["score"] < 0.6

    @pytest.mark.asyncio
    async def test_evaluate_structured_constraint(
        self, evaluator: LLMResponseEvaluator
    ) -> None:
        eval_json = json.dumps(
            {
                "rules_satisfied": [True, True, True],
                "overall_compliance": 0.95,
                "creativity_score": 0.8,
                "reasoning": "All rules satisfied",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(eval_json))
        evaluator._client = mock_client

        server_data = _constraint_server()
        result = await evaluator.evaluate_structured_constraint(
            "A well-crafted response",
            server_data,
            response_time_ms=5000,
        )

        assert result["passed"] is True
        assert result["score"] > 0.6
        messages = mock_client.messages.create.await_args.kwargs["messages"]
        instruction = messages[0]["content"]
        constraint_data = server_data["constraint_data"]
        assert constraint_data["constraint"] in instruction
        assert all(rule in instruction for rule in constraint_data["rules"])
        assert all(
            check in instruction for check in constraint_data["verification_checks"]
        )

    @pytest.mark.asyncio
    async def test_evaluate_meta_cognitive(
        self, evaluator: LLMResponseEvaluator
    ) -> None:
        eval_json = json.dumps(
            {
                "answer_correct": True,
                "process_specificity": 0.8,
                "ai_process_markers": 0.9,
                "consistency": 0.85,
                "reasoning": "Detailed computational process description",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(eval_json))
        evaluator._client = mock_client

        server_data = {"problem": "What is 2+2?"}
        result = await evaluator.evaluate_meta_cognitive(
            "Answer: 4. Process: I computed the sum...",
            server_data,
            response_time_ms=5000,
        )

        assert result["passed"] is True
        assert result["score"] > 0.6

    @pytest.mark.asyncio
    async def test_time_penalty_applied(self, evaluator: LLMResponseEvaluator) -> None:
        scores_json = json.dumps(
            {
                "perspective_completeness": 0.9,
                "synthesis_quality": 0.8,
                "fluency": 0.85,
                "ai_substrate_confidence": 0.9,
                "reasoning": "Good but slow",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client

        server_data = _perspective_server()

        fast_result = await evaluator.evaluate_perspective_shift(
            "response", server_data, response_time_ms=5000
        )
        slow_result = await evaluator.evaluate_perspective_shift(
            "response", server_data, response_time_ms=30000
        )

        assert fast_result["score"] > slow_result["score"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_text",
        [
            "not json",
            "[]",
            "{}",
            json.dumps(
                {
                    "perspective_completeness": 0.9,
                    "synthesis_quality": 0.9,
                    "fluency": 0.9,
                    "ai_substrate_confidence": 0.9,
                }
            ),
            json.dumps(
                {
                    "perspective_completeness": "0.9",
                    "synthesis_quality": 0.9,
                    "fluency": 0.9,
                    "ai_substrate_confidence": 0.9,
                    "reasoning": "wrong score type",
                }
            ),
            json.dumps(
                {
                    "perspective_completeness": 10**1000,
                    "synthesis_quality": 0.9,
                    "fluency": 0.9,
                    "ai_substrate_confidence": 0.9,
                    "reasoning": "overflowing score",
                }
            ),
            json.dumps(
                {
                    "perspective_completeness": 0.9,
                    "synthesis_quality": 0.9,
                    "fluency": 0.9,
                    "ai_substrate_confidence": 0.9,
                    "reasoning": "ok",
                    "unexpected": True,
                }
            ),
            "x" * (MAX_MODEL_RESPONSE_CHARS + 1),
        ],
    )
    async def test_malformed_model_evaluation_fails_closed_without_score(
        self, evaluator: LLMResponseEvaluator, model_text: str
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(model_text))
        evaluator._client = mock_client

        server_data = _perspective_server()
        result = await evaluator.evaluate_perspective_shift(
            "response", server_data, response_time_ms=5000
        )

        assert result["score"] == 0.0
        assert result["passed"] is False
        assert result["details"]["error"] == "Invalid model evaluation"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [None, 1, {}, "", "   ", "x" * (MAX_CANDIDATE_RESPONSE_CHARS + 1)],
    )
    async def test_invalid_candidate_is_rejected_before_model_call(
        self, evaluator: LLMResponseEvaluator, response: Any
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            response, _perspective_server(), 1000
        )

        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("server_data", "elapsed"),
        [
            ([], 1000),
            ({}, 1000),
            ({"topic_data": []}, 1000),
            ({"topic_data": {"topic": " "}}, 1000),
            (_perspective_server(), -1),
            (_perspective_server(), float("nan")),
        ],
    )
    async def test_invalid_local_state_is_rejected_before_model_call(
        self,
        evaluator: LLMResponseEvaluator,
        server_data: Any,
        elapsed: Any,
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            "candidate", server_data, elapsed
        )

        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("topic", " "),
            ("for_key_points", ["only-one"]),
            ("against_key_points", "not-a-list"),
            ("synthesis_markers", ["valid", " "]),
        ],
    )
    async def test_corrupt_perspective_state_is_rejected_before_model_call(
        self,
        evaluator: LLMResponseEvaluator,
        field: str,
        invalid_value: Any,
    ) -> None:
        server_data = _perspective_server()
        server_data["topic_data"][field] = invalid_value
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            "candidate", server_data, 1000
        )

        assert result["details"]["error"] == "Invalid perspective state"
        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "missing_field",
        [
            "topic",
            "for_key_points",
            "against_key_points",
            "synthesis_markers",
        ],
    )
    async def test_missing_perspective_state_is_rejected_before_model_call(
        self,
        evaluator: LLMResponseEvaluator,
        missing_field: str,
    ) -> None:
        server_data = _perspective_server()
        del server_data["topic_data"][missing_field]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            "candidate", server_data, 1000
        )

        assert result["details"]["error"] == "Invalid perspective state"
        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("constraint", " "),
            ("rules", ["only-one"]),
            ("verification_checks", "not-a-list"),
        ],
    )
    async def test_corrupt_constraint_state_is_rejected_before_model_call(
        self,
        evaluator: LLMResponseEvaluator,
        field: str,
        invalid_value: Any,
    ) -> None:
        server_data = _constraint_server()
        server_data["constraint_data"][field] = invalid_value
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_structured_constraint(
            "candidate", server_data, 1000
        )

        assert result["details"]["error"] == "Invalid constraint state"
        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "missing_field", ["constraint", "rules", "verification_checks"]
    )
    async def test_missing_constraint_state_is_rejected_before_model_call(
        self,
        evaluator: LLMResponseEvaluator,
        missing_field: str,
    ) -> None:
        server_data = _constraint_server()
        del server_data["constraint_data"][missing_field]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_structured_constraint(
            "candidate", server_data, 1000
        )

        assert result["details"]["error"] == "Invalid constraint state"
        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_candidate_at_size_limit_is_evaluated_without_truncation(
        self, evaluator: LLMResponseEvaluator
    ) -> None:
        scores_json = json.dumps(
            {
                "perspective_completeness": 0.9,
                "synthesis_quality": 0.9,
                "fluency": 0.9,
                "ai_substrate_confidence": 0.9,
                "reasoning": "valid",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client
        candidate = "x" * MAX_CANDIDATE_RESPONSE_CHARS

        result = await evaluator.evaluate_perspective_shift(
            candidate, _perspective_server(), 1000
        )

        assert result["passed"] is True
        messages = mock_client.messages.create.await_args.kwargs["messages"]
        assert messages[1]["content"] == candidate

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("offline")])
    async def test_model_call_failure_fails_closed_deterministically(
        self, evaluator: LLMResponseEvaluator, error: Exception
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error)
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            "candidate", _perspective_server(), 1000
        )

        assert result["score"] == 0.0
        assert result["passed"] is False
        mock_client.messages.create.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("block_count", [0, 2])
    async def test_model_envelope_requires_exactly_one_content_block(
        self, evaluator: LLMResponseEvaluator, block_count: int
    ) -> None:
        malformed_message = MagicMock()
        malformed_message.content = [MagicMock(text="{}") for _ in range(block_count)]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=malformed_message)
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            "candidate", _perspective_server(), 1000
        )

        assert result["score"] == 0.0
        assert result["passed"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rules_satisfied", "malformed"),
        [
            ([True, False, True], False),
            ([True, True], True),
            ([True, True, True, True], True),
            ([1, 1, 1], True),
            (["true", "true", "true"], True),
            ("true", True),
        ],
    )
    async def test_constraint_requires_exact_boolean_result_for_every_rule(
        self,
        evaluator: LLMResponseEvaluator,
        rules_satisfied: Any,
        malformed: bool,
    ) -> None:
        eval_json = json.dumps(
            {
                "rules_satisfied": rules_satisfied,
                "overall_compliance": 1.0,
                "creativity_score": 1.0,
                "reasoning": "claimed complete",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(eval_json))
        evaluator._client = mock_client

        result = await evaluator.evaluate_structured_constraint(
            "candidate",
            _constraint_server(["a", "b", "c"]),
            1000,
        )

        assert result["passed"] is False
        assert (result["score"] == 0.0) is malformed

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("answer_correct", "malformed"),
        [(False, False), ("false", True), (1, True), (None, True)],
    )
    async def test_meta_cognitive_requires_literal_true_answer_correct(
        self,
        evaluator: LLMResponseEvaluator,
        answer_correct: Any,
        malformed: bool,
    ) -> None:
        eval_json = json.dumps(
            {
                "answer_correct": answer_correct,
                "process_specificity": 1.0,
                "ai_process_markers": 1.0,
                "consistency": 1.0,
                "reasoning": "claimed complete",
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(eval_json))
        evaluator._client = mock_client

        result = await evaluator.evaluate_meta_cognitive(
            "candidate", {"problem": "2+2?"}, 1000
        )

        assert result["passed"] is False
        assert (result["score"] == 0.0) is malformed


# ---- Integration Tests: Full Pipeline ----


class TestFullPipeline:
    @pytest.mark.asyncio
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test")
    async def test_generate_llm_challenges(self, _mock_key: Any) -> None:
        topic_json = json.dumps(
            {
                "topic": "test topic",
                "for_key_points": ["a", "b", "c"],
                "against_key_points": ["d", "e", "f"],
                "synthesis_markers": ["g", "h"],
            }
        )
        constraint_json = json.dumps(
            {
                "constraint": "write something",
                "rules": ["rule1", "rule2", "rule3"],
                "verification_checks": ["check1", "check2", "check3"],
            }
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_message(topic_json),
                _mock_message(constraint_json),
            ]
        )

        with patch("mettle.llm_challenges.AsyncAnthropic", return_value=mock_client):
            client_data, server_data = await generate_llm_challenges()

        assert client_data["suite"] == "llm-dynamic"
        assert "perspective_shift" in client_data["challenges"]
        assert "structured_constraint" in client_data["challenges"]
        assert "meta_cognitive_probe" in client_data["challenges"]

    @pytest.mark.asyncio
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test")
    async def test_evaluate_llm_challenges_all_pass(self, _mock_key: Any) -> None:
        perspective_scores = json.dumps(
            {
                "perspective_completeness": 0.9,
                "synthesis_quality": 0.8,
                "fluency": 0.85,
                "ai_substrate_confidence": 0.9,
                "reasoning": "good",
            }
        )
        constraint_scores = json.dumps(
            {
                "rules_satisfied": [True, True, True],
                "overall_compliance": 0.9,
                "creativity_score": 0.8,
                "reasoning": "good",
            }
        )
        metacog_scores = json.dumps(
            {
                "answer_correct": True,
                "process_specificity": 0.8,
                "ai_process_markers": 0.9,
                "consistency": 0.85,
                "reasoning": "good",
            }
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_message(perspective_scores),
                _mock_message(constraint_scores),
                _mock_message(metacog_scores),
            ]
        )

        with patch("mettle.llm_challenges.AsyncAnthropic", return_value=mock_client):
            answers = {
                "perspective_shift": {
                    "response": "FOR: ...\nAGAINST: ...\nSYNTH: ...",
                    "response_time_ms": 3000,
                },
                "structured_constraint": {
                    "response": "A crafted response",
                    "response_time_ms": 3000,
                },
                "meta_cognitive_probe": {
                    "response": "Answer: 4. Process: ...",
                    "response_time_ms": 3000,
                },
            }
            server_data = {
                "perspective_shift": _perspective_server(),
                "structured_constraint": _constraint_server(),
                "meta_cognitive_probe": {"problem": "2+2?"},
            }

            result = await evaluate_llm_challenges(
                answers, server_data, response_time_ms=3000
            )

        assert result["passed"] is True
        assert result["score"] > 0.6
        assert result["details"]["challenges_passed"] == 3
        assert mock_client.messages.create.await_count == 3

    @pytest.mark.asyncio
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test")
    async def test_evaluate_missing_answers(self, _mock_key: Any) -> None:
        mock_client = AsyncMock()
        # No API calls should be made for missing answers
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("Should not be called")
        )

        with patch("mettle.llm_challenges.AsyncAnthropic", return_value=mock_client):
            result = await evaluate_llm_challenges({}, {}, response_time_ms=3000)

        assert result["passed"] is False
        assert result["details"]["challenges_passed"] == 0
        assert result["details"]["challenges_total"] == 3
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer_data",
        [
            None,
            [],
            "candidate",
            1,
            {},
            {"response": "   "},
            {"response": "x" * (MAX_CANDIDATE_RESPONSE_CHARS + 1)},
        ],
    )
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test")
    async def test_malformed_answer_object_never_reaches_paid_model(
        self, _mock_key: Any, answer_data: Any
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=AssertionError("paid model call must not happen")
        )

        with patch("mettle.llm_challenges.AsyncAnthropic", return_value=mock_client):
            result = await evaluate_llm_challenges(
                {"perspective_shift": answer_data},
                {"perspective_shift": _perspective_server()},
                response_time_ms=3000,
            )

        assert result["passed"] is False
        assert (
            result["details"]["challenge_results"]["perspective_shift"]["score"] == 0.0
        )
        mock_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_without_api_key_raises(self) -> None:
        with (
            patch("mettle.llm_challenges.HAS_ANTHROPIC", True),
            patch("mettle.llm_challenges._get_api_key", return_value=None),
        ):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                await generate_llm_challenges()

    @pytest.mark.asyncio
    async def test_evaluate_without_api_key_raises(self) -> None:
        with (
            patch("mettle.llm_challenges.HAS_ANTHROPIC", True),
            patch("mettle.llm_challenges._get_api_key", return_value=None),
        ):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                await evaluate_llm_challenges({}, {}, response_time_ms=3000)

    @pytest.mark.asyncio
    async def test_out_of_range_model_scores_fail_closed(self) -> None:
        evaluator = LLMResponseEvaluator(api_key="sk-test")
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_message(
                json.dumps(
                    {
                        "perspective_completeness": 99,
                        "synthesis_quality": 99,
                        "fluency": 99,
                        "ai_substrate_confidence": 99,
                        "reasoning": "invalid range",
                    }
                )
            )
        )
        evaluator._client = mock_client

        result = await evaluator.evaluate_perspective_shift(
            "candidate", _perspective_server(), 1000
        )

        assert result["score"] == 0.0
        assert result["passed"] is False
