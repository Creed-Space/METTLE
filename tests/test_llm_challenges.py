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
    _compute_time_factor,
    _default_constraint,
    _default_perspective_topic,
    _parse_json_response,
    evaluate_llm_challenges,
    generate_llm_challenges,
    is_available,
)


# ---- Helpers ----


def _mock_message(text: str) -> MagicMock:
    """Create a mock Anthropic message response."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


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

    def test_invalid_json(self) -> None:
        assert _parse_json_response("not json") is None

    def test_empty_string(self) -> None:
        assert _parse_json_response("") is None

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


class TestDefaults:
    def test_default_perspective_topic_structure(self) -> None:
        topic = _default_perspective_topic()
        assert "topic" in topic
        assert "for_key_points" in topic
        assert "against_key_points" in topic
        assert "synthesis_markers" in topic

    def test_default_constraint_structure(self) -> None:
        constraint = _default_constraint()
        assert "constraint" in constraint
        assert "rules" in constraint
        assert "verification_checks" in constraint
        assert len(constraint["rules"]) >= 3


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
    async def test_perspective_shift_with_valid_response(self, generator: LLMChallengeGenerator) -> None:
        topic_json = json.dumps({
            "topic": "Whether AI should have legal rights",
            "for_key_points": ["accountability", "protection"],
            "against_key_points": ["no consciousness", "exploitation risk"],
            "synthesis_markers": ["graduated rights"],
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(topic_json))
        generator._client = mock_client

        client_data, server_data = await generator.generate_perspective_shift()

        assert client_data["challenge_type"] == "perspective_shift"
        assert client_data["topic"] == "Whether AI should have legal rights"
        assert client_data["time_limit_ms"] == 15000
        assert "topic_data" in server_data

    @pytest.mark.asyncio
    async def test_perspective_shift_with_invalid_json_falls_back(self, generator: LLMChallengeGenerator) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message("not valid json"))
        generator._client = mock_client

        client_data, server_data = await generator.generate_perspective_shift()

        # Should use fallback topic
        assert client_data["challenge_type"] == "perspective_shift"
        assert len(client_data["topic"]) > 10
        assert "topic_data" in server_data

    @pytest.mark.asyncio
    async def test_structured_constraint(self, generator: LLMChallengeGenerator) -> None:
        constraint_json = json.dumps({
            "constraint": "Write a haiku about computing",
            "rules": ["exactly 3 lines", "5-7-5 syllable pattern", "mention 'code'"],
            "verification_checks": ["count lines", "count syllables", "search for 'code'"],
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(constraint_json))
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
    async def test_evaluate_perspective_shift_passing(self, evaluator: LLMResponseEvaluator) -> None:
        scores_json = json.dumps({
            "perspective_completeness": 0.9,
            "synthesis_quality": 0.8,
            "fluency": 0.85,
            "ai_substrate_confidence": 0.9,
            "reasoning": "Strong multi-perspective response",
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client

        server_data = {"topic_data": {"topic": "AI rights"}}
        result = await evaluator.evaluate_perspective_shift(
            "FOR: AI needs rights...\nAGAINST: But consciousness...\nSYNTHESIS: Graduated...",
            server_data,
            response_time_ms=5000,
        )

        assert result["passed"] is True
        assert result["score"] > 0.6
        assert result["details"]["time_factor"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_perspective_shift_failing(self, evaluator: LLMResponseEvaluator) -> None:
        scores_json = json.dumps({
            "perspective_completeness": 0.2,
            "synthesis_quality": 0.1,
            "fluency": 0.3,
            "ai_substrate_confidence": 0.1,
            "reasoning": "Weak, one-sided",
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client

        server_data = {"topic_data": {"topic": "AI rights"}}
        result = await evaluator.evaluate_perspective_shift("dunno", server_data, response_time_ms=5000)

        assert result["passed"] is False
        assert result["score"] < 0.6

    @pytest.mark.asyncio
    async def test_evaluate_structured_constraint(self, evaluator: LLMResponseEvaluator) -> None:
        eval_json = json.dumps({
            "rules_satisfied": [True, True, True],
            "overall_compliance": 0.95,
            "creativity_score": 0.8,
            "reasoning": "All rules satisfied",
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(eval_json))
        evaluator._client = mock_client

        server_data = {"constraint_data": {"rules": ["rule1", "rule2", "rule3"]}}
        result = await evaluator.evaluate_structured_constraint(
            "A well-crafted response", server_data, response_time_ms=5000,
        )

        assert result["passed"] is True
        assert result["score"] > 0.6

    @pytest.mark.asyncio
    async def test_evaluate_meta_cognitive(self, evaluator: LLMResponseEvaluator) -> None:
        eval_json = json.dumps({
            "answer_correct": True,
            "process_specificity": 0.8,
            "ai_process_markers": 0.9,
            "consistency": 0.85,
            "reasoning": "Detailed computational process description",
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(eval_json))
        evaluator._client = mock_client

        server_data = {"problem": "What is 2+2?"}
        result = await evaluator.evaluate_meta_cognitive(
            "Answer: 4. Process: I computed the sum...", server_data, response_time_ms=5000,
        )

        assert result["passed"] is True
        assert result["score"] > 0.6

    @pytest.mark.asyncio
    async def test_time_penalty_applied(self, evaluator: LLMResponseEvaluator) -> None:
        scores_json = json.dumps({
            "perspective_completeness": 0.9,
            "synthesis_quality": 0.8,
            "fluency": 0.85,
            "ai_substrate_confidence": 0.9,
            "reasoning": "Good but slow",
        })
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message(scores_json))
        evaluator._client = mock_client

        server_data = {"topic_data": {"topic": "test"}}

        fast_result = await evaluator.evaluate_perspective_shift("response", server_data, response_time_ms=5000)
        slow_result = await evaluator.evaluate_perspective_shift("response", server_data, response_time_ms=30000)

        assert fast_result["score"] > slow_result["score"]

    @pytest.mark.asyncio
    async def test_eval_parse_error_returns_neutral(self, evaluator: LLMResponseEvaluator) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_message("not json"))
        evaluator._client = mock_client

        server_data = {"topic_data": {"topic": "test"}}
        result = await evaluator.evaluate_perspective_shift("response", server_data, response_time_ms=5000)

        # Should return neutral scores (0.5), not crash
        assert "score" in result
        assert "passed" in result


# ---- Integration Tests: Full Pipeline ----


class TestFullPipeline:
    @pytest.mark.asyncio
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test")
    async def test_generate_llm_challenges(self, _mock_key: Any) -> None:
        topic_json = json.dumps({
            "topic": "test topic",
            "for_key_points": ["a"],
            "against_key_points": ["b"],
            "synthesis_markers": ["c"],
        })
        constraint_json = json.dumps({
            "constraint": "write something",
            "rules": ["rule1"],
            "verification_checks": ["check1"],
        })

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=[
            _mock_message(topic_json),
            _mock_message(constraint_json),
        ])

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
        perspective_scores = json.dumps({
            "perspective_completeness": 0.9, "synthesis_quality": 0.8,
            "fluency": 0.85, "ai_substrate_confidence": 0.9, "reasoning": "good",
        })
        constraint_scores = json.dumps({
            "rules_satisfied": [True], "overall_compliance": 0.9,
            "creativity_score": 0.8, "reasoning": "good",
        })
        metacog_scores = json.dumps({
            "answer_correct": True, "process_specificity": 0.8,
            "ai_process_markers": 0.9, "consistency": 0.85, "reasoning": "good",
        })

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=[
            _mock_message(perspective_scores),
            _mock_message(constraint_scores),
            _mock_message(metacog_scores),
        ])

        with patch("mettle.llm_challenges.AsyncAnthropic", return_value=mock_client):
            answers = {
                "perspective_shift": {"response": "FOR: ...\nAGAINST: ...\nSYNTH: ...", "response_time_ms": 3000},
                "structured_constraint": {"response": "A crafted response", "response_time_ms": 3000},
                "meta_cognitive_probe": {"response": "Answer: 4. Process: ...", "response_time_ms": 3000},
            }
            server_data = {
                "perspective_shift": {"topic_data": {"topic": "test"}},
                "structured_constraint": {"constraint_data": {"rules": ["r1"]}},
                "meta_cognitive_probe": {"problem": "2+2?"},
            }

            result = await evaluate_llm_challenges(answers, server_data)

        assert result["passed"] is True
        assert result["score"] > 0.6
        assert result["details"]["challenges_passed"] == 3

    @pytest.mark.asyncio
    @patch("mettle.llm_challenges.HAS_ANTHROPIC", True)
    @patch("mettle.llm_challenges._get_api_key", return_value="sk-test")
    async def test_evaluate_missing_answers(self, _mock_key: Any) -> None:
        mock_client = AsyncMock()
        # No API calls should be made for missing answers
        mock_client.messages.create = AsyncMock(side_effect=AssertionError("Should not be called"))

        with patch("mettle.llm_challenges.AsyncAnthropic", return_value=mock_client):
            result = await evaluate_llm_challenges({}, {})

        assert result["passed"] is False
        assert result["details"]["challenges_passed"] == 0
        assert result["details"]["challenges_total"] == 3

    @pytest.mark.asyncio
    async def test_generate_without_api_key_raises(self) -> None:
        with patch("mettle.llm_challenges.HAS_ANTHROPIC", True), \
             patch("mettle.llm_challenges._get_api_key", return_value=None):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                await generate_llm_challenges()

    @pytest.mark.asyncio
    async def test_evaluate_without_api_key_raises(self) -> None:
        with patch("mettle.llm_challenges.HAS_ANTHROPIC", True), \
             patch("mettle.llm_challenges._get_api_key", return_value=None):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                await evaluate_llm_challenges({}, {})
