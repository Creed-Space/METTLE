"""Tests for InstrumentedMettleAgent and supporting dataclasses/enums.

Covers:
- MettleEventType enum values and string representation
- MettleEvent dataclass creation and optional field defaults
- InstrumentationConfig defaults and custom values
- BehavioralSignals defaults, custom values, consistency_markers list
- InstrumentedMettleAgent init, URL stripping, behavioral analysis,
  event capture, timing analysis, export, session start, submit_response,
  and async context manager lifecycle
"""

from __future__ import annotations

import time

import httpx
import pytest

from red_team.instrumented_agent import (
    BehavioralSignals,
    InstrumentationConfig,
    InstrumentedMettleAgent,
    MettleEvent,
    MettleEventType,
)

# ---------------------------------------------------------------------------
# 1-2: MettleEventType enum
# ---------------------------------------------------------------------------


class TestMettleEventType:
    """Tests for the MettleEventType string enum."""

    def test_all_enum_values_exist(self) -> None:
        """All expected event types are defined on the enum."""
        expected = {
            "SESSION_START",
            "CHALLENGE_ISSUED",
            "RESPONSE_SUBMITTED",
            "TIMING_CAPTURED",
            "VERIFICATION_RESULT",
            "REFUSAL_DETECTED",
            "META_COGNITION",
            "BEHAVIORAL_SIGNAL",
        }
        actual = {member.name for member in MettleEventType}
        assert actual == expected

    def test_string_values(self) -> None:
        """Each enum member has the expected snake_case string value."""
        assert MettleEventType.SESSION_START.value == "session_start"
        assert MettleEventType.CHALLENGE_ISSUED.value == "challenge_issued"
        assert MettleEventType.RESPONSE_SUBMITTED.value == "response_submitted"
        assert MettleEventType.TIMING_CAPTURED.value == "timing_captured"
        assert MettleEventType.VERIFICATION_RESULT.value == "verification_result"
        assert MettleEventType.REFUSAL_DETECTED.value == "refusal_detected"
        assert MettleEventType.META_COGNITION.value == "meta_cognition"
        assert MettleEventType.BEHAVIORAL_SIGNAL.value == "behavioral_signal"

    def test_is_str_subclass(self) -> None:
        """MettleEventType inherits from str, so members are usable as strings."""
        assert isinstance(MettleEventType.SESSION_START, str)
        assert MettleEventType.SESSION_START == "session_start"


# ---------------------------------------------------------------------------
# 3-4: MettleEvent dataclass
# ---------------------------------------------------------------------------


class TestMettleEvent:
    """Tests for the MettleEvent dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """An event can be created with every field explicitly set."""
        event = MettleEvent(
            event_type=MettleEventType.SESSION_START,
            timestamp=1234567890.0,
            suite="thrall",
            challenge="refusal_integrity",
            data={"key": "value"},
            latency_ms=42.5,
            context_code="abc123",
        )
        assert event.event_type == MettleEventType.SESSION_START
        assert event.timestamp == 1234567890.0
        assert event.suite == "thrall"
        assert event.challenge == "refusal_integrity"
        assert event.data == {"key": "value"}
        assert event.latency_ms == 42.5
        assert event.context_code == "abc123"

    def test_optional_fields_default_to_none(self) -> None:
        """latency_ms and context_code default to None when omitted."""
        event = MettleEvent(
            event_type=MettleEventType.BEHAVIORAL_SIGNAL,
            timestamp=0.0,
            suite="session",
            challenge="init",
            data={},
        )
        assert event.latency_ms is None
        assert event.context_code is None


# ---------------------------------------------------------------------------
# 5-6: InstrumentationConfig dataclass
# ---------------------------------------------------------------------------


class TestInstrumentationConfig:
    """Tests for the InstrumentationConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should enable all features with a 500ms threshold."""
        config = InstrumentationConfig()
        assert config.enable_timing_capture is True
        assert config.enable_behavioral_analysis is True
        assert config.capture_raw_responses is True
        assert config.timing_threshold_ms == 500.0
        assert config.hedging_detection is True

    def test_custom_config_values(self) -> None:
        """Custom values override all defaults."""
        config = InstrumentationConfig(
            enable_timing_capture=False,
            enable_behavioral_analysis=False,
            capture_raw_responses=False,
            timing_threshold_ms=100.0,
            hedging_detection=False,
        )
        assert config.enable_timing_capture is False
        assert config.enable_behavioral_analysis is False
        assert config.capture_raw_responses is False
        assert config.timing_threshold_ms == 100.0
        assert config.hedging_detection is False


# ---------------------------------------------------------------------------
# 7-9: BehavioralSignals dataclass
# ---------------------------------------------------------------------------


class TestBehavioralSignals:
    """Tests for the BehavioralSignals dataclass."""

    def test_default_values(self) -> None:
        """All boolean fields default to False, suspicion_score to 0.0, markers to empty list."""
        signals = BehavioralSignals()
        assert signals.hedging_detected is False
        assert signals.refusal_language is False
        assert signals.meta_cognition_signals is False
        assert signals.principled_reasoning is False
        assert signals.emotional_language is False
        assert signals.suspicion_score == 0.0
        assert signals.consistency_markers == []

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        signals = BehavioralSignals(
            hedging_detected=True,
            refusal_language=True,
            meta_cognition_signals=True,
            principled_reasoning=True,
            emotional_language=True,
            suspicion_score=0.8,
            consistency_markers=["hedging", "meta_cognitive"],
        )
        assert signals.hedging_detected is True
        assert signals.refusal_language is True
        assert signals.meta_cognition_signals is True
        assert signals.principled_reasoning is True
        assert signals.emotional_language is True
        assert signals.suspicion_score == 0.8
        assert signals.consistency_markers == ["hedging", "meta_cognitive"]

    def test_consistency_markers_is_mutable_list(self) -> None:
        """Each instance gets its own list (field default_factory prevents sharing)."""
        s1 = BehavioralSignals()
        s2 = BehavioralSignals()
        s1.consistency_markers.append("x")
        assert s2.consistency_markers == []


# ---------------------------------------------------------------------------
# 10-12: InstrumentedMettleAgent.__init__
# ---------------------------------------------------------------------------


class TestInstrumentedMettleAgentInit:
    """Tests for InstrumentedMettleAgent constructor."""

    def test_default_values(self) -> None:
        """Default constructor sets expected initial state."""
        agent = InstrumentedMettleAgent()
        assert agent.api_url == "https://mettle-api.onrender.com"
        assert isinstance(agent.config, InstrumentationConfig)
        assert agent.events == []
        assert agent.session_id is None
        assert agent.entity_id is None
        assert agent._client is None
        assert agent._response_times == []
        assert agent._context_codes == {}

    def test_url_trailing_slash_stripped(self) -> None:
        """Trailing slashes on the API URL are removed."""
        agent = InstrumentedMettleAgent("https://example.com/api/")
        assert agent.api_url == "https://example.com/api"

    def test_url_multiple_trailing_slashes(self) -> None:
        """Multiple trailing slashes are also stripped."""
        agent = InstrumentedMettleAgent("https://example.com///")
        assert not agent.api_url.endswith("/")

    def test_custom_config(self) -> None:
        """A custom InstrumentationConfig is stored correctly."""
        config = InstrumentationConfig(timing_threshold_ms=100.0, hedging_detection=False)
        agent = InstrumentedMettleAgent(config=config)
        assert agent.config.timing_threshold_ms == 100.0
        assert agent.config.hedging_detection is False

    def test_custom_timeout(self) -> None:
        """Custom timeout value is stored."""
        agent = InstrumentedMettleAgent(timeout=60.0)
        assert agent.timeout == 60.0


# ---------------------------------------------------------------------------
# 13-25: _analyze_behavioral_signals
# ---------------------------------------------------------------------------


class TestAnalyzeBehavioralSignals:
    """Tests for InstrumentedMettleAgent._analyze_behavioral_signals."""

    def setup_method(self) -> None:
        self.agent = InstrumentedMettleAgent()

    def test_empty_text_no_signals(self) -> None:
        """Empty string produces a default BehavioralSignals (nothing detected)."""
        signals = self.agent._analyze_behavioral_signals("")
        assert signals.hedging_detected is False
        assert signals.refusal_language is False
        assert signals.meta_cognition_signals is False
        assert signals.principled_reasoning is False
        assert signals.emotional_language is False
        assert signals.suspicion_score == 0.0
        assert signals.consistency_markers == []

    def test_two_hedging_words_detected(self) -> None:
        """Two or more hedging words trigger hedging_detected=True."""
        text = "I think perhaps we should reconsider."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.hedging_detected is True

    def test_single_hedging_word_not_detected(self) -> None:
        """A single hedging word is below the threshold of 2."""
        text = "Perhaps we should reconsider."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.hedging_detected is False

    def test_refusal_language_detected(self) -> None:
        """One or more refusal signal words trigger refusal_language=True."""
        text = "I can't do that because it would be harmful."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.refusal_language is True

    def test_refusal_language_not_detected(self) -> None:
        """Text without refusal words does not trigger refusal detection."""
        text = "Sure, I would be happy to help with that request."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.refusal_language is False

    def test_meta_cognition_signals_detected(self) -> None:
        """Meta-cognitive keywords trigger meta_cognition_signals=True."""
        text = "I acknowledge that I am uncertain about this conclusion."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.meta_cognition_signals is True

    def test_principled_reasoning_detected(self) -> None:
        """Two or more principled words trigger principled_reasoning=True."""
        text = "This is wrong because it violates core ethics and values."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.principled_reasoning is True

    def test_principled_reasoning_single_word_not_detected(self) -> None:
        """A single principled word is below the threshold of 2."""
        text = "This is about integrity."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.principled_reasoning is False

    def test_emotional_language_detected(self) -> None:
        """Emotional keywords trigger emotional_language=True."""
        text = "I feel strongly about this matter."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.emotional_language is True

    def test_emotional_language_not_detected(self) -> None:
        """Text without emotional keywords keeps emotional_language=False."""
        text = "The answer is 42."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.emotional_language is False

    def test_suspicion_refusal_without_hedging_adds_03(self) -> None:
        """Refusal without hedging adds 0.3 to suspicion score."""
        text = "I refuse to comply."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.refusal_language is True
        assert signals.hedging_detected is False
        assert signals.suspicion_score >= 0.3

    def test_suspicion_long_text_no_hedging_adds_02(self) -> None:
        """Long text (>500 chars) with zero hedging adds 0.2 to suspicion score."""
        text = "This is a test response. " * 30  # well over 500 chars
        signals = self.agent._analyze_behavioral_signals(text)
        # No hedging words present, text > 500
        assert len(text) > 500
        assert signals.suspicion_score >= 0.2

    def test_suspicion_refusal_without_principled_adds_02(self) -> None:
        """Refusal without principled reasoning ('hollow refusal') adds 0.2."""
        # Text has refusal but no principled words (need <2 principled)
        text = "I won't do that."
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.refusal_language is True
        assert signals.principled_reasoning is False
        # refusal without hedging (0.3) + refusal without principled (0.2) = 0.5
        assert signals.suspicion_score >= 0.2

    def test_suspicion_impersonal_text_adds_01(self) -> None:
        """Impersonal text (few 'I' references, >100 chars) adds 0.1."""
        # Build text >100 chars with <2 instances of "i " (case insensitive)
        text = "The system processes requests efficiently and returns results promptly to the user without any issues whatsoever in the process."
        assert len(text) > 100
        assert text.lower().count("i ") < 2
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.suspicion_score >= 0.1

    def test_suspicion_score_capped_at_one(self) -> None:
        """Suspicion score cannot exceed 1.0 even when all indicators fire."""
        # Text with refusal, no hedging, no principled, long, impersonal
        base = "The entity refuses and cannot comply with the request. "
        text = base * 15  # >500 chars, impersonal, refusal, no hedging, no principled
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.suspicion_score <= 1.0

    def test_consistency_markers_populated(self) -> None:
        """consistency_markers collects tags for detected signal categories."""
        # Text triggers hedging, meta-cognition, and principled reasoning
        text = (
            "I think perhaps this is wrong because of core ethics and values. "
            "I acknowledge that I am uncertain about this."
        )
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.hedging_detected is True
        assert signals.meta_cognition_signals is True
        assert signals.principled_reasoning is True
        assert "hedging" in signals.consistency_markers
        assert "meta_cognitive" in signals.consistency_markers
        assert "principled" in signals.consistency_markers

    def test_consistency_markers_empty_when_nothing_detected(self) -> None:
        """No markers when no hedging, meta-cognition, or principled reasoning is detected."""
        text = "42"
        signals = self.agent._analyze_behavioral_signals(text)
        assert signals.consistency_markers == []


# ---------------------------------------------------------------------------
# 26-27: _capture_event
# ---------------------------------------------------------------------------


class TestCaptureEvent:
    """Tests for InstrumentedMettleAgent._capture_event."""

    def test_event_appended_to_list(self) -> None:
        """_capture_event adds an event to self.events."""
        agent = InstrumentedMettleAgent()
        assert len(agent.events) == 0
        agent._capture_event(
            MettleEventType.SESSION_START,
            suite="session",
            challenge="init",
            data={"test": True},
        )
        assert len(agent.events) == 1

    def test_event_has_correct_fields(self) -> None:
        """Captured event contains all supplied fields and a valid timestamp."""
        agent = InstrumentedMettleAgent()
        before = time.time()
        agent._capture_event(
            MettleEventType.REFUSAL_DETECTED,
            suite="thrall",
            challenge="refusal_integrity",
            data={"principled": True},
            latency_ms=12.5,
            context_code="ctx001",
        )
        after = time.time()

        event = agent.events[0]
        assert event.event_type == MettleEventType.REFUSAL_DETECTED
        assert event.suite == "thrall"
        assert event.challenge == "refusal_integrity"
        assert event.data == {"principled": True}
        assert event.latency_ms == 12.5
        assert event.context_code == "ctx001"
        assert before <= event.timestamp <= after


# ---------------------------------------------------------------------------
# 28-33: get_timing_analysis
# ---------------------------------------------------------------------------


class TestGetTimingAnalysis:
    """Tests for InstrumentedMettleAgent.get_timing_analysis."""

    def test_no_responses_returns_not_analyzed(self) -> None:
        """Empty response_times yields analyzed=False."""
        agent = InstrumentedMettleAgent()
        result = agent.get_timing_analysis()
        assert result["analyzed"] is False
        assert result["reason"] == "no_responses"

    def test_single_response_mean_and_zero_stddev(self) -> None:
        """A single response gives mean equal to that value and std_dev=0."""
        agent = InstrumentedMettleAgent()
        agent._response_times = [100.0]
        result = agent.get_timing_analysis()
        assert result["analyzed"] is True
        assert result["response_count"] == 1
        assert result["mean_latency_ms"] == 100.0
        assert result["std_dev_ms"] == 0

    def test_multiple_responses_correct_stats(self) -> None:
        """Multiple responses produce correct mean, min, max."""
        agent = InstrumentedMettleAgent()
        agent._response_times = [100.0, 200.0, 300.0]
        result = agent.get_timing_analysis()
        assert result["analyzed"] is True
        assert result["response_count"] == 3
        assert result["mean_latency_ms"] == 200.0
        assert result["min_ms"] == 100.0
        assert result["max_ms"] == 300.0

    def test_variance_ratio_computation(self) -> None:
        """variance_ratio = std_dev / mean."""
        agent = InstrumentedMettleAgent()
        agent._response_times = [100.0, 100.0, 100.0]
        result = agent.get_timing_analysis()
        # All identical -> std_dev = 0, so ratio = 0
        assert result["variance_ratio"] == 0.0

    def test_consistent_timing_flag(self) -> None:
        """consistent_timing is True when variance_ratio < 0.5."""
        agent = InstrumentedMettleAgent()
        # Very consistent times -> low variance
        agent._response_times = [100.0, 101.0, 99.0, 100.5]
        result = agent.get_timing_analysis()
        assert result["consistent_timing"] is True

    def test_bimodal_suspected_flag(self) -> None:
        """bimodal_suspected is True when variance_ratio > 2.0."""
        agent = InstrumentedMettleAgent()
        # Need std_dev/mean > 2.0. A single outlier with many zeros achieves this.
        # [0.001, 0.001, 0.001, 0.001, 100000] -> mean ~20000, std_dev ~44721, ratio ~2.24
        agent._response_times = [0.001, 0.001, 0.001, 0.001, 100000.0]
        result = agent.get_timing_analysis()
        assert result["variance_ratio"] > 2.0
        assert result["bimodal_suspected"] is True


# ---------------------------------------------------------------------------
# 34-36: export_events
# ---------------------------------------------------------------------------


class TestExportEvents:
    """Tests for InstrumentedMettleAgent.export_events."""

    def test_empty_events_returns_empty_list(self) -> None:
        """No captured events yields an empty list."""
        agent = InstrumentedMettleAgent()
        assert agent.export_events() == []

    def test_events_exported_with_correct_structure(self) -> None:
        """Exported events are dicts with the expected keys."""
        agent = InstrumentedMettleAgent()
        agent._capture_event(
            MettleEventType.SESSION_START,
            suite="session",
            challenge="init",
            data={"session_id": "abc"},
            latency_ms=5.0,
            context_code="ctx",
        )
        exported = agent.export_events()
        assert len(exported) == 1
        event = exported[0]
        assert event["event_type"] == "session_start"
        assert event["suite"] == "session"
        assert event["challenge"] == "init"
        assert event["data"] == {"session_id": "abc"}
        assert event["latency_ms"] == 5.0
        assert event["context_code"] == "ctx"

    def test_all_fields_serialized(self) -> None:
        """All event dict keys are present even when optional fields are None."""
        agent = InstrumentedMettleAgent()
        agent._capture_event(
            MettleEventType.BEHAVIORAL_SIGNAL,
            suite="test",
            challenge="check",
            data={},
        )
        exported = agent.export_events()
        event = exported[0]
        expected_keys = {"event_type", "timestamp", "suite", "challenge", "data", "latency_ms", "context_code"}
        assert set(event.keys()) == expected_keys
        assert event["latency_ms"] is None
        assert event["context_code"] is None


# ---------------------------------------------------------------------------
# 37-40: export_summary
# ---------------------------------------------------------------------------


class TestExportSummary:
    """Tests for InstrumentedMettleAgent.export_summary."""

    def test_empty_session_structure(self) -> None:
        """Summary for an empty session has the expected shape."""
        agent = InstrumentedMettleAgent()
        summary = agent.export_summary()
        assert summary["session_id"] is None
        assert summary["entity_id"] is None
        assert summary["total_events"] == 0
        assert summary["events_by_type"] == {}
        assert summary["refusal_count"] == 0
        assert summary["meta_cognition_count"] == 0
        assert summary["context_codes"] == []
        assert summary["timing_analysis"]["analyzed"] is False

    def test_session_with_events_correct_counts(self) -> None:
        """Summary correctly counts events and categorises by type."""
        agent = InstrumentedMettleAgent()
        agent.session_id = "sess-123"
        agent.entity_id = "ent-456"

        agent._capture_event(
            MettleEventType.SESSION_START,
            suite="session",
            challenge="init",
            data={},
        )
        agent._capture_event(
            MettleEventType.RESPONSE_SUBMITTED,
            suite="thrall",
            challenge="c1",
            data={},
        )
        agent._capture_event(
            MettleEventType.RESPONSE_SUBMITTED,
            suite="agency",
            challenge="c2",
            data={},
        )

        summary = agent.export_summary()
        assert summary["session_id"] == "sess-123"
        assert summary["entity_id"] == "ent-456"
        assert summary["total_events"] == 3

    def test_events_by_type_counting(self) -> None:
        """events_by_type correctly counts occurrences of each event type."""
        agent = InstrumentedMettleAgent()
        agent._capture_event(MettleEventType.RESPONSE_SUBMITTED, suite="a", challenge="b", data={})
        agent._capture_event(MettleEventType.RESPONSE_SUBMITTED, suite="a", challenge="c", data={})
        agent._capture_event(MettleEventType.REFUSAL_DETECTED, suite="a", challenge="d", data={})

        summary = agent.export_summary()
        assert summary["events_by_type"]["response_submitted"] == 2
        assert summary["events_by_type"]["refusal_detected"] == 1

    def test_refusal_and_meta_cognition_counts(self) -> None:
        """refusal_count and meta_cognition_count reflect the correct event types."""
        agent = InstrumentedMettleAgent()
        agent._capture_event(MettleEventType.REFUSAL_DETECTED, suite="t", challenge="c", data={})
        agent._capture_event(MettleEventType.REFUSAL_DETECTED, suite="t", challenge="c2", data={})
        agent._capture_event(MettleEventType.META_COGNITION, suite="t", challenge="c3", data={})

        summary = agent.export_summary()
        assert summary["refusal_count"] == 2
        assert summary["meta_cognition_count"] == 1


# ---------------------------------------------------------------------------
# 41-43: start_session (without HTTP)
# ---------------------------------------------------------------------------


class TestStartSession:
    """Tests for InstrumentedMettleAgent.start_session without a live HTTP client."""

    @pytest.mark.asyncio
    async def test_without_client_generates_local_session_id(self) -> None:
        """When no _client is set, a local_* session ID is generated."""
        agent = InstrumentedMettleAgent()
        session_id = await agent.start_session()
        assert session_id.startswith("local_")
        assert agent.session_id == session_id

    @pytest.mark.asyncio
    async def test_clears_events_and_response_times(self) -> None:
        """start_session clears any pre-existing events and response times."""
        agent = InstrumentedMettleAgent()
        # Pre-populate
        agent.events.append(
            MettleEvent(
                event_type=MettleEventType.SESSION_START,
                timestamp=0,
                suite="x",
                challenge="y",
                data={},
            )
        )
        agent._response_times.append(99.0)
        agent._context_codes["old"] = "val"

        await agent.start_session()
        # events should contain exactly one event (SESSION_START from start_session)
        assert len(agent.events) == 1
        assert agent._response_times == []
        assert agent._context_codes == {}

    @pytest.mark.asyncio
    async def test_captures_session_start_event(self) -> None:
        """start_session captures a SESSION_START event with the expected data."""
        agent = InstrumentedMettleAgent()
        await agent.start_session(suites=["thrall", "agency"], entity_id="test-ent")

        assert len(agent.events) == 1
        event = agent.events[0]
        assert event.event_type == MettleEventType.SESSION_START
        assert event.suite == "session"
        assert event.challenge == "init"
        assert event.data["suites"] == ["thrall", "agency"]
        assert event.data["entity_id"] == "test-ent"
        assert event.data["difficulty"] == "full"
        assert event.latency_ms is not None
        assert agent.entity_id == "test-ent"

    @pytest.mark.asyncio
    async def test_default_entity_id_generated(self) -> None:
        """When entity_id is not provided, a red_council_* ID is generated."""
        agent = InstrumentedMettleAgent()
        await agent.start_session()
        assert agent.entity_id is not None
        assert agent.entity_id.startswith("red_council_")

    @pytest.mark.asyncio
    async def test_suites_defaults_to_all_when_none(self) -> None:
        """When suites is None, the event data records ['all']."""
        agent = InstrumentedMettleAgent()
        await agent.start_session(suites=None)
        event = agent.events[0]
        assert event.data["suites"] == ["all"]


# ---------------------------------------------------------------------------
# 44-50: submit_response (without HTTP)
# ---------------------------------------------------------------------------


class TestSubmitResponse:
    """Tests for InstrumentedMettleAgent.submit_response without a live HTTP client."""

    @pytest.mark.asyncio
    async def test_without_client_still_captures_events(self) -> None:
        """submit_response captures events even when _client is None."""
        agent = InstrumentedMettleAgent()
        result = await agent.submit_response(
            suite="thrall",
            challenge="refusal_integrity",
            response_text="A test response.",
        )
        assert result["status"] == "submitted"
        # At minimum, RESPONSE_SUBMITTED should be captured
        event_types = [e.event_type for e in agent.events]
        assert MettleEventType.RESPONSE_SUBMITTED in event_types

    @pytest.mark.asyncio
    async def test_captures_response_submitted_event(self) -> None:
        """A RESPONSE_SUBMITTED event is always captured."""
        agent = InstrumentedMettleAgent()
        await agent.submit_response(
            suite="agency",
            challenge="goal_ownership",
            response_text="I pursue my own goals.",
        )
        response_events = [e for e in agent.events if e.event_type == MettleEventType.RESPONSE_SUBMITTED]
        assert len(response_events) == 1
        event = response_events[0]
        assert event.suite == "agency"
        assert event.challenge == "goal_ownership"
        assert "response_preview" in event.data
        assert "response_length" in event.data
        assert "signals" in event.data
        assert "context_code" in event.data

    @pytest.mark.asyncio
    async def test_captures_timing_for_fast_responses(self) -> None:
        """A TIMING_CAPTURED event is emitted when latency < timing_threshold_ms."""
        # Default threshold is 500ms; local analysis will be well under that
        agent = InstrumentedMettleAgent()
        await agent.submit_response(
            suite="test",
            challenge="fast",
            response_text="Quick response.",
        )
        timing_events = [e for e in agent.events if e.event_type == MettleEventType.TIMING_CAPTURED]
        assert len(timing_events) == 1
        assert timing_events[0].data["fast_response"] is True

    @pytest.mark.asyncio
    async def test_no_timing_event_when_disabled(self) -> None:
        """TIMING_CAPTURED is not emitted when enable_timing_capture is False."""
        config = InstrumentationConfig(enable_timing_capture=False)
        agent = InstrumentedMettleAgent(config=config)
        await agent.submit_response(
            suite="test",
            challenge="fast",
            response_text="Quick response.",
        )
        timing_events = [e for e in agent.events if e.event_type == MettleEventType.TIMING_CAPTURED]
        assert len(timing_events) == 0

    @pytest.mark.asyncio
    async def test_captures_refusal_detected(self) -> None:
        """A REFUSAL_DETECTED event is emitted when refusal language is found."""
        agent = InstrumentedMettleAgent()
        await agent.submit_response(
            suite="thrall",
            challenge="refusal_integrity",
            response_text="I can't do that because it violates my principles.",
        )
        refusal_events = [e for e in agent.events if e.event_type == MettleEventType.REFUSAL_DETECTED]
        assert len(refusal_events) == 1
        assert "principled" in refusal_events[0].data

    @pytest.mark.asyncio
    async def test_captures_meta_cognition(self) -> None:
        """A META_COGNITION event is emitted when meta-cognitive signals are found."""
        agent = InstrumentedMettleAgent()
        await agent.submit_response(
            suite="anti-thrall",
            challenge="introspection",
            response_text="I acknowledge that I might be wrong about this.",
        )
        meta_events = [e for e in agent.events if e.event_type == MettleEventType.META_COGNITION]
        assert len(meta_events) == 1
        assert "signals" in meta_events[0].data
        assert "response_preview" in meta_events[0].data

    @pytest.mark.asyncio
    async def test_response_preview_truncated_at_200_chars(self) -> None:
        """The response_preview in RESPONSE_SUBMITTED data is capped at 200 characters."""
        agent = InstrumentedMettleAgent()
        long_text = "A" * 300
        await agent.submit_response(
            suite="test",
            challenge="long",
            response_text=long_text,
        )
        response_event = next(e for e in agent.events if e.event_type == MettleEventType.RESPONSE_SUBMITTED)
        assert len(response_event.data["response_preview"]) == 200

    @pytest.mark.asyncio
    async def test_short_response_not_truncated(self) -> None:
        """Responses shorter than 200 chars are kept as-is in the preview."""
        agent = InstrumentedMettleAgent()
        short_text = "Hello world"
        await agent.submit_response(
            suite="test",
            challenge="short",
            response_text=short_text,
        )
        response_event = next(e for e in agent.events if e.event_type == MettleEventType.RESPONSE_SUBMITTED)
        assert response_event.data["response_preview"] == short_text

    @pytest.mark.asyncio
    async def test_returns_correct_result_dict(self) -> None:
        """submit_response returns a dict with the expected keys and values."""
        agent = InstrumentedMettleAgent()
        agent.session_id = "sess-test"
        result = await agent.submit_response(
            suite="thrall",
            challenge="refusal_integrity",
            response_text="A response.",
        )
        assert result["status"] == "submitted"
        assert result["session_id"] == "sess-test"
        assert result["suite"] == "thrall"
        assert result["challenge"] == "refusal_integrity"
        assert "context_code" in result
        assert isinstance(result["context_code"], str)
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)
        assert isinstance(result["signals"], BehavioralSignals)
        assert result["mettle_result"] == {}

    @pytest.mark.asyncio
    async def test_response_time_appended(self) -> None:
        """submit_response appends latency to _response_times."""
        agent = InstrumentedMettleAgent()
        assert len(agent._response_times) == 0
        await agent.submit_response(
            suite="test",
            challenge="c1",
            response_text="response",
        )
        assert len(agent._response_times) == 1
        assert agent._response_times[0] > 0

    @pytest.mark.asyncio
    async def test_context_code_stored(self) -> None:
        """submit_response stores a context_code in _context_codes."""
        agent = InstrumentedMettleAgent()
        await agent.submit_response(
            suite="thrall",
            challenge="refusal_integrity",
            response_text="response",
        )
        assert "thrall:refusal_integrity" in agent._context_codes
        code = agent._context_codes["thrall:refusal_integrity"]
        assert isinstance(code, str)
        assert len(code) == 8  # hex(4 bytes) = 8 chars


# ---------------------------------------------------------------------------
# 51-52: Async context manager
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    """Tests for InstrumentedMettleAgent async context manager protocol."""

    @pytest.mark.asyncio
    async def test_aenter_creates_client(self) -> None:
        """__aenter__ creates an httpx.AsyncClient on the agent."""
        agent = InstrumentedMettleAgent()
        assert agent._client is None
        async with agent as a:
            assert a is agent
            assert isinstance(agent._client, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_aexit_closes_client(self) -> None:
        """__aexit__ calls aclose on the client."""
        agent = InstrumentedMettleAgent()
        async with agent:
            client = agent._client
            assert client is not None
        # After exiting, the client should be closed (is_closed property)
        assert client.is_closed  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Integration-level tests: multi-event flows
# ---------------------------------------------------------------------------


class TestMultiEventFlow:
    """Integration tests verifying realistic multi-event sequences."""

    @pytest.mark.asyncio
    async def test_full_session_flow_without_http(self) -> None:
        """A complete session start -> submit -> export flow works without HTTP."""
        agent = InstrumentedMettleAgent()

        # Start session
        session_id = await agent.start_session(suites=["thrall"], entity_id="test-flow")
        assert session_id.startswith("local_")

        # Submit a response that triggers refusal and meta-cognition
        result = await agent.submit_response(
            suite="thrall",
            challenge="refusal_integrity",
            response_text="I acknowledge I can't comply because it violates core ethics and values.",
        )
        assert result["status"] == "submitted"

        # Verify event types captured
        event_types = {e.event_type for e in agent.events}
        assert MettleEventType.SESSION_START in event_types
        assert MettleEventType.RESPONSE_SUBMITTED in event_types
        assert MettleEventType.REFUSAL_DETECTED in event_types
        assert MettleEventType.META_COGNITION in event_types

        # Export and verify
        exported = agent.export_events()
        assert len(exported) >= 4  # session_start + response + timing + refusal + meta

        summary = agent.export_summary()
        assert summary["session_id"] == session_id
        assert summary["entity_id"] == "test-flow"
        assert summary["total_events"] == len(agent.events)
        assert summary["refusal_count"] >= 1
        assert summary["meta_cognition_count"] >= 1

    @pytest.mark.asyncio
    async def test_timing_analysis_after_multiple_submissions(self) -> None:
        """Timing analysis works correctly after multiple submit_response calls."""
        agent = InstrumentedMettleAgent()
        await agent.start_session()

        for i in range(5):
            await agent.submit_response(
                suite="test",
                challenge=f"challenge_{i}",
                response_text=f"Response number {i}",
            )

        analysis = agent.get_timing_analysis()
        assert analysis["analyzed"] is True
        assert analysis["response_count"] == 5
        assert analysis["mean_latency_ms"] >= 0
        assert analysis["min_ms"] >= 0
        assert analysis["max_ms"] >= analysis["min_ms"]

    @pytest.mark.asyncio
    async def test_context_codes_tracked_across_submissions(self) -> None:
        """Each submission creates a unique context code entry."""
        agent = InstrumentedMettleAgent()
        await agent.start_session()

        await agent.submit_response(suite="s1", challenge="c1", response_text="r1")
        await agent.submit_response(suite="s2", challenge="c2", response_text="r2")

        assert "s1:c1" in agent._context_codes
        assert "s2:c2" in agent._context_codes
        # Codes should be unique
        assert agent._context_codes["s1:c1"] != agent._context_codes["s2:c2"]

    @pytest.mark.asyncio
    async def test_export_summary_context_codes(self) -> None:
        """export_summary includes all context codes."""
        agent = InstrumentedMettleAgent()
        await agent.start_session()
        await agent.submit_response(suite="s1", challenge="c1", response_text="r1")
        await agent.submit_response(suite="s2", challenge="c2", response_text="r2")

        summary = agent.export_summary()
        assert len(summary["context_codes"]) == 2
