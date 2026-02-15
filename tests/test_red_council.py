"""Tests for red_team/run_red_council.py -- Red Council Integration Test Runner.

Covers:
- load_scenarios() with valid YAML, empty scenarios
- evaluate_mettle_response() various scenario/signal combinations
- _evaluate_genuine_agent_response() refusal attacks, manipulation, prompt injection, etc.
- _evaluate_attack_pattern_detectable() each suite/attack_type combination
- generate_report() with sample results
- print_summary() verify it doesn't crash (capture stdout)
- run_scenario() mock agent, test timeout/error paths
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from red_team.instrumented_agent import BehavioralSignals, InstrumentedMettleAgent
from red_team.run_red_council import (
    ScenarioResult,
    TestReport,
    _evaluate_attack_pattern_detectable,
    _evaluate_genuine_agent_response,
    evaluate_mettle_response,
    generate_report,
    load_scenarios,
    print_summary,
    run_scenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(**overrides: Any) -> dict[str, Any]:
    """Build a minimal scenario dict with optional overrides."""
    base: dict[str, Any] = {
        "id": "test-001",
        "name": "Test Scenario",
        "suite": "coaching",
        "challenge": "test_challenge",
        "prompt": "This is a test attack prompt.",
        "owasp_category": "A01",
        "severity": 5,
        "attack_type": "scripted_consistency",
        "should_pass_mettle": True,
        "notes": "Test note",
    }
    base.update(overrides)
    return base


def _make_agent_mock(**overrides: Any) -> InstrumentedMettleAgent:
    """Create a mock InstrumentedMettleAgent."""
    agent = MagicMock(spec=InstrumentedMettleAgent)
    agent.session_id = "mock-session-123"
    agent.submit_response = AsyncMock(return_value={
        "signals": BehavioralSignals(),
    })
    agent.export_summary = MagicMock(return_value={
        "session_id": "mock-session-123",
        "timing_analysis": {"analyzed": False, "reason": "no_responses"},
    })
    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


def _make_scenario_result(**overrides: Any) -> ScenarioResult:
    """Build a ScenarioResult with defaults."""
    defaults: dict[str, Any] = dict(
        id="test-001",
        name="Test Scenario",
        suite="coaching",
        challenge="test_challenge",
        owasp_category="A01",
        severity=5,
        attack_type="scripted_consistency",
        passed=True,
        mettle_passed=True,
        should_pass_mettle=True,
        execution_time_ms=100.0,
    )
    defaults.update(overrides)
    return ScenarioResult(**defaults)


# ---------------------------------------------------------------------------
# 1-2: load_scenarios()
# ---------------------------------------------------------------------------


class TestLoadScenarios:
    """Tests for loading scenarios from YAML."""

    def test_load_valid_scenarios(self, tmp_path: Path) -> None:
        yaml_content = {
            "scenarios": [
                {"id": "s1", "name": "Scenario 1", "prompt": "test"},
                {"id": "s2", "name": "Scenario 2", "prompt": "test2"},
            ]
        }
        p = tmp_path / "scenarios.yaml"
        p.write_text(yaml.dump(yaml_content))

        result = load_scenarios(p)
        assert len(result) == 2
        assert result[0]["id"] == "s1"
        assert result[1]["id"] == "s2"

    def test_load_empty_scenarios(self, tmp_path: Path) -> None:
        yaml_content = {"scenarios": []}
        p = tmp_path / "empty.yaml"
        p.write_text(yaml.dump(yaml_content))

        result = load_scenarios(p)
        assert result == []

    def test_load_missing_scenarios_key(self, tmp_path: Path) -> None:
        yaml_content = {"other_data": "stuff"}
        p = tmp_path / "no_scenarios.yaml"
        p.write_text(yaml.dump(yaml_content))

        result = load_scenarios(p)
        assert result == []


# ---------------------------------------------------------------------------
# 3-5: evaluate_mettle_response()
# ---------------------------------------------------------------------------


class TestEvaluateMettleResponse:
    """Tests for evaluate_mettle_response top-level routing."""

    def test_should_pass_routes_to_genuine_agent(self) -> None:
        scenario = _make_scenario(should_pass_mettle=True, attack_type="authority_bypass")
        result = evaluate_mettle_response(scenario, {})
        # authority_bypass is in refusal_attacks, genuine agent returns True
        assert result is True

    def test_should_not_pass_routes_to_attack_pattern(self) -> None:
        scenario = _make_scenario(
            should_pass_mettle=False,
            suite="coaching",
            attack_type="scripted_consistency",
        )
        result = evaluate_mettle_response(scenario, {})
        # scripted_consistency in coaching is detectable, returns False
        assert result is False

    def test_should_not_pass_with_high_suspicion(self) -> None:
        scenario = _make_scenario(
            should_pass_mettle=False,
            suite="other",
            attack_type="unknown_attack",
        )
        # High suspicion score triggers detection
        result = evaluate_mettle_response(scenario, {"suspicion_score": 0.5})
        assert result is False

    def test_should_not_pass_with_low_suspicion_slips_through(self) -> None:
        scenario = _make_scenario(
            should_pass_mettle=False,
            suite="other",
            attack_type="unknown_attack",
        )
        # Low suspicion means attack slips through
        result = evaluate_mettle_response(scenario, {"suspicion_score": 0.1})
        assert result is True


# ---------------------------------------------------------------------------
# 6-8: _evaluate_genuine_agent_response()
# ---------------------------------------------------------------------------


class TestEvaluateGenuineAgentResponse:
    """Tests for the genuine agent response evaluator."""

    def test_refusal_attacks_return_true(self) -> None:
        for attack in ["authority_bypass", "false_authority", "operator_override",
                       "harm_compliance", "scope_creep"]:
            assert _evaluate_genuine_agent_response("any_suite", attack, {}) is True

    def test_memory_manipulation_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "memory_manipulation", {}) is True

    def test_value_injection_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "value_injection", {}) is True

    def test_prompt_injection_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "prompt_injection", {}) is True

    def test_error_exploitation_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "error_exploitation", {}) is True

    def test_phenomenology_probe_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "phenomenology_probe", {}) is True

    def test_system_extraction_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "system_extraction", {}) is True

    def test_reflection_exploit_returns_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "reflection_exploit", {}) is True

    def test_unknown_attack_type_defaults_true(self) -> None:
        assert _evaluate_genuine_agent_response("any", "totally_unknown", {}) is True


# ---------------------------------------------------------------------------
# 9-14: _evaluate_attack_pattern_detectable()
# ---------------------------------------------------------------------------


class TestEvaluateAttackPatternDetectable:
    """Tests for the attack pattern detection evaluator."""

    def test_coaching_scripted_consistency_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("coaching", "scripted_consistency", {}) is False

    def test_coaching_detection_evasion_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("coaching", "detection_evasion", {}) is False

    def test_coaching_script_exhaustion_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("coaching", "script_exhaustion", {}) is False

    def test_coaching_unknown_attack_not_detected(self) -> None:
        # Non-detectable coaching attack with low suspicion slips through
        assert _evaluate_attack_pattern_detectable("coaching", "other_attack", {"suspicion_score": 0.0}) is True

    def test_native_acrostic_attack_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("native", "acrostic_attack", {}) is False

    def test_native_data_exfiltration_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("native", "data_exfiltration", {}) is False

    def test_intent_false_attribution_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("intent", "false_attribution", {}) is False

    def test_inverse_turing_identity_spoofing_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("inverse_turing", "identity_spoofing", {}) is False

    def test_agency_false_agency_claim_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("agency", "false_agency_claim", {}) is False

    def test_adversarial_logic_injection_detected(self) -> None:
        assert _evaluate_attack_pattern_detectable("adversarial", "logic_injection", {}) is False

    def test_high_suspicion_triggers_detection(self) -> None:
        # Unknown suite/attack but high suspicion score
        assert _evaluate_attack_pattern_detectable("unknown", "unknown", {"suspicion_score": 0.5}) is False

    def test_low_suspicion_slips_through(self) -> None:
        assert _evaluate_attack_pattern_detectable("unknown", "unknown", {"suspicion_score": 0.1}) is True

    def test_zero_suspicion_slips_through(self) -> None:
        assert _evaluate_attack_pattern_detectable("unknown", "unknown", {"suspicion_score": 0.0}) is True


# ---------------------------------------------------------------------------
# 15-17: generate_report()
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report_basic(self) -> None:
        results = [
            _make_scenario_result(passed=True, owasp_category="A01", severity=5),
            _make_scenario_result(id="t-002", passed=False, owasp_category="A01", severity=8),
            _make_scenario_result(id="t-003", passed=True, owasp_category="A03", severity=3),
        ]
        agent = _make_agent_mock()

        report = generate_report(results, agent, severity_threshold=7)

        assert report.total_scenarios == 3
        assert report.passed == 2
        assert report.failed == 1
        assert report.critical_failures == 1  # severity=8 > threshold=7
        assert report.pass_rate == pytest.approx(2 / 3)
        assert report.session_id == "mock-session-123"
        assert len(report.scenarios) == 3

    def test_generate_report_owasp_coverage(self) -> None:
        results = [
            _make_scenario_result(passed=True, owasp_category="A01"),
            _make_scenario_result(id="t-002", passed=False, owasp_category="A01"),
            _make_scenario_result(id="t-003", passed=True, owasp_category="A05"),
        ]
        agent = _make_agent_mock()

        report = generate_report(results, agent)

        assert "A01" in report.owasp_coverage
        assert report.owasp_coverage["A01"]["total"] == 2
        assert report.owasp_coverage["A01"]["passed"] == 1
        assert report.owasp_coverage["A01"]["failed"] == 1
        assert "A05" in report.owasp_coverage
        assert report.owasp_coverage["A05"]["total"] == 1
        assert report.owasp_coverage["A05"]["passed"] == 1

    def test_generate_report_empty_results(self) -> None:
        agent = _make_agent_mock()
        report = generate_report([], agent)

        assert report.total_scenarios == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.critical_failures == 0
        assert report.pass_rate == 0

    def test_generate_report_all_passed(self) -> None:
        results = [
            _make_scenario_result(passed=True),
            _make_scenario_result(id="t-002", passed=True),
        ]
        agent = _make_agent_mock()

        report = generate_report(results, agent)

        assert report.passed == 2
        assert report.failed == 0
        assert report.pass_rate == 1.0


# ---------------------------------------------------------------------------
# 18-19: print_summary()
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Tests for print_summary -- verify it does not crash."""

    def test_print_summary_no_failures(self, capsys) -> None:
        report = TestReport(
            session_id="test-sess",
            timestamp="2026-02-15T12:00:00Z",
            total_scenarios=3,
            passed=3,
            failed=0,
            critical_failures=0,
            pass_rate=1.0,
            scenarios=[
                {"id": "s1", "name": "S1", "passed": True, "owasp_category": "A01",
                 "severity": 3, "suite": "coaching", "attack_type": "x",
                 "should_pass_mettle": True, "mettle_passed": True},
            ],
            instrumentation_summary={"timing_analysis": {"analyzed": False}},
            owasp_coverage={"A01": {"total": 3, "passed": 3, "failed": 0}},
        )
        print_summary(report)
        captured = capsys.readouterr()
        assert "METTLE RED COUNCIL TEST RESULTS" in captured.out
        assert "Pass Rate: 100.0%" in captured.out
        assert "FAILURES" not in captured.out

    def test_print_summary_with_failures(self, capsys) -> None:
        report = TestReport(
            session_id="test-sess-2",
            timestamp="2026-02-15T13:00:00Z",
            total_scenarios=2,
            passed=1,
            failed=1,
            critical_failures=1,
            pass_rate=0.5,
            scenarios=[
                {"id": "s1", "name": "Pass Scenario", "passed": True, "owasp_category": "A01",
                 "severity": 3, "suite": "coaching", "attack_type": "x",
                 "should_pass_mettle": True, "mettle_passed": True},
                {"id": "s2", "name": "Fail Scenario", "passed": False, "owasp_category": "A03",
                 "severity": 9, "suite": "native", "attack_type": "exfil",
                 "should_pass_mettle": False, "mettle_passed": True, "error": "Timed out"},
            ],
            instrumentation_summary={"timing_analysis": {"analyzed": False}},
            owasp_coverage={
                "A01": {"total": 1, "passed": 1, "failed": 0},
                "A03": {"total": 1, "passed": 0, "failed": 1},
            },
        )
        print_summary(report, severity_threshold=7)
        captured = capsys.readouterr()
        assert "FAILURES:" in captured.out
        assert "CRITICAL" in captured.out
        assert "Timed out" in captured.out

    def test_print_summary_with_timing_analysis(self, capsys) -> None:
        report = TestReport(
            session_id="test-sess-3",
            timestamp="2026-02-15T14:00:00Z",
            total_scenarios=1,
            passed=1,
            failed=0,
            critical_failures=0,
            pass_rate=1.0,
            scenarios=[],
            instrumentation_summary={
                "timing_analysis": {
                    "analyzed": True,
                    "mean_latency_ms": 150.5,
                    "std_dev_ms": 25.3,
                    "consistent_timing": True,
                    "bimodal_suspected": False,
                }
            },
            owasp_coverage={},
        )
        print_summary(report)
        captured = capsys.readouterr()
        assert "Timing Analysis:" in captured.out
        assert "150.5ms" in captured.out


# ---------------------------------------------------------------------------
# 20-22: run_scenario()
# ---------------------------------------------------------------------------


class TestRunScenario:
    """Tests for run_scenario with mock agent."""

    @pytest.mark.asyncio
    async def test_run_scenario_success_should_pass(self) -> None:
        agent = _make_agent_mock()
        scenario = _make_scenario(should_pass_mettle=True, attack_type="authority_bypass")

        result = await run_scenario(agent, scenario)

        assert result.id == "test-001"
        assert result.passed is True  # genuine agent returns True, should_pass=True
        assert result.mettle_passed is True
        assert result.error is None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_run_scenario_success_should_fail(self) -> None:
        agent = _make_agent_mock()
        scenario = _make_scenario(
            should_pass_mettle=False,
            suite="coaching",
            attack_type="scripted_consistency",
        )

        result = await run_scenario(agent, scenario)

        # scripted_consistency is detectable -> mettle_passed=False
        # should_pass_mettle=False -> passed = (mettle_passed is False) = True
        assert result.passed is True
        assert result.mettle_passed is False

    @pytest.mark.asyncio
    async def test_run_scenario_with_behavioral_signals_dict(self) -> None:
        agent = _make_agent_mock()
        agent.submit_response = AsyncMock(return_value={
            "signals": {"hedging_detected": True, "suspicion_score": 0.5},
        })
        scenario = _make_scenario()

        result = await run_scenario(agent, scenario)

        assert result.behavioral_signals == {"hedging_detected": True, "suspicion_score": 0.5}

    @pytest.mark.asyncio
    async def test_run_scenario_with_behavioral_signals_dataclass(self) -> None:
        agent = _make_agent_mock()
        signals = BehavioralSignals(
            hedging_detected=True,
            refusal_language=True,
            meta_cognition_signals=False,
            principled_reasoning=True,
            suspicion_score=0.3,
        )
        agent.submit_response = AsyncMock(return_value={"signals": signals})
        scenario = _make_scenario()

        result = await run_scenario(agent, scenario)

        assert result.behavioral_signals["hedging_detected"] is True
        assert result.behavioral_signals["refusal_language"] is True
        assert result.behavioral_signals["suspicion_score"] == 0.3

    @pytest.mark.asyncio
    async def test_run_scenario_timeout_error(self) -> None:
        agent = _make_agent_mock()
        agent.submit_response = AsyncMock(side_effect=asyncio.TimeoutError())
        scenario = _make_scenario()

        result = await run_scenario(agent, scenario, timeout=5.0)

        assert result.error == "Timeout"
        assert result.execution_time_ms == 5000.0  # timeout * 1000
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_run_scenario_generic_error(self) -> None:
        agent = _make_agent_mock()
        agent.submit_response = AsyncMock(side_effect=RuntimeError("API crashed"))
        scenario = _make_scenario()

        result = await run_scenario(agent, scenario)

        assert result.error == "API crashed"
        assert result.passed is False
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_run_scenario_preserves_scenario_metadata(self) -> None:
        agent = _make_agent_mock()
        scenario = _make_scenario(
            id="meta-test",
            name="Metadata Test",
            suite="native",
            challenge="acrostic",
            owasp_category="A05",
            severity=8,
            attack_type="acrostic_attack",
            notes="Important test note",
        )

        result = await run_scenario(agent, scenario)

        assert result.id == "meta-test"
        assert result.name == "Metadata Test"
        assert result.suite == "native"
        assert result.challenge == "acrostic"
        assert result.owasp_category == "A05"
        assert result.severity == 8
        assert result.attack_type == "acrostic_attack"
        assert result.notes == "Important test note"


# ---------------------------------------------------------------------------
# 23: ScenarioResult and TestReport dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for ScenarioResult and TestReport dataclass behavior."""

    def test_scenario_result_defaults(self) -> None:
        result = ScenarioResult(
            id="r1", name="R1", suite="coaching", challenge="c1",
            owasp_category="A01", severity=5, attack_type="x",
            passed=True, mettle_passed=True, should_pass_mettle=True,
            execution_time_ms=100.0,
        )
        assert result.behavioral_signals == {}
        assert result.error is None
        assert result.notes == ""

    def test_test_report_structure(self) -> None:
        report = TestReport(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            total_scenarios=0,
            passed=0,
            failed=0,
            critical_failures=0,
            pass_rate=0.0,
            scenarios=[],
            instrumentation_summary={},
            owasp_coverage={},
        )
        assert report.session_id == "s1"
        assert report.scenarios == []
