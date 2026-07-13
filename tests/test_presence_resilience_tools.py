"""Tests for the Presence latency and real Redis resilience instruments."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.testing.run_presence_latency_trials import (
    Cohort,
    build_report,
    parse_cohort,
    percentile,
)
from scripts.testing.presence_trial_support import validate_trial_url


def test_trial_url_requires_tls_except_for_loopback() -> None:
    assert validate_trial_url("https://mettle.example/") == "https://mettle.example"
    assert validate_trial_url("http://127.0.0.1:8123") == "http://127.0.0.1:8123"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_trial_url("http://mettle.example")


def test_latency_report_keeps_synthetic_boundary_explicit() -> None:
    cohorts = [Cohort("direct", 0, 1), Cohort("paced", 250, 1)]
    samples = [
        {"cohort": "direct", "server_response_time_ms": 20},
        {"cohort": "direct", "server_response_time_ms": 30},
        {"cohort": "paced", "server_response_time_ms": 280},
        {"cohort": "paced", "server_response_time_ms": 290},
    ]
    report = build_report(
        base_url="https://mettle.example", cohorts=cohorts, samples=samples
    )
    assert report["attestation_signature_verified"] is True
    assert report["summaries"]["direct"]["p95_ms"] == 30
    assert report["separation"]["paced"] == {
        "direct_max_to_paced_min_gap_ms": 250,
        "observed_overlap": False,
        "descriptive_midpoint_ms": 155,
    }
    assert any(
        "must not be enforced" in limit for limit in report["interpretation_limits"]
    )
    assert percentile([30, 10, 20], 0.5) == 20
    assert parse_cohort("paced:250:3") == Cohort("paced", 250, 3)


@pytest.mark.skipif(
    shutil.which("redis-server") is None,
    reason="redis-server is required for the real persistence trial",
)
def test_real_redis_crash_recovery_through_live_api(tmp_path: Path) -> None:
    output = tmp_path / "redis-resilience.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/testing/run_redis_resilience_trial.py",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is True
    assert evidence["during_outage"]["http_status"] == 503
    assert evidence["after_restart"]["state_preserved"] is True
    assert evidence["resumed_completion"]["issuer_signature_verified"] is True
    assert evidence["resumed_completion"]["presence_sequence"] == 5
    assert evidence["presentation_recovery"]["replay_http_status"] == 400


@pytest.mark.skipif(
    shutil.which("redis-server") is None,
    reason="redis-server is required for the real failover trial",
)
def test_real_redis_replica_promotion_through_live_api(tmp_path: Path) -> None:
    output = tmp_path / "redis-failover.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/testing/run_redis_failover_trial.py",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is True
    assert evidence["during_failover"]["http_status"] == 503
    assert evidence["after_promotion"]["role"] == "master"
    assert evidence["after_promotion"]["state_preserved"] is True
    assert evidence["resumed_completion"]["issuer_signature_verified"] is True
    assert evidence["resumed_completion"]["presence_sequence"] == 5
    assert evidence["promoted_node_restart"]["replay_http_status"] == 400
