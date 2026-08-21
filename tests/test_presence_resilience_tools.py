"""Tests for the Presence latency and real Redis resilience instruments."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import asyncio
from pathlib import Path

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from mettle.session_manager import SessionManager, SessionStateError, _key, _rate_key

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


@pytest.mark.asyncio
@pytest.mark.skipif(
    shutil.which("redis-server") is None,
    reason="redis-server is required for the real Lua corruption regression",
)
async def test_real_redis_wrong_type_cannot_partially_commit_terminal_state(
    tmp_path: Path,
) -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [
            "redis-server",
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    redis = Redis(
        host="127.0.0.1",
        port=port,
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=1.0,
    )
    try:
        for _ in range(100):
            try:
                if await redis.ping():
                    break
            except (RedisConnectionError, OSError):
                await asyncio.sleep(0.02)
        else:
            pytest.fail("isolated Redis did not become ready")

        session_id = "wrong-type-terminal"
        user_id = "wrong-type-user"
        original = json.dumps({"status": "in_progress"})
        answers = json.dumps({"secret": "retained"})
        await redis.set(_key(session_id), original)
        await redis.set(_key(session_id, "answers"), answers)
        # A corrupt string at the set key previously made SREM fail only after
        # the terminal metadata SET had already executed inside Lua.
        await redis.set(_rate_key(user_id, "active"), "wrong-type")
        terminal = {
            "session_id": session_id,
            "user_id": user_id,
            "status": "cancelled",
        }
        manager = SessionManager(redis)

        with pytest.raises(SessionStateError, match="active-session index"):
            async with manager._session_lock(session_id):
                await manager._persist_terminal_session(terminal)

        assert await redis.get(_key(session_id)) == original
        assert await redis.get(_key(session_id, "answers")) == answers
        assert await redis.get(_rate_key(user_id, "active")) == "wrong-type"
    finally:
        await redis.aclose()
        process.terminate()
        process.wait(timeout=5)
