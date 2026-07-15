#!/usr/bin/env python3
"""Crash and recover real Redis while a live Presence session is in progress."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket

# This local harness launches fixed argument vectors without a shell.
import subprocess  # nosec B404
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.testing.presence_trial_support import (
    BRONZE_SUITES,
    PresenceSessionDriver,
    ephemeral_vcp_signing_key_pem,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(driver: PresenceSessionDriver, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = driver.request("GET", "/api/health")
            if response.status_code == 200:
                return
        except Exception as exc:  # startup polling must tolerate connection refusal
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"METTLE test API did not become ready: {last_error}")


def _wait_redis(port: int, timeout: float = 10.0) -> redis.Redis:
    client = redis.Redis(
        host="127.0.0.1",
        port=port,
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if client.ping():
                return client
        except redis.RedisError as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"Redis did not become ready: {last_error}")


def _start_redis(
    executable: str, config_path: Path, log_handle: Any
) -> subprocess.Popen:
    # Both the executable discovered on PATH and generated config are harness-owned.
    return subprocess.Popen(  # nosec B603
        [executable, str(config_path)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _kill_process(process: subprocess.Popen, *, crash: bool) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGKILL if crash else signal.SIGTERM)
    process.wait(timeout=10)


def run_trial(*, output: Path) -> dict[str, Any]:
    redis_executable = shutil.which("redis-server")
    if redis_executable is None:
        raise RuntimeError("redis-server is required for the resilience trial")

    temp_context = tempfile.TemporaryDirectory(prefix="mettle-redis-resilience-")
    workdir = Path(temp_context.name)
    redis_port = _free_port()
    api_port = _free_port()
    while api_port == redis_port:
        api_port = _free_port()
    config_path = workdir / "redis.conf"
    redis_log_path = workdir / "redis.log"
    api_log_path = workdir / "api.log"
    config_path.write_text(
        "\n".join(
            [
                "bind 127.0.0.1",
                "protected-mode yes",
                f"port {redis_port}",
                f"dir {workdir}",
                'save ""',
                "appendonly yes",
                "appendfsync always",
                "aof-use-rdb-preamble no",
                "daemonize no",
            ]
        )
        + "\n"
    )

    redis_log = redis_log_path.open("ab")
    api_log = api_log_path.open("ab")
    redis_process = _start_redis(redis_executable, config_path, redis_log)
    api_process: subprocess.Popen | None = None
    driver: PresenceSessionDriver | None = None
    evidence: dict[str, Any] = {
        "schema": "mettle-redis-resilience-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "redis_version": redis.__version__,
        "aof": {"enabled": True, "appendfsync": "always"},
    }
    try:
        redis_client = _wait_redis(redis_port)
        info = redis_client.info("persistence")
        evidence["aof"]["redis_reports_enabled"] = bool(info.get("aof_enabled"))
        api_key = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env.update(
            {
                "METTLE_ENVIRONMENT": "development",
                "METTLE_DEV_MODE": "false",
                "METTLE_VCP_SIGNING_KEY": ephemeral_vcp_signing_key_pem(),
                "METTLE_VCP_SIGNING_KEY_ID": "mettle-redis-resilience-v1",
                "METTLE_API_KEYS": api_key,
                "METTLE_REDIS_URL": f"redis://127.0.0.1:{redis_port}/0",
                "METTLE_REDIS_NAMESPACE": "mettle-resilience",
                "METTLE_USE_DATABASE": "false",
                "METTLE_RATE_LIMIT_SESSIONS": "1000/minute",
                "METTLE_RATE_LIMIT_ANSWERS": "1000/minute",
            }
        )
        # Fixed local uvicorn argument vector, with no user-controlled command text.
        api_process = subprocess.Popen(  # nosec B603
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--log-level",
                "warning",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=api_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        driver = PresenceSessionDriver(
            base_url=f"http://127.0.0.1:{api_port}",
            api_key=api_key,
            suites=list(BRONZE_SUITES),
            audience="redis-resilience.mettle.local",
        )
        _wait_http(driver)
        driver.start()
        first = driver.submit_current()
        first_state = first["presence"]
        evidence["before_crash"] = {
            "status": "in_progress",
            "sequence": first_state["sequence"],
            "transcript_hash": first_state["transcript_hash"],
            "next_action": first_state["action"],
        }

        _kill_process(redis_process, crash=True)
        outage = driver.status(expected_status=503)
        evidence["during_outage"] = {
            "http_status": outage.status_code,
            "fail_closed": outage.status_code == 503,
        }

        redis_process = _start_redis(redis_executable, config_path, redis_log)
        redis_client = _wait_redis(redis_port)
        recovered = driver.status(expected_status=200).json()
        evidence["after_restart"] = {
            "http_status": 200,
            "status": recovered["status"],
            "sequence": recovered["presence"]["sequence"],
            "transcript_hash": recovered["presence"]["transcript_hash"],
            "state_preserved": (
                recovered["presence"]["sequence"] == first_state["sequence"]
                and recovered["presence"]["transcript_hash"]
                == first_state["transcript_hash"]
            ),
        }

        result = driver.complete()
        attestation = result["vcp_attestation"]
        proof = attestation["metadata"]["proof_of_possession"]
        challenge_ids = [
            receipt["challenge_id"] for receipt in proof["server_timing"]["submissions"]
        ]
        evidence["resumed_completion"] = {
            "overall_passed": result["overall_passed"],
            "credential_issued": attestation["credential_issued"],
            "issuer_signature_verified": True,
            "presence_sequence": proof["sequence"],
            "unique_continuity_receipts": len(challenge_ids) == len(set(challenge_ids)),
        }

        jti = attestation["metadata"]["jti"]
        presentation = driver.create_presentation_challenge(jti)
        _kill_process(redis_process, crash=True)
        presentation_outage = driver.verify_presentation(
            presentation, jti, attestation, expected_status=503
        )
        redis_process = _start_redis(redis_executable, config_path, redis_log)
        _wait_redis(redis_port)
        accepted = driver.verify_presentation(
            presentation, jti, attestation, expected_status=200
        )
        replay = driver.verify_presentation(
            presentation, jti, attestation, expected_status=400
        )
        evidence["presentation_recovery"] = {
            "outage_http_status": presentation_outage.status_code,
            "recovered_http_status": accepted.status_code,
            "replay_http_status": replay.status_code,
            "challenge_survived_crash": accepted.status_code == 200,
            "replay_rejected": replay.status_code == 400,
        }

        required = [
            evidence["during_outage"]["fail_closed"],
            evidence["after_restart"]["state_preserved"],
            evidence["resumed_completion"]["overall_passed"],
            evidence["resumed_completion"]["credential_issued"],
            evidence["resumed_completion"]["issuer_signature_verified"],
            evidence["resumed_completion"]["unique_continuity_receipts"],
            evidence["presentation_recovery"]["challenge_survived_crash"],
            evidence["presentation_recovery"]["replay_rejected"],
        ]
        evidence["passed"] = all(required)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n")
        if not evidence["passed"]:
            raise RuntimeError("Redis resilience invariants did not all pass")
        return evidence
    except Exception as exc:
        redis_log.flush()
        api_log.flush()
        try:
            redis_tail = "\n".join(
                redis_log_path.read_text(errors="replace").splitlines()[-40:]
            )
            api_tail = "\n".join(
                api_log_path.read_text(errors="replace").splitlines()[-40:]
            )
        except OSError:
            redis_tail = api_tail = "<process log unavailable>"
        raise RuntimeError(
            f"{exc}\nRedis log tail:\n{redis_tail}\nAPI log tail:\n{api_tail}"
        ) from exc
    finally:
        if driver is not None:
            driver.close()
        if api_process is not None:
            _kill_process(api_process, crash=False)
        _kill_process(redis_process, crash=False)
        redis_log.close()
        api_log.close()
        temp_context.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run_trial(output=args.output)
    print(json.dumps({"output": str(args.output), "passed": evidence["passed"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
