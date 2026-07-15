#!/usr/bin/env python3
"""Promote a real Redis replica while a Presence session is in progress."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil

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
from scripts.testing.run_redis_resilience_trial import (
    _free_port,
    _kill_process,
    _wait_http,
    _wait_redis,
)


def _distinct_ports(count: int) -> list[int]:
    ports: set[int] = set()
    while len(ports) < count:
        ports.add(_free_port())
    return list(ports)


def _redis_config(path: Path, port: int) -> Path:
    path.mkdir(parents=True)
    config = path / "redis.conf"
    config.write_text(
        "\n".join(
            [
                "bind 127.0.0.1",
                "protected-mode yes",
                f"port {port}",
                f"dir {path}",
                'save ""',
                "appendonly yes",
                "appendfsync always",
                "aof-use-rdb-preamble no",
                "daemonize no",
            ]
        )
        + "\n"
    )
    return config


def _start_redis(
    executable: str,
    config: Path,
    log_handle: Any,
    *,
    replica_of_port: int | None = None,
) -> subprocess.Popen:
    command = [executable, str(config)]
    if replica_of_port is not None:
        command.extend(["--replicaof", "127.0.0.1", str(replica_of_port)])
    # Both the executable discovered on PATH and generated config are harness-owned.
    return subprocess.Popen(  # nosec B603
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _start_proxy(
    *, listen_port: int, backend_port: int, log_handle: Any
) -> subprocess.Popen:
    # Fixed local proxy argument vector, with integer ports chosen by the harness.
    return subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "scripts/testing/tcp_failover_proxy.py",
            "--listen-port",
            str(listen_port),
            "--backend-port",
            str(backend_port),
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_replica(replica_client: redis.Redis, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = replica_client.info("replication")
        if info.get("role") == "slave" and info.get("master_link_status") == "up":
            return
        time.sleep(0.05)
    raise RuntimeError("Redis replica did not synchronize with the primary")


def _wait_master(client: redis.Redis, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.info("replication").get("role") == "master":
            return
        time.sleep(0.05)
    raise RuntimeError("Promoted Redis replica did not become writable")


def _wait_session_recovery(
    driver: PresenceSessionDriver, timeout: float = 15.0
) -> dict[str, Any]:
    if driver.session_id is None:
        raise RuntimeError("Presence session has not been started")
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    while time.monotonic() < deadline:
        response = driver.request("GET", f"/api/mettle/sessions/{driver.session_id}")
        last_status = response.status_code
        if response.status_code == 200:
            return response.json()
        if response.status_code != 503:
            raise RuntimeError(
                f"Unexpected HTTP {response.status_code} while waiting for Redis recovery"
            )
        time.sleep(0.1)
    raise RuntimeError(
        f"METTLE did not recover after failover, last HTTP {last_status}"
    )


def run_trial(*, output: Path) -> dict[str, Any]:
    redis_executable = shutil.which("redis-server")
    if redis_executable is None:
        raise RuntimeError("redis-server is required for the failover trial")

    temp_context = tempfile.TemporaryDirectory(prefix="mettle-redis-failover-")
    workdir = Path(temp_context.name)
    primary_port, replica_port, proxy_port, api_port = _distinct_ports(4)
    primary_config = _redis_config(workdir / "primary", primary_port)
    replica_config = _redis_config(workdir / "replica", replica_port)
    log_path = workdir / "processes.log"
    log_handle = log_path.open("ab")
    primary = _start_redis(redis_executable, primary_config, log_handle)
    replica: subprocess.Popen | None = None
    proxy: subprocess.Popen | None = None
    api_process: subprocess.Popen | None = None
    driver: PresenceSessionDriver | None = None
    evidence: dict[str, Any] = {
        "schema": "mettle-redis-failover-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topology": "primary-replica-stable-endpoint",
        "aof": {"enabled": True, "appendfsync": "always"},
    }
    try:
        primary_client = _wait_redis(primary_port)
        replica = _start_redis(
            redis_executable,
            replica_config,
            log_handle,
            replica_of_port=primary_port,
        )
        replica_client = _wait_redis(replica_port)
        _wait_replica(replica_client)
        proxy = _start_proxy(
            listen_port=proxy_port, backend_port=primary_port, log_handle=log_handle
        )
        api_key = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env.update(
            {
                "METTLE_ENVIRONMENT": "development",
                "METTLE_DEV_MODE": "false",
                "METTLE_VCP_SIGNING_KEY": ephemeral_vcp_signing_key_pem(),
                "METTLE_VCP_SIGNING_KEY_ID": "mettle-redis-failover-v1",
                "METTLE_API_KEYS": api_key,
                "METTLE_REDIS_URL": f"redis://127.0.0.1:{proxy_port}/0",
                "METTLE_REDIS_NAMESPACE": "mettle-failover",
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
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        driver = PresenceSessionDriver(
            base_url=f"http://127.0.0.1:{api_port}",
            api_key=api_key,
            suites=list(BRONZE_SUITES),
            audience="redis-failover.mettle.local",
        )
        _wait_http(driver)
        driver.start()
        first = driver.submit_current()
        first_presence = first["presence"]
        replicas_acked = int(primary_client.wait(1, 5000))
        evidence["before_failover"] = {
            "replicas_acked": replicas_acked,
            "sequence": first_presence["sequence"],
            "transcript_hash": first_presence["transcript_hash"],
            "next_action": first_presence["action"],
        }
        if replicas_acked != 1:
            raise RuntimeError(
                "Primary did not confirm replica durability before failover"
            )

        _kill_process(primary, crash=True)
        outage = driver.status(expected_status=503)
        evidence["during_failover"] = {
            "http_status": outage.status_code,
            "fail_closed": outage.status_code == 503,
        }

        replica_client.execute_command("REPLICAOF", "NO", "ONE")
        _wait_master(replica_client)
        _kill_process(proxy, crash=False)
        proxy = _start_proxy(
            listen_port=proxy_port, backend_port=replica_port, log_handle=log_handle
        )
        recovered = _wait_session_recovery(driver)
        evidence["after_promotion"] = {
            "http_status": 200,
            "role": replica_client.info("replication").get("role"),
            "sequence": recovered["presence"]["sequence"],
            "transcript_hash": recovered["presence"]["transcript_hash"],
            "state_preserved": (
                recovered["presence"]["sequence"] == first_presence["sequence"]
                and recovered["presence"]["transcript_hash"]
                == first_presence["transcript_hash"]
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
        _kill_process(replica, crash=True)
        restart_outage = driver.verify_presentation(
            presentation, jti, attestation, expected_status=503
        )
        replica = _start_redis(redis_executable, replica_config, log_handle)
        restarted_client = _wait_redis(replica_port)
        _wait_master(restarted_client)
        _wait_session_recovery(driver)
        accepted = driver.verify_presentation(
            presentation, jti, attestation, expected_status=200
        )
        replay = driver.verify_presentation(
            presentation, jti, attestation, expected_status=400
        )
        evidence["promoted_node_restart"] = {
            "outage_http_status": restart_outage.status_code,
            "recovered_http_status": accepted.status_code,
            "replay_http_status": replay.status_code,
            "challenge_survived_crash": accepted.status_code == 200,
            "replay_rejected": replay.status_code == 400,
        }

        required = [
            evidence["during_failover"]["fail_closed"],
            evidence["after_promotion"]["role"] == "master",
            evidence["after_promotion"]["state_preserved"],
            evidence["resumed_completion"]["overall_passed"],
            evidence["resumed_completion"]["credential_issued"],
            evidence["resumed_completion"]["issuer_signature_verified"],
            evidence["resumed_completion"]["unique_continuity_receipts"],
            evidence["promoted_node_restart"]["challenge_survived_crash"],
            evidence["promoted_node_restart"]["replay_rejected"],
        ]
        evidence["passed"] = all(required)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n")
        if not evidence["passed"]:
            raise RuntimeError("Redis failover invariants did not all pass")
        return evidence
    except Exception as exc:
        log_handle.flush()
        try:
            log_tail = "\n".join(
                log_path.read_text(errors="replace").splitlines()[-80:]
            )
        except OSError:
            log_tail = "<process log unavailable>"
        raise RuntimeError(f"{exc}\nProcess log tail:\n{log_tail}") from exc
    finally:
        if driver is not None:
            driver.close()
        if api_process is not None:
            _kill_process(api_process, crash=False)
        if proxy is not None:
            _kill_process(proxy, crash=False)
        if replica is not None:
            _kill_process(replica, crash=False)
        _kill_process(primary, crash=False)
        log_handle.close()
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
