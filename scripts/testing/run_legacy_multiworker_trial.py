#!/usr/bin/env python3
"""Prove the legacy API progresses across two independent API workers.

The trial starts two Uvicorn processes on separate ports with one shared Redis,
starts a session through worker A, alternates answers between A and B, and reads
the completed result from A.  No process-local session state can satisfy that
sequence.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import redis

ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT_SECONDS = 30


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Session-Token"] = token
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_ready(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"worker exited with status {process.returncode}")
        try:
            status, payload = _request(base_url, "GET", "/api/health/ready")
            if status == 200 and payload.get("status") == "ready":
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"worker did not become ready at {base_url}")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    redis_url = os.environ.get("METTLE_REDIS_URL", "redis://127.0.0.1:6379/15")
    redis_client = redis.Redis.from_url(redis_url, socket_timeout=2)
    if not redis_client.ping():
        raise RuntimeError("Redis did not answer the preflight ping")

    ports = (_free_port(), _free_port())
    bases = tuple(f"http://127.0.0.1:{port}" for port in ports)
    env = os.environ.copy()
    env.update(
        {
            "METTLE_REDIS_URL": redis_url,
            "METTLE_SECRET_KEY": "multiworker-trial-secret-key-32-bytes",  # pragma: allowlist secret
            "METTLE_ADMIN_API_KEY": "multiworker-trial-admin-key-32-bytes",  # pragma: allowlist secret
            "METTLE_USE_DATABASE": "false",
        }
    )

    session_ids: list[str] = []
    processes: list[subprocess.Popen[str]] = []
    with tempfile.TemporaryDirectory(prefix="mettle-multiworker-") as temp_dir:
        log_paths = [Path(temp_dir) / f"worker-{index}.log" for index in (1, 2)]
        log_handles = [path.open("w+") for path in log_paths]
        try:
            for port, log_handle in zip(ports, log_handles, strict=True):
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "uvicorn",
                            "main:app",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(port),
                            "--workers",
                            "1",
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                )

            for base, process in zip(bases, processes, strict=True):
                _wait_ready(base, process)

            status, started = _request(
                bases[0],
                "POST",
                "/api/session/start",
                body={"difficulty": "basic", "entity_id": "multiworker-trial"},
            )
            if status != 200:
                raise RuntimeError(f"session start failed with HTTP {status}")
            session_id = started["session_id"]
            session_ids.append(session_id)
            token = started["session_token"]
            challenge = started["current_challenge"]

            for base in (bases[1], bases[0], bases[1]):
                status, answered = _request(
                    base,
                    "POST",
                    "/api/session/answer",
                    body={
                        "session_id": session_id,
                        "challenge_id": challenge["id"],
                        "answer": "bounded multiworker trial response",
                    },
                    token=token,
                )
                if status != 200:
                    raise RuntimeError(f"answer failed with HTTP {status}")
                challenge = answered.get("next_challenge")
                if challenge is None:
                    break

            status, result = _request(
                bases[0],
                "GET",
                f"/api/session/{session_id}/result",
                token=token,
            )
            if status != 200 or result.get("total") != 3:
                raise RuntimeError("completed result was not readable across workers")

            print(
                json.dumps(
                    {
                        "status": "passed",
                        "workers": 2,
                        "cross_worker_transitions": 4,
                        "session_total": result["total"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        except Exception:
            for log_handle, log_path in zip(log_handles, log_paths, strict=True):
                log_handle.flush()
                print(f"--- {log_path.name} ---", file=sys.stderr)
                print(log_path.read_text()[-4000:], file=sys.stderr)
            raise
        finally:
            for process in processes:
                _stop(process)
            for log_handle in log_handles:
                log_handle.close()
            for session_id in session_ids:
                redis_client.delete(f"mettle:legacy:session:{session_id}")
                redis_client.delete(f"mettle:legacy:session:{session_id}:lock")
            redis_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
