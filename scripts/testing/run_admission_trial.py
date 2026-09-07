"""Exercise the published quick API example across isolated loopback processes.

Test-only: real generated challenges, original response windows, ephemeral issuer
keys, and the existing reference client. No production credentials or deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import multiprocessing as mp
import os
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BADGE_LIFETIME_SECONDS = 5


def load_example(issuer: str) -> dict:
    source = (ROOT / "static/docs.html").read_text()
    match = re.search(r'<code id="quickstart-python">(.*?)</code>', source, re.S)
    if match is None:
        raise RuntimeError("Published quick API example was not found")
    namespace: dict = {}
    exec(compile(html.unescape(match[1]), "quickstart-python", "exec"), namespace)  # noqa: S102
    namespace["ISSUER"] = issuer
    return namespace


def isolated_environment(directory: str, extra: dict[str, str]) -> None:
    # Workers receive no ambient service keys. Empty cwd prevents .env loading.
    kept = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR")
        if name in os.environ
    }
    os.environ.clear()
    os.environ.update(kept)
    os.environ.update(extra)
    os.chdir(directory)


def room_app(issuer: str):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    example = load_example(issuer)
    app = FastAPI()

    class Admission(BaseModel):
        badge: str | None = Field(default=None, max_length=8192)

    @app.get("/health")
    def health():
        return {"ready": True}

    def members(body):
        if not example["check_badge"](body.badge):
            raise HTTPException(403, "Admission refused")
        return {"room": "mettle-trial-room", "admitted": True}

    # Resolve the local request model before FastAPI inspects annotations.
    members.__annotations__["body"] = Admission
    app.post("/members")(members)
    return app


def server_worker(sock, stop, directory: str, config: dict, issuer: str | None):
    isolated_environment(directory, config)
    # Trial logs contain no credentials or raw challenge answers.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        import uvicorn

        if issuer is None:
            from main import app
        else:
            app = room_app(issuer)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                access_log=False,
                log_level="error",
                timeout_graceful_shutdown=3,
            )
        )

        def stop_when_requested():
            stop.wait()
            server.should_exit = True

        threading.Thread(target=stop_when_requested, daemon=True).start()
        server.run(sockets=[sock])
        sock.close()


class LocalServer:
    def __init__(self, directory: str, config: dict, issuer: str | None = None):
        self.context = mp.get_context("spawn")
        self.stop_event = self.context.Event()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
            self.process = self.context.Process(
                target=server_worker,
                args=(
                    sock,
                    self.stop_event,
                    directory,
                    config,
                    issuer,
                ),
            )
            self.process.start()
        path = "/api/health/ready" if issuer is None else "/health"
        deadline = time.monotonic() + 45
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if not self.process.is_alive():
                        raise RuntimeError("Trial server exited during startup")
                    try:
                        if client.get(self.url + path).status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
            raise TimeoutError("Trial server did not become ready")
        except Exception:
            self.stop()
            raise

    def stop(self):
        self.stop_event.set()
        self.process.join(timeout=12)
        if self.process.is_alive():
            raise RuntimeError("Trial server did not stop cooperatively")
        if self.process.exitcode != 0:
            raise RuntimeError("Trial server exited unsuccessfully")


def client_worker(connection, directory: str, issuer: str, room: str, fail: bool):
    isolated_environment(directory, {})
    try:
        from scripts.testing.solver import solve_challenge

        seen = []

        def answer(challenge):
            # Assert answer secrecy and record only timings/types, never content.
            if "expected_answer" in challenge.get("data", {}):
                raise RuntimeError("Server exposed an expected answer")
            seen.append(
                {"type": challenge["type"], "time_limit_ms": challenge["time_limit_ms"]}
            )
            return (
                "deliberately wrong trial answer"
                if fail
                else solve_challenge(challenge)
            )

        badge = load_example(issuer)["earn_badge"](answer)
        with httpx.Client(timeout=12, trust_env=False) as client:
            admitted = client.post(room + "/members", json={"badge": badge})
        # Private pipe only. The parent never writes the badge into the receipt.
        connection.send(
            {
                "ok": True,
                "badge": badge,
                "status": admitted.status_code,
                "challenges": seen,
                "pid": os.getpid(),
            }
        )
    except Exception as exc:
        connection.send({"ok": False, "error_type": type(exc).__name__})
    finally:
        connection.close()


def run_client(directory: str, issuer: str, room: str, fail: bool = False) -> dict:
    context = mp.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=client_worker, args=(sending, directory, issuer, room, fail)
    )
    process.start()
    sending.close()
    try:
        if not receiving.poll(75):
            raise TimeoutError("Trial client did not complete")
        result = receiving.recv()
    finally:
        receiving.close()
        process.join(timeout=12)
    if process.is_alive() or process.exitcode != 0:
        raise RuntimeError("Trial client did not exit normally")
    if not result["ok"]:
        raise RuntimeError("Trial client failed: " + result["error_type"])
    return result


def run_trial() -> dict:
    from mettle.protocol import CREDENTIAL_CLOCK_SKEW_SECONDS

    report: dict[str, Any] = {
        "status": "running",
        "scope": "isolated loopback admission trial",
        "client": "test-only reference client using public challenges",
        "cases": {},
        "challenge_windows_changed": False,
        "badge_lifetime_seconds": BADGE_LIFETIME_SECONDS,
        "clock_skew_seconds": CREDENTIAL_CLOCK_SKEW_SECONDS,
        "example_sha256": hashlib.sha256(
            (ROOT / "static/docs.html").read_bytes()
        ).hexdigest(),
    }
    admin_key = secrets.token_urlsafe(32)
    config = {
        "METTLE_ENVIRONMENT": "development",
        "METTLE_DEV_MODE": "true",
        "METTLE_DATABASE_URL": "sqlite:///:memory:",
        "METTLE_USE_DATABASE": "false",
        "METTLE_REDIS_URL": "",
        "METTLE_SECRET_KEY": secrets.token_urlsafe(32),
        "METTLE_ADMIN_API_KEY": admin_key,
        "METTLE_BADGE_EXPIRY_SECONDS": str(BADGE_LIFETIME_SECONDS),
    }
    issuer = room = None
    with tempfile.TemporaryDirectory(prefix="mettle-admission-") as directory:
        try:
            issuer = LocalServer(directory, config)
            room = LocalServer(directory, {}, issuer.url)
            report["processes"] = {
                "issuer": issuer.process.pid,
                "room": room.process.pid,
            }
            with httpx.Client(timeout=12, trust_env=False) as client:

                def check_case(name, badge, expected):
                    response = client.post(room.url + "/members", json={"badge": badge})
                    report["cases"][name] = {
                        "http_status": response.status_code,
                        "expected": expected,
                    }
                    if response.status_code != expected:
                        raise RuntimeError("Unexpected admission outcome: " + name)

                good = run_client(directory, issuer.url, room.url)
                if not good["badge"] or good["status"] != 200:
                    raise RuntimeError("Reference client did not gain admission")
                report["processes"]["client"] = good["pid"]
                report["challenge_windows"] = good["challenges"]
                report["cases"]["valid"] = {
                    "http_status": good["status"],
                    "expected": 200,
                }
                check_case("missing", None, 403)
                check_case("invalid", "invalid-trial-badge", 403)
                parts = good["badge"].split(".")
                parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
                check_case("tampered", ".".join(parts), 403)
                failed = run_client(directory, issuer.url, room.url, fail=True)
                if failed["badge"] is not None or failed["status"] != 403:
                    raise RuntimeError("Failed test gained admission")
                report["cases"]["failed_test"] = {
                    "http_status": failed["status"],
                    "expected": 403,
                }
                revoked = client.post(
                    issuer.url + "/api/badge/revoke",
                    headers={"X-Admin-Key": admin_key},
                    json={
                        "token": good["badge"],
                        "reason": "Isolated admission trial revocation",
                    },
                )
                if (
                    revoked.status_code != 200
                    or revoked.json().get("revoked") is not True
                ):
                    raise RuntimeError("Trial badge was not revoked")
                check_case("revoked", good["badge"], 403)

                expiring = run_client(directory, issuer.url, room.url)
                if expiring["status"] != 200 or not expiring["badge"]:
                    raise RuntimeError(
                        "Expiry control did not initially gain admission"
                    )
                receipt = client.post(
                    issuer.url + "/api/badge/verify", json={"token": expiring["badge"]}
                ).json()
                expires_at = datetime.fromisoformat(receipt["expires_at"]).timestamp()
                time.sleep(
                    max(0, expires_at + CREDENTIAL_CLOCK_SKEW_SECONDS + 1 - time.time())
                )
                expired = client.post(
                    issuer.url + "/api/badge/verify", json={"token": expiring["badge"]}
                ).json()
                if (
                    expired.get("valid") is not False
                    or expired.get("error") != "Badge has expired"
                ):
                    raise RuntimeError("Expiry was not observed by the real verifier")
                check_case("expired", expiring["badge"], 403)

                fresh = run_client(directory, issuer.url, room.url)
                if fresh["status"] != 200 or not fresh["badge"]:
                    raise RuntimeError(
                        "Outage control did not initially gain admission"
                    )
                issuer.stop()
                issuer = None
                check_case("issuer_unavailable", fresh["badge"], 403)
            report["status"] = "passed"
        except Exception as exc:
            report["status"] = "failed"
            report["error_type"] = type(exc).__name__
        finally:
            for server in (room, issuer):
                if server is not None:
                    server.stop()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["limits"] = [
        "Local HTTP, no production deployment or TLS trial",
        "Reference client, no model-provider or human discrimination study",
        "Quick bearer badges, no holder-bound Presence claim",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_trial()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
