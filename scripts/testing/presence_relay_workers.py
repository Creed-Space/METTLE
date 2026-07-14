"""Isolated holder and solver process clients for Presence relay trials.

The holder subprocess owns the Ed25519 private key and receives only the
bounded fields needed to sign one protocol action. The solver subprocess sees
only the client-visible challenge. Neither worker inherits the parent process'
METTLE credentials.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, Self


REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDER_WORKER = Path(__file__).with_name("relay_holder_worker.py")
SOLVER_WORKER = Path(__file__).with_name("relay_solver_worker.py")


class WorkerProtocolError(RuntimeError):
    """An isolated trial worker failed its JSON-lines contract."""


def worker_environment() -> dict[str, str]:
    """Return a minimal worker environment without parent application secrets."""
    allowed = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


class JsonLineWorker:
    """Synchronous, request-correlated client for one fixed worker program."""

    def __init__(self, worker_path: Path, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Worker timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self._next_request_id = 1
        self._closed = False
        self._process = subprocess.Popen(  # nosec B603
            [sys.executable, "-I", str(worker_path)],
            cwd=REPO_ROOT,
            env=worker_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            close_fds=True,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self._process.pid

    def _readline(self) -> str:
        stdout = self._process.stdout
        if stdout is None:
            raise WorkerProtocolError("Worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout_seconds):
                raise WorkerProtocolError(
                    f"Worker {self.pid} timed out after {self.timeout_seconds:g}s"
                )
            line = stdout.readline()
        finally:
            selector.close()
        if line:
            return line
        return_code = self._process.poll()
        detail = ""
        if return_code is not None and self._process.stderr is not None:
            detail = self._process.stderr.read(2000).strip()
        raise WorkerProtocolError(
            f"Worker {self.pid} exited before replying"
            + (f" ({detail})" if detail else "")
        )

    def request(self, action: str, **payload: Any) -> dict[str, Any]:
        if self._closed:
            raise WorkerProtocolError("Worker is closed")
        stdin = self._process.stdin
        if stdin is None:
            raise WorkerProtocolError("Worker stdin is unavailable")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"id": request_id, "action": action, **payload}
        try:
            stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerProtocolError(f"Worker {self.pid} pipe is closed") from exc
        try:
            response = json.loads(self._readline())
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError(
                f"Worker {self.pid} returned invalid JSON"
            ) from exc
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise WorkerProtocolError(
                f"Worker {self.pid} returned an uncorrelated response"
            )
        if response.get("ok") is not True:
            error = response.get("error", "unspecified worker error")
            raise WorkerProtocolError(f"Worker {self.pid}: {error}")
        return response

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self.request("shutdown")
                self._process.wait(timeout=2)
        except (WorkerProtocolError, subprocess.TimeoutExpired):
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        finally:
            self._closed = True
            for stream in (
                self._process.stdin,
                self._process.stdout,
                self._process.stderr,
            ):
                if stream is not None:
                    stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class HolderWorkerClient(JsonLineWorker):
    """Client for a holder process whose private key never crosses IPC."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        super().__init__(HOLDER_WORKER, timeout_seconds=timeout_seconds)
        response = self.request("public_key")
        public_key_pem = response.get("public_key_pem")
        if not isinstance(public_key_pem, str) or "PRIVATE" in public_key_pem:
            self.close()
            raise WorkerProtocolError("Holder returned an invalid public key")
        self.public_key_pem = public_key_pem

    def sign_submission(
        self,
        *,
        session_id: str,
        action: str,
        nonce: str,
        previous_transcript_hash: str,
        payload_hash: str,
    ) -> tuple[str, float]:
        started = time.perf_counter()
        response = self.request(
            "sign_submission",
            session_id=session_id,
            submission_action=action,
            nonce=nonce,
            previous_transcript_hash=previous_transcript_hash,
            payload_hash=payload_hash,
        )
        roundtrip_ms = (time.perf_counter() - started) * 1000
        signature = response.get("signature")
        if not isinstance(signature, str):
            raise WorkerProtocolError("Holder omitted its submission signature")
        return signature, round(roundtrip_ms, 3)

    def sign_presentation(
        self,
        *,
        challenge_id: str,
        nonce: str,
        audience: str,
        credential_jti: str,
        expires_at: str,
    ) -> tuple[str, float]:
        started = time.perf_counter()
        response = self.request(
            "sign_presentation",
            challenge_id=challenge_id,
            nonce=nonce,
            audience=audience,
            credential_jti=credential_jti,
            expires_at=expires_at,
        )
        roundtrip_ms = (time.perf_counter() - started) * 1000
        signature = response.get("signature")
        if not isinstance(signature, str):
            raise WorkerProtocolError("Holder omitted its presentation signature")
        return signature, round(roundtrip_ms, 3)


class SolverWorkerClient(JsonLineWorker):
    """Client for a solver process that receives only public challenge data."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        super().__init__(SOLVER_WORKER, timeout_seconds=timeout_seconds)

    def solve(
        self, suite: str, challenge: dict[str, Any]
    ) -> tuple[dict[str, Any], float, float]:
        started = time.perf_counter()
        response = self.request("solve", suite=suite, challenge=challenge)
        roundtrip_ms = (time.perf_counter() - started) * 1000
        answers = response.get("answers")
        solve_time_ms = response.get("solve_time_ms")
        if not isinstance(answers, dict) or not isinstance(solve_time_ms, (int, float)):
            raise WorkerProtocolError("Solver returned an invalid answer response")
        return answers, float(solve_time_ms), round(roundtrip_ms, 3)
