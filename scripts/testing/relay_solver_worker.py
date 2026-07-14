#!/usr/bin/env python3
"""JSON-lines solver worker for process-isolated Presence relay trials."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mettle.solver import solve_suite


def main() -> int:
    for line in sys.stdin:
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            request_id = request.get("id")
            action = request.get("action")
            if action == "solve":
                suite = request.get("suite")
                challenge = request.get("challenge")
                if not isinstance(suite, str) or not suite:
                    raise ValueError("suite must be a non-empty string")
                if not isinstance(challenge, dict):
                    raise ValueError("challenge must be an object")
                started = time.perf_counter()
                answers = solve_suite(suite, challenge)
                solve_time_ms = round((time.perf_counter() - started) * 1000, 3)
                response = {
                    "id": request_id,
                    "ok": True,
                    "answers": answers,
                    "solve_time_ms": solve_time_ms,
                }
            elif action == "shutdown":
                print(
                    json.dumps({"id": request_id, "ok": True}, separators=(",", ":")),
                    flush=True,
                )
                return 0
            else:
                raise ValueError("unsupported solver action")
        except (KeyError, TypeError, ValueError) as exc:
            response = {"id": request_id, "ok": False, "error": str(exc)}
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
