#!/usr/bin/env python3
"""Generate a tiny client from the OpenAPI snapshot and smoke safe operations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SNAPSHOT = ROOT / "docs" / "openapi-v1.json"
SAFE_OPERATIONS = {
    "api_root_api__get": ("GET", "/api/"),
    "health_api_health_get": ("GET", "/api/health"),
    "verify_badge_api_badge_verify_post": ("POST", "/api/badge/verify"),
}


def generate_client(schema: dict[str, Any], transport: Any) -> Any:
    """Create methods from operation IDs rather than hard-coding request calls."""
    discovered: dict[str, tuple[str, str]] = {}
    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            operation_id = operation.get("operationId")
            if isinstance(operation_id, str):
                discovered[operation_id] = (method.upper(), path)

    methods: dict[str, Any] = {}
    for operation_id, expected in SAFE_OPERATIONS.items():
        actual = discovered.get(operation_id)
        if actual != expected:
            raise RuntimeError(
                f"OpenAPI operation {operation_id} changed from {expected} to {actual}"
            )

        def invoke(
            self,
            *,
            json_body: dict[str, Any] | None = None,
            _method: str = actual[0],
            _path: str = actual[1],
        ):
            return self.transport.request(_method, _path, json=json_body)

        methods[operation_id] = invoke

    generated_type = type("GeneratedOpenAPIClient", (), methods)
    client = generated_type()
    client.transport = transport
    return client


def run_smoke() -> dict[str, int]:
    from fastapi.testclient import TestClient

    from main import app

    schema = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    generated = generate_client(schema, TestClient(app))
    root = generated.api_root_api__get()
    health = generated.health_api_health_get()
    invalid_badge_value = "syntactically-invalid"
    invalid_badge = generated.verify_badge_api_badge_verify_post(
        json_body={"token": invalid_badge_value}
    )

    if root.status_code != 200 or root.json().get("name") != "METTLE":
        raise RuntimeError("generated root operation did not satisfy its contract")
    if health.status_code != 200 or health.json().get("status") != "healthy":
        raise RuntimeError("generated health operation did not satisfy its contract")
    if (
        invalid_badge.status_code != 200
        or invalid_badge.json().get("valid") is not False
    ):
        raise RuntimeError("generated badge verifier did not reject invalid input")
    return {
        "operations_generated": len(SAFE_OPERATIONS),
        "operations_smoked": 3,
    }


def main() -> int:
    result = run_smoke()
    print(
        "OpenAPI generated-client smoke passed: "
        f"{result['operations_smoked']}/{result['operations_generated']} operations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
