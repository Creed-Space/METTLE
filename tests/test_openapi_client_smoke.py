"""Tests for the snapshot-derived client smoke."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from scripts.smoke_openapi_client import SAFE_OPERATIONS, generate_client, run_smoke


def test_generated_client_contains_only_declared_safe_operations() -> None:
    schema = json.loads(Path("docs/openapi-v1.json").read_text(encoding="utf-8"))
    client = generate_client(schema, TestClient(app))

    assert set(SAFE_OPERATIONS) <= set(dir(client))


def test_generated_client_smoke_calls_live_application() -> None:
    assert run_smoke() == {"operations_generated": 3, "operations_smoked": 3}
