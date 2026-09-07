"""Boundary checks for the isolated HTTP admission instrument."""

from fastapi.testclient import TestClient

from scripts.testing import run_admission_trial as trial


def test_instrument_loads_the_actual_published_example():
    example = trial.load_example("http://127.0.0.1:12345")
    assert example["ISSUER"] == "http://127.0.0.1:12345"
    assert callable(example["earn_badge"])
    assert callable(example["check_badge"])


def test_room_requires_a_positive_verifier_decision(monkeypatch):
    seen = []

    def check(badge):
        seen.append(badge)
        return badge == "accepted-test-fixture"

    monkeypatch.setattr(trial, "load_example", lambda _: {"check_badge": check})
    with TestClient(trial.room_app("http://127.0.0.1:12345")) as client:
        assert client.post("/members", json={}).status_code == 403
        assert client.post("/members", json={"badge": "wrong"}).status_code == 403
        response = client.post("/members", json={"badge": "accepted-test-fixture"})
        assert response.status_code == 200
        assert response.json() == {"room": "mettle-trial-room", "admitted": True}
        assert client.post("/members", json={"badge": "x" * 8193}).status_code == 422
    assert seen == [None, "wrong", "accepted-test-fixture"]


def test_worker_environment_excludes_ambient_credentials(tmp_path, monkeypatch):
    # A stand-in mapping avoids changing the test runner's real environment.
    environment = {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path),
        "METTLE_SECRET_KEY": "test-only",  # pragma: allowlist secret, public test fixture
        "OPENAI_API_KEY": "test-only",  # pragma: allowlist secret, public test fixture
        "HTTPS_PROXY": "http://unexpected-proxy",
    }
    monkeypatch.setattr(trial.os, "environ", environment)
    monkeypatch.chdir(tmp_path)
    trial.isolated_environment(str(tmp_path), {"METTLE_DEV_MODE": "true"})
    assert environment == {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path),
        "METTLE_DEV_MODE": "true",
    }
