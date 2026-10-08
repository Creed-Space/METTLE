"""Tag-bound Render production promotion tests."""

from __future__ import annotations

import http.client
import inspect
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from scripts import deploy_render_release as release


SOURCE_REVISION = "a" * 40


def _contract(*, api_auto_deploy: bool = False) -> dict[str, Any]:
    return {
        "blueprint_sha256": {"render.yaml": "b" * 64},
        "deployment_sha256": "c" * 64,
        "services": {
            "mettle-api": {
                "binding": {
                    "service_id": "srv-api",
                    "promote_on_release": True,
                },
                "blueprint": {"autoDeploy": api_auto_deploy},
            },
            "mettle-mcp": {
                "binding": {
                    "service_id": "srv-mcp",
                    "promote_on_release": True,
                },
                "blueprint": {"autoDeploy": False},
            },
            "mettle-holder-staging": {
                "binding": {"service_id": "srv-holder"},
                "blueprint": {"autoDeploy": False},
            },
        },
    }


def _live(deploy_id: str) -> dict[str, object]:
    return {
        "id": deploy_id,
        "status": "live",
        "commit": {"id": SOURCE_REVISION},
        "trigger": "api",
        "startedAt": "2026-08-14T00:00:00Z",
        "finishedAt": "2026-08-14T00:01:00Z",
    }


def test_release_targets_are_exact_and_disable_mutable_auto_deploy() -> None:
    assert release.release_targets(_contract()) == [
        {"name": "mettle-api", "service_id": "srv-api"},
        {"name": "mettle-mcp", "service_id": "srv-mcp"},
    ]

    with pytest.raises(ValueError, match="disable autoDeploy"):
        release.release_targets(_contract(api_auto_deploy=True))


def test_promote_release_binds_both_services_to_one_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        calls.append((method, path, payload))
        service = "api" if "srv-api" in path else "mcp"
        if method == "GET":
            return 200, [
                {
                    "deploy": {
                        "id": f"dep-{service}-previous",
                        "status": "live",
                        "commit": {"id": "0" * 40},
                    }
                }
            ]
        return 201, _live(f"dep-{service}-release")

    monkeypatch.setattr(release, "_request_json", request)
    receipt = release.promote_release(
        _contract(), SOURCE_REVISION, "v0.4.0", "secret-token", poll_seconds=0
    )

    assert receipt["result"] == "live"
    assert receipt["source_revision"] == SOURCE_REVISION
    services = receipt["services"]
    assert isinstance(services, list)
    assert [service["name"] for service in services] == [
        "mettle-api",
        "mettle-mcp",
    ]
    posts = [call for call in calls if call[0] == "POST"]
    assert posts == [
        (
            "POST",
            "/services/srv-api/deploys",
            {"commitId": SOURCE_REVISION, "clearCache": "do_not_clear"},
        ),
        (
            "POST",
            "/services/srv-mcp/deploys",
            {"commitId": SOURCE_REVISION, "clearCache": "do_not_clear"},
        ),
    ]
    assert "secret-token" not in repr(receipt)


def test_terminal_provider_failure_cannot_produce_a_live_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = {
        "id": "dep-failed",
        "status": "update_failed",
        "commit": {"id": SOURCE_REVISION},
        "trigger": "api",
    }

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del payload
        if method == "GET":
            return 200, [
                {"deploy": failed},
                {
                    "deploy": {
                        "id": "dep-previous",
                        "status": "live",
                        "commit": {"id": "0" * 40},
                    }
                },
            ]
        return 201, failed

    monkeypatch.setattr(release, "_request_json", request)
    with pytest.raises(release.RenderPromotionError) as raised:
        release.promote_release(
            _contract(), SOURCE_REVISION, "v0.4.0", "secret-token", poll_seconds=0
        )
    assert raised.value.rollbacks == [
        {
            "name": "mettle-api",
            "service_id": "srv-api",
            "action": "already_live",
            "rollback_deploy_id": "dep-previous",
            "restored_commit_id": "0" * 40,
            "status": "live",
        }
    ]


def test_later_service_failure_rolls_back_already_promoted_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    api_is_promoted = False

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        nonlocal api_is_promoted
        calls.append((method, path, payload))
        if method == "GET":
            service = "api" if "srv-api" in path else "mcp"
            deployments: list[dict[str, object]] = [
                {
                    "deploy": {
                        "id": f"dep-{service}-previous",
                        "status": "live",
                        "commit": {"id": "0" * 40},
                    }
                }
            ]
            if service == "api" and api_is_promoted:
                deployments.insert(0, {"deploy": _live("dep-api-release")})
            if service == "mcp" and api_is_promoted:
                deployments.insert(
                    0,
                    {
                        "deploy": {
                            "id": "dep-mcp-failed",
                            "status": "build_failed",
                            "commit": {"id": SOURCE_REVISION},
                            "trigger": "api",
                        }
                    },
                )
            return 200, deployments
        if path == "/services/srv-api/deploys":
            api_is_promoted = True
            return 201, _live("dep-api-release")
        if path == "/services/srv-mcp/deploys":
            return 201, {
                "id": "dep-mcp-failed",
                "status": "build_failed",
                "commit": {"id": SOURCE_REVISION},
                "trigger": "api",
            }
        if path == "/services/srv-mcp/rollback":
            rollback = _live("dep-mcp-rollback")
            rollback["commit"] = {"id": "0" * 40}
            rollback["trigger"] = "rollback"
            return 201, rollback
        if path == "/services/srv-api/rollback":
            rollback = _live("dep-api-rollback")
            rollback["commit"] = {"id": "0" * 40}
            rollback["trigger"] = "rollback"
            return 201, rollback
        raise AssertionError(path)

    monkeypatch.setattr(release, "_request_json", request)

    with pytest.raises(release.RenderPromotionError) as raised:
        release.promote_release(
            _contract(), SOURCE_REVISION, "v0.4.0", "secret-token", poll_seconds=0
        )

    assert raised.value.rollbacks == [
        {
            "name": "mettle-mcp",
            "service_id": "srv-mcp",
            "action": "already_live",
            "rollback_deploy_id": "dep-mcp-previous",
            "restored_commit_id": "0" * 40,
            "status": "live",
        },
        {
            "name": "mettle-api",
            "service_id": "srv-api",
            "action": "rollback",
            "rollback_deploy_id": "dep-api-rollback",
            "restored_commit_id": "0" * 40,
            "status": "live",
        },
    ]
    assert (
        "POST",
        "/services/srv-api/rollback",
        {"deployId": "dep-api-previous"},
    ) in calls


def test_timeout_after_deploy_trigger_rolls_back_the_current_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        calls.append((method, path, payload))
        if method == "GET":
            return 200, [
                {
                    "deploy": {
                        "id": "dep-api-previous",
                        "status": "live",
                        "commit": {"id": "0" * 40},
                    }
                }
            ]
        if path == "/services/srv-api/deploys":
            return 201, {
                "id": "dep-api-attempt",
                "status": "building",
                "commit": {"id": SOURCE_REVISION},
                "trigger": "api",
            }
        if path == "/services/srv-api/rollback":
            rollback = _live("dep-api-rollback")
            rollback["commit"] = {"id": "0" * 40}
            rollback["trigger"] = "rollback"
            return 201, rollback
        raise AssertionError(path)

    monkeypatch.setattr(release, "_request_json", request)

    with pytest.raises(release.RenderPromotionError) as raised:
        release.promote_release(
            _contract(),
            SOURCE_REVISION,
            "v0.4.0",
            "secret-token",
            timeout_seconds=0,
            poll_seconds=0,
        )

    assert raised.value.rollbacks == [
        {
            "name": "mettle-api",
            "service_id": "srv-api",
            "action": "rollback",
            "rollback_deploy_id": "dep-api-rollback",
            "restored_commit_id": "0" * 40,
            "status": "live",
        }
    ]
    assert (
        "POST",
        "/services/srv-api/rollback",
        {"deployId": "dep-api-previous"},
    ) in calls


@pytest.mark.parametrize(
    "path",
    ["https://attacker.example/", "//attacker.example/v1", "services/srv-api"],
)
def test_render_bearer_cannot_be_sent_outside_fixed_origin(path: str) -> None:
    with pytest.raises(ValueError, match="fixed HTTPS origin"):
        release._request_json(path, "secret-token")


@pytest.mark.parametrize(
    "error_type", [urllib.error.URLError, OSError, http.client.HTTPException]
)
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_transport_failure_recovers_every_attempted_service(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    rollback_fails: bool,
) -> None:
    posts: list[str] = []
    original_request = release._request_json

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del payload
        service = "api" if "srv-api" in path else "mcp"
        if method == "GET":
            previous = _live(f"dep-{service}-previous")
            previous["commit"] = {"id": "0" * 40}
            deployments = [{"deploy": previous}]
            if service == "api" and posts:
                deployments.insert(0, {"deploy": _live("dep-api-release")})
            return 200, deployments
        posts.append(path)
        if path == "/services/srv-mcp/deploys":
            if issubclass(error_type, http.client.HTTPException):
                # A truncated response must cross the real request boundary.
                return original_request(path, _token, method=method)
            raise error_type("secret-token: connection lost after trigger")
        if path.endswith("/rollback"):
            if service == "mcp" and rollback_fails:
                raise error_type("secret-token: rollback connection lost")
            restored = _live(f"dep-{service}-rollback")
            restored["commit"] = {"id": "0" * 40}
            restored["trigger"] = "rollback"
            return 201, restored
        return 201, _live("dep-api-release")

    class TruncatedResponse:
        status = 201

        def __enter__(self) -> TruncatedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return release.RENDER_API + "/services/srv-mcp/deploys"

        def read(self) -> bytes:
            raise http.client.IncompleteRead(b"secret-token", 32)

    monkeypatch.setattr(
        release._HTTPS_OPENER, "open", lambda *_a, **_kw: TruncatedResponse()
    )
    monkeypatch.setattr(release, "_request_json", request)
    outcome = "rollback incomplete" if rollback_fails else "services restored"
    with pytest.raises(release.RenderPromotionError, match=outcome) as raised:
        release.promote_release(
            _contract(), SOURCE_REVISION, "v0.5.6", "secret-token", poll_seconds=0
        )

    assert posts == [
        "/services/srv-api/deploys",
        "/services/srv-mcp/deploys",
        "/services/srv-mcp/rollback",
        "/services/srv-api/rollback",
    ]
    assert [item["name"] for item in raised.value.rollbacks] == [
        "mettle-mcp",
        "mettle-api",
    ]
    assert [item["status"] for item in raised.value.rollbacks] == [
        "error" if rollback_fails else "live",
        "live",
    ]
    assert "secret-token" not in str(raised.value)
    assert "secret-token" not in repr(raised.value.rollbacks)


@pytest.mark.parametrize(
    "error_type", [urllib.error.URLError, OSError, http.client.HTTPException]
)
def test_transport_errors_are_secret_safe(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise error_type("secret-token: internal transport detail")

    monkeypatch.setattr(release._HTTPS_OPENER, "open", fail_open)
    with pytest.raises(release.RenderAPIError, match="transport failed") as raised:
        release._request_json("/services/srv-api/deploys", "secret-token")
    assert "secret-token" not in str(raised.value)


@pytest.mark.parametrize("body", [b"\xff", b"{secret-token"])
def test_malformed_response_is_a_recoverable_secret_safe_api_error(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    response = MagicMock()
    response.status = 201
    response.geturl.return_value = release.RENDER_API + "/services/srv-api/deploys"
    response.read.return_value = body
    opener = MagicMock()
    opener.return_value.__enter__.return_value = response
    monkeypatch.setattr(release._HTTPS_OPENER, "open", opener)

    with pytest.raises(release.RenderAPIError, match="malformed JSON") as raised:
        release._request_json("/services/srv-api/deploys", "secret-token")
    assert "secret-token" not in str(raised.value)


@pytest.mark.parametrize("pending_visible", [False, True])
def test_retry_lost_response_cannot_use_a_historical_failed_attempt(
    monkeypatch: pytest.MonkeyPatch, pending_visible: bool
) -> None:
    posts: list[str] = []

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del payload
        service = "api" if "srv-api" in path else "mcp"
        if method == "GET":
            previous = _live(f"dep-{service}-previous")
            previous["commit"] = {"id": "0" * 40}
            deployments = [{"deploy": previous}]
            if service == "api" and posts:
                deployments.insert(0, {"deploy": _live("dep-api-release")})
            if service == "mcp":
                old = _live("dep-mcp-old-failure")
                old.update(status="build_failed", createdAt="2026-10-07T00:00:00Z")
                deployments.insert(0, {"deploy": old})
                if pending_visible and "/services/srv-mcp/deploys" in posts:
                    pending = _live("dep-mcp-new-attempt")
                    pending.update(status="building", createdAt="2026-10-08T00:00:00Z")
                    deployments.insert(0, {"deploy": pending})
            return 200, deployments
        posts.append(path)
        if path == "/services/srv-mcp/deploys":
            raise urllib.error.URLError("secret-token: accepted trigger response lost")
        if path.endswith("/rollback"):
            restored = _live(f"dep-{service}-rollback")
            restored["commit"] = {"id": "0" * 40}
            restored["trigger"] = "rollback"
            return 201, restored
        return 201, _live("dep-api-release")

    monkeypatch.setattr(release, "_request_json", request)
    with pytest.raises(
        release.RenderPromotionError, match="services restored"
    ) as raised:
        release.promote_release(
            _contract(), SOURCE_REVISION, "v0.5.6", "secret-token", poll_seconds=0
        )
    assert posts == [
        "/services/srv-api/deploys",
        "/services/srv-mcp/deploys",
        "/services/srv-mcp/rollback",
        "/services/srv-api/rollback",
    ]
    assert [item["action"] for item in raised.value.rollbacks] == [
        "rollback",
        "rollback",
    ]
    assert "secret-token" not in repr(raised.value.rollbacks)


def test_queued_retry_waits_for_a_new_deployment_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued: set[str] = set()
    observations: dict[str, int] = {"api": 0, "mcp": 0}

    def request(
        path: str,
        _token: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> tuple[int, object | None]:
        del payload
        service = "api" if "srv-api" in path else "mcp"
        if method == "POST":
            assert path.endswith("/deploys")
            queued.add(service)
            return 202, None
        previous = _live(f"dep-{service}-previous")
        previous["commit"] = {"id": "0" * 40}
        old = _live(f"dep-{service}-old-failure")
        old["status"] = "build_failed"
        deployments = [{"deploy": old}, {"deploy": previous}]
        if service in queued:
            observations[service] += 1
            if observations[service] == 2:
                deployments.insert(0, {"deploy": _live(f"dep-{service}-new-attempt")})
        return 200, deployments

    monkeypatch.setattr(release, "_request_json", request)
    receipt = release.promote_release(
        _contract(), SOURCE_REVISION, "v0.5.6", "secret-token", poll_seconds=0
    )
    services = receipt["services"]
    assert isinstance(services, list)
    assert [item["deploy_id"] for item in services] == [
        "dep-api-new-attempt",
        "dep-mcp-new-attempt",
    ]
    assert observations == {"api": 2, "mcp": 2}


def test_production_job_reserves_time_for_both_promotions_and_rollbacks() -> None:
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / ".github/workflows/release.yml"
        ).read_text()
    )
    operation_seconds = (
        inspect.signature(release.promote_release).parameters["timeout_seconds"].default
    )
    job_minutes = workflow["jobs"]["deploy-render-production"]["timeout-minutes"]
    assert job_minutes * 60 >= 4 * operation_seconds + 15 * 60
