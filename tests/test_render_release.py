"""Tag-bound Render production promotion tests."""

from __future__ import annotations

from typing import Any

import pytest

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
