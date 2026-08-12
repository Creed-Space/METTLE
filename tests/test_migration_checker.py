"""Tests for the non-sensitive migration status command."""

from __future__ import annotations

import json

from scripts import check_migrations


def test_check_report_is_read_only_and_hides_the_dsn(monkeypatch) -> None:
    applied: list[bool] = []
    monkeypatch.setattr(
        check_migrations.database,
        "DATABASE_URL",
        "postgresql://secret@db.example/mettle",
    )
    monkeypatch.setattr(check_migrations.database, "LATEST_SCHEMA_VERSION", 2)
    monkeypatch.setattr(check_migrations.database, "get_schema_version", lambda: 2)
    monkeypatch.setattr(check_migrations.database, "check_health", lambda: True)
    monkeypatch.setattr(
        check_migrations.database, "init_db", lambda: applied.append(True)
    )

    report = check_migrations.build_report()

    assert applied == []
    assert report == {
        "schema": "mettle-migration-status-v1",
        "database_scheme": "postgresql",
        "database_healthy": True,
        "current_version": 2,
        "latest_version": 2,
        "current": True,
        "action": "check",
    }
    assert "secret" not in json.dumps(report)
    assert "db.example" not in json.dumps(report)


def test_apply_runs_migrations_before_reading_status(monkeypatch) -> None:
    events: list[str] = []

    def read_version() -> int:
        events.append("read")
        return 2

    monkeypatch.setattr(check_migrations.database, "DATABASE_URL", "sqlite:///local.db")
    monkeypatch.setattr(check_migrations.database, "LATEST_SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        check_migrations.database, "init_db", lambda: events.append("apply")
    )
    monkeypatch.setattr(
        check_migrations.database,
        "get_schema_version",
        read_version,
    )
    monkeypatch.setattr(check_migrations.database, "check_health", lambda: True)

    report = check_migrations.build_report(apply=True)

    assert events == ["apply", "read"]
    assert report["action"] == "apply"
    assert report["current"] is True


def test_main_fails_when_schema_is_not_current(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_migrations,
        "build_report",
        lambda **_kwargs: {
            "schema": "mettle-migration-status-v1",
            "current": False,
        },
    )
    assert check_migrations.main(["--check"]) == 1
    assert json.loads(capsys.readouterr().out)["current"] is False
