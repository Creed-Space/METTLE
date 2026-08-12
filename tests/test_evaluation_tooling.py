"""Tests for privacy-minimal policy and harvesting evaluation tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scripts.evaluate_policy_dataset import aggregate, load_rows
from scripts.testing.evaluate_challenge_harvesting import evaluate


def test_policy_aggregate_computes_false_decision_rates(tmp_path) -> None:
    path = tmp_path / "held-out.jsonl"
    rows = [
        {
            "dataset_version": "test-only",
            "subject_class": "becoming-mind",
            "suite": "adversarial",
            "expected_pass": True,
            "observed_pass": False,
        },
        {
            "dataset_version": "test-only",
            "subject_class": "becoming-mind",
            "suite": "adversarial",
            "expected_pass": False,
            "observed_pass": True,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = aggregate(load_rows(path))

    assert result["records"] == 2
    assert result["results"][0]["false_reject_rate"] == 1.0
    assert result["results"][0]["false_accept_rate"] == 1.0
    assert result["results"][0]["insufficient_data"] is True

    schema = json.loads(
        Path("evaluation/aggregate-schema-v1.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)


def test_evaluation_schemas_are_valid() -> None:
    for path in (
        Path("evaluation/input-schema-v1.json"),
        Path("evaluation/aggregate-schema-v1.json"),
    ):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_policy_input_rejects_raw_answer_or_identity_fields(tmp_path) -> None:
    path = tmp_path / "unsafe.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset_version": "unsafe",
                "subject_class": "becoming-mind",
                "suite": "adversarial",
                "expected_pass": True,
                "observed_pass": True,
                "answer": "must not be retained",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="privacy-sensitive"):
        load_rows(path)


def test_harvesting_report_has_quantified_rotation_triggers() -> None:
    report = evaluate(100)

    assert report["suite_policy_version"]
    assert len(report["metrics"]) == 3
    assert all(0 <= item["collision_rate"] <= 1 for item in report["metrics"])
    assert report["rotation_triggers"]["adaptive_replay_coverage_max"] == 0.05
