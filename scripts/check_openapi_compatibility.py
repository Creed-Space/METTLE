#!/usr/bin/env python3
"""Check the public OpenAPI contract against its reviewed snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "openapi-v1.json"
HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


def current_schema() -> dict[str, Any]:
    """Build the application schema under a non-production configuration."""
    os.environ.setdefault("METTLE_ENVIRONMENT", "test")
    os.environ.setdefault("METTLE_SECRET_KEY", "openapi-snapshot-secret")
    os.environ.setdefault("METTLE_ADMIN_API_KEY", "openapi-snapshot-admin")
    sys.path.insert(0, str(ROOT))
    from main import app

    return app.openapi()


def canonical(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Return conservative compatibility breaks requiring an explicit version plan."""
    findings: list[str] = []
    old_paths = old.get("paths", {})
    new_paths = new.get("paths", {})
    for path, old_path_item in old_paths.items():
        if path not in new_paths:
            findings.append(f"removed path {path}")
            continue
        for method in HTTP_METHODS & set(old_path_item):
            if method not in new_paths[path]:
                findings.append(f"removed operation {method.upper()} {path}")
                continue
            old_operation = old_path_item[method]
            new_operation = new_paths[path][method]
            old_responses = set(old_operation.get("responses", {}))
            new_responses = set(new_operation.get("responses", {}))
            for response in sorted(old_responses - new_responses):
                findings.append(
                    f"removed response {method.upper()} {path} status {response}"
                )

            def parameter_key(parameter: dict[str, Any]) -> tuple[str, str]:
                return str(parameter.get("in", "")), str(parameter.get("name", ""))

            old_parameters = {
                parameter_key(parameter): parameter
                for parameter in [
                    *old_path_item.get("parameters", []),
                    *old_operation.get("parameters", []),
                ]
            }
            new_parameters = {
                parameter_key(parameter): parameter
                for parameter in [
                    *new_paths[path].get("parameters", []),
                    *new_operation.get("parameters", []),
                ]
            }
            for key, old_parameter in old_parameters.items():
                if key not in new_parameters:
                    findings.append(
                        f"removed parameter {method.upper()} {path} {key[0]}:{key[1]}"
                    )
                    continue
                new_parameter = new_parameters[key]
                if not old_parameter.get("required") and new_parameter.get("required"):
                    findings.append(
                        f"new required parameter {method.upper()} {path} {key[0]}:{key[1]}"
                    )
                old_type = old_parameter.get("schema", {}).get("type")
                new_type = new_parameter.get("schema", {}).get("type")
                if old_type and new_type and old_type != new_type:
                    findings.append(
                        f"changed parameter type {method.upper()} {path} "
                        f"{key[0]}:{key[1]} {old_type}->{new_type}"
                    )
            for key, new_parameter in new_parameters.items():
                if key not in old_parameters and new_parameter.get("required"):
                    findings.append(
                        f"new required parameter {method.upper()} {path} {key[0]}:{key[1]}"
                    )
    old_schemas = old.get("components", {}).get("schemas", {})
    new_schemas = new.get("components", {}).get("schemas", {})
    for name, old_model in old_schemas.items():
        if name not in new_schemas:
            findings.append(f"removed schema {name}")
            continue
        old_required = set(old_model.get("required", []))
        new_required = set(new_schemas[name].get("required", []))
        for field in sorted(new_required - old_required):
            findings.append(f"new required field {name}.{field}")
        old_properties = old_model.get("properties", {})
        new_properties = new_schemas[name].get("properties", {})
        for field in sorted(set(old_properties) - set(new_properties)):
            findings.append(f"removed field {name}.{field}")
        for field in sorted(set(old_properties) & set(new_properties)):
            old_type = old_properties[field].get("type")
            new_type = new_properties[field].get("type")
            if old_type and new_type and old_type != new_type:
                findings.append(
                    f"changed field type {name}.{field} {old_type}->{new_type}"
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Replace the snapshot after reviewing compatibility and version impact",
    )
    args = parser.parse_args()
    schema = current_schema()
    rendered = canonical(schema)

    if args.update:
        SNAPSHOT.write_text(rendered, encoding="utf-8")
        print(f"Updated {SNAPSHOT.relative_to(ROOT)}")
        return 0
    if not SNAPSHOT.exists():
        print(
            "OpenAPI snapshot is missing; review and run with --update", file=sys.stderr
        )
        return 1
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if canonical(old) == rendered:
        print("OpenAPI snapshot matches the application schema")
        return 0

    breaks = breaking_changes(old, schema)
    if breaks:
        print("Breaking OpenAPI changes detected:", file=sys.stderr)
        for finding in breaks:
            print(f"  - {finding}", file=sys.stderr)
    else:
        print("Additive or descriptive OpenAPI drift detected", file=sys.stderr)
    print(
        "Review the contract and version impact, then run "
        "python3 scripts/check_openapi_compatibility.py --update",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
