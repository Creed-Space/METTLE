"""Conservative OpenAPI break classification and public-schema tests."""

from scripts.check_openapi_compatibility import breaking_changes, current_schema


def test_breaking_change_classifier_covers_operations_parameters_and_models() -> None:
    old = {
        "paths": {
            "/things/{thing_id}": {
                "get": {
                    "parameters": [
                        {
                            "in": "path",
                            "name": "thing_id",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "in": "query",
                            "name": "verbose",
                            "required": False,
                            "schema": {"type": "boolean"},
                        },
                    ],
                    "responses": {"200": {}, "404": {}},
                }
            }
        },
        "components": {
            "schemas": {
                "Thing": {
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                    },
                }
            }
        },
    }
    new = {
        "paths": {
            "/things/{thing_id}": {
                "get": {
                    "parameters": [
                        {
                            "in": "path",
                            "name": "thing_id",
                            "required": True,
                            "schema": {"type": "integer"},
                        },
                        {
                            "in": "query",
                            "name": "verbose",
                            "required": True,
                            "schema": {"type": "boolean"},
                        },
                    ],
                    "responses": {"200": {}},
                }
            }
        },
        "components": {
            "schemas": {
                "Thing": {
                    "required": ["id", "added"],
                    "properties": {
                        "id": {"type": "integer"},
                        "added": {"type": "string"},
                    },
                }
            }
        },
    }

    findings = breaking_changes(old, new)

    assert "removed response GET /things/{thing_id} status 404" in findings
    assert (
        "changed parameter type GET /things/{thing_id} path:thing_id string->integer"
        in findings
    )
    assert "new required parameter GET /things/{thing_id} query:verbose" in findings
    assert "new required field Thing.added" in findings
    assert "removed field Thing.label" in findings
    assert "changed field type Thing.id string->integer" in findings


def test_additive_optional_field_is_not_breaking() -> None:
    old = {
        "paths": {},
        "components": {
            "schemas": {"Thing": {"properties": {"id": {"type": "string"}}}}
        },
    }
    new = {
        "paths": {},
        "components": {
            "schemas": {
                "Thing": {
                    "properties": {
                        "id": {"type": "string"},
                        "optional": {"type": "string"},
                    }
                }
            }
        },
    }

    assert breaking_changes(old, new) == []


def test_validation_error_schema_omits_rejected_payload_details() -> None:
    properties = current_schema()["components"]["schemas"]["ValidationError"][
        "properties"
    ]

    assert set(properties) == {"loc", "msg", "type"}
