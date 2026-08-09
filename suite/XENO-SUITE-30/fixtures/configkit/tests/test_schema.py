"""Baseline behaviour of schema validation."""

from __future__ import annotations

from typing import Any

import pytest

from configkit.errors import ValidationError
from configkit.schema import Field, Schema


def build_schema() -> Schema:
    """A small schema exercising nesting, defaults and required fields."""
    return Schema(
        [
            Field("name", str, required=True),
            Field("db.host", str, default="localhost"),
            Field("db.port", int, default=5432),
            Field("db.tls", bool, default=False),
            Field("timeout", float, default=1.5),
        ]
    )


def test_validate_fills_defaults_for_absent_fields() -> None:
    result = build_schema().validate({"name": "svc"})
    assert result["db"] == {"host": "localhost", "port": 5432, "tls": False}
    assert result["timeout"] == 1.5


def test_validate_leaves_a_none_default_absent() -> None:
    schema = Schema([Field("db.dsn", str)])
    assert schema.validate({}) == {}


def test_validate_raises_on_a_missing_required_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_schema().validate({})
    assert excinfo.value.field == "name"


def test_validate_raises_on_a_wrong_type() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_schema().validate({"name": "svc", "db": {"port": "5432"}})
    assert excinfo.value.field == "db.port"


def test_validate_preserves_undeclared_keys_and_does_not_mutate_input() -> None:
    data: dict[str, Any] = {"name": "svc", "extra": {"k": "v"}}
    result = build_schema().validate(data)
    assert result["extra"] == {"k": "v"}
    assert data == {"name": "svc", "extra": {"k": "v"}}


def test_validate_accepts_an_int_for_a_float_field_but_not_a_bool() -> None:
    assert build_schema().validate({"name": "svc", "timeout": 2})["timeout"] == 2
    with pytest.raises(ValidationError):
        build_schema().validate({"name": "svc", "timeout": True})


def test_schema_reports_its_field_names() -> None:
    schema = build_schema()
    assert schema.field_names() == ("name", "db.host", "db.port", "db.tls", "timeout")
    assert schema.required_names() == ("name",)


def test_schema_rejects_duplicate_field_names() -> None:
    with pytest.raises(ValueError):
        Schema([Field("a", str), Field("a", int)])
