"""ZS-25 acceptance spec: schema-typed environment overrides end to end.

Four pieces must land together:

``configkit.schema``
    ``Schema.type_for(dotted_path)`` returns the declared type of a field, or
    ``None`` when the path is not declared.
``configkit.env``
    ``env_overrides(prefix, environ=None, schema=None)`` coerces each value to
    the type the schema declares (``int``, ``float``, ``bool`` from
    ``true/false/1/0/yes/no`` case-insensitively, ``str`` untouched) and raises
    ``ValidationError`` when a value cannot be coerced. With ``schema=None``
    every value stays a string.
``configkit.loader``
    ``load_config(paths, schema, env_prefix, environ=None)`` layers the files,
    applies the typed environment overrides, then runs ``schema.validate``.
``configkit``
    ``load_config`` is re-exported from the package root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import configkit
from configkit import Field, Schema, load_config
from configkit.env import env_overrides
from configkit.errors import ValidationError
from configkit.loader import load_config as loader_load_config


def build_schema() -> Schema:
    """Schema shared by the cases below."""
    return Schema(
        [
            Field("name", str, required=True),
            Field("db.host", str, default="localhost"),
            Field("db.port", int, default=5432),
            Field("db.tls", bool, default=False),
            Field("db.timeout", float, default=1.5),
        ]
    )


def test_type_for_resolves_declared_dotted_paths() -> None:
    schema = build_schema()
    assert schema.type_for("db.port") is int
    assert schema.type_for("db.tls") is bool
    assert schema.type_for("db.timeout") is float
    assert schema.type_for("name") is str


def test_type_for_returns_none_for_undeclared_paths() -> None:
    schema = build_schema()
    assert schema.type_for("db.missing") is None
    assert schema.type_for("nope") is None


def test_env_overrides_coerces_int_and_float_using_the_schema() -> None:
    environ = {"APP__DB__PORT": "6543", "APP__DB__TIMEOUT": "2.5"}
    result = env_overrides("APP", environ, schema=build_schema())
    assert result["db"]["port"] == 6543
    assert result["db"]["timeout"] == 2.5


@pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", "YES"])
def test_env_overrides_coerces_truthy_bool_spellings(raw: str) -> None:
    result = env_overrides("APP", {"APP__DB__TLS": raw}, schema=build_schema())
    assert result["db"]["tls"] is True


@pytest.mark.parametrize("raw", ["false", "FALSE", "False", "0", "no", "NO"])
def test_env_overrides_coerces_falsy_bool_spellings(raw: str) -> None:
    result = env_overrides("APP", {"APP__DB__TLS": raw}, schema=build_schema())
    assert result["db"]["tls"] is False


def test_env_overrides_leaves_str_fields_untouched() -> None:
    result = env_overrides("APP", {"APP__DB__HOST": "1234"}, schema=build_schema())
    assert result["db"]["host"] == "1234"


def test_env_overrides_leaves_undeclared_paths_as_strings() -> None:
    result = env_overrides("APP", {"APP__EXTRA__COUNT": "7"}, schema=build_schema())
    assert result["extra"]["count"] == "7"


def test_env_overrides_without_a_schema_keeps_every_value_a_string() -> None:
    environ = {"APP__DB__PORT": "6543", "APP__DB__TLS": "true"}
    result = env_overrides("APP", environ, schema=None)
    assert result["db"]["port"] == "6543"
    assert result["db"]["tls"] == "true"


def test_env_overrides_raises_validation_error_on_an_uncoercible_int() -> None:
    with pytest.raises(ValidationError):
        env_overrides("APP", {"APP__DB__PORT": "not-a-number"}, schema=build_schema())


def test_env_overrides_raises_validation_error_on_an_uncoercible_bool() -> None:
    with pytest.raises(ValidationError):
        env_overrides("APP", {"APP__DB__TLS": "maybe"}, schema=build_schema())


def test_load_config_layers_files_then_env_then_defaults(tmp_path: Path) -> None:
    base = tmp_path / "base.ini"
    base.write_text("name = svc\n[db]\nhost = localhost\n", encoding="utf-8")
    site = tmp_path / "site.json"
    site.write_text(json.dumps({"db": {"host": "site.internal"}}), encoding="utf-8")

    result = load_config(
        [base, site],
        schema=build_schema(),
        env_prefix="APP",
        environ={"APP__DB__PORT": "6543", "APP__DB__TLS": "yes"},
    )

    assert result["name"] == "svc"
    assert result["db"]["host"] == "site.internal"
    assert result["db"]["port"] == 6543
    assert result["db"]["tls"] is True
    assert result["db"]["timeout"] == 1.5


def test_load_config_reports_schema_violations(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"db": {"host": "h"}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config([base], schema=build_schema(), env_prefix="APP", environ={})


def test_load_config_is_exported_from_the_package_root() -> None:
    assert "load_config" in configkit.__all__
    assert configkit.load_config is loader_load_config
