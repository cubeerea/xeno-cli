"""The public surface of the package and how the layers compose today."""

from __future__ import annotations

import json
from pathlib import Path

import configkit
from configkit import Field, Schema, deep_merge, env_overrides, load_layers


def test_public_names_are_exported_and_sorted() -> None:
    assert configkit.__all__ == sorted(configkit.__all__)
    for name in configkit.__all__:
        assert hasattr(configkit, name)


def test_errors_share_a_common_base() -> None:
    assert issubclass(configkit.ParseError, configkit.ConfigError)
    assert issubclass(configkit.ValidationError, configkit.ConfigError)


def test_files_then_env_then_schema_compose_by_hand(tmp_path: Path) -> None:
    base = tmp_path / "base.ini"
    base.write_text("name = svc\n[db]\nhost = localhost\n", encoding="utf-8")
    site = tmp_path / "site.json"
    site.write_text(json.dumps({"db": {"host": "site.internal"}}), encoding="utf-8")

    layered = load_layers([base, site])
    layered = deep_merge(layered, env_overrides("APP", {"APP__DB__PORT": "5432"}))
    schema = Schema([Field("name", str, required=True), Field("db.tls", str, default="off")])
    result = schema.validate(layered)

    assert result["db"] == {"host": "site.internal", "port": "5432", "tls": "off"}
    assert result["name"] == "svc"
