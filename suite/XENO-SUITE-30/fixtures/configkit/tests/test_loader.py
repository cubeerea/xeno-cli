"""Baseline behaviour of the file loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configkit.errors import ParseError
from configkit.loader import load_file, load_layers, parse_ini


def test_parse_ini_builds_nested_sections_of_strings() -> None:
    text = "\n".join(
        [
            "name = base",
            "# a comment",
            "; another comment",
            "[db]",
            "host = localhost",
            "port = 5432",
            "",
            "[db.pool]",
            "size = 10",
        ]
    )
    assert parse_ini(text) == {
        "name": "base",
        "db": {"host": "localhost", "port": "5432", "pool": {"size": "10"}},
    }


def test_parse_ini_keeps_values_verbatim_after_the_first_equals() -> None:
    parsed = parse_ini("[url]\ndsn = postgres://u:p@h/db?x=1")
    assert parsed["url"]["dsn"] == "postgres://u:p@h/db?x=1"


def test_parse_ini_rejects_a_line_without_an_equals_sign() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_ini("[db]\nhost localhost", source="cfg.ini")
    assert excinfo.value.line == 2


def test_parse_ini_rejects_an_unterminated_section_header() -> None:
    with pytest.raises(ParseError):
        parse_ini("[db\nhost = localhost")


def test_load_file_reads_json_objects(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"db": {"port": 5432}}), encoding="utf-8")
    assert load_file(path) == {"db": {"port": 5432}}


def test_load_file_rejects_a_non_object_json_document(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ParseError):
        load_file(path)


def test_load_file_rejects_an_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("db: {}", encoding="utf-8")
    with pytest.raises(ParseError):
        load_file(path)


def test_load_layers_merges_files_in_order(tmp_path: Path) -> None:
    base = tmp_path / "base.ini"
    base.write_text("[db]\nhost = localhost\nport = 5432\n", encoding="utf-8")
    prod = tmp_path / "prod.json"
    prod.write_text(json.dumps({"db": {"host": "prod.internal"}}), encoding="utf-8")
    merged = load_layers([base, prod])
    assert merged == {"db": {"host": "prod.internal", "port": "5432"}}
