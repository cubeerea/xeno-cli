from __future__ import annotations

from pathlib import Path

import pytest

from etl.sources import JsonLinesSource, ListSource, Source, SourceError


def test_list_source_yields_every_row() -> None:
    source = ListSource([{"id": 1}, {"id": 2}])
    assert list(source.rows()) == [{"id": 1}, {"id": 2}]
    assert len(source) == 2


def test_list_source_is_replayable() -> None:
    source = ListSource([{"id": 1}])
    assert list(source.rows()) == list(source.rows())


def test_list_source_copies_rows_defensively() -> None:
    original = {"id": 1}
    source = ListSource([original])
    emitted = next(iter(source.rows()))
    emitted["id"] = 99
    assert original == {"id": 1}
    assert list(source.rows()) == [{"id": 1}]


def test_list_source_satisfies_source_protocol() -> None:
    assert isinstance(ListSource([]), Source)


def test_jsonl_source_reads_objects(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text('{"id": 1, "name": "ada"}\n{"id": 2, "name": "bob"}\n', encoding="utf-8")
    assert list(JsonLinesSource(path).rows()) == [
        {"id": 1, "name": "ada"},
        {"id": 2, "name": "bob"},
    ]


def test_jsonl_source_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text('\n{"id": 1}\n\n   \n{"id": 2}\n', encoding="utf-8")
    assert [row["id"] for row in JsonLinesSource(path).rows()] == [1, 2]


def test_jsonl_source_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text('{"id": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(SourceError):
        list(JsonLinesSource(path).rows())


def test_jsonl_source_rejects_non_object_lines(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(SourceError):
        list(JsonLinesSource(path).rows())
