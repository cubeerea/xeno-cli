"""ZS-15 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from etl.pipeline import Pipeline
from etl.sinks import MemorySink
from etl.sources import CsvSource, ListSource, Row
from etl.transforms import cast_fields


def _write_csv(tmp_path: Path, text: str, name: str = "in.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _boom_on_id_two(row: Row) -> Row | None:
    if row.get("id") in {2, "2"}:
        raise RuntimeError("row 2 is cursed")
    return row


def test_csv_source_yields_rows_keyed_by_header(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "id,name\n1,ada\n2,bob\n")
    assert list(CsvSource(path).rows()) == [
        {"id": "1", "name": "ada"},
        {"id": "2", "name": "bob"},
    ]


def test_csv_source_supports_a_custom_delimiter(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "id;name\n1;ada\n")
    assert list(CsvSource(path, delimiter=";").rows()) == [{"id": "1", "name": "ada"}]


def test_csv_source_on_empty_file_yields_nothing(tmp_path: Path) -> None:
    assert list(CsvSource(_write_csv(tmp_path, "")).rows()) == []


def test_csv_source_header_only_file_yields_nothing(tmp_path: Path) -> None:
    assert list(CsvSource(_write_csv(tmp_path, "id,name\n")).rows()) == []


def test_csv_source_is_replayable(tmp_path: Path) -> None:
    source = CsvSource(_write_csv(tmp_path, "id\n1\n2\n"))
    assert list(source.rows()) == list(source.rows())


def test_csv_source_feeds_a_pipeline(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "id,name\n1,ada\n2,bob\n")
    sink = MemorySink()
    stats = Pipeline(CsvSource(path), [cast_fields({"id": int})], sink).run()
    assert stats.rows_in == 2
    assert stats.rows_out == 2
    assert sink.rows == [{"id": 1, "name": "ada"}, {"id": 2, "name": "bob"}]


def test_pipeline_default_on_error_is_raise() -> None:
    with pytest.raises(RuntimeError, match="cursed"):
        Pipeline(ListSource([{"id": 1}, {"id": 2}]), [_boom_on_id_two], MemorySink()).run()


def test_pipeline_on_error_raise_is_explicitly_supported() -> None:
    with pytest.raises(RuntimeError, match="cursed"):
        Pipeline(
            ListSource([{"id": 2}]),
            [_boom_on_id_two],
            MemorySink(),
            on_error="raise",
        ).run()


def test_pipeline_on_error_skip_counts_and_continues() -> None:
    sink = MemorySink()
    stats = Pipeline(
        ListSource([{"id": 1}, {"id": 2}, {"id": 3}]),
        [_boom_on_id_two],
        sink,
        on_error="skip",
    ).run()
    assert stats.rows_in == 3
    assert stats.rows_out == 2
    assert stats.rows_skipped == 1
    assert sink.rows == [{"id": 1}, {"id": 3}]


def test_on_error_skip_still_counts_transform_drops() -> None:
    def drop_id_three(row: Row) -> Row | None:
        return None if row.get("id") == 3 else row

    stats = Pipeline(
        ListSource([{"id": 1}, {"id": 2}, {"id": 3}]),
        [_boom_on_id_two, drop_id_three],
        MemorySink(),
        on_error="skip",
    ).run()
    assert stats.rows_out == 1
    assert stats.rows_skipped == 2


def test_csv_source_and_skip_policy_together(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "id,name\n1,ada\n2,bob\n3,cy\n")
    sink = MemorySink()
    stats = Pipeline(
        CsvSource(path),
        [_boom_on_id_two, cast_fields({"id": int})],
        sink,
        on_error="skip",
    ).run()
    assert stats.rows_in == 3
    assert stats.rows_out == 2
    assert stats.rows_skipped == 1
    assert sink.rows == [{"id": 1, "name": "ada"}, {"id": 3, "name": "cy"}]
    assert sink.closed
