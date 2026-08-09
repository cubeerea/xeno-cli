from __future__ import annotations

import pytest

from etl.pipeline import Pipeline, RunStats
from etl.sinks import MemorySink
from etl.sources import ListSource, Row
from etl.transforms import add_constant, cast_fields, filter_rows


def _boom(row: Row) -> Row | None:
    raise RuntimeError(f"bad row: {row!r}")


def test_run_moves_every_row_to_the_sink() -> None:
    sink = MemorySink()
    stats = Pipeline(ListSource([{"id": 1}, {"id": 2}]), [], sink).run()
    assert sink.rows == [{"id": 1}, {"id": 2}]
    assert (stats.rows_in, stats.rows_out, stats.rows_skipped) == (2, 2, 0)


def test_run_applies_transforms_in_order() -> None:
    sink = MemorySink()
    Pipeline(
        ListSource([{"n": "3"}]),
        [cast_fields({"n": int}), add_constant("seen", True)],
        sink,
    ).run()
    assert sink.rows == [{"n": 3, "seen": True}]


def test_run_counts_dropped_rows_as_skipped() -> None:
    sink = MemorySink()
    stats = Pipeline(
        ListSource([{"keep": True}, {"keep": False}, {"keep": True}]),
        [filter_rows(lambda row: bool(row["keep"]))],
        sink,
    ).run()
    assert stats.rows_in == 3
    assert stats.rows_out == 2
    assert stats.rows_skipped == 1
    assert len(sink) == 2


def test_dropped_rows_never_reach_the_sink() -> None:
    sink = MemorySink()
    Pipeline(ListSource([{"id": 1}]), [filter_rows(lambda row: False)], sink).run()
    assert sink.rows == []


def test_run_closes_the_sink() -> None:
    sink = MemorySink()
    Pipeline(ListSource([{"id": 1}]), [], sink).run()
    assert sink.closed


def test_transform_exception_propagates_by_default() -> None:
    sink = MemorySink()
    with pytest.raises(RuntimeError, match="bad row"):
        Pipeline(ListSource([{"id": 1}]), [_boom], sink).run()


def test_empty_source_produces_zero_stats() -> None:
    stats = Pipeline(ListSource([]), [], MemorySink()).run()
    assert stats.as_dict()["rows_in"] == 0
    assert stats.as_dict()["rows_out"] == 0


def test_run_stats_as_dict_is_json_friendly() -> None:
    stats = RunStats(rows_in=3, rows_out=2, rows_skipped=1)
    as_dict = stats.as_dict()
    assert as_dict["rows_in"] == 3
    assert as_dict["rows_out"] == 2
    assert as_dict["rows_skipped"] == 1
