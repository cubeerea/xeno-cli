"""ZS-23 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etl.config import build_pipeline
from etl.pipeline import Pipeline
from etl.sinks import JsonLinesSink, MemorySink
from etl.sources import ListSource, Row


def _lines(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _boom(row: Row) -> Row | None:
    if row.get("id") == 3:
        raise RuntimeError("row 3 is cursed")
    return row


def test_jsonl_sink_flushes_every_batch_size_rows(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonLinesSink(path, batch_size=2)
    sink.write({"id": 1})
    assert not path.exists() or _lines(path) == []
    sink.write({"id": 2})
    assert _lines(path) == [{"id": 1}, {"id": 2}]
    sink.write({"id": 3})
    assert _lines(path) == [{"id": 1}, {"id": 2}]


def test_jsonl_sink_flushes_remainder_on_close(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonLinesSink(path, batch_size=2)
    for index in range(3):
        sink.write({"id": index})
    sink.close()
    assert _lines(path) == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_jsonl_sink_default_batch_size_is_100(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonLinesSink(path)
    for index in range(99):
        sink.write({"id": index})
    assert not path.exists() or _lines(path) == []
    sink.write({"id": 99})
    assert len(_lines(path)) == 100


def test_jsonl_sink_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonLinesSink(path, batch_size=10)
    sink.write({"id": 1})
    sink.close()
    sink.close()
    assert _lines(path) == [{"id": 1}]


def test_jsonl_sink_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonLinesSink(path, batch_size=1)
    sink.write({"id": 1, "name": "ada"})
    sink.write({"id": 2, "name": "bob"})
    sink.close()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_build_pipeline_wires_a_jsonl_sink(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    pipeline = build_pipeline(
        {
            "source": {"type": "list", "rows": [{"id": 1}, {"id": 2}, {"id": 3}]},
            "sink": {"type": "jsonl", "path": str(path), "batch_size": 2},
        }
    )
    assert isinstance(pipeline.sink, JsonLinesSink)
    stats = pipeline.run()
    assert stats.rows_out == 3
    assert _lines(path) == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_run_stats_reports_rows_written_for_jsonl_sink(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    stats = Pipeline(
        ListSource([{"id": 1}, {"id": 2}, {"id": 3}]),
        [],
        JsonLinesSink(path, batch_size=2),
    ).run()
    assert stats.rows_written == 3
    assert stats.as_dict()["rows_written"] == 3


def test_run_stats_reports_rows_written_for_memory_sink() -> None:
    stats = Pipeline(ListSource([{"id": 1}, {"id": 2}]), [], MemorySink()).run()
    assert stats.rows_written == 2


def test_sink_is_closed_when_a_transform_raises(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonLinesSink(path, batch_size=100)
    with pytest.raises(RuntimeError, match="cursed"):
        Pipeline(ListSource([{"id": 1}, {"id": 2}, {"id": 3}]), [_boom], sink).run()
    assert _lines(path) == [{"id": 1}, {"id": 2}]


def test_memory_sink_is_closed_when_a_transform_raises() -> None:
    sink = MemorySink()
    with pytest.raises(RuntimeError, match="cursed"):
        Pipeline(ListSource([{"id": 3}]), [_boom], sink).run()
    assert sink.closed
