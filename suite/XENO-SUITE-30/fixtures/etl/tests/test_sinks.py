from __future__ import annotations

import json
from pathlib import Path

import pytest

from etl.sinks import JsonSink, MemorySink, Sink, SinkError


def test_memory_sink_collects_rows() -> None:
    sink = MemorySink()
    sink.write({"id": 1})
    sink.write({"id": 2})
    assert sink.rows == [{"id": 1}, {"id": 2}]
    assert len(sink) == 2


def test_memory_sink_copies_rows() -> None:
    sink = MemorySink()
    row = {"id": 1}
    sink.write(row)
    row["id"] = 99
    assert sink.rows == [{"id": 1}]


def test_memory_sink_tracks_closed() -> None:
    sink = MemorySink()
    assert not sink.closed
    sink.close()
    assert sink.closed


def test_memory_sink_rejects_write_after_close() -> None:
    sink = MemorySink()
    sink.close()
    with pytest.raises(SinkError):
        sink.write({"id": 1})


def test_sinks_satisfy_sink_protocol(tmp_path: Path) -> None:
    assert isinstance(MemorySink(), Sink)
    assert isinstance(JsonSink(tmp_path / "out.json"), Sink)


def test_json_sink_writes_array_on_close(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    sink = JsonSink(path)
    sink.write({"id": 1})
    sink.write({"id": 2})
    assert not path.exists()
    sink.close()
    assert json.loads(path.read_text(encoding="utf-8")) == [{"id": 1}, {"id": 2}]


def test_json_sink_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "out.json"
    sink = JsonSink(path)
    sink.write({"id": 1})
    sink.close()
    assert json.loads(path.read_text(encoding="utf-8")) == [{"id": 1}]


def test_json_sink_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    sink = JsonSink(path)
    sink.write({"id": 1})
    sink.close()
    sink.close()
    assert json.loads(path.read_text(encoding="utf-8")) == [{"id": 1}]


def test_json_sink_rejects_write_after_close(tmp_path: Path) -> None:
    sink = JsonSink(tmp_path / "out.json")
    sink.close()
    with pytest.raises(SinkError):
        sink.write({"id": 1})
