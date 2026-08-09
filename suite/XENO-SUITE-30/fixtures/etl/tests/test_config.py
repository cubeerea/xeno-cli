from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from etl.config import ConfigError, PipelineConfig, build_pipeline, build_sink, build_source
from etl.sinks import JsonSink, MemorySink
from etl.sources import JsonLinesSource, ListSource


def _memory_config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "source": {"type": "list", "rows": [{"n": "1"}, {"n": "2"}]},
        "sink": {"type": "memory"},
    }
    config.update(extra)
    return config


def test_pipeline_config_defaults_to_memory_sink() -> None:
    parsed = PipelineConfig.from_mapping({"source": {"type": "list", "rows": []}})
    assert parsed.sink == {"type": "memory"}
    assert parsed.transforms == ()


def test_pipeline_config_requires_a_source() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.from_mapping({"sink": {"type": "memory"}})


def test_pipeline_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.from_mapping({"source": {"type": "list"}, "bogus": 1})


def test_pipeline_config_rejects_non_list_transforms() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.from_mapping({"source": {"type": "list"}, "transforms": {}})


def test_build_source_dispatches_on_type(tmp_path: Path) -> None:
    assert isinstance(build_source({"type": "list", "rows": []}), ListSource)
    assert isinstance(
        build_source({"type": "jsonl", "path": str(tmp_path / "a.jsonl")}), JsonLinesSource
    )


def test_build_sink_dispatches_on_type(tmp_path: Path) -> None:
    assert isinstance(build_sink({"type": "memory"}), MemorySink)
    assert isinstance(build_sink({"type": "json", "path": str(tmp_path / "a.json")}), JsonSink)


def test_unknown_source_type_raises() -> None:
    with pytest.raises(ConfigError, match="unknown source type"):
        build_source({"type": "warehouse"})


def test_unknown_sink_type_raises() -> None:
    with pytest.raises(ConfigError, match="unknown sink type"):
        build_sink({"type": "carrier-pigeon"})


def test_missing_type_key_raises() -> None:
    with pytest.raises(ConfigError, match="missing a 'type' key"):
        build_source({"path": "x.jsonl"})


def test_unknown_transform_type_raises() -> None:
    with pytest.raises(ConfigError, match="unknown transform type"):
        build_pipeline(_memory_config(transforms=[{"type": "levitate"}]))


def test_unknown_caster_raises() -> None:
    with pytest.raises(ConfigError, match="unknown caster"):
        build_pipeline(_memory_config(transforms=[{"type": "cast", "fields": {"n": "complex"}}]))


def test_build_pipeline_runs_end_to_end() -> None:
    pipeline = build_pipeline(
        _memory_config(
            transforms=[
                {"type": "cast", "fields": {"n": "int"}},
                {"type": "rename", "mapping": {"n": "count"}},
                {"type": "add_constant", "field": "src", "value": "unit-test"},
            ]
        )
    )
    stats = pipeline.run()
    sink = pipeline.sink
    assert isinstance(sink, MemorySink)
    assert sink.rows == [
        {"count": 1, "src": "unit-test"},
        {"count": 2, "src": "unit-test"},
    ]
    assert stats.rows_out == 2


def test_build_pipeline_filter_block_skips_rows() -> None:
    pipeline = build_pipeline(
        {
            "source": {"type": "list", "rows": [{"s": "ok"}, {"s": "no"}, {"s": "ok"}]},
            "transforms": [{"type": "filter", "field": "s", "equals": "ok"}],
        }
    )
    stats = pipeline.run()
    assert stats.rows_out == 2
    assert stats.rows_skipped == 1


def test_build_pipeline_wires_jsonl_source_to_json_sink(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    out = tmp_path / "out.json"
    stats = build_pipeline(
        {
            "source": {"type": "jsonl", "path": str(src)},
            "sink": {"type": "json", "path": str(out)},
        }
    ).run()
    assert stats.rows_in == 2
    assert json.loads(out.read_text(encoding="utf-8")) == [{"id": 1}, {"id": 2}]
