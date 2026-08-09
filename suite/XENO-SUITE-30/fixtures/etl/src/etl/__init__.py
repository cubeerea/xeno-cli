"""A small, dependency-free, row-oriented ETL pipeline.

Rows are plain dictionaries. A pipeline pulls them from a :class:`Source`,
threads them through a chain of :data:`Transform` callables, and hands the
survivors to a :class:`Sink`.
"""

from etl.config import CASTERS, ConfigError, PipelineConfig, build_pipeline
from etl.pipeline import Pipeline, RunStats
from etl.sinks import JsonSink, MemorySink, Sink, SinkError
from etl.sources import JsonLinesSource, ListSource, Row, Source, SourceError
from etl.transforms import (
    Transform,
    TransformError,
    add_constant,
    apply_all,
    cast_fields,
    filter_rows,
    rename_fields,
)

__all__ = [
    "CASTERS",
    "ConfigError",
    "JsonLinesSource",
    "JsonSink",
    "ListSource",
    "MemorySink",
    "Pipeline",
    "PipelineConfig",
    "Row",
    "RunStats",
    "Sink",
    "SinkError",
    "Source",
    "SourceError",
    "Transform",
    "TransformError",
    "add_constant",
    "apply_all",
    "build_pipeline",
    "cast_fields",
    "filter_rows",
    "rename_fields",
]

__version__ = "0.1.0"
