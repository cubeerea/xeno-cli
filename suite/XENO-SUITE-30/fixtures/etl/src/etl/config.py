"""Build pipelines from plain dictionaries, e.g. parsed JSON or TOML.

A configuration mapping looks like::

    {
        "source": {"type": "jsonl", "path": "in.jsonl"},
        "transforms": [{"type": "cast", "fields": {"age": "int"}}],
        "sink": {"type": "json", "path": "out.json"},
    }
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from etl.pipeline import Pipeline
from etl.sinks import JsonSink, MemorySink, Sink
from etl.sources import JsonLinesSource, ListSource, Row, Source
from etl.transforms import Transform, add_constant, cast_fields, filter_rows, rename_fields


class ConfigError(ValueError):
    """Raised when a configuration mapping cannot be turned into a pipeline."""


CASTERS: dict[str, Callable[[Any], Any]] = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
}
"""Caster names usable in a ``{"type": "cast"}`` transform block."""

_KNOWN_KEYS = frozenset({"sink", "source", "transforms"})


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise :class:`ConfigError`."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _spec_type(spec: Mapping[str, Any], what: str) -> str:
    """Return the ``type`` discriminator of a spec block."""
    if "type" not in spec:
        raise ConfigError(f"{what} spec is missing a 'type' key")
    return str(spec["type"])


def _require_path(spec: Mapping[str, Any], what: str) -> Path:
    """Return the ``path`` entry of a spec block as a :class:`Path`."""
    path = spec.get("path")
    if not isinstance(path, str | Path):
        raise ConfigError(f"{what} spec requires a string 'path'")
    return Path(path)


def _field_equals(field: str, expected: Any) -> Callable[[Row], bool]:
    """Return a predicate matching rows whose ``field`` equals ``expected``."""

    def _predicate(row: Row) -> bool:
        return bool(row.get(field) == expected)

    return _predicate


@dataclass(frozen=True)
class PipelineConfig:
    """The validated, normalised form of a raw configuration mapping."""

    source: Mapping[str, Any]
    transforms: tuple[Mapping[str, Any], ...]
    sink: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PipelineConfig:
        """Validate ``raw`` and return a :class:`PipelineConfig`."""
        config = _require_mapping(raw, "config")
        unknown = sorted(set(config) - _KNOWN_KEYS)
        if unknown:
            raise ConfigError(f"unknown config keys: {unknown}")
        if "source" not in config:
            raise ConfigError("config requires a 'source' block")
        raw_transforms = config.get("transforms", [])
        if not isinstance(raw_transforms, list):
            raise ConfigError("'transforms' must be a list of spec mappings")
        return cls(
            source=_require_mapping(config["source"], "source"),
            transforms=tuple(
                _require_mapping(spec, f"transform #{index}")
                for index, spec in enumerate(raw_transforms)
            ),
            sink=_require_mapping(config.get("sink", {"type": "memory"}), "sink"),
        )


def build_source(spec: Mapping[str, Any]) -> Source:
    """Construct a source from a ``{"type": ...}`` spec."""
    kind = _spec_type(spec, "source")
    if kind == "list":
        rows = spec.get("rows", [])
        if not isinstance(rows, list):
            raise ConfigError("list source 'rows' must be a list")
        return ListSource(rows)
    if kind == "jsonl":
        return JsonLinesSource(_require_path(spec, "source"))
    raise ConfigError(f"unknown source type: {kind!r}")


def build_transform(spec: Mapping[str, Any]) -> Transform:
    """Construct a single transform from a ``{"type": ...}`` spec."""
    kind = _spec_type(spec, "transform")
    if kind == "rename":
        mapping = _require_mapping(spec.get("mapping", {}), "rename 'mapping'")
        return rename_fields({str(old): str(new) for old, new in mapping.items()})
    if kind == "cast":
        fields = _require_mapping(spec.get("fields", {}), "cast 'fields'")
        casters: dict[str, Callable[[Any], Any]] = {}
        for name, caster in fields.items():
            if str(caster) not in CASTERS:
                raise ConfigError(f"unknown caster: {caster!r}")
            casters[str(name)] = CASTERS[str(caster)]
        return cast_fields(casters)
    if kind == "add_constant":
        return add_constant(_require_field(spec, "add_constant"), spec.get("value"))
    if kind == "filter":
        return filter_rows(_field_equals(_require_field(spec, "filter"), spec.get("equals")))
    raise ConfigError(f"unknown transform type: {kind!r}")


def _require_field(spec: Mapping[str, Any], what: str) -> str:
    """Return the ``field`` entry of a transform spec."""
    field = spec.get("field")
    if not isinstance(field, str):
        raise ConfigError(f"{what} transform requires a string 'field'")
    return field


def build_sink(spec: Mapping[str, Any]) -> Sink:
    """Construct a sink from a ``{"type": ...}`` spec."""
    kind = _spec_type(spec, "sink")
    if kind == "memory":
        return MemorySink()
    if kind == "json":
        indent = spec.get("indent", 2)
        if indent is not None and not isinstance(indent, int):
            raise ConfigError("json sink 'indent' must be an integer or null")
        return JsonSink(_require_path(spec, "sink"), indent=indent)
    raise ConfigError(f"unknown sink type: {kind!r}")


def build_pipeline(config: Mapping[str, Any]) -> Pipeline:
    """Construct a ready-to-run :class:`~etl.pipeline.Pipeline` from ``config``."""
    parsed = PipelineConfig.from_mapping(config)
    return Pipeline(
        source=build_source(parsed.source),
        transforms=[build_transform(spec) for spec in parsed.transforms],
        sink=build_sink(parsed.sink),
    )
