"""Declarative validation of configuration mappings.

Nesting is expressed with **dotted field names**: a field called ``"db.port"``
addresses ``data["db"]["port"]``. This keeps a schema flat and easy to compare
against the dotted paths produced by :mod:`configkit.env`.

Type checking is deliberately forgiving in exactly one place: a field declared
as ``float`` also accepts an ``int``. Booleans are never accepted for ``int``
or ``float`` fields, and ``int`` is never accepted for a ``bool`` field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from configkit.errors import ValidationError
from configkit.merge import get_path, has_path, set_path


@dataclass(frozen=True)
class Field:
    """A single declared configuration field.

    ``name`` is a dotted path, ``type`` is the expected Python type, ``required``
    makes a missing value an error, and ``default`` is written into the result
    when the field is absent (a ``None`` default means "leave it absent").
    """

    name: str
    type: type
    required: bool = False
    default: Any = None

    @property
    def parts(self) -> tuple[str, ...]:
        """The dotted name split into path components."""
        return tuple(self.name.split("."))


class Schema:
    """An ordered collection of :class:`Field` declarations."""

    def __init__(self, fields: Iterable[Field]) -> None:
        self.fields: tuple[Field, ...] = tuple(fields)
        seen: set[str] = set()
        for field in self.fields:
            if not field.name:
                raise ValueError("field names must be non-empty")
            if field.name in seen:
                raise ValueError(f"duplicate field name: {field.name!r}")
            seen.add(field.name)

    def __repr__(self) -> str:
        names = ", ".join(field.name for field in self.fields)
        return f"Schema({names})"

    def field_names(self) -> tuple[str, ...]:
        """Return every declared dotted field name, in declaration order."""
        return tuple(field.name for field in self.fields)

    def required_names(self) -> tuple[str, ...]:
        """Return the dotted names of every required field."""
        return tuple(field.name for field in self.fields if field.required)

    def validate(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Return a copy of ``data`` with defaults filled in and types checked.

        Raises :class:`~configkit.errors.ValidationError` when a required field
        is missing or when a present value has the wrong type. Keys that are not
        declared in the schema are preserved untouched.
        """
        result = _copy_mapping(data)
        for field in self.fields:
            parts = list(field.parts)
            if not has_path(result, parts):
                if field.required:
                    raise ValidationError("missing required field", field=field.name)
                if field.default is not None:
                    set_path(result, parts, field.default)
                continue
            value = get_path(result, parts)
            if not type_matches(value, field.type):
                raise ValidationError(
                    f"expected {field.type.__name__}, got {type(value).__name__}",
                    field=field.name,
                )
        return result


def type_matches(value: object, expected: type) -> bool:
    """Return ``True`` when ``value`` satisfies the declared ``expected`` type."""
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, expected)


def _copy_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``data`` deeply enough that writing defaults cannot mutate the input."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        result[key] = _copy_mapping(value) if isinstance(value, Mapping) else value
    return result
