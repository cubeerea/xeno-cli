"""Row-level transformations.

A :data:`Transform` is any callable taking one row and returning either a new
row or ``None``. Returning ``None`` means "drop this row": the pipeline stops
applying further transforms and counts the row as skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from etl.sources import Row

Transform = Callable[[Row], "Row | None"]
"""A callable applied to a single row; ``None`` drops the row."""


class TransformError(ValueError):
    """Raised when a transform cannot process the row it was handed."""


def rename_fields(mapping: Mapping[str, str]) -> Transform:
    """Return a transform that renames fields according to ``old -> new``.

    Fields absent from ``mapping`` keep their original name. Field order is
    preserved.
    """

    def _rename(row: Row) -> Row | None:
        renamed: Row = {}
        for key, value in row.items():
            renamed[mapping.get(key, key)] = value
        return renamed

    return _rename


def cast_fields(mapping: Mapping[str, Callable[[Any], Any]]) -> Transform:
    """Return a transform applying a caster callable to each named field.

    Fields missing from the row are left alone. A caster that raises
    :class:`TypeError` or :class:`ValueError` is re-raised as
    :class:`TransformError`.
    """

    def _cast(row: Row) -> Row | None:
        casted: Row = dict(row)
        for name, caster in mapping.items():
            if name not in casted:
                continue
            try:
                casted[name] = caster(casted[name])
            except (TypeError, ValueError) as exc:
                raise TransformError(f"cannot cast field {name!r}: {casted[name]!r}") from exc
        return casted

    return _cast


def filter_rows(predicate: Callable[[Row], bool]) -> Transform:
    """Return a transform that drops every row for which ``predicate`` is false."""

    def _filter(row: Row) -> Row | None:
        return row if predicate(row) else None

    return _filter


def add_constant(field: str, value: Any) -> Transform:
    """Return a transform that sets ``field`` to ``value`` on every row."""

    def _add(row: Row) -> Row | None:
        annotated: Row = dict(row)
        annotated[field] = value
        return annotated

    return _add


def apply_all(transforms: Sequence[Transform], row: Row) -> Row | None:
    """Apply ``transforms`` in order, short-circuiting on the first drop.

    Returns the resulting row, or ``None`` if any transform dropped it.
    """
    current: Row | None = row
    for transform in transforms:
        if current is None:
            return None
        current = transform(current)
    return current
