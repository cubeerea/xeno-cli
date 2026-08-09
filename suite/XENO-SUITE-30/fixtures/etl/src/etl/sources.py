"""Input adapters: everything that can stream rows into a pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

Row = dict[str, Any]
"""A single record flowing through the pipeline."""


class SourceError(RuntimeError):
    """Raised when a source cannot produce well-formed rows."""


@runtime_checkable
class Source(Protocol):
    """Anything that can be read row by row."""

    def rows(self) -> Iterator[Row]:
        """Yield one :data:`Row` at a time, in source order."""


class ListSource:
    """A source backed by an in-memory iterable of rows.

    Rows are copied on the way in and on the way out so that downstream
    transforms can never mutate the caller's data.
    """

    def __init__(self, rows: Iterable[Row]) -> None:
        self._rows: list[Row] = [dict(row) for row in rows]

    def rows(self) -> Iterator[Row]:
        """Yield a shallow copy of each stored row."""
        for row in self._rows:
            yield dict(row)

    def __len__(self) -> int:
        return len(self._rows)


class JsonLinesSource:
    """Read newline-delimited JSON objects from ``path``.

    Blank lines are ignored. Any line that is not a JSON object raises
    :class:`SourceError` tagged with the offending line number.
    """

    def __init__(self, path: Path | str, *, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def rows(self) -> Iterator[Row]:
        """Yield one row per non-blank line of the file."""
        with self.path.open("r", encoding=self.encoding) as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise SourceError(f"{self.path}:{lineno}: invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise SourceError(
                        f"{self.path}:{lineno}: expected a JSON object, "
                        f"got {type(payload).__name__}"
                    )
                yield {str(key): value for key, value in payload.items()}

    def __repr__(self) -> str:
        return f"JsonLinesSource({str(self.path)!r})"
