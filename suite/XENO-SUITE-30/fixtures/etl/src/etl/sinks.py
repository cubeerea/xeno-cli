"""Output adapters: everything that can consume rows from a pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from etl.sources import Row


class SinkError(RuntimeError):
    """Raised when a sink is used incorrectly, e.g. written to after closing."""


@runtime_checkable
class Sink(Protocol):
    """Anything that accepts rows one at a time and is closed exactly once."""

    def write(self, row: Row) -> None:
        """Accept a single row."""

    def close(self) -> None:
        """Flush any buffered state. Implementations must be idempotent."""


class MemorySink:
    """Collect rows in a list; useful for tests and dry runs."""

    def __init__(self) -> None:
        self.rows: list[Row] = []
        self.closed = False

    def write(self, row: Row) -> None:
        """Append a copy of ``row`` to :attr:`rows`."""
        if self.closed:
            raise SinkError("cannot write to a closed MemorySink")
        self.rows.append(dict(row))

    def close(self) -> None:
        """Mark the sink closed; safe to call repeatedly."""
        self.closed = True

    def __len__(self) -> int:
        return len(self.rows)


class JsonSink:
    """Buffer rows and write them as one JSON array when the sink is closed.

    The parent directory is created on demand. Closing twice is a no-op, so a
    caller may close defensively without truncating the file it just wrote.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        indent: int | None = 2,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.indent = indent
        self.encoding = encoding
        self._buffer: list[Row] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    def write(self, row: Row) -> None:
        """Buffer ``row`` until the sink is closed."""
        if self._closed:
            raise SinkError(f"cannot write to closed JsonSink({str(self.path)!r})")
        self._buffer.append(dict(row))

    def close(self) -> None:
        """Serialise every buffered row to ``path`` as a JSON array."""
        if self._closed:
            return
        self._closed = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding=self.encoding) as handle:
            json.dump(self._buffer, handle, indent=self.indent)

    def __repr__(self) -> str:
        return f"JsonSink({str(self.path)!r})"
