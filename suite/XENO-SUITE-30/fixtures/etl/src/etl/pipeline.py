"""Wire a source, an ordered chain of transforms, and a sink together."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from etl.sinks import Sink
from etl.sources import Source
from etl.transforms import Transform, apply_all


@dataclass(frozen=True)
class RunStats:
    """Counts describing a single :meth:`Pipeline.run`.

    ``rows_in`` is everything the source produced, ``rows_out`` everything the
    sink accepted, and ``rows_skipped`` every row a transform dropped by
    returning ``None``.
    """

    rows_in: int = 0
    rows_out: int = 0
    rows_skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain, JSON-friendly mapping."""
        return {key: int(value) for key, value in asdict(self).items()}


class Pipeline:
    """Pull rows from ``source``, push them through ``transforms``, write to ``sink``.

    The sink is closed once the source is exhausted. Exceptions raised by a
    transform propagate to the caller.
    """

    def __init__(self, source: Source, transforms: Sequence[Transform], sink: Sink) -> None:
        self.source = source
        self.transforms: tuple[Transform, ...] = tuple(transforms)
        self.sink = sink

    def run(self) -> RunStats:
        """Drain the source into the sink and report what happened."""
        rows_in = 0
        rows_out = 0
        rows_skipped = 0
        for row in self.source.rows():
            rows_in += 1
            result = apply_all(self.transforms, row)
            if result is None:
                rows_skipped += 1
                continue
            self.sink.write(result)
            rows_out += 1
        self.sink.close()
        return RunStats(rows_in=rows_in, rows_out=rows_out, rows_skipped=rows_skipped)

    def __repr__(self) -> str:
        return (
            f"Pipeline(source={type(self.source).__name__}, "
            f"transforms={len(self.transforms)}, sink={type(self.sink).__name__})"
        )
