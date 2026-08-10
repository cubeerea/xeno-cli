"""Structured, replayable run logging (PRD S13 Phase 0, S15.2).

One JSONL event per line, appended and flushed immediately. Flushing per event
is deliberate: the runs worth inspecting most closely are the ones that halted
unexpectedly, and a buffered final event is exactly the one that gets lost.

Events are the instrument behind M2.1/M2.2 and the input to the deferred replay
and graph-visualization tooling (PRD S13, "beyond release v1"), so the schema is
flat, additive, and never rewritten in place.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from xeno.core.state import stable_json


class EventKind(StrEnum):
    RUN_START = "run.start"
    RUN_END = "run.end"
    NODE_ENTER = "node.enter"
    NODE_EXIT = "node.exit"
    MODEL_CALL = "model.call"
    MODEL_ERROR = "model.error"
    TIER_ESCALATION = "tier.escalation"
    FALLBACK = "provider.fallback"
    CACHE_PROBE = "cache.probe"
    SECRET_REDACTED = "secret.redacted"
    LADDER_ADVANCE = "ladder.advance"
    BREAKER_FIRED = "breaker.fired"
    CHECKPOINT = "checkpoint"
    VERDICT = "verdict"


class RunLog:
    """Append-only JSONL writer. Thread-safe; one instance per run."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._seq = 0
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def event(self, kind: EventKind | str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = {
                "seq": self._seq,
                "ts": time.time(),
                "run_id": self.run_id,
                "kind": str(kind),
                **payload,
            }
            self._fh.write(stable_json(record) + "\n")
            self._fh.flush()
        return record

    @contextmanager
    def span(self, kind_enter: EventKind, kind_exit: EventKind, **payload: Any) -> Iterator[None]:
        """Emit paired enter/exit events with a measured duration."""
        started = time.perf_counter()
        self.event(kind_enter, **payload)
        try:
            yield
        except Exception as exc:
            self.event(
                kind_exit,
                **payload,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            self.event(
                kind_exit,
                **payload,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                ok=True,
            )

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: Path) -> list[dict[str, Any]]:
    """Load a run log for replay or scorecard aggregation.

    Malformed trailing lines are skipped rather than fatal: a run killed
    mid-write leaves a partial last line, and that run's earlier events are
    still the ones you need to read.
    """
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class NullRunLog(RunLog):
    """No-op sink for tests and dry runs.

    Deliberately does NOT call `super().__init__`, which would open a real
    file handle — but it must still populate every attribute the base class
    declares, or an attribute only the null sink lacks would fail exclusively
    in the dry-run path that exists to be safe.
    """

    def __init__(self) -> None:
        self.path = Path(os.devnull)
        self.run_id = "null"
        self._seq = 0
        self._lock = threading.Lock()

    def event(self, kind: EventKind | str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
        return {"seq": self._seq, "kind": str(kind), **payload}

    def close(self) -> None:
        return
