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
from collections.abc import Callable, Iterator
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
    #: A response survived the corrective retry still unparseable, and its raw
    #: text was written to the run workspace (`xeno.graph.nodeops`). Not a
    #: MODEL_ERROR: the call succeeded and the tokens were billed, so filing a
    #: prompt defect in with the transport failures would have anything
    #: reading this log counting the wrong failure. It is also the only event
    #: carrying a `path`, which is what a post-mortem has to point at.
    RESPONSE_UNPARSED = "response.unparsed"
    #: The call succeeded but the answer did not arrive intact — cut off at
    #: the token limit, or left entirely in a reasoning model's hidden
    #: channel. Distinct from RESPONSE_UNPARSED, which is about a complete
    #: answer in the wrong shape: this one says nothing about whether the
    #: model can follow the format.
    MODEL_INCOMPLETE = "model.incomplete"
    TIER_ESCALATION = "tier.escalation"
    FALLBACK = "provider.fallback"
    CACHE_PROBE = "cache.probe"
    SECRET_REDACTED = "secret.redacted"
    LADDER_ADVANCE = "ladder.advance"
    BREAKER_FIRED = "breaker.fired"
    CHECKPOINT = "checkpoint"
    #: A milestone's own tests were written and passed (`xeno.graph.lachesis`).
    #: Distinct from CHECKPOINT because it is the only green in a run that
    #: includes the test command actually running — a scorecard that counted
    #: it as just another checkpoint could not tell the two apart.
    MILESTONE = "milestone"
    VERDICT = "verdict"
    #: The worktree's manifests changed mid-run and the toolchain was
    #: re-derived (`xeno.graph.toolchain`). Worth its own kind rather than a
    #: node event: it is the one thing that can change what "the gates" even
    #: means partway through a run, so a scorecard reading this log must be
    #: able to find it without inferring it from a node name.
    TOOLCHAIN_REFRESH = "toolchain.refresh"


class RunLog:
    """Append-only JSONL writer. Thread-safe; one instance per run.

    An optional `observer` is notified of every record after it is durably
    written. That ordering is the point: the log on disk is the record of
    record, and a live view is a reader of it, never a second source of
    truth that could disagree. It also keeps the terminal out of
    `xeno.graph` entirely — nodes emit events, and whether anything is
    drawing them is not their concern.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self._seq = 0
        self._lock = threading.Lock()
        self._observer = observer
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
        # Outside the lock: an observer that draws to a terminal must never
        # be able to stall the writer, and a view that raises must not take
        # the run down with it — losing the picture is not losing the run.
        if self._observer is not None:
            try:
                self._observer(record)
            except Exception:
                self._observer = None
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
        self._observer = None

    def event(self, kind: EventKind | str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
        return {"seq": self._seq, "kind": str(kind), **payload}

    def close(self) -> None:
        return
