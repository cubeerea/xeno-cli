"""The live run view: a pinned graph with the run's story scrolling above it.

Everything here is driven by `RunLog` events rather than by the nodes calling
a UI. That direction matters: the run log is already the authoritative record
of what happened (PRD S13/S15.2), so a view built on it cannot drift from
what the run actually did, and nothing in `xeno.graph` has to know a terminal
exists. It also means `--quiet`, the JSONL log, and a future replay are all
the same feature seen from different ends.

Log lines are printed through `Live.console`, which scrolls them above the
pinned region instead of into a fixed-height box — so the graph and the
counters stay put at the bottom while the history above stays scrollable in
the terminal's own buffer.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

from xeno.core.runlog import EventKind
from xeno.core.types import RUNG_BUDGETS
from xeno.ui.graph import render_graph

#: Nodes whose entry is worth a line of its own. `rollback` is deterministic
#: and `checkpoint` is reported by its own event, so neither announces itself.
_NODE_LABELS: dict[str, str] = {
    "argus_skeleton": "argus",
    "odysseus": "odysseus",
    "lachesis_expand": "lachesis",
    "argus_research": "argus",
    "daedalus": "daedalus",
    "chiron": "chiron",
    "lachesis_verify": "lachesis",
    "talos": "talos",
    "cerberus": "cerberus",
    "rollback": "rollback",
}


class RunView(Protocol):
    """What `xeno.cli` needs from a view, so quiet mode is a different object
    rather than an `if` at every call site."""

    def handle(self, record: dict[str, Any]) -> None: ...

    def running(self) -> AbstractContextManager[object]: ...


class NullRunView:
    """Quiet mode. Also what a non-TTY gets, so piping `xeno run` into a file
    produces a clean transcript instead of escape codes."""

    def handle(self, record: dict[str, Any]) -> None:
        return

    @contextmanager
    def running(self) -> Iterator[NullRunView]:
        yield self


class LiveRunView:
    """Consumes run-log records; renders the graph, the counters, and a
    scrolling narration of what each node did."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: Live | None = None
        self._started = time.monotonic()

        self._active: str | None = None
        self._visited: set[str] = set()
        self._failed: set[str] = set()
        self._rung = 0
        self._attempt = 0
        self._usd = 0.0
        self._tokens = 0
        self._task_cursor = 0
        self._task_count = 0
        self._milestone_cursor = 0
        self._milestone_count = 0

    # ---- lifecycle --------------------------------------------------------

    @contextmanager
    def running(self) -> Iterator[LiveRunView]:
        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _say(self, line: Text) -> None:
        """Print above the pinned region."""
        if self._live is not None:
            self._live.console.print(line)
        else:
            self._console.print(line)

    # ---- rendering --------------------------------------------------------

    def _render(self) -> RenderableType:
        return Group(
            render_graph(
                active=self._active,
                visited=self._visited,
                failed=self._failed,
                ladder_note=self._ladder_note(),
                width=self._console.width,
            ),
            Text(""),
            self._status(),
        )

    def _ladder_note(self) -> str:
        if self._rung == 0 and self._attempt == 0:
            return ""
        budget = RUNG_BUDGETS.get(self._rung)
        if budget is None:
            return f"L{self._rung}"
        return f"L{self._rung} · attempt {self._attempt}/{budget}"

    def _status(self) -> Text:
        out = Text()
        if self._milestone_count:
            shown = min(self._milestone_cursor + 1, self._milestone_count)
            out.append(f"milestone {shown}/{self._milestone_count}", style="cyan")
            out.append("  ·  ", style="grey42")
        if self._task_count:
            # The plan grows one milestone at a time, so this total is what is
            # known right now rather than what the run will end up executing.
            out.append(f"task {min(self._task_cursor + 1, self._task_count)}/{self._task_count}")
            out.append("  ·  ", style="grey42")

        rung = self._ladder_note() or "L0"
        out.append(rung, style="yellow" if self._rung else "grey42")
        out.append("  ·  ", style="grey42")

        out.append(f"${self._usd:.4f}", style="magenta")
        out.append(f" · {self._tokens:,} tok", style="grey42")
        out.append("  ·  ", style="grey42")
        out.append(_elapsed(time.monotonic() - self._started), style="grey42")
        return out

    # ---- event handling ---------------------------------------------------

    def handle(self, record: dict[str, Any]) -> None:
        kind = record.get("kind", "")
        handler = _HANDLERS.get(kind)
        if handler is not None:
            handler(self, record)
        self._refresh()


# ---- per-event handlers ---------------------------------------------------
#
# Split out as plain functions keyed by event kind rather than a long
# if/elif, so adding an event kind to the run log means adding one entry
# here and nothing else.


def _on_node_enter(view: LiveRunView, record: dict[str, Any]) -> None:
    node = str(record.get("node", ""))
    view._active = node
    view._visited.add(node)


def _on_node_exit(view: LiveRunView, record: dict[str, Any]) -> None:
    node = str(record.get("node", ""))
    view._active = None
    label = _NODE_LABELS.get(node, node)

    if node == "talos":
        passed = record.get("passed")
        if passed:
            view._failed.discard("talos")
            view._say(_line("✓", label, str(record.get("detail") or "all gates passed"), "green"))
        else:
            view._failed.add("talos")
            detail = record.get("detail") or f"{record.get('failed_command', 'gates')} failed"
            view._say(_line("✗", label, str(detail), "red"))
        return

    if record.get("declined"):
        reason = str(record.get("decline_reason") or "no patch proposed")
        view._say(_line("~", label, f"declined: {reason}", "yellow"))
        return

    if record.get("halted"):
        view._say(_line("✗", label, "halted", "red"))
        return

    detail = record.get("detail")
    if detail:
        view._say(_line("✓", label, str(detail), "green"))


def _on_model_call(view: LiveRunView, record: dict[str, Any]) -> None:
    view._usd += float(record.get("usd") or 0.0)
    view._tokens += int(record.get("input_tokens") or 0) + int(record.get("output_tokens") or 0)


def _on_ladder(view: LiveRunView, record: dict[str, Any]) -> None:
    rung = str(record.get("rung", "L0"))
    view._rung = int(rung.removeprefix("L") or 0)
    view._attempt = int(record.get("attempt") or 1)
    reason = record.get("reason")
    suffix = f" ({reason})" if reason else ""
    view._say(_line("↻", "ladder", f"escalated to {rung}{suffix}", "yellow"))


def _absorb_cursors(view: LiveRunView, record: dict[str, Any]) -> None:
    view._visited.add("checkpoint")
    view._failed.discard("talos")
    view._rung = 0
    view._attempt = 0
    view._task_cursor = int(record.get("task_cursor") or 0)
    view._task_count = int(record.get("task_count") or 0)
    view._milestone_cursor = int(record.get("milestone_cursor") or 0)
    view._milestone_count = int(record.get("milestone_count") or 0)


def _on_checkpoint(view: LiveRunView, record: dict[str, Any]) -> None:
    _absorb_cursors(view, record)
    index = int(record.get("task_index") or 0)
    view._say(_line("✓", "checkpoint", f"task {index + 1} committed", "green"))


def _on_milestone(view: LiveRunView, record: dict[str, Any]) -> None:
    """A milestone's own tests were written and passed — the only green in a
    run that includes the test command actually running, so it says so."""
    _absorb_cursors(view, record)
    done = view._milestone_cursor
    total = view._milestone_count
    view._say(_line("★", "milestone", f"{done}/{total} verified — tests pass", "bold green"))


def _on_breaker(view: LiveRunView, record: dict[str, Any]) -> None:
    view._say(
        _line("!", "breaker", f"{record.get('code')} — {record.get('detail')}", "bold red")
    )


def _on_verdict(view: LiveRunView, record: dict[str, Any]) -> None:
    verdict = str(record.get("verdict") or "none")
    colour = {"approve": "bold green", "escalate": "bold yellow"}.get(verdict, "bold red")
    view._say(_line("»", "cerberus", f"verdict: {verdict}", colour))


def _on_toolchain(view: LiveRunView, record: dict[str, Any]) -> None:
    if not record.get("ok"):
        view._say(_line("~", "toolchain", str(record.get("detail") or "refresh failed"), "yellow"))
        return
    commands = record.get("commands") or []
    established = "established" if record.get("newly_established") else "updated"
    view._say(
        _line("✓", "toolchain", f"{established}: {', '.join(str(c) for c in commands)}", "cyan")
    )


def _on_unparsed(view: LiveRunView, record: dict[str, Any]) -> None:
    """A response nobody could parse, named by the file it was saved to.

    Worth its own line even though a halt usually follows: that line says only
    that the response was unusable, and the path is the whole difference
    between a dead end and something to open. It is also the ONLY signal for
    the nodes that tolerate a malformed response and carry on without halting
    at all (Argus, Chiron).
    """
    view._say(
        _line(
            "~",
            str(record.get("node") or "model"),
            f"unparseable response → {record.get('path')}",
            "yellow",
        )
    )


def _on_run_end(view: LiveRunView, record: dict[str, Any]) -> None:
    view._active = None
    if record.get("ok"):
        view._visited.add("you")


_HANDLERS: dict[str, Any] = {
    EventKind.NODE_ENTER.value: _on_node_enter,
    EventKind.NODE_EXIT.value: _on_node_exit,
    EventKind.MODEL_CALL.value: _on_model_call,
    EventKind.LADDER_ADVANCE.value: _on_ladder,
    EventKind.CHECKPOINT.value: _on_checkpoint,
    EventKind.MILESTONE.value: _on_milestone,
    EventKind.BREAKER_FIRED.value: _on_breaker,
    EventKind.VERDICT.value: _on_verdict,
    EventKind.TOOLCHAIN_REFRESH.value: _on_toolchain,
    EventKind.RESPONSE_UNPARSED.value: _on_unparsed,
    EventKind.RUN_END.value: _on_run_end,
}


def _line(marker: str, label: str, detail: str, style: str) -> Text:
    out = Text()
    out.append(f" {marker} ", style=style)
    out.append(f"{label:<11}", style="bold")
    out.append(detail)
    return out


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"
