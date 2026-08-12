"""The run graph, drawn.

One picture of the whole state machine with the current node lit up. Two
rows, because the graph has two nested loops and squeezing them onto one line
stopped being legible at nine nodes: the top row runs once per RUN (orient,
map the roadmap, expand the next milestone) and the bottom row runs once per
TASK, closing with the milestone's own verification.

The layout is computed rather than written out as a string literal: the
ladder's elbows have to line up with the columns of the two nodes they
connect (`argus_research` and `talos`), the connector between the rows has to
meet a specific cell, and hand-aligned ASCII stops being true the first time
a label changes.

Cell widths reserve room for the active marker's brackets whether or not
they are drawn, so a node becoming active never reflows the diagram — a
picture that shifts sideways every time the run advances is harder to read
than one that stays put.
"""

from __future__ import annotations

from enum import StrEnum

from rich.text import Text

#: (event node name, label, sublabel). The node names match what
#: `xeno.graph.build` emits as `node.enter`/`node.exit`, so the view never
#: has to translate between what the graph calls a node and what the picture
#: calls it.
#:
#: Row 1: once per run, plus a return here at the start of every milestone.
_ROADMAP_CELLS: tuple[tuple[str, str, str], ...] = (
    ("argus_skeleton", "Argus", "skeleton"),
    ("odysseus", "Odysseus", "roadmap"),
    ("lachesis_expand", "Lachesis", "expand"),
)

#: Row 2: once per task, then the milestone's verification and the review.
#: "ckpt" and "you" are not graph nodes — they are the deterministic commit
#: between tasks and the human gate that follows the run.
_LOOP_CELLS: tuple[tuple[str, str, str], ...] = (
    ("argus_research", "Argus", "research"),
    ("daedalus", "Daedalus", "write"),
    ("talos", "Talos", "gates"),
    ("checkpoint", "ckpt", ""),
    ("lachesis_verify", "Lachesis", "verify"),
    ("cerberus", "Cerberus", "review"),
    ("you", "you", ""),
)

_PLAIN_LINK = " ─▶ "
#: Talos is where the graph forks: pass continues right, fail drops into the
#: ladder. The `┬` is the fork, and its column anchors the loop below.
_FORK_LINK = " ─┬─▶ "
_FORK_AFTER = "talos"

_LADDER_NODE = "chiron"
_LADDER_LABEL = "Chiron"
_LADDER_SUB = "patch"

#: Where the roadmap row hands off to the task loop.
_HANDOFF_FROM = "lachesis_expand"
_HANDOFF_TO = "argus_research"
_HANDOFF_CAPTION = "once per milestone"


class Phase(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"


_STYLES: dict[Phase, str] = {
    Phase.PENDING: "grey42",
    Phase.ACTIVE: "bold cyan",
    Phase.DONE: "green",
    Phase.FAILED: "bold red",
}


def _phase(key: str, active: str | None, visited: set[str], failed: set[str]) -> Phase:
    if key == active:
        return Phase.ACTIVE
    if key in failed:
        return Phase.FAILED
    if key in visited:
        return Phase.DONE
    return Phase.PENDING


def render_graph(
    *,
    active: str | None = None,
    visited: set[str] | None = None,
    failed: set[str] | None = None,
    ladder_note: str = "",
    width: int = 100,
) -> Text:
    """Draw the graph with `active` lit up.

    `visited` marks nodes the run has already been through and `failed`
    marks ones that reported a failure, so the picture carries a little
    history rather than only a cursor. Falls back to a single compact line
    when the terminal is too narrow for the full drawing.
    """
    visited = visited or set()
    failed = failed or set()

    if width < full_width():
        return _render_compact(active, visited, failed, ladder_note)
    return _render_full(active, visited, failed, ladder_note)


# ---- layout ---------------------------------------------------------------


def _cell_width(label: str, sub: str) -> int:
    # +2 so the active marker's brackets fit without reflowing the row.
    return max(len(label) + 2, len(sub))


def _link_after(key: str) -> str:
    return _FORK_LINK if key == _FORK_AFTER else _PLAIN_LINK


def _columns(cells: tuple[tuple[str, str, str], ...]) -> tuple[dict[str, tuple[int, int]], int]:
    """Start/width per cell in this row, plus the fork column when the row
    contains the fork — in one pass, so the two can never disagree about
    where anything sits."""
    spans: dict[str, tuple[int, int]] = {}
    col = 0
    fork_col = 0
    for index, (key, label, sub) in enumerate(cells):
        cell_w = _cell_width(label, sub)
        spans[key] = (col, cell_w)
        col += cell_w
        if index < len(cells) - 1:
            link = _link_after(key)
            if key == _FORK_AFTER:
                fork_col = col + link.index("┬")
            col += len(link)
    return spans, fork_col


def _row_width(cells: tuple[tuple[str, str, str], ...]) -> int:
    spans, _ = _columns(cells)
    start, cell_w = spans[cells[-1][0]]
    return start + cell_w


def full_width() -> int:
    """The widest row. Exposed for the test that asserts a node becoming
    active does not reflow the drawing."""
    return max(_row_width(_ROADMAP_CELLS), _row_width(_LOOP_CELLS))


def _center(text: str, span: tuple[int, int]) -> int:
    start, cell_w = span
    return start + max(0, (cell_w - len(text)) // 2)


# ---- renderers ------------------------------------------------------------


def _node_row(
    cells: tuple[tuple[str, str, str], ...],
    active: str | None,
    visited: set[str],
    failed: set[str],
) -> Text:
    out = Text()
    for index, (key, label, sub) in enumerate(cells):
        phase = _phase(key, active, visited, failed)
        marked = f"[{label}]" if phase is Phase.ACTIVE else f" {label} "
        pad = _cell_width(label, sub) - len(marked)
        out.append(marked, style=_STYLES[phase])
        if pad > 0:
            out.append(" " * pad)
        if index < len(cells) - 1:
            out.append(_link_after(key), style="grey42")
    return out


def _sub_row(cells: tuple[tuple[str, str, str], ...]) -> str:
    spans, _ = _columns(cells)
    line = [" "] * _row_width(cells)
    for key, _label, sub in cells:
        if not sub:
            continue
        at = _center(sub, spans[key])
        line[at : at + len(sub)] = list(sub)
    return "".join(line).rstrip()


def _render_full(
    active: str | None, visited: set[str], failed: set[str], ladder_note: str
) -> Text:
    out = Text()

    out.append_text(_node_row(_ROADMAP_CELLS, active, visited, failed))
    out.append("\n")
    out.append(_sub_row(_ROADMAP_CELLS), style="grey42")
    out.append("\n")

    out.append_text(_handoff(active))
    out.append("\n")

    out.append_text(_node_row(_LOOP_CELLS, active, visited, failed))
    out.append("\n")
    out.append(_sub_row(_LOOP_CELLS), style="grey42")
    out.append("\n")

    out.append_text(_ladder(active, ladder_note))
    return out


def _handoff(active: str | None) -> Text:
    """The elbow carrying the expanded milestone down into the task loop.

    Drawn from the roadmap row's own column arithmetic to the loop row's, so
    the two ends actually meet whatever either row's labels are.
    """
    roadmap_spans, _ = _columns(_ROADMAP_CELLS)
    loop_spans, _ = _columns(_LOOP_CELLS)
    from_col = _center("│", roadmap_spans[_HANDOFF_FROM])
    to_col = _center("▼", loop_spans[_HANDOFF_TO])

    style = "grey42"
    stem = [" "] * (from_col + 1)
    stem[from_col] = "│"

    elbow = [" "] * (from_col + 1)
    elbow[to_col] = "┌"
    for col in range(to_col + 1, from_col):
        elbow[col] = "─"
    elbow[from_col] = "┘"
    # The run returns to the roadmap row once per milestone, which the arrows
    # alone cannot say. Written into the elbow's dashes rather than added as
    # another row: it is a caption for this edge, not a step of its own.
    caption = f" {_HANDOFF_CAPTION} "
    at = to_col + 1 + max(0, (from_col - to_col - 1 - len(caption)) // 2)
    if at + len(caption) < from_col:
        elbow[at : at + len(caption)] = list(caption)

    head = [" "] * (to_col + 1)
    head[to_col] = "▼"

    out = Text()
    out.append("".join(stem), style=style)
    out.append("\n")
    out.append("".join(elbow), style=style)
    out.append("\n")
    out.append("".join(head), style=style)
    return out


def _ladder(active: str | None, ladder_note: str) -> Text:
    """The three rows of the escalation loop, anchored to the two columns of
    the task-loop row it connects."""
    spans, fork_col = _columns(_LOOP_CELLS)
    research_col = _center("▲", spans[_HANDOFF_TO])
    is_active = active == _LADDER_NODE
    style = "bold yellow" if is_active else "grey42"

    row1 = [" "] * (fork_col + 1)
    row1[research_col] = "▲"
    row1[fork_col] = "│"

    row3 = [" "] * _row_width(_LOOP_CELLS)
    at = research_col + 7
    row3[at : at + len(_LADDER_SUB)] = list(_LADDER_SUB)
    note = ladder_note or "bounded ladder: L0…L5"
    note_at = at + len(_LADDER_SUB) + 3
    row3[note_at : note_at + len(note)] = list(f"({note})")

    out = Text()
    out.append("".join(row1), style=style)
    out.append("\n")
    out.append(_ladder_row(research_col, fork_col, is_active), style=style)
    out.append("\n")
    out.append("".join(row3).rstrip(), style=style)
    return out


def _ladder_row(research_col: int, fork_col: int, active: bool) -> str:
    """`└──── Chiron ◀────┘`, stretched to meet both anchors."""
    inner = fork_col - research_col - 1
    label = f"[{_LADDER_LABEL}]" if active else f" {_LADDER_LABEL} "
    body = f"{label}◀"
    left = 6
    right = inner - left - len(body)
    if right < 0:  # very tight layouts: give the dashes back to the left side
        left = max(0, left + right)
        right = 0
    return " " * research_col + "└" + "─" * left + body + "─" * right + "┘"


def _render_compact(
    active: str | None, visited: set[str], failed: set[str], ladder_note: str
) -> Text:
    """One line, for terminals too narrow for the drawing."""
    out = Text()
    for index, (key, label, _sub) in enumerate((*_ROADMAP_CELLS, *_LOOP_CELLS)):
        if index:
            out.append("▸", style="grey42")
        out.append(label, style=_STYLES[_phase(key, active, visited, failed)])
    if active == _LADDER_NODE:
        out.append("  ↺ ", style="grey42")
        out.append(_LADDER_LABEL, style="bold yellow")
    if ladder_note:
        out.append(f"  ({ladder_note})", style="grey42")
    return out
