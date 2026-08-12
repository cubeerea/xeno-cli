"""The live run view and the graph drawing (`xeno.ui`).

Rendered into a fixed-width `Console` capture rather than a real terminal, so
alignment is asserted on actual output instead of eyeballed. The view is fed
raw run-log records, which is exactly what `RunLog`'s observer hands it.
"""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console

from xeno.ui.graph import full_width, render_graph
from xeno.ui.live import LiveRunView, NullRunView

WIDE = 120


#: `no_color=True` drops colour but still emits bold/dim, so a raw capture
#: measures escape codes rather than what a reader sees. Every assertion here
#: is about visible layout, so the codes come out first.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(renderable: Any) -> str:
    console = Console(width=WIDE, no_color=True, highlight=False)
    with console.capture() as cap:
        console.print(renderable)
    return _ANSI.sub("", cap.get())


# ---- the drawing ----------------------------------------------------------


def test_every_node_appears_in_the_full_drawing() -> None:
    text = _plain(render_graph(width=WIDE))
    for label in (
        "Argus",
        "Odysseus",
        "Lachesis",
        "Daedalus",
        "Talos",
        "ckpt",
        "Cerberus",
        "Chiron",
        "you",
    ):
        assert label in text


def test_both_lachesis_jobs_get_their_own_cell() -> None:
    """Expansion and verification bookend a milestone from opposite ends of
    the task loop, so one shared cell would light up in the wrong place for
    half the run."""
    text = _plain(render_graph(width=WIDE))
    assert text.count("Lachesis") == 2
    assert "expand" in text
    assert "verify" in text


def test_expanding_and_verifying_light_different_cells() -> None:
    expanding = _plain(render_graph(active="lachesis_expand", width=WIDE))
    verifying = _plain(render_graph(active="lachesis_verify", width=WIDE))
    assert expanding.index("[Lachesis]") != verifying.index("[Lachesis]")


def test_the_handoff_connects_the_roadmap_row_to_the_task_loop() -> None:
    """The elbow's two ends are computed from the two rows' own column
    arithmetic, so this asserts they actually meet the cells they claim to."""
    lines = _plain(render_graph(width=WIDE)).splitlines()
    expand_row = next(line for line in lines if "Lachesis" in line and "Odysseus" in line)
    stem_row = next(line for line in lines if line.strip() == "│")
    elbow_row = next(line for line in lines if "┌" in line and "┘" in line)
    head_row = next(line for line in lines if line.strip() == "▼")
    research_row = next(line for line in lines if "research" in line)

    # The stem drops out of the Lachesis/expand cell...
    start = expand_row.index("Lachesis")
    assert start <= stem_row.index("│") < start + len("Lachesis")
    assert stem_row.index("│") == elbow_row.index("┘")
    # ...and the arrowhead lands on the column Argus/research is centred in.
    assert head_row.index("▼") == elbow_row.index("┌")
    assert research_row.index("research") <= head_row.index("▼")


def test_the_active_node_is_the_only_one_bracketed() -> None:
    text = _plain(render_graph(active="daedalus", width=WIDE))
    assert "[Daedalus]" in text
    assert "[Talos]" not in text


def test_becoming_active_does_not_reflow_the_drawing() -> None:
    """Cell widths reserve room for the brackets whether or not they are
    drawn — a picture that shifts sideways as the run advances is harder to
    read than one that stays put."""
    idle = _plain(render_graph(width=WIDE)).splitlines()
    busy = _plain(render_graph(active="daedalus", width=WIDE)).splitlines()
    assert [len(line) for line in idle] == [len(line) for line in busy]


def test_the_ladder_elbows_line_up_with_the_nodes_they_connect() -> None:
    """The `▲` must sit under Argus/research and the `┘` under Talos's fork,
    or the drawing is lying about which nodes the ladder joins."""
    lines = _plain(render_graph(width=WIDE)).splitlines()
    up_row = next(line for line in lines if "▲" in line)
    elbow_row = next(line for line in lines if "└" in line)
    fork_row = next(line for line in lines if "─┬─▶" in line)

    assert up_row.index("▲") == elbow_row.index("└")
    assert fork_row.index("┬") == up_row.index("│") == elbow_row.index("┘")


def test_a_narrow_terminal_falls_back_to_one_line() -> None:
    narrow = _plain(render_graph(active="talos", width=40))
    assert len(narrow.strip().splitlines()) == 1
    assert "Talos" in narrow


def test_the_full_drawing_fits_a_hundred_columns() -> None:
    """Wide enough to be worth drawing, narrow enough for a normal terminal.

    This is why the picture is two rows: nine nodes on one line ran to 122
    columns, past the point where most terminals would ever show it.
    """
    assert full_width() <= 100


# ---- the view -------------------------------------------------------------


def _view() -> LiveRunView:
    return LiveRunView(Console(width=WIDE, no_color=True, highlight=False))


def test_cost_and_tokens_accumulate_across_model_calls() -> None:
    view = _view()
    view.handle({"kind": "model.call", "usd": 0.001, "input_tokens": 100, "output_tokens": 10})
    view.handle({"kind": "model.call", "usd": 0.002, "input_tokens": 200, "output_tokens": 20})
    status = _plain(view._status())
    assert "$0.0030" in status
    assert "330 tok" in status


def test_the_ladder_rung_and_attempt_are_reported() -> None:
    view = _view()
    view.handle({"kind": "ladder.advance", "rung": "L1", "attempt": 2})
    status = _plain(view._status())
    assert "L1" in status
    assert "2/3" in status, "attempts are shown against the rung's budget"


def test_a_checkpoint_clears_the_ladder_and_advances_the_task() -> None:
    """A passing task resets the rung — carrying L3 into the next task would
    misreport a healthy task as one attempt from halting."""
    view = _view()
    view.handle({"kind": "ladder.advance", "rung": "L3", "attempt": 1})
    view.handle({"kind": "checkpoint", "task_index": 0, "task_cursor": 1, "task_count": 3})
    status = _plain(view._status())
    assert "task 2/3" in status
    assert "L3" not in status


def test_a_failing_gate_marks_talos_and_a_pass_clears_it() -> None:
    view = _view()
    view.handle({"kind": "node.exit", "node": "talos", "passed": False, "detail": "lint failed"})
    assert "talos" in view._failed
    view.handle({"kind": "node.exit", "node": "talos", "passed": True, "detail": "ok"})
    assert "talos" not in view._failed


def test_the_active_node_tracks_enter_and_exit() -> None:
    view = _view()
    view.handle({"kind": "node.enter", "node": "chiron"})
    assert view._active == "chiron"
    view.handle({"kind": "node.exit", "node": "chiron", "halted": False, "detail": ""})
    assert view._active is None
    assert "chiron" in view._visited


def test_an_unknown_event_kind_is_ignored() -> None:
    """The run log is explicitly additive (PRD S15.2), so a view that
    crashed on a new event kind would make adding one a breaking change."""
    view = _view()
    view.handle({"kind": "something.new", "whatever": 1})


def test_the_null_view_accepts_everything_and_renders_nothing() -> None:
    view = NullRunView()
    with view.running():
        view.handle({"kind": "node.enter", "node": "talos"})


# ---- gate-failure summaries ----------------------------------------------


def test_a_fenced_triage_excerpt_does_not_become_the_summary() -> None:
    """Talos's triage is a model call and models fence their output, so the
    literal first line is often '```python' — which tells a reader watching
    the live view nothing about why the gates failed."""
    from xeno.core.state import EvalReport
    from xeno.graph.build import _talos_detail

    report = EvalReport(
        passed=False,
        failed_command="lint",
        first_failure="```python\nsrc/x.py:1:89: E501 Line too long\n```",
    )
    assert _talos_detail(report) == "lint failed — src/x.py:1:89: E501 Line too long"


def test_a_gate_failure_with_no_excerpt_still_names_the_command() -> None:
    from xeno.core.state import EvalReport
    from xeno.graph.build import _talos_detail

    assert _talos_detail(EvalReport(passed=False, failed_command="test")) == "test failed"


def test_the_milestone_counter_appears_alongside_the_task_counter() -> None:
    """The plan grows one milestone at a time, so "task 2/3" alone would be
    a total that keeps changing — the milestone count is the stable one."""
    view = _view()
    view.handle(
        {
            "kind": "checkpoint",
            "task_index": 0,
            "task_cursor": 1,
            "task_count": 3,
            "milestone_cursor": 0,
            "milestone_count": 2,
        }
    )
    status = _plain(view._status())
    assert "milestone 1/2" in status
    assert "task 2/3" in status


def test_a_verified_milestone_is_reported_distinctly_from_a_checkpoint() -> None:
    """It is the only green in a run where the test command actually ran."""
    view = _view()
    lines: list[str] = []
    view._say = lines.append  # type: ignore[method-assign]
    view.handle(
        {"kind": "milestone", "milestone_cursor": 1, "milestone_count": 2, "task_cursor": 2}
    )
    text = _plain(lines[0])
    assert "1/2 verified" in text
    assert "tests pass" in text


def test_an_unparseable_response_points_at_the_saved_file() -> None:
    """The node.exit line that follows says only that the run halted. The
    path is the difference between a dead end and something to open."""
    view = _view()
    lines: list[str] = []
    view._say = lines.append  # type: ignore[method-assign]
    view.handle(
        {
            "kind": "response.unparsed",
            "node": "specifier",
            "attempts": 2,
            "path": "/repo/.xeno/runs/r/workspace/unparsed_specifier_1.txt",
        }
    )
    text = _plain(lines[0])
    assert "specifier" in text
    assert "unparsed_specifier_1.txt" in text
