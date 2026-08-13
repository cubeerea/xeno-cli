"""The CLI's human-gate UI (PRD S8.1, S13 Phase 4) — the one point in the
whole system that talks to a human. Tested directly against `_human_gate`/
`_print_escalate_report` with a hand-built `AgentState`, not through a full
`xeno run` (which needs Docker/router mocking already exercised in
`test_graph.py`) — narrowly about the display/decision layer, per this
project's "no premature abstraction" style.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from xeno.cli import _STARTER_CONFIG, _human_gate, _print_escalate_report, app
from xeno.core.config import XenoConfig, load_config
from xeno.core.state import MAX_FIELD_BYTES, AgentState, Handle, stable_json


def _handle(tmp_path: Path, name: str, content: str) -> Handle:
    path = tmp_path / name
    path.write_text(content)
    return Handle.for_file(path, summary=name)


@pytest.fixture
def approved_state(tmp_path: Path) -> AgentState:
    state = AgentState(run_id="t", goal="add a widget")
    state.review_diff_handle = _handle(tmp_path, "diff.patch", "diff --git a/x b/x\n+1\n")
    state.cerberus_notes = _handle(tmp_path, "notes.txt", "looks good")
    state.commit_message = "feat: add widget"
    return state


def test_human_gate_approve(monkeypatch: pytest.MonkeyPatch, approved_state: AgentState) -> None:
    import xeno.cli as cli

    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "approve")
    _human_gate(approved_state, "xeno/x-abc")
    assert approved_state.human_approved is True


def test_human_gate_reject(monkeypatch: pytest.MonkeyPatch, approved_state: AgentState) -> None:
    import xeno.cli as cli

    answers = iter(["reject", ""])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(answers))
    _human_gate(approved_state, "xeno/x-abc")
    assert approved_state.human_approved is False


def test_human_gate_inspect_then_approve(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    answers = iter(["inspect", "approve"])
    monkeypatch.setattr("xeno.cli.Prompt.ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(Console, "pager", lambda self: contextlib.nullcontext())
    _human_gate(approved_state, "xeno/x-abc")
    assert approved_state.human_approved is True


def test_print_escalate_report_labels_unreviewed_when_cerberus_never_ran(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = AgentState(run_id="t", goal="add a widget")
    state.halt_reason = "cerberus: tier 'flagship' chain exhausted after 1 attempt(s)"
    _print_escalate_report(state, "xeno/x-abc")
    out = capsys.readouterr().out
    assert "UNREVIEWED" in out
    assert "has NOT been reviewed" in out


def test_print_escalate_report_shows_cerberus_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = AgentState(run_id="t", goal="add a widget")
    state.halt_reason = "cerberus: escalated for human judgment"
    state.cerberus_notes = _handle(tmp_path, "report.txt", "the blocking question is X")
    _print_escalate_report(state, "xeno/x-abc")
    out = capsys.readouterr().out
    assert "the blocking question is X" in out
    assert "UNREVIEWED" not in out


def test_starter_config_declares_git_block_and_loads() -> None:
    assert "git:" in _STARTER_CONFIG


def test_starter_config_round_trips_through_init(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    config = load_config(tmp_path / "xeno.yaml")
    assert isinstance(config, XenoConfig)
    assert config.git.branch_prefix == "xeno/"
    assert config.git.open_pr is False


# ---- the gate's free-text channel (B4) -------------------------------------
#
# Until this landed, the sole human gate recorded a single bit. Cerberus can
# say why it rejected something and route back; a person could only say no,
# and the one thing the harness could never learn was the one thing only a
# human knew. `.xeno/memory.md` is what eventually reads this; the run that
# collected it has already approved and has nothing left to route into.


def _gate(monkeypatch: pytest.MonkeyPatch, state: AgentState, *answers: str) -> None:
    replies = iter(answers)
    monkeypatch.setattr("xeno.cli.Prompt.ask", lambda *a, **k: next(replies))
    _human_gate(state, "xeno/x-abc")


def test_a_rejection_can_carry_a_reason(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    _gate(monkeypatch, approved_state, "reject", "the retry loop should be bounded")
    assert approved_state.human_approved is False
    assert approved_state.human_objection == "the retry loop should be bounded"


def test_an_approval_is_never_asked_for_a_reason(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    """One prompt in the tuple: a second `Prompt.ask` raises StopIteration.
    Approving is the common path and must stay one keypress."""
    _gate(monkeypatch, approved_state, "approve")
    assert approved_state.human_objection is None


def test_the_reason_is_optional(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    """A gate that will not let you leave without writing an essay gets an
    essay that says "no", and a field full of "no" is worse than an empty one
    because it looks like data."""
    _gate(monkeypatch, approved_state, "reject", "   ")
    assert approved_state.human_approved is False
    assert approved_state.human_objection is None


def test_declining_to_explain_does_not_undo_the_rejection(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    """Ctrl-C at the reason prompt. The verdict is already recorded; losing it
    here would ship a change the human just refused."""

    def ask(*args: object, **kwargs: object) -> str:
        if kwargs.get("default") == "":
            raise KeyboardInterrupt
        return "reject"

    monkeypatch.setattr("xeno.cli.Prompt.ask", ask)
    _human_gate(approved_state, "xeno/x-abc")
    assert approved_state.human_approved is False


def test_an_overlong_reason_is_truncated_rather_than_lost(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    """AgentState validates field size at construction, so assigning this raw
    would raise AFTER the human finished typing — and the typing is the whole
    reason the prompt exists."""
    essay = "the retry loop should be bounded and configurable " * 200
    _gate(monkeypatch, approved_state, "reject", essay)

    assert approved_state.human_objection is not None
    assert approved_state.human_objection.startswith("the retry loop should be bounded")
    assert "[truncated]" in approved_state.human_objection


def test_a_truncated_reason_still_fits_the_state_field(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    """The assertion that matters: the object was constructed, so the PRD S6.3
    validator accepted it."""
    _gate(monkeypatch, approved_state, "reject", "x " * 5_000)
    assert approved_state.human_objection is not None
    round_tripped = AgentState(
        run_id="t", goal="g", human_objection=approved_state.human_objection
    )
    assert round_tripped.human_objection == approved_state.human_objection


def test_the_cap_is_measured_on_the_encoded_form() -> None:
    """A cap computed on len(text) passes here and fails in the validator on
    any objection containing an em dash or a quote, which is most of the
    sentences a person actually types about software."""
    from xeno.cli import bound_objection

    quoted = '"' * 4_000
    bounded = bound_objection(quoted)
    assert len(stable_json(bounded).encode()) <= MAX_FIELD_BYTES
    assert len(bounded) < len(quoted)


def test_a_reason_that_fits_is_passed_through_verbatim() -> None:
    from xeno.cli import bound_objection

    assert bound_objection("  use magic links, not passwords  ") == (
        "use magic links, not passwords"
    )
    assert "[truncated]" not in bound_objection("short")


def test_truncation_cuts_on_a_word_boundary() -> None:
    """Keeping the first sentences intact is the point — that is where a
    reason of any length puts its actual content."""
    from xeno.cli import bound_objection

    bounded = bound_objection("alpha bravo charlie " * 500)
    body = bounded.removesuffix("\n[truncated]")
    assert not body.endswith(("alph", "brav", "charli")), "cut mid-word"
    assert body.split()[-1] in {"alpha", "bravo", "charlie"}


# ---- what the rest of the run does with it ---------------------------------


def test_finalize_does_not_squash_a_declined_change(tmp_path: Path) -> None:
    from xeno.cli import _finalize
    from xeno.core.types import Verdict

    state = AgentState(run_id="t", goal="g")
    state.review_verdict = Verdict.APPROVE
    state.human_approved = False
    state.human_objection = "the retry loop should be bounded"

    # No vcs monkeypatching: a squash would raise on this bare tmp_path, so
    # the test fails loudly if the decline is ever ignored.
    assert _finalize(state, load_config(None), tmp_path, "xeno/x-abc") == 1


def test_finalize_shows_the_stated_reason_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It goes to memory.md, but it also has to survive in scrollback — the
    person who typed it is standing right there."""
    from xeno.cli import _finalize
    from xeno.core.types import Verdict

    state = AgentState(run_id="t", goal="g")
    state.review_verdict = Verdict.APPROVE
    state.human_approved = False
    state.human_objection = "the retry loop should be bounded"

    _finalize(state, load_config(None), tmp_path, "xeno/x-abc")
    assert "retry loop" in capsys.readouterr().out
