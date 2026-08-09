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
from xeno.core.state import AgentState, Handle


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
    assert _human_gate(approved_state, Path("/tmp/worktree"), "xeno/x-abc") == "approve"


def test_human_gate_reject(monkeypatch: pytest.MonkeyPatch, approved_state: AgentState) -> None:
    import xeno.cli as cli

    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "reject")
    assert _human_gate(approved_state, Path("/tmp/worktree"), "xeno/x-abc") == "reject"


def test_human_gate_inspect_then_approve(
    monkeypatch: pytest.MonkeyPatch, approved_state: AgentState
) -> None:
    answers = iter(["inspect", "approve"])
    monkeypatch.setattr("xeno.cli.Prompt.ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(Console, "pager", lambda self: contextlib.nullcontext())
    assert _human_gate(approved_state, Path("/tmp/worktree"), "xeno/x-abc") == "approve"


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
