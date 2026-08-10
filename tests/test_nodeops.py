"""The operations every model-calling node shares (`xeno.graph.nodeops`).

These were six copy-pasted blocks before, each covered only indirectly through
whichever node happened to exercise it. Now that one implementation backs
Argus, Odysseus, Daedalus, Chiron, Cerberus, and toolchain discovery alike, the
two invariants that were implicit in all six copies get pinned directly: every
turn reaches the prompt-cache history whether or not it parsed (PRD T8), and a
model-authored path can never resolve outside the worktree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from xeno.core.state import AgentState
from xeno.core.types import NodeRole
from xeno.graph.nodeops import (
    MAX_FORMAT_ATTEMPTS,
    WorktreeEscape,
    complete_with_format_retry,
    write_file_blocks,
)
from xeno.graph.prompts import FileBlock


@dataclass(frozen=True)
class _Parsed:
    text: str
    malformed: bool


@dataclass(frozen=True)
class _Result:
    text: str


class _FakeBuilder:
    """Records what a real `PromptBuilder` would have been told, without the
    system-fingerprint machinery a real one enforces per process."""

    def __init__(self) -> None:
        self.built: list[str] = []
        self.turns: list[tuple[str, str]] = []

    def build(self, current_turn: str) -> str:
        self.built.append(current_turn)
        return current_turn

    def append_turn(self, role: str, content: str) -> None:
        self.turns.append((role, content))


class _FakeRouter:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete(self, node: NodeRole, prompt: str, *, state: AgentState) -> _Result:
        del node, prompt, state
        self.calls += 1
        return _Result(text=self._replies.pop(0))


def _retry(router: _FakeRouter, builder: _FakeBuilder, malformed: set[str]) -> _Parsed:
    return complete_with_format_retry(
        router=router,  # type: ignore[arg-type]
        builder=builder,  # type: ignore[arg-type]
        node=NodeRole.CODER,
        state=AgentState(run_id="r", goal="g"),
        current_turn="do the thing",
        correction="that was malformed, resend",
        parse=lambda text: _Parsed(text=text, malformed=text in malformed),
    )


def test_a_parseable_first_response_costs_exactly_one_call() -> None:
    router = _FakeRouter(["good"])
    builder = _FakeBuilder()

    output = _retry(router, builder, malformed=set())

    assert output.text == "good"
    assert router.calls == 1
    assert builder.built == ["do the thing"]


def test_a_malformed_response_is_retried_once_with_the_correction() -> None:
    router = _FakeRouter(["bad", "good"])
    builder = _FakeBuilder()

    output = _retry(router, builder, malformed={"bad"})

    assert output.text == "good"
    assert router.calls == 2
    assert builder.built == ["do the thing", "that was malformed, resend"]


def test_the_malformed_exchange_still_reaches_the_prompt_history() -> None:
    """The correction attempt has to SEE the response it is correcting, and
    the cached turn history must reflect what was actually sent (PRD T8)."""
    router = _FakeRouter(["bad", "good"])
    builder = _FakeBuilder()

    _retry(router, builder, malformed={"bad"})

    assert builder.turns == [
        ("user", "do the thing"),
        ("assistant", "bad"),
        ("user", "that was malformed, resend"),
        ("assistant", "good"),
    ]


def test_retries_are_bounded_and_the_last_result_is_returned_still_malformed() -> None:
    """One corrective nudge, not an open loop — the caller decides what a
    surviving malformed result means for it."""
    router = _FakeRouter(["bad", "still bad"])
    builder = _FakeBuilder()

    output = _retry(router, builder, malformed={"bad", "still bad"})

    assert router.calls == MAX_FORMAT_ATTEMPTS == 2
    assert output.malformed


def test_blocks_are_written_and_diffed_against_what_was_there_before(tmp_path: Path) -> None:
    (tmp_path / "kept.py").write_text("old = 1\n")

    written, diff = write_file_blocks(
        tmp_path,
        [
            FileBlock(path="kept.py", content="new = 2\n"),
            FileBlock(path="sub/added.py", content="x\n"),
        ],
    )

    assert written == [tmp_path / "kept.py", tmp_path / "sub" / "added.py"]
    assert (tmp_path / "kept.py").read_text() == "new = 2\n"
    assert (tmp_path / "sub" / "added.py").read_text() == "x\n"
    assert "-old = 1" in diff
    assert "+new = 2" in diff
    assert "+x" in diff


def test_a_path_escaping_the_worktree_raises_before_it_is_written(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    outside = tmp_path / "outside.py"

    with pytest.raises(WorktreeEscape) as exc:
        write_file_blocks(worktree, [FileBlock(path="../outside.py", content="pwned\n")])

    assert exc.value.path == "../outside.py"
    assert not outside.exists()


def test_earlier_blocks_survive_a_later_escape(tmp_path: Path) -> None:
    """Deliberate: the caller halts the run, and a half-written worktree a
    human can inspect beats silently rolling back writes the model believed
    it had made."""
    worktree = tmp_path / "wt"
    worktree.mkdir()

    with pytest.raises(WorktreeEscape):
        write_file_blocks(
            worktree,
            [
                FileBlock(path="first.py", content="written\n"),
                FileBlock(path="../escaped.py", content="nope\n"),
            ],
        )

    assert (worktree / "first.py").read_text() == "written\n"
