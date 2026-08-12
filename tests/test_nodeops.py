"""The operations every model-calling node shares (`xeno.graph.nodeops`).

These were six copy-pasted blocks before, each covered only indirectly through
whichever node happened to exercise it. Now that one implementation backs
Argus, Odysseus, Daedalus, Chiron, Cerberus, and toolchain discovery alike, the
two invariants that were implicit in all six copies get pinned directly: every
turn reaches the prompt-cache history whether or not it parsed (PRD T8), and a
model-authored path can never resolve outside the worktree.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from xeno.core.paths import RunPaths
from xeno.core.runlog import EventKind, NullRunLog
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
    format_hint: str = ""


@dataclass(frozen=True)
class _Result:
    """Mirrors the fields of `RouterResult` that the helper reads. Kept as a
    stand-in rather than the real thing because building one needs a
    CallRecord and a ModelSpec, neither of which any assertion here is
    about."""

    text: str
    finish_reason: str | None = None
    truncated: bool = False
    silent: bool = False
    reasoning: str = ""


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


class _RecordingLog(NullRunLog):
    """A null sink that remembers. The helper announces a saved response
    through the router's log, and "written but never mentioned" is exactly
    the failure a silent sink would hide."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def event(self, kind: EventKind | str, **payload: Any) -> dict[str, Any]:
        record = super().event(kind, **payload)
        self.records.append(record)
        return record


class _FakeRouter:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0
        self.runlog = _RecordingLog()

    def complete(self, node: NodeRole, prompt: str, *, state: AgentState) -> _Result:
        del node, prompt, state
        self.calls += 1
        return _Result(text=self._replies.pop(0))


def _paths(root: Path) -> RunPaths:
    return RunPaths(repo_root=root, run_id="r").ensure()


def _retry(
    router: _FakeRouter,
    builder: _FakeBuilder,
    malformed: set[str],
    *,
    paths: RunPaths | None = None,
    state: AgentState | None = None,
    tmp_path: Path | None = None,
) -> _Parsed:
    if paths is None:
        paths = _paths(tmp_path if tmp_path is not None else Path(tempfile.mkdtemp()))
    return complete_with_format_retry(
        router=router,  # type: ignore[arg-type]
        builder=builder,  # type: ignore[arg-type]
        node=NodeRole.CODER,
        state=state if state is not None else AgentState(run_id="r", goal="g"),
        paths=paths,
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


# ---- a response nobody could parse is kept ---------------------------------
#
# Run 20260811T124116 halted on "the response contained neither tag" after
# spending 3,174 output tokens across two attempts, and the text of neither
# attempt existed anywhere on disk afterwards. The halt itself was right —
# there genuinely was nothing to act on — but it left no way to find out
# WHY, which is the whole reason that class of bug took four investigations
# to pin down.


def test_a_response_that_never_parses_is_written_to_the_run_workspace(
    tmp_path: Path,
) -> None:
    state = AgentState(run_id="r", goal="g")

    _retry(
        _FakeRouter(["prose, no tags", "still prose"]),
        _FakeBuilder(),
        malformed={"prose, no tags", "still prose"},
        state=state,
        tmp_path=tmp_path,
    )

    assert state.unparsed_response is not None
    saved = state.unparsed_response.read_text()
    assert "prose, no tags" in saved
    assert "still prose" in saved, "the corrective attempt is half the evidence"
    assert state.unparsed_response.verify(), "the handle describes what is on disk"


def test_a_recovered_format_failure_leaves_nothing_behind(tmp_path: Path) -> None:
    """A response the correction fixed went on to produce its real artifact,
    and the run log already shows the two calls. Saving raw text for every
    slip would grow the workspace to duplicate what is already in it."""
    paths = _paths(tmp_path)
    state = AgentState(run_id="r", goal="g")

    _retry(
        _FakeRouter(["bad", "good"]), _FakeBuilder(), malformed={"bad"}, paths=paths, state=state
    )

    assert state.unparsed_response is None
    assert not list(paths.workspace.glob("unparsed_*.txt"))


def test_a_later_parse_clears_the_pointer_to_an_earlier_failure(tmp_path: Path) -> None:
    """Argus and Chiron tolerate a malformed response and carry on. If the
    pointer survived that, an unrelated halt three nodes later would print the
    path of a response nobody halted over — a panel naming the wrong file is
    worse than one naming none."""
    paths = _paths(tmp_path)
    state = AgentState(run_id="r", goal="g")

    _retry(
        _FakeRouter(["junk", "junk"]), _FakeBuilder(), malformed={"junk"}, paths=paths, state=state
    )
    assert state.unparsed_response is not None

    _retry(_FakeRouter(["good"]), _FakeBuilder(), malformed=set(), paths=paths, state=state)

    assert state.unparsed_response is None
    assert list(paths.workspace.glob("unparsed_*.txt")), "cleared in state, kept on disk"


def test_the_saved_response_is_announced_in_the_run_log(tmp_path: Path) -> None:
    """A file written where nobody is told to look is barely better than no
    file. The event carries the path so a post-mortem reading events.jsonl can
    point at it without knowing the naming scheme."""
    state = AgentState(run_id="r", goal="g")
    router = _FakeRouter(["junk", "junk"])

    _retry(router, _FakeBuilder(), malformed={"junk"}, state=state, tmp_path=tmp_path)

    events = [r for r in router.runlog.records if r["kind"] == EventKind.RESPONSE_UNPARSED]
    assert len(events) == 1
    assert state.unparsed_response is not None
    assert events[0]["path"] == str(state.unparsed_response.path)
    assert events[0]["attempts"] == MAX_FORMAT_ATTEMPTS


def test_a_second_failure_by_the_same_node_does_not_overwrite_the_first(
    tmp_path: Path,
) -> None:
    """A node is called many times in a run. Two failures collapsing onto one
    filename would destroy the earlier evidence — the same "nothing on disk"
    outcome, arrived at differently."""
    paths = _paths(tmp_path)
    state = AgentState(run_id="r", goal="g")

    _retry(_FakeRouter(["a", "a"]), _FakeBuilder(), malformed={"a"}, paths=paths, state=state)
    first = state.unparsed_response
    _retry(_FakeRouter(["b", "b"]), _FakeBuilder(), malformed={"b"}, paths=paths, state=state)

    assert first is not None and state.unparsed_response is not None
    assert first.path != state.unparsed_response.path
    assert "a" in first.read_text()


def test_the_correction_carries_what_was_actually_wrong(tmp_path: Path) -> None:
    """The generic correction alone tells a model that emitted good tags with
    one unreadable attribute that no tag was found — a correction it cannot
    act on, because the thing being complained about was already right."""
    builder = _FakeBuilder()

    complete_with_format_retry(
        router=_FakeRouter(["bad", "bad"]),  # type: ignore[arg-type]
        builder=builder,  # type: ignore[arg-type]
        node=NodeRole.CODER,
        state=AgentState(run_id="r", goal="g"),
        paths=_paths(tmp_path),
        current_turn="do the thing",
        correction="that was malformed, resend",
        parse=lambda text: _Parsed(
            text=text, malformed=True, format_hint="the acceptance= attribute is unreadable"
        ),
    )

    assert "the acceptance= attribute is unreadable" in builder.built[1]


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
