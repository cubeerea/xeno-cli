"""Lachesis (`xeno.graph.lachesis`) and the test-file boundary it enforces.

The two jobs are driven directly rather than through the whole graph: what is
under test here is the plan splicing, the refusal rule, and the gate-profile
handoff, all of which are decided inside the node. The milestone loop's
routing gets its coverage in `test_graph.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xeno.core.config import Limits, ModelSpec, NodeSpec, ProviderSpec, XenoConfig
from xeno.core.ledger import CostLedger
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import DEFAULT_NODE_TIERS, GateProfile, Tier
from xeno.core.usage import Usage
from xeno.graph.lachesis import MAX_PLAN_OBJECTIONS, MAX_VERIFY_REFUSALS, make_lachesis_nodes
from xeno.graph.law import ProjectLaw
from xeno.graph.plan import Milestone, Plan, PlanTask, Roadmap, read_plan, write_plan, write_roadmap
from xeno.graph.testfiles import is_test_file
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers.base import CompletionResult, Provider
from xeno.router.router import Router

TASKS_TWO = (
    '<xeno-task acceptance="a exists">write a</xeno-task>'
    '<xeno-task acceptance="b exists">write b</xeno-task>'
)
TESTS_OK = '<xeno-file path="tests/test_a.py">def test_a():\n    assert True\n</xeno-file>'
TESTS_WITH_SOURCE = (
    '<xeno-file path="tests/test_a.py">def test_a():\n    assert True\n</xeno-file>'
    '<xeno-file path="pkg/a.py">a = 1\n</xeno-file>'
)


class _Scripted(Provider):
    """Replies in order, recording the turns it was sent."""

    def __init__(self, name: str, spec: ProviderSpec, *, replies: list[str]) -> None:
        super().__init__(name, spec)
        self.replies = list(replies)
        self.turns: list[str] = []
        self.cache_capable = True

    def complete(self, prompt, model, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.turns.append(prompt.current_turn)
        text = self.replies.pop(0) if self.replies else self.replies_fallback
        return CompletionResult(
            text=text,
            model=model.model,
            usage=Usage(input_tokens=10, output_tokens=5),
            latency_ms=1.0,
        )

    replies_fallback = TESTS_OK

    def health_check(self) -> tuple[bool, str]:
        return True, "fake"


@pytest.fixture
def config() -> XenoConfig:
    return XenoConfig(
        providers={"ollama": ProviderSpec(family="ollama", base_url="http://x")},
        tiers={t: (ModelSpec(provider="ollama", model="m"),) for t in Tier},
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
        limits=Limits(),
    )


class _Harness:
    def __init__(self, expand, verify, fake, refusals: list[bool]) -> None:  # type: ignore[no-untyped-def]
        self.expand = expand
        self.verify = verify
        self.fake = fake
        self.refusals = refusals


def _harness(
    config: XenoConfig,
    tmp_path: Path,
    replies: list[str],
    *,
    established: bool = True,
) -> _Harness:
    router = Router(config, ledger=CostLedger(run_id="t"))
    fake = _Scripted("ollama", config.providers["ollama"], replies=replies)
    router._providers["ollama"] = fake
    worktree = tmp_path / "worktree"
    worktree.mkdir(exist_ok=True)
    refusals: list[bool] = []
    expand, verify = make_lachesis_nodes(
        router=router,
        config=config,
        keyring=CacheKeyring(run_id="t", worktree_root=worktree),
        paths=RunPaths(repo_root=tmp_path, run_id="t").ensure(),
        worktree=worktree,
        law=ProjectLaw(repo_root=tmp_path),
        touched_files=[],
        toolchain_established=lambda: established,
        report_refused=refusals.append,
    )
    return _Harness(expand, verify, fake, refusals)


def _state(tmp_path: Path, milestones: list[Milestone], cursor: int = 0) -> AgentState:
    path = tmp_path / f"roadmap_{cursor}.json"
    write_roadmap(path, Roadmap(milestones=milestones))
    state = AgentState(run_id="t", goal="build a thing")
    state.roadmap = Handle.for_file(path, summary="roadmap")
    state.milestone_cursor = cursor
    state.milestone_count = len(milestones)
    return state


def _with_plan(state: AgentState, tmp_path: Path, tasks: list[PlanTask], cursor: int) -> AgentState:
    path = tmp_path / "plan_existing.json"
    write_plan(path, Plan(tasks=tasks))
    state.plan = Handle.for_file(path, summary="plan")
    state.task_cursor = cursor
    state.task_count = len(tasks)
    return state


_M1 = Milestone(description="build the core", outcome="it converts")
_M2 = Milestone(description="wire the CLI", outcome="it runs")


# ---- JOB 1: expansion -----------------------------------------------------


def test_expanding_the_first_milestone_creates_the_plan(config: XenoConfig, tmp_path: Path) -> None:
    h = _harness(config, tmp_path, [TASKS_TWO])
    state = h.expand(_state(tmp_path, [_M1, _M2]))

    assert state.plan is not None
    assert [t.description for t in read_plan(state.plan).tasks] == ["write a", "write b"]
    assert state.task_count == 2
    assert state.milestone_task_start == 0


def test_the_turn_carries_the_job_selector_and_the_milestone(
    config: XenoConfig, tmp_path: Path
) -> None:
    h = _harness(config, tmp_path, [TASKS_TWO])
    h.expand(_state(tmp_path, [_M1, _M2]))

    turn = h.fake.turns[0]
    assert turn.startswith("JOB 1 - EXPAND.")
    assert "build the core" in turn
    assert "it converts" in turn
    assert "Milestone 1 of 2" in turn
    assert "wire the CLI" not in turn, "a later milestone is not this one's business"


def test_expanding_the_next_milestone_appends_to_the_plan(
    config: XenoConfig, tmp_path: Path
) -> None:
    """The plan is a growing materialization of the roadmap, not a rewrite:
    the previous milestone's tasks are already checkpointed."""
    h = _harness(config, tmp_path, [TASKS_TWO])
    state = _state(tmp_path, [_M1, _M2], cursor=1)
    state = _with_plan(state, tmp_path, [PlanTask(description="done", acceptance="x")], cursor=1)

    state = h.expand(state)

    assert [t.description for t in read_plan(state.plan).tasks] == ["done", "write a", "write b"]
    assert state.milestone_task_start == 1, "its tests are written against its own work only"


def test_an_l4_reexpansion_replaces_the_stuck_task_and_keeps_the_finished_ones(
    config: XenoConfig, tmp_path: Path
) -> None:
    h = _harness(config, tmp_path, [TASKS_TWO])
    state = _state(tmp_path, [_M1])
    state = _with_plan(
        state,
        tmp_path,
        [
            PlanTask(description="done", acceptance="x"),
            PlanTask(description="stuck", acceptance="y"),
        ],
        cursor=1,
    )
    state.ladder_rung = 4
    state.eval_report = None
    from xeno.core.state import EvalReport

    state.eval_report = EvalReport(passed=False, failed_command="lint", first_failure="E501")

    state = h.expand(state)

    tasks = [t.description for t in read_plan(state.plan).tasks]
    assert tasks == ["done", "write a", "write b"], "the stuck task is gone, the finished one stays"
    assert "Re-expand this" in h.fake.turns[-1]
    assert "E501" in h.fake.turns[-1], "the re-expansion is told what it is working around"


def test_a_reexpansion_does_not_move_the_milestone_mark(
    config: XenoConfig, tmp_path: Path
) -> None:
    """It is still the same milestone. Moving the mark would drop the tasks
    completed before the stuck one out of the slice its tests are written
    against."""
    h = _harness(config, tmp_path, [TASKS_TWO])
    state = _state(tmp_path, [_M1])
    state = _with_plan(
        state,
        tmp_path,
        [
            PlanTask(description="done", acceptance="x"),
            PlanTask(description="stuck", acceptance="y"),
        ],
        cursor=1,
    )
    state.milestone_task_start = 0
    state.ladder_rung = 4
    from xeno.core.state import EvalReport

    state.eval_report = EvalReport(passed=False, failed_command="lint")

    state = h.expand(state)
    assert state.milestone_task_start == 0


def test_expansion_sets_the_implementation_gate_profile(
    config: XenoConfig, tmp_path: Path
) -> None:
    """Tasks in a milestone cannot be gated on tests that describe code they
    have not written yet."""
    h = _harness(config, tmp_path, [TASKS_TWO])
    state = _state(tmp_path, [_M1])
    state.gate_profile = GateProfile.FULL

    assert h.expand(state).gate_profile is GateProfile.IMPLEMENTATION


def test_expanding_past_the_last_milestone_halts_rather_than_raising(
    config: XenoConfig, tmp_path: Path
) -> None:
    h = _harness(config, tmp_path, [TASKS_TWO])
    state = _state(tmp_path, [_M1])
    state.milestone_cursor = 5

    state = h.expand(state)
    assert state.halted
    assert "no milestone at index 5" in (state.halt_reason or "")


def test_an_objection_during_expansion_goes_back_to_odysseus(
    config: XenoConfig, tmp_path: Path
) -> None:
    """An objection is a defect report about a milestone Odysseus wrote, and
    Odysseus is the only node that may rewrite one. It is carried in state
    rather than ending the run."""
    objection = "<xeno-objection>this milestone is impossible</xeno-objection>"
    h = _harness(config, tmp_path, [objection])
    state = h.expand(_state(tmp_path, [_M1]))

    assert not state.halted
    assert state.plan_objection == "this milestone is impossible"
    assert state.plan_objection_count == 1
    # Nothing was expanded, so nothing downstream may run against this state.
    assert state.plan is None
    assert state.task_count == 0


def test_the_objection_budget_halts_once_odysseus_has_had_its_chances(
    config: XenoConfig, tmp_path: Path
) -> None:
    """The count IS the diagnosis. A first objection is the planning-time
    information asymmetry resolving; one that survives Odysseus re-planning
    against the objection itself is a real fault, and the halt has to say
    which of the two it saw."""
    objection = "<xeno-objection>no parser exists to extend</xeno-objection>"
    h = _harness(config, tmp_path, [objection] * MAX_PLAN_OBJECTIONS)
    state = _state(tmp_path, [_M1])

    for _ in range(MAX_PLAN_OBJECTIONS - 1):
        state = h.expand(state)
        assert not state.halted
        # Cleared by Odysseus in a real run; cleared here so each round is a
        # fresh objection rather than the same one read twice.
        state.plan_objection = None

    state = h.expand(state)
    assert state.halted
    reason = state.halt_reason or ""
    assert "no parser exists to extend" in reason
    assert f"{MAX_PLAN_OBJECTIONS}/{MAX_PLAN_OBJECTIONS}" in reason
    assert "odysseus re-planned" in reason
    assert state.plan_objection is None, "an exhausted budget must not also route to odysseus"


def test_a_successful_expansion_clears_the_objection_budget(
    config: XenoConfig, tmp_path: Path
) -> None:
    """The budget is for one milestone Odysseus could not fix, not for the run.

    Objecting once per milestone is the normal case — this node is the first
    to see a milestone against code that exists — so a run-level tally would
    halt a long roadmap for behaving exactly as designed.
    """
    objection = "<xeno-objection>nothing to extend yet</xeno-objection>"
    h = _harness(config, tmp_path, [objection, TASKS_TWO, objection])
    state = _state(tmp_path, [_M1, _M2])

    state = h.expand(state)
    assert state.plan_objection_count == 1

    state.plan_objection = None  # Odysseus consumes it and re-plans
    state = h.expand(state)
    assert state.plan_objection_count == 0, "it expanded; the budget is for stuck milestones"
    assert state.plan is not None

    state = h.expand(state)
    assert state.plan_objection_count == 1, "a later milestone starts from a full budget"
    assert not state.halted


def test_an_unparseable_expansion_leaves_the_raw_response_on_disk(
    config: XenoConfig, tmp_path: Path
) -> None:
    """Reproduces a real halt (run 20260811T124116): "neither a valid
    <xeno-task> block nor an <xeno-objection>", two attempts spent, and no
    record anywhere of what the model actually said. Halting is correct —
    there is nothing to act on — but it has to leave the evidence behind."""
    prose = "Sure! Here is the plan:\n1. Build the thing\n2. Test the thing"
    h = _harness(config, tmp_path, [prose, prose])

    state = h.expand(_state(tmp_path, [_M1]))

    assert state.halted
    assert "contained no <xeno-task> block" in (state.halt_reason or "")
    assert state.unparsed_response is not None
    assert "Build the thing" in state.unparsed_response.read_text()
    # The parser's "found no usable tags" fallback reports itself AS an
    # objection, and a response nobody could read says nothing about whether
    # the milestone is buildable. Routing it to Odysseus would have the
    # planner rewrite a milestone over a format failure.
    assert state.plan_objection is None
    assert state.plan_objection_count == 0


def test_an_expansion_whose_criteria_contain_apostrophes_is_not_lost(
    config: XenoConfig, tmp_path: Path
) -> None:
    """The specific shape that halted a live run: every acceptance criterion
    was about a parser, and a mechanical criterion about a parser names its
    input — so every one of them carried a quote character."""
    reply = (
        "<xeno-task acceptance=\"the tokenizer's output is a list\">tokenize</xeno-task>"
        '<xeno-task acceptance=\'parse("2+2") returns an Add node\'>parse</xeno-task>'
    )
    h = _harness(config, tmp_path, [reply])

    state = h.expand(_state(tmp_path, [_M1]))

    assert not state.halted
    assert state.task_count == 2


def test_the_first_milestone_is_not_told_that_earlier_code_exists(
    config: XenoConfig, tmp_path: Path
) -> None:
    """The prompt used to assert "earlier milestones are already built and
    their code is in the codebase map above" on EVERY expansion, including
    the first, where it is false. Paired with "use the real names, a task
    referring to code that does not exist cannot be completed", milestone 1
    of a fresh project had no legal answer — which is one of the two live
    explanations for that run's prose response."""
    h = _harness(config, tmp_path, [TASKS_TWO])
    h.expand(_state(tmp_path, [_M1, _M2]))

    turn = h.fake.turns[0]
    assert "This is the FIRST milestone" in turn
    assert "Earlier milestones are already built" not in turn


def test_a_later_milestone_is_told_the_opposite(config: XenoConfig, tmp_path: Path) -> None:
    h = _harness(config, tmp_path, [TASKS_TWO])
    h.expand(_state(tmp_path, [_M1, _M2], cursor=1))

    turn = h.fake.turns[0]
    assert "Earlier milestones are already built" in turn
    assert "This is the FIRST milestone" not in turn


def test_the_response_format_is_restated_in_the_turn_itself(
    config: XenoConfig, tmp_path: Path
) -> None:
    """The system prompt's copy sits before the codebase map, which on a real
    repository is thousands of tokens wide — the run that prompted this had
    6,200 tokens of unrelated README between the instruction and the ask.
    Keeping a copy in the turn is a structural guarantee of adjacency."""
    h = _harness(config, tmp_path, [TASKS_TWO])
    h.expand(_state(tmp_path, [_M1]))

    turn = h.fake.turns[0]
    assert "REQUIRED RESPONSE FORMAT" in turn
    assert turn.index("REQUIRED RESPONSE FORMAT") > turn.index("CURRENT MILESTONE")


# ---- JOB 2: verification --------------------------------------------------


def _verified(
    config: XenoConfig, tmp_path: Path, replies: list[str]
) -> tuple[_Harness, AgentState]:
    h = _harness(config, tmp_path, [TASKS_TWO, *replies])
    state = h.expand(_state(tmp_path, [_M1]))
    state.task_cursor = state.task_count  # every task done, as at a real verify
    return h, h.verify(state)


def test_verification_writes_the_tests_and_opens_the_full_gate_profile(
    config: XenoConfig, tmp_path: Path
) -> None:
    h, state = _verified(config, tmp_path, [TESTS_OK])

    assert (tmp_path / "worktree" / "tests" / "test_a.py").is_file()
    assert state.gate_profile is GateProfile.FULL
    assert state.diff_handle is not None
    assert h.refusals == [False]


def test_the_verification_turn_lists_the_milestones_own_tasks(
    config: XenoConfig, tmp_path: Path
) -> None:
    h, _state_out = _verified(config, tmp_path, [TESTS_OK])
    turn = h.fake.turns[-1]

    assert turn.startswith("JOB 2 - VERIFY.")
    assert "write a" in turn and "write b" in turn
    assert "build the core" in turn


def test_a_response_mixing_source_into_tests_is_refused_whole(
    config: XenoConfig, tmp_path: Path
) -> None:
    """Not partially applied. Writing "just the test half" would land a test
    written against a change that never happened."""
    h, state = _verified(config, tmp_path, [TESTS_WITH_SOURCE])

    assert not (tmp_path / "worktree" / "tests" / "test_a.py").exists()
    assert not (tmp_path / "worktree" / "pkg" / "a.py").exists()
    assert h.refusals == [True]
    assert not state.halted, "one refusal loops back, it does not end the run"
    assert state.gate_profile is not GateProfile.FULL, "nothing was proven, so nothing opens up"


def test_a_refusal_is_replayed_into_the_next_attempt(config: XenoConfig, tmp_path: Path) -> None:
    """A node that is never told its output was overruled just reproposes
    it — observed burning a whole rung against a one-character fix."""
    h = _harness(config, tmp_path, [TASKS_TWO, TESTS_WITH_SOURCE, TESTS_OK])
    state = h.expand(_state(tmp_path, [_M1]))
    state.task_cursor = state.task_count

    state = h.verify(state)
    state = h.verify(state)

    retry_turn = h.fake.turns[-1]
    assert "NOTHING was written" in retry_turn
    assert "pkg/a.py" in retry_turn
    assert (tmp_path / "worktree" / "tests" / "test_a.py").is_file()


def test_a_refusal_is_not_replayed_a_second_time(config: XenoConfig, tmp_path: Path) -> None:
    """Read-and-clear: it describes the call immediately before it."""
    h = _harness(config, tmp_path, [TASKS_TWO, TESTS_WITH_SOURCE, TESTS_OK, TESTS_OK])
    state = h.expand(_state(tmp_path, [_M1]))
    state.task_cursor = state.task_count

    state = h.verify(state)
    state = h.verify(state)
    state = h.verify(state)

    assert "NOTHING was written" not in h.fake.turns[-1]


def test_repeated_refusals_halt_rather_than_loop_forever(
    config: XenoConfig, tmp_path: Path
) -> None:
    """Nothing about a refused write advances the ladder or a breaker's
    counters, so this bound is the only thing standing between a stubborn
    model and an infinite loop."""
    h = _harness(config, tmp_path, [TASKS_TWO] + [TESTS_WITH_SOURCE] * MAX_VERIFY_REFUSALS)
    state = h.expand(_state(tmp_path, [_M1]))
    state.task_cursor = state.task_count

    for _ in range(MAX_VERIFY_REFUSALS):
        state = h.verify(state)

    assert state.halted
    assert "outside the test tree" in (state.halt_reason or "")


def test_expanding_the_next_milestone_clears_the_refusal_count(
    config: XenoConfig, tmp_path: Path
) -> None:
    """Each milestone's verification starts from a clean slate — otherwise a
    single bad milestone would poison every one after it."""
    h = _harness(
        config,
        tmp_path,
        [TASKS_TWO, TESTS_WITH_SOURCE, TESTS_WITH_SOURCE, TASKS_TWO, TESTS_WITH_SOURCE],
    )
    state = h.expand(_state(tmp_path, [_M1, _M2]))
    state.task_cursor = state.task_count
    state = h.verify(state)
    state = h.verify(state)

    state.milestone_cursor = 1
    state = h.expand(state)
    state.task_cursor = state.task_count
    state = h.verify(state)

    assert not state.halted, "the count restarted with the new milestone"


# ---- the shared definition of a test file ---------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_convert.py",
        "test/foo.py",
        "src/__tests__/thing.js",
        "spec/models_spec.rb",
        "conftest.py",
        "pkg/conftest.py",
        "cmd/server_test.go",
        "src/widget.test.tsx",
        "src/widget.spec.ts",
        "test_root_level.py",
    ],
)
def test_recognized_as_a_test_file(path: str) -> None:
    assert is_test_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/main.py",
        "pkg/latest.py",
        "src/contest/entry.py",
        "docs/testing.md",
        "src/protest.rs",
        "pyproject.toml",
    ],
)
def test_not_a_test_file(path: str) -> None:
    """The substring version read `src/contest/` as a test tree and missed
    `server_test.go` entirely. Both now decide whether a write is allowed,
    not merely how a diff is labelled."""
    assert not is_test_file(path)
