"""The specifier node: expands one milestone into tasks, then writes the
tests that prove it.

Two jobs on one node, because they are one judgement made at two moments —
what this increment of the build should do, and whether it did it. Splitting
them across two nodes would put the definition of "done" in one place and the
check for it in another, which is precisely the split that lets the two
disagree.

Why this node exists at all: Odysseus writes the roadmap before any code
exists, so every concrete detail it could put in a task — a module name, a
signature, a test target — would be a guess. Lachesis is called immediately
BEFORE each milestone is built, with the preceding milestones' code already
on disk and in the codebase map, so from the second milestone onward it is
naming real code rather than predicting it.

That same asymmetry is why an objection here is not a halt. Being the first
node to see a milestone against the code that actually exists makes this the
first node able to find one that cannot be built — and the node that can fix
it is the one that wrote it, so the objection routes back to Odysseus with
its text attached (`MAX_PLAN_OBJECTIONS`), rather than ending the run on
information nobody has acted on yet.

Test authorship is exclusive to this node (`xeno.graph.testfiles`). Daedalus
and Chiron are both refused any write that touches a test file, so the check
can never be edited by whatever is being checked; conversely every write from
here is refused unless it IS a test file. The tests are the specification
from the moment they land: when the milestone's verification gates fail, the
escalation ladder has no way to weaken the test, so it is forced to fix the
code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.runlog import EventKind
from xeno.core.state import AgentState, Handle
from xeno.core.types import GateProfile, NodeRole
from xeno.graph.context import build_codebase_map
from xeno.graph.law import ProjectLaw
from xeno.graph.nodeops import (
    WorktreeEscape,
    complete_with_format_retry,
    write_file_blocks,
)
from xeno.graph.plan import (
    Milestone,
    Plan,
    PlanTask,
    current_milestone_or_none,
    read_plan,
    write_plan,
)
from xeno.graph.prompts import (
    CURRENT_MILESTONE_PREFIX,
    GREENFIELD_SCAFFOLD_ACCEPTANCE,
    GREENFIELD_SCAFFOLD_TASK,
    LACHESIS_EXPAND_FORMAT_CORRECTION,
    LACHESIS_EXPAND_FORMAT_REMINDER,
    LACHESIS_EXPAND_PREFIX,
    LACHESIS_GREENFIELD_PREFIX,
    LACHESIS_REEXPAND_PREFIX,
    LACHESIS_REFUSED_PREFIX,
    LACHESIS_SYSTEM,
    LACHESIS_VERIFY_FORMAT_CORRECTION,
    LACHESIS_VERIFY_PREFIX,
    MILESTONE_STANDING_FIRST,
    MILESTONE_STANDING_LATER,
    parse_task_list_output,
    parse_test_files_output,
)
from xeno.graph.testfiles import is_test_file
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: PRD S7.2 L4: the rung at which a stuck task is re-expanded rather than
#: retried. Reaching it is how the expand job tells "expand the next
#: milestone" from "the last expansion produced a task nobody could finish".
_L4_RUNG = 4

#: How many times a verification write may be refused for touching a non-test
#: file before the run halts. A refusal loops straight back to this node with
#: the reason attached, so a model that can take the correction takes it on
#: the next call; one that cannot would otherwise loop forever, since nothing
#: about a refused write advances the ladder or any breaker's counters.
MAX_VERIFY_REFUSALS = 3

#: How many times this node may object to the roadmap before the run stops
#: trying. An objection routes back to Odysseus with the text attached
#: (`xeno.graph.build`'s "replan" edge), which is a cycle, and every cycle in
#: this graph carries its own bound.
#:
#: The count is also the diagnosis, which is why the bound is not 1. A FIRST
#: objection is usually not an error by anyone: Odysseus planned against a
#: repository that did not exist yet and this node is looking at what was
#: actually built, so the objection is that asymmetry resolving — the reason
#: this node is called per-milestone at all. One that survives a re-plan
#: Odysseus made WITH the objection in hand is a genuine planning failure.
#: One that survives three says the problem is upstream of both nodes, in the
#: goal or the repository, which is the conclusion worth handing a human.
#:
#: Counted CONSECUTIVELY, not per run: a successful expansion resets it, so
#: the budget is spent on one milestone nobody could fix rather than on a
#: long roadmap's worth of individually-resolved first objections.
MAX_PLAN_OBJECTIONS = 3

#: Objections are model prose and otherwise unbounded. Capped on the way into
#: `AgentState.plan_objection`, which is an inline field under the 4 KB rule
#: (PRD S6.3) rather than a Handle.
_MAX_OBJECTION_CHARS = 1000


class _VerifyState:
    """Refusal bookkeeping shared between the two jobs.

    Lives here rather than in `_LoopContext` because it is entirely internal
    to this node: expand resets it because a new milestone's verification
    starts with a clean slate, and nothing outside needs to see it except the
    one routing decision that asks whether to loop.
    """

    def __init__(self) -> None:
        self.refusal = ""
        self.refusals = 0

    def reset(self) -> None:
        self.refusal = ""
        self.refusals = 0

    def refuse(self, reason: str) -> None:
        self.refusal = reason
        self.refusals += 1

    def take_refusal(self) -> str:
        """Read-and-clear: a refusal describes the call immediately before
        it, so replaying it into a later, unrelated call misleads."""
        refusal, self.refusal = self.refusal, ""
        return refusal


def _milestone_block(milestone: Milestone, state: AgentState) -> str:
    return CURRENT_MILESTONE_PREFIX.format(
        description=milestone.description,
        outcome=milestone.outcome,
        position=state.milestone_cursor + 1,
        total=state.milestone_count,
        standing=(
            MILESTONE_STANDING_LATER
            if state.milestone_cursor > 0
            else MILESTONE_STANDING_FIRST
        ),
    )


def _build_expand_turn(
    state: AgentState,
    milestone: Milestone,
    *,
    is_greenfield: bool,
    is_reexpand: bool,
) -> str:
    lines = [LACHESIS_EXPAND_PREFIX, f"Goal: {state.goal}", "", _milestone_block(milestone, state)]

    if is_greenfield:
        lines += ["", LACHESIS_GREENFIELD_PREFIX]

    if is_reexpand:
        report = state.eval_report
        assert report is not None, "a re-expansion only happens after a task has failed"
        lines += ["", LACHESIS_REEXPAND_PREFIX]
        lines.append(f"Stuck task failed with: failed_command={report.failed_command!r}")
        if report.first_failure:
            lines.append(f"Critical failure detail: {report.first_failure}")

    # Last, and in the current turn rather than the system prompt, which is
    # where it also lives. On a real repository the codebase map sits between
    # the two (`ASSEMBLY_ORDER`), so the system prompt's copy can be several
    # thousand tokens back by the time the model generates — the run that
    # prompted this had 6,200 tokens of unrelated README in between. Putting
    # it here is a structural guarantee of adjacency that no wording change
    # to the system prompt can provide.
    lines += ["", LACHESIS_EXPAND_FORMAT_REMINDER]
    return "\n".join(lines)


def _build_verify_turn(state: AgentState, milestone: Milestone, last_refusal: str) -> str:
    lines = [LACHESIS_VERIFY_PREFIX, f"Goal: {state.goal}", "", _milestone_block(milestone, state)]

    tasks = _milestone_tasks(state)
    if tasks:
        lines.append("")
        lines.append("The tasks that were implemented for this milestone:")
        lines.extend(f"- {task.description}" for task in tasks)

    if last_refusal:
        lines += ["", LACHESIS_REFUSED_PREFIX, last_refusal]

    lines += [
        "",
        "Write the tests that prove this milestone delivered its stated outcome.",
    ]
    return "\n".join(lines)


def _object_to_roadmap(state: AgentState, objection: str) -> AgentState:
    """Send an unbuildable milestone back to Odysseus, or stop trying.

    This node files a defect report; it does not write the replacement. If it
    could, the specifier would be authoring the roadmap through the escape
    hatch that exists for refusing one, and the run log would no longer say
    who planned what — the same separation that keeps Talos from flipping a
    verdict and Cerberus from writing code.

    Exhausting the budget is what makes this a halt worth reading. The old
    behaviour halted on the first objection, which said only that some
    milestone was unbuildable; reaching `MAX_PLAN_OBJECTIONS` says it stayed
    unbuildable after Odysseus re-planned against the objection itself.
    """
    text = " ".join(objection.split())[:_MAX_OBJECTION_CHARS]
    state.plan_objection_count += 1
    if state.plan_objection_count >= MAX_PLAN_OBJECTIONS:
        state.halt_reason = (
            f"lachesis objection {state.plan_objection_count}/{MAX_PLAN_OBJECTIONS} on "
            f"milestone {state.milestone_cursor + 1} of {state.milestone_count}: {text} "
            f"(odysseus re-planned with the earlier objections in hand and the milestone "
            f"is still not expandable — the goal or the repository is the likelier fault)"
        )
        return state
    # Left unset on the halting branch above, so `route_after_expand` can read
    # the two as mutually exclusive without ordering them by hand.
    state.plan_objection = text
    return state


def _milestone_tasks(state: AgentState) -> list[PlanTask]:
    """The tasks belonging to the milestone being verified.

    `milestone_task_start` is the plan cursor at the moment this milestone was
    expanded, so the slice from there to the end is exactly this milestone's
    work — the plan is append-only within a milestone, and the previous
    milestones' tasks sit before that mark.
    """
    if state.plan is None:
        return []
    return read_plan(state.plan).tasks[state.milestone_task_start :]


def make_lachesis_nodes(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    law: ProjectLaw,
    touched_files: list[Path],
    toolchain_established: Callable[[], bool],
    report_refused: Callable[[bool], None],
) -> tuple[Callable[[AgentState], AgentState], Callable[[AgentState], AgentState]]:
    """Build the (expand, verify) pair.

    One `PromptBuilder` serves both jobs, like `xeno.graph.argus`'s three:
    the SYSTEM breakpoint stays byte-identical across every call this node
    makes (PRD T8), and the accumulated history means the node that wrote a
    milestone's tasks is the same one, with the same memory, that later writes
    its tests.

    `report_refused` tells `xeno.graph.build` whether the verify call actually
    wrote anything, for the same reason Chiron's `report_declined` exists: a
    LangGraph node can only return `AgentState`, and the caller must not run
    the post-write bookkeeping for a call that wrote nothing.
    """
    builder = PromptBuilder(
        node=NodeRole.SPECIFIER,
        keyring=keyring,
        system_text=LACHESIS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    verify_state = _VerifyState()
    expand_calls = 0
    verify_calls = 0

    def _refresh_map(state: AgentState) -> None:
        # PRD S9.6.5: refreshed immediately before build(), never stale. No
        # focus: this node's whole advantage over planning up front is that it
        # can see what actually got built, and a narrowed view would give that
        # back.
        #
        # Serves both jobs, so law and map stay in lockstep across expand and
        # verify — the two calls that must agree on what "done" means.
        builder.set_project_law(law.render(state))
        builder.set_codebase_map(build_codebase_map(worktree), require_fresh=False)

    def expand(state: AgentState) -> AgentState:
        nonlocal expand_calls
        expand_calls += 1

        milestone = current_milestone_or_none(state.roadmap, state.milestone_cursor)
        if milestone is None:
            state.halt_reason = (
                f"lachesis: no milestone at index {state.milestone_cursor} of "
                f"{state.milestone_count}"
            )
            return state

        # A repository with no manifest has nothing for the gates to run, and
        # that is established deterministically (`has_manifests`) rather than
        # judged — so the scaffold is prepended below by construction, and the
        # prompt only explains why it is already there.
        is_greenfield = not toolchain_established() and state.plan is None
        is_reexpand = state.ladder_rung == _L4_RUNG
        current_turn = _build_expand_turn(
            state, milestone, is_greenfield=is_greenfield, is_reexpand=is_reexpand
        )
        _refresh_map(state)

        try:
            output = complete_with_format_retry(
                router=router,
                builder=builder,
                node=NodeRole.SPECIFIER,
                state=state,
                paths=paths,
                current_turn=current_turn,
                correction=LACHESIS_EXPAND_FORMAT_CORRECTION,
                parse=parse_task_list_output,
            )
        except ChainExhaustedError as exc:
            state.halt_reason = f"lachesis expand: {exc}"
            return state

        if output.malformed:
            # Checked BEFORE the objection below, because the parser's
            # "found no usable tags" fallback reports itself as an objection
            # (`parse_task_list_output`) and this is the one place where the
            # difference decides an edge. A format failure says nothing about
            # whether the milestone can be built — sending it to Odysseus
            # would have the planner rewriting a milestone on the strength of
            # a response the harness could not read.
            state.halt_reason = f"lachesis expand: {output.objection}"
            return state

        if output.is_objection:
            # Not a halt: an objection is a defect report about the milestone
            # Odysseus wrote, and Odysseus is the one node that can act on it.
            return _object_to_roadmap(state, output.objection or "")

        if output.dropped:
            # An expansion that parsed is not the same as one that parsed
            # WHOLE. Blocks the scanner could not read are dropped
            # individually, so the plan below looks complete and is simply
            # short — the one failure here with no other symptom, and the
            # reason this line exists at all.
            router.runlog.event(
                EventKind.RESPONSE_UNPARSED,
                node=NodeRole.SPECIFIER.value,
                dropped=output.dropped,
                kept=len(output.tasks),
                detail="task block(s) in an otherwise usable expansion could not be read",
            )

        new_tasks: tuple[PlanTask, ...] = output.tasks
        if is_greenfield:
            new_tasks = (
                PlanTask(
                    description=GREENFIELD_SCAFFOLD_TASK,
                    acceptance=GREENFIELD_SCAFFOLD_ACCEPTANCE,
                ),
                *new_tasks,
            )

        # One splice rule covers both callers. Expanding the NEXT milestone
        # has the cursor sitting just past the last task, so everything before
        # it is kept and the new tasks are appended; an L4 re-expansion has
        # the cursor on the stuck task, so everything before it is kept and
        # the new tasks replace it. Same operation, and deliberately so —
        # "keep what is already checkpointed, replace what is not" is the only
        # rule either case actually wants.
        kept: tuple[PlanTask, ...] = ()
        if state.plan is not None:
            kept = tuple(read_plan(state.plan).tasks[: state.task_cursor])
        tasks = (*kept, *new_tasks)

        plan = Plan(tasks=list(tasks))
        plan_path = paths.workspace / f"plan_{expand_calls}.json"
        write_plan(plan_path, plan)
        state.plan = Handle.for_file(
            plan_path, summary=f"plan v{expand_calls}: {len(plan.tasks)} task(s)"
        )
        state.task_count = len(plan.tasks)
        if not is_reexpand:
            # Only a genuinely new milestone moves the mark; a re-expansion is
            # still the same milestone, and moving it would lose the earlier
            # tasks from the slice its verification is written against.
            state.milestone_task_start = len(kept)
        state.gate_profile = GateProfile.IMPLEMENTATION
        verify_state.reset()
        # CONSECUTIVE objections, which is the only count that means anything.
        # A first objection per milestone is the normal case this node exists
        # to produce — it is the first to see a milestone against code that
        # actually exists — so a run-level tally would halt an eight-milestone
        # roadmap on milestone three for working exactly as intended. What
        # `MAX_PLAN_OBJECTIONS` bounds is one milestone that Odysseus could
        # not fix, and a successful expansion is proof that it did.
        state.plan_objection_count = 0

        return state

    def verify(state: AgentState) -> AgentState:
        nonlocal verify_calls
        verify_calls += 1

        milestone = current_milestone_or_none(state.roadmap, state.milestone_cursor)
        if milestone is None:
            state.halt_reason = (
                f"lachesis: no milestone at index {state.milestone_cursor} of "
                f"{state.milestone_count}"
            )
            return state

        current_turn = _build_verify_turn(state, milestone, verify_state.take_refusal())
        _refresh_map(state)

        try:
            output = complete_with_format_retry(
                router=router,
                builder=builder,
                node=NodeRole.SPECIFIER,
                state=state,
                paths=paths,
                current_turn=current_turn,
                correction=LACHESIS_VERIFY_FORMAT_CORRECTION,
                parse=parse_test_files_output,
            )
        except ChainExhaustedError as exc:
            state.halt_reason = f"lachesis verify: {exc}"
            return state

        if output.is_objection:
            state.halt_reason = f"lachesis objection: {output.objection}"
            return state

        non_tests = [b.path for b in output.files if not is_test_file(Path(b.path).as_posix())]
        if non_tests:
            # Refused whole rather than partially applied, mirroring Chiron's
            # test-file rule in the opposite direction: a response that mixes
            # a test with an implementation file is one where the node has
            # decided to fix the code itself, and writing "just the test half"
            # of that would land a test written against a change that never
            # happened.
            verify_state.refuse(f"non-test path(s): {', '.join(non_tests)}")
            report_refused(True)
            if verify_state.refusals >= MAX_VERIFY_REFUSALS:
                state.halt_reason = (
                    f"lachesis: {verify_state.refusals} verification attempts in a row wrote "
                    f"outside the test tree (last: {', '.join(non_tests)})"
                )
            return state

        try:
            written, diff_text = write_file_blocks(worktree, output.files)
        except WorktreeEscape as exc:
            state.halt_reason = f"lachesis attempted to write outside the worktree: {exc.path}"
            return state

        keyring.mark_worktree_written(written, reason=f"lachesis call {verify_calls}")
        # Accumulate, not replace: Talos's lint/type scope has to cover the
        # tests as well as the source they exercise, or a test that does not
        # even parse would only be caught by the test runner itself.
        for path in written:
            if path not in touched_files:
                touched_files.append(path)

        diff_path = paths.workspace / f"lachesis_diff_{verify_calls}.patch"
        diff_path.write_text(diff_text)
        state.diff_handle = Handle.for_file(
            diff_path,
            summary=f"lachesis call {verify_calls}: {', '.join(b.path for b in output.files)}",
        )
        # From here the milestone is gated on everything the toolchain has,
        # tests included — which is the whole point of having written them.
        state.gate_profile = GateProfile.FULL
        report_refused(False)

        return state

    return expand, verify
