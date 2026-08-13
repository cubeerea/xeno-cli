"""The planner node (PRD S10, S13 Phase 3): turns a goal into an ordered
ROADMAP of milestones.

Odysseus decides the shape of the build, not its contents. It is called
exactly once in a normal run, before anything exists — which is precisely why
it no longer writes tasks. A task written at that moment can only guess at
the module names, signatures, and test targets it will need, and a guess
baked into an acceptance criterion becomes a gate failure the ladder then
spends its whole budget on. `xeno.graph.lachesis` expands each milestone into
tasks immediately before that milestone runs, when the preceding milestone's
code is on disk and can be named rather than guessed at.

Odysseus never touches the worktree — it has no shell access and reads
nothing directly off disk (PRD S10, same separation of powers as every
other node). Its only inputs are the goal and Argus's repo-skeleton prose,
which `xeno.graph.build` supplies by reference rather than through
`AgentState` — the skeleton is a one-shot planning aid used at exactly three
points in a run (the initial roadmap, a Cerberus rejection, and a Lachesis
objection), never re-read by any other node, so it does not earn a place in
state next to the Handles that genuinely need to survive across the whole run
(PRD S6.3).

Both re-entries arrive the same way: as someone else's written objection plus
the milestones already built. Odysseus is the only node that may author a
milestone, so a node that cannot proceed reports why and this one decides what
to do about it — the reporter never writes the replacement itself.
"""

from __future__ import annotations

from collections.abc import Callable

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole
from xeno.graph.law import ProjectLaw
from xeno.graph.nodeops import complete_with_format_retry
from xeno.graph.plan import (
    Milestone,
    Roadmap,
    current_milestone_or_none,
    read_roadmap,
    write_roadmap,
)
from xeno.graph.prompts import (
    ODYSSEUS_CERBERUS_REJECT_PREFIX,
    ODYSSEUS_FORMAT_CORRECTION,
    ODYSSEUS_PLAN_OBJECTION_PREFIX,
    ODYSSEUS_PLAN_PREFIX,
    ODYSSEUS_SYSTEM,
    parse_roadmap_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router


def _build_current_turn(
    state: AgentState,
    skeleton_text: str,
    *,
    is_cerberus_reject: bool,
    plan_objection: str,
) -> str:
    lines = [ODYSSEUS_PLAN_PREFIX, f"Goal: {state.goal}"]
    if skeleton_text:
        lines += ["", "Repository structure:", skeleton_text]

    if is_cerberus_reject:
        # E17 (PRD S8.3): every milestone already passed its own tests, so
        # there is no `eval_report` failure to describe — the objection is
        # Cerberus's own written verdict, not a gate result.
        assert state.cerberus_notes is not None, "a PLANNER rejection always carries objections"
        lines += ["", ODYSSEUS_CERBERUS_REJECT_PREFIX, state.cerberus_notes.read_text()]

    if plan_objection:
        # The objection alone does not say what was objected TO: Lachesis
        # refuses one milestone, and this node's own copy of the roadmap is
        # several turns back in the accumulated history by now. Naming the
        # milestone here is what makes a re-plan a repair rather than a
        # re-roll of the same prompt.
        milestone = current_milestone_or_none(state.roadmap, state.milestone_cursor)
        lines += [
            "",
            ODYSSEUS_PLAN_OBJECTION_PREFIX.format(
                position=state.milestone_cursor + 1,
                total=state.milestone_count,
                description=milestone.description if milestone else "(not found in the roadmap)",
            ),
            plan_objection,
        ]

    return "\n".join(lines)


def make_odysseus_node(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    law: ProjectLaw,
    skeleton: Callable[[], Handle | None],
) -> Callable[[AgentState], AgentState]:
    """Build the Odysseus node. `skeleton` reads back whatever
    `xeno.graph.argus`'s skeleton node most recently produced — a callable
    for the same reason `xeno.graph.talos`'s `intervention` parameter is
    one: the same node instance is reused across the run, and the value can
    change between calls."""
    builder = PromptBuilder(
        node=NodeRole.PLANNER,
        keyring=keyring,
        system_text=ODYSSEUS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    call_index = 0

    def node(state: AgentState) -> AgentState:
        nonlocal call_index
        call_index += 1

        is_cerberus_reject = state.reject_destination is NodeRole.PLANNER
        state.reject_destination = None
        # Read-and-clear, exactly like `reject_destination` above: this call
        # IS the answer to the objection, and one left set would send the
        # expansion that follows straight back here on a settled question.
        plan_objection = state.plan_objection or ""
        state.plan_objection = None

        builder.set_project_law(law.render(state))

        skeleton_handle = skeleton()
        skeleton_text = skeleton_handle.read_text() if skeleton_handle else ""
        current_turn = _build_current_turn(
            state,
            skeleton_text,
            is_cerberus_reject=is_cerberus_reject,
            plan_objection=plan_objection,
        )

        try:
            output = complete_with_format_retry(
                router=router,
                builder=builder,
                node=NodeRole.PLANNER,
                state=state,
                paths=paths,
                current_turn=current_turn,
                correction=ODYSSEUS_FORMAT_CORRECTION,
                parse=parse_roadmap_output,
            )
        except ChainExhaustedError as exc:
            state.halt_reason = f"odysseus: {exc}"
            return state

        if output.is_objection:
            state.halt_reason = f"odysseus objection: {output.objection}"
            return state

        new_milestones: tuple[Milestone, ...] = output.milestones
        if is_cerberus_reject or plan_objection:
            # One splice rule serves both re-entries: milestones before the
            # cursor are already built, tested, and checkpointed (PRD S13:
            # "the tasks already completed stay as they are"), so they are
            # kept and everything from the cursor on is replaced.
            #
            # The two differ only in where the cursor happens to be. A
            # Cerberus rejection arrives with it past the last milestone, so
            # "replace from here" appends; a Lachesis objection arrives with
            # it ON the milestone that could not be expanded, so the same rule
            # rewrites that one. Deliberately not two branches — this is the
            # same "keep what is checkpointed, replace what is not" rule
            # `xeno.graph.lachesis` applies one level down to the plan.
            assert state.roadmap is not None, "neither re-entry happens before a roadmap exists"
            completed = read_roadmap(state.roadmap).milestones[: state.milestone_cursor]
            new_milestones = (*completed, *new_milestones)

        roadmap = Roadmap(milestones=list(new_milestones))
        roadmap_path = paths.workspace / f"roadmap_{call_index}.json"
        write_roadmap(roadmap_path, roadmap)
        state.roadmap = Handle.for_file(
            roadmap_path, summary=f"roadmap v{call_index}: {len(roadmap.milestones)} milestone(s)"
        )
        state.milestone_count = len(roadmap.milestones)

        return state

    return node
