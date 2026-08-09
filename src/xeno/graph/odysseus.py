"""The planner node (PRD S10, S13 Phase 3): breaks a goal into an ordered
plan of small tasks, each with a mechanical acceptance criterion.

Odysseus never touches the worktree — it has no shell access and reads
nothing directly off disk (PRD S10, same separation of powers as every
other node). Its only inputs are the goal and Argus's repo-skeleton prose,
which `xeno.graph.build` supplies by reference rather than through
`AgentState` — the skeleton is a one-shot planning aid used at exactly two
points in a run (the initial plan and an L4 re-plan), never re-read by any
other node, so it does not earn a place in state next to the Handles that
genuinely need to survive across the whole run (PRD S6.3).
"""

from __future__ import annotations

from collections.abc import Callable

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole
from xeno.graph.plan import Plan, PlanTask, read_plan, write_plan
from xeno.graph.prompts import (
    ODYSSEUS_CERBERUS_REJECT_PREFIX,
    ODYSSEUS_FORMAT_CORRECTION,
    ODYSSEUS_REPLAN_PREFIX,
    ODYSSEUS_SYSTEM,
    parse_odysseus_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: Same rationale as Daedalus's: one corrective nudge, not an open loop.
_MAX_FORMAT_ATTEMPTS = 2

#: PRD S7.2: L4 is a re-plan. `AgentState.ladder_rung` reaching this value is
#: how the node tells "initial plan" from "revise the stuck task" apart —
#: there is no separate boolean in state for it, since the rung already
#: carries that information and a second flag could drift out of sync with it.
_L4_RUNG = 4


def _build_current_turn(
    state: AgentState, skeleton_text: str, *, is_l4_replan: bool, is_cerberus_reject: bool
) -> str:
    lines = [f"Goal: {state.goal}"]
    if skeleton_text:
        lines += ["", "Repository structure:", skeleton_text]

    if is_cerberus_reject:
        # E17 (PRD S8.3): every task already passed Talos's gates, so there is
        # no `eval_report` failure to describe — the objection is Cerberus's
        # own written verdict, not a gate result.
        assert state.cerberus_notes is not None, "a PLANNER rejection always carries objections"
        lines.append("")
        lines.append(ODYSSEUS_CERBERUS_REJECT_PREFIX)
        lines.append(state.cerberus_notes.read_text())
        return "\n".join(lines)

    if not is_l4_replan:
        return "\n".join(lines)

    report = state.eval_report
    assert report is not None, "a replan only happens after a task has failed"
    lines.append("")
    lines.append(ODYSSEUS_REPLAN_PREFIX)
    lines.append(
        f"Stuck task failed with: parse_ok={report.parse_ok} "
        f"lint_errors={report.lint_errors} type_errors={report.type_errors} "
        f"tests_failed={report.tests_failed}/{report.tests_run}"
    )
    if report.first_failure:
        lines.append(f"Critical failure detail: {report.first_failure}")
    return "\n".join(lines)


def make_odysseus_node(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
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

        is_l4_replan = state.ladder_rung == _L4_RUNG
        is_cerberus_reject = state.reject_destination is NodeRole.PLANNER
        is_replan = is_l4_replan or is_cerberus_reject
        state.reject_destination = None
        skeleton_handle = skeleton()
        skeleton_text = skeleton_handle.read_text() if skeleton_handle else ""
        current_turn = _build_current_turn(
            state, skeleton_text, is_l4_replan=is_l4_replan, is_cerberus_reject=is_cerberus_reject
        )

        output = None
        for attempt in range(_MAX_FORMAT_ATTEMPTS):
            turn_text = current_turn if attempt == 0 else ODYSSEUS_FORMAT_CORRECTION
            prompt = builder.build(turn_text)
            try:
                result = router.complete(NodeRole.PLANNER, prompt, state=state)
            except ChainExhaustedError as exc:
                state.halt_reason = f"odysseus: {exc}"
                return state

            builder.append_turn("user", turn_text)
            builder.append_turn("assistant", result.text)

            output = parse_odysseus_output(result.text)
            if not output.malformed:
                break

        assert output is not None
        if output.is_objection:
            state.halt_reason = f"odysseus objection: {output.objection}"
            return state

        new_tasks: tuple[PlanTask, ...] = output.tasks
        if is_replan:
            # Tasks before the cursor are already checkpointed and done
            # (PRD S13: "the tasks already completed stay as they are") —
            # only the stuck task onward gets replaced.
            assert state.plan is not None, "a replan only happens after an initial plan exists"
            old_plan = read_plan(state.plan)
            completed = old_plan.tasks[: state.task_cursor]
            new_tasks = (*completed, *new_tasks)

        plan = Plan(tasks=list(new_tasks))
        plan_path = paths.workspace / f"plan_{call_index}.json"
        write_plan(plan_path, plan)
        state.plan = Handle.for_file(
            plan_path, summary=f"plan v{call_index}: {len(plan.tasks)} task(s)"
        )
        state.task_count = len(plan.tasks)

        return state

    return node
