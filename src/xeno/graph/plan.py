"""The roadmap and the plan, on disk (PRD S13 Phase 3, T2 filesystem-as-memory).

Two levels, written by two different nodes, because they are knowable at two
different times:

* A ROADMAP is a list of milestones — coarse increments of the goal, with no
  file names and no acceptance criteria. Odysseus writes it once, up front,
  when nothing but the goal is known.
* A PLAN is the flat list of concrete tasks actually executed. Lachesis
  writes it one milestone at a time, immediately before that milestone runs,
  so from the second milestone onward it is writing tasks against code that
  already exists rather than code it is guessing at.

The plan grows; it is never written whole. That is the entire point of the
split: a task written at t=0 against an empty repository can only guess at
the module names, function signatures, and test targets it will need, and a
guess baked into an acceptance criterion becomes a gate failure the ladder
then spends its whole budget on.

Both live behind Handles for the same reason (the 4 KB field rule, PRD S6.3)
— `AgentState.plan` and `AgentState.roadmap` are pointers, and this module is
the schema every side of them agrees on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from xeno.core.state import AgentState, Handle


class PlanTask(BaseModel):
    """One unit of work. `acceptance` is a MECHANICAL criterion (PRD S14's
    "Add rate limiting... with tests" example: "5 tasks, each with a
    mechanical acceptance criterion") — something Talos's gates can check,
    not a vague restatement of the task."""

    description: str = Field(max_length=2000)
    acceptance: str = Field(max_length=500)


class Plan(BaseModel):
    tasks: list[PlanTask] = Field(min_length=1)


class Milestone(BaseModel):
    """One coarse increment of the goal.

    Deliberately thinner than a `PlanTask`: no acceptance criterion, because
    the whole reason milestones exist is that whoever writes them cannot yet
    know what a mechanical check on them would look like. `outcome` is prose
    describing what is observably true once the milestone is done — it is what
    Lachesis writes the milestone's tests against, not something a gate reads.
    """

    description: str = Field(max_length=2000)
    outcome: str = Field(max_length=500)


class Roadmap(BaseModel):
    milestones: list[Milestone] = Field(min_length=1)


def write_plan(path: Path, plan: Plan) -> None:
    path.write_text(plan.model_dump_json(indent=2))


def read_plan(handle: Handle) -> Plan:
    return Plan.model_validate_json(handle.read_text())


def write_roadmap(path: Path, roadmap: Roadmap) -> None:
    path.write_text(roadmap.model_dump_json(indent=2))


def read_roadmap(handle: Handle) -> Roadmap:
    return Roadmap.model_validate_json(handle.read_text())


def current_task(plan: Plan, task_cursor: int) -> PlanTask:
    return plan.tasks[task_cursor]


def current_task_or_none(handle: Handle | None, task_cursor: int) -> PlanTask | None:
    """The current task, for callers that must not assume one exists.

    Returns `None` both before the first expansion and — importantly — once
    the cursor has run past the end of the plan, which is the normal state
    during a milestone's verification pass: every task is done, and the work
    in flight belongs to the milestone rather than to any single task.
    """
    if handle is None:
        return None
    plan = read_plan(handle)
    if not 0 <= task_cursor < len(plan.tasks):
        return None
    return plan.tasks[task_cursor]


def current_milestone_or_none(handle: Handle | None, milestone_cursor: int) -> Milestone | None:
    if handle is None:
        return None
    roadmap = read_roadmap(handle)
    if not 0 <= milestone_cursor < len(roadmap.milestones):
        return None
    return roadmap.milestones[milestone_cursor]


@dataclass(frozen=True, slots=True)
class Focus:
    """What the write nodes are working on right now, whichever level it
    lives at. Daedalus and Chiron are handed one of these rather than a
    `PlanTask` so that neither has to grow a branch for "the plan ran out" —
    during a verification pass there is genuinely no current task, but there
    is always a current *thing being built*."""

    description: str
    acceptance: str


def current_focus(state: AgentState) -> Focus | None:
    """The current task if one is in flight, otherwise the current milestone.

    The fallback is what keeps Chiron oriented when a milestone's tests fail:
    the cursor is past the last task by then, and without this the debugger
    would be handed the run's whole goal and left to guess which increment of
    it just broke.
    """
    task = current_task_or_none(state.plan, state.task_cursor)
    if task is not None:
        return Focus(description=task.description, acceptance=task.acceptance)

    milestone = current_milestone_or_none(state.roadmap, state.milestone_cursor)
    if milestone is not None:
        return Focus(
            description=f"Verifying the milestone: {milestone.description}",
            acceptance=milestone.outcome,
        )
    return None
