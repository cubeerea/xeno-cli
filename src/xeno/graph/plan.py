"""Odysseus's plan, on disk (PRD S13 Phase 3, T2 filesystem-as-memory).

A plan is too large to live in `AgentState` directly (the 4 KB field rule,
PRD S6.3) — that is precisely why `AgentState.plan` is a `Handle` rather
than a list of tasks. This module is the schema both sides of that Handle
agree on: Odysseus writes it, everything downstream (Argus, Daedalus,
Chiron, the checkpoint step) reads it back to find the current task.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from xeno.core.state import Handle


class PlanTask(BaseModel):
    """One unit of work. `acceptance` is a MECHANICAL criterion (PRD S14's
    "Add rate limiting... with tests" example: "5 tasks, each with a
    mechanical acceptance criterion") — something Talos's gates can check,
    not a vague restatement of the task."""

    description: str = Field(max_length=2000)
    acceptance: str = Field(max_length=500)


class Plan(BaseModel):
    tasks: list[PlanTask] = Field(min_length=1)


def write_plan(path: Path, plan: Plan) -> None:
    path.write_text(plan.model_dump_json(indent=2))


def read_plan(handle: Handle) -> Plan:
    return Plan.model_validate_json(handle.read_text())


def current_task(plan: Plan, task_cursor: int) -> PlanTask:
    return plan.tasks[task_cursor]
