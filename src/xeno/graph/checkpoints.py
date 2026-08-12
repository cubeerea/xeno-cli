"""Checkpointing and L3 rollback (PRD S7.2, S8.4, S13 Phase 3).

Both functions here are deterministic — no model call, no routing decision
— so they are plain functions `xeno.graph.build` calls directly from a
wrapping node step, not node factories with their own `make_*` constructor
like every model-calling node in this package.
"""

from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path

from xeno.core import vcs
from xeno.core.breakers import reset_for_new_task
from xeno.core.ledger import CostLedger
from xeno.core.state import AgentState, CommitRef
from xeno.core.types import GateProfile
from xeno.graph.plan import current_task, read_plan, read_roadmap

#: `CommitRef.message` and the git commit message it mirrors share this cap
#: (PRD S6.3-style field discipline, even though `CommitRef` isn't itself an
#: `AgentState` field).
_MAX_MESSAGE_CHARS = 200


class CheckpointKind(StrEnum):
    """What a green evaluation just proved, and therefore what advanced."""

    TASK = "task"
    MILESTONE = "milestone"
    #: A green run reviewed, rejected, and fixed (PRD S8.3). Neither cursor
    #: moves: the plan and the roadmap are both already exhausted, and the
    #: work being committed belongs to the review round rather than to any
    #: task or milestone.
    REVIEW = "review"


def checkpoint_step(state: AgentState, worktree: Path, ledger: CostLedger) -> CheckpointKind:
    """Commit whatever the green evaluation just proved, and advance.

    Call this ONLY after Talos reports a pass — `xeno.graph.build`'s
    `talos_step` is the sole caller. Which of the three things it is
    committing is decided here rather than by the caller, because the answer
    comes entirely from state the caller would have to re-derive: the gate
    profile says whether this evaluation included the test command, and the
    two cursors say whether there is anything left for them to point at.

    `reset_for_new_task` clears the ladder/CB-4 baseline/CB-5 window for
    whatever starts next; advancing the cursor must happen in the same step,
    or a later halt would leave `state.checkpoints` and the cursor disagreeing
    about what is "current."

    This is also the only point in a run where a unit of work is definitively
    complete, so it is where M1.3's per-task cost window closes
    (`CostLedger.complete_task`) — a run that halts mid-task correctly
    contributes nothing to the median.
    """
    kind = _kind(state)
    _commit(state, worktree, ledger, _message(state, kind))
    if kind is CheckpointKind.TASK:
        state.task_cursor += 1
    elif kind is CheckpointKind.MILESTONE:
        state.milestone_cursor += 1
    return kind


def _kind(state: AgentState) -> CheckpointKind:
    """A task if one is in flight, else a milestone, else a review fix.

    Deliberately keyed off the cursors first and the gate profile second. A
    task Cerberus appended on an E16 rejection (PRD S8.3) runs under the FULL
    profile left over from the milestone's verification, but it is still a
    plan task and still has to advance the cursor — leaving the cursor behind
    would point every later node at a task that was already done.
    """
    if state.plan is not None and state.task_cursor < len(read_plan(state.plan).tasks):
        return CheckpointKind.TASK
    if (
        state.gate_profile is GateProfile.FULL
        and state.roadmap is not None
        and state.milestone_cursor < state.milestone_count
    ):
        return CheckpointKind.MILESTONE
    return CheckpointKind.REVIEW


def _message(state: AgentState, kind: CheckpointKind) -> str:
    if kind is CheckpointKind.TASK:
        assert state.plan is not None
        task = current_task(read_plan(state.plan), state.task_cursor)
        text = f"xeno: task {state.task_cursor} - {task.description}"
    elif kind is CheckpointKind.MILESTONE:
        assert state.roadmap is not None
        milestone = read_roadmap(state.roadmap).milestones[state.milestone_cursor]
        text = f"xeno: milestone {state.milestone_cursor} verified - {milestone.description}"
    else:
        text = f"xeno: review fix {len(state.checkpoints)}"
    return text[:_MAX_MESSAGE_CHARS]


def _commit(state: AgentState, worktree: Path, ledger: CostLedger, message: str) -> None:
    sha = vcs.commit(worktree, message)
    state.checkpoints = [
        *state.checkpoints,
        CommitRef(sha=sha, task_index=state.task_cursor, message=message, created_at=time.time()),
    ]
    if state.eval_report is not None:
        # Carried up to the run level before `reset_for_new_task` clears the
        # task's own bookkeeping — Cerberus needs it at the END of the run.
        state.touched_test_files = sorted(
            {*state.touched_test_files, *state.eval_report.touched_test_files}
        )

    ledger.complete_task()
    reset_for_new_task(state)


def rollback_step(state: AgentState, worktree: Path, *, initial_sha: str) -> None:
    """L3 (PRD S7.2): discard the current attempt before it is rewritten from
    scratch.

    The target is always `state.checkpoints[-1]` when one exists, never a
    search by task index: `checkpoint_step` appends a checkpoint and
    advances its cursor in the same atomic step, so whatever is last in
    `state.checkpoints` is by construction "the state right before the
    CURRENT unit of work started" — the very definition of L3's rollback
    point. `initial_sha` (from `vcs.init_repo`, PRD S13) is only the fallback
    for the run's first task, which has no prior checkpoint to roll back to.
    """
    target = state.checkpoints[-1].sha if state.checkpoints else initial_sha
    vcs.reset_hard(worktree, target)
