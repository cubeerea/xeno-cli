"""Checkpointing and L3 rollback (PRD S7.2, S8.4, S13 Phase 3).

Both functions here are deterministic — no model call, no routing decision
— so they are plain functions `xeno.graph.build` calls directly from a
wrapping node step, not node factories with their own `make_*` constructor
like every model-calling node in this package.
"""

from __future__ import annotations

import time
from pathlib import Path

from xeno.core import vcs
from xeno.core.breakers import reset_for_new_task
from xeno.core.ledger import CostLedger
from xeno.core.state import AgentState, CommitRef
from xeno.graph.plan import current_task, read_plan

#: `CommitRef.message` and the git commit message it mirrors share this cap
#: (PRD S6.3-style field discipline, even though `CommitRef` isn't itself an
#: `AgentState` field).
_MAX_MESSAGE_CHARS = 200


def checkpoint_step(state: AgentState, worktree: Path, ledger: CostLedger) -> None:
    """Commit the just-passed task's state and advance to the next one.

    Call this ONLY after Talos reports a pass — `xeno.graph.build`'s
    `talos_step` is the sole caller. `reset_for_new_task` clears the
    ladder/CB-4 baseline/CB-5 window for the task about to start; advancing
    `task_cursor` must happen in the same step, or a later halt mid-way
    through the next task would leave `state.checkpoints` and `task_cursor`
    disagreeing about which task is "current."

    This is also the only point in a run where a task is definitively
    complete, so it is where M1.3's per-task cost window closes
    (`CostLedger.complete_task`) — a run that halts mid-task correctly
    contributes nothing to the median.
    """
    assert state.plan is not None, "checkpointing only happens once a plan exists"
    plan = read_plan(state.plan)
    task = current_task(plan, state.task_cursor)
    message = f"xeno: task {state.task_cursor} - {task.description}"[:_MAX_MESSAGE_CHARS]

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
    state.task_cursor += 1


def rollback_step(state: AgentState, worktree: Path, *, initial_sha: str) -> None:
    """L3 (PRD S7.2): discard the current task's uncommitted attempt before
    Daedalus rewrites it from scratch.

    The target is always `state.checkpoints[-1]` when one exists, never a
    search by task index: `checkpoint_step` appends a checkpoint and
    advances `task_cursor` in the same atomic step, so whatever is last in
    `state.checkpoints` is by construction "the state right before the
    CURRENT task started" — the very definition of L3's rollback point.
    `initial_sha` (from `vcs.init_repo`, PRD S13) is only the fallback for
    the run's first task, which has no prior checkpoint to roll back to.
    """
    target = state.checkpoints[-1].sha if state.checkpoints else initial_sha
    vcs.reset_hard(worktree, target)
