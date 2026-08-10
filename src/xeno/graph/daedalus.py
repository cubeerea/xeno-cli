"""The coder node (PRD S10): writes files, never declares itself correct.

Separation of powers is load-bearing here: this node has no shell access and
cannot run tests (PRD T3, S10) — it calls the router, parses a file-block
response, writes files directly into the isolated worktree via ordinary
library calls, and stops. Whether the result is correct is Talos's call, not
this node's.

A first failure routes to Chiron (PRD S7.2 L1), not back here — but L3 does
re-enter this node, after `xeno.graph.checkpoints.rollback_step` has discarded
the current task's attempt, so a re-entry always starts from a clean
checkpoint rather than from the failed state it is replacing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole
from xeno.graph.context import build_codebase_map
from xeno.graph.nodeops import (
    WorktreeEscape,
    complete_with_format_retry,
    write_file_blocks,
)
from xeno.graph.prompts import (
    DAEDALUS_CERBERUS_REJECTION_PREFIX,
    DAEDALUS_FORMAT_CORRECTION,
    DAEDALUS_SYSTEM,
    parse_daedalus_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router


def _build_current_turn(state: AgentState) -> str:
    """Ordinary task work is just the goal. A Cerberus E16 REJECT_AND_RETURN
    (PRD S8.3) appends the reviewer's objections instead — a fully green run
    was rejected on implementation grounds, so this call is a targeted fix,
    not a from-scratch implementation."""
    if state.reject_destination is not NodeRole.CODER:
        return state.goal
    assert state.cerberus_notes is not None, "a CODER rejection always carries objections"
    return "\n".join(
        [state.goal, "", DAEDALUS_CERBERUS_REJECTION_PREFIX, state.cerberus_notes.read_text()]
    )


def make_daedalus_node(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    touched_files: list[Path],
) -> Callable[[AgentState], AgentState]:
    """Build the Daedalus node."""
    builder = PromptBuilder(
        node=NodeRole.CODER,
        keyring=keyring,
        system_text=DAEDALUS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    call_index = 0

    def node(state: AgentState) -> AgentState:
        nonlocal call_index
        call_index += 1
        state.iterations_this_task += 1

        # We always pass freshly-computed text immediately before build(), so
        # we are always the refresher PromptBuilder's contract expects
        # (PRD S9.6.5) — never a stale reader of a previously built map.
        focus = [h.path for h in state.context_handles] or None
        builder.set_codebase_map(build_codebase_map(worktree, focus=focus), require_fresh=False)

        current_turn = _build_current_turn(state)
        state.reject_destination = None

        try:
            output = complete_with_format_retry(
                router=router,
                builder=builder,
                node=NodeRole.CODER,
                state=state,
                current_turn=current_turn,
                correction=DAEDALUS_FORMAT_CORRECTION,
                parse=parse_daedalus_output,
            )
        except ChainExhaustedError as exc:
            state.halt_reason = f"daedalus: {exc}"
            return state

        if output.is_objection:
            # A genuine <xeno-objection> halts immediately (PRD S10: an
            # underspecified task gets a blocking objection, not an invented
            # requirement — there is no Odysseus to re-plan to until Phase
            # 3). A malformed response that survived the one corrective
            # retry gets the same treatment: there is nothing safe to act on
            # either way.
            state.halt_reason = f"daedalus objection: {output.objection}"
            return state

        try:
            written, diff_text = write_file_blocks(worktree, output.files)
        except WorktreeEscape as exc:
            state.halt_reason = f"daedalus attempted to write outside the worktree: {exc.path}"
            return state

        keyring.mark_worktree_written(written, reason=f"daedalus call {call_index}")
        touched_files[:] = written

        diff_path = paths.workspace / f"diff_{call_index}.patch"
        diff_path.write_text(diff_text)
        state.diff_handle = Handle.for_file(
            diff_path,
            summary=f"daedalus call {call_index}: {', '.join(b.path for b in output.files)}",
        )

        return state

    return node
