"""The coder node (PRD S10): writes files, never declares itself correct.

Separation of powers is load-bearing here: this node has no shell access and
cannot run tests (PRD T3, S10) — it calls the router, parses a file-block
response, writes files directly into the isolated worktree via ordinary
library calls, and stops. Whether the result is correct is Talos's call, not
this node's.

`xeno.graph.build`'s graph calls this node exactly once per run: a failure
routes to Chiron (PRD S7.2 L1), never back here. L3's "rollback and rewrite
from scratch" needs the checkpoint substrate that lands in Phase 3 (PRD S13),
so there is currently no path that re-enters Daedalus with a prior failure in
state.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from pathlib import Path

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole
from xeno.graph.context import build_codebase_map
from xeno.graph.prompts import (
    DAEDALUS_CERBERUS_REJECTION_PREFIX,
    DAEDALUS_FORMAT_CORRECTION,
    DAEDALUS_SYSTEM,
    DaedalusOutput,
    parse_daedalus_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: One corrective nudge, not an open-ended retry loop: CB-1 bounds the graph
#: as a whole, and a format failure that survives a single explicit
#: correction is more likely a capability gap than a one-off slip.
_MAX_FORMAT_ATTEMPTS = 2


def _compute_diff(worktree: Path, touched: list[Path], before: dict[Path, str]) -> str:
    chunks: list[str] = []
    for path in touched:
        rel = path.relative_to(worktree).as_posix()
        old = before.get(path, "")
        new = path.read_text() if path.exists() else ""
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        chunks.append("".join(diff))
    return "\n".join(chunks)


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

        output: DaedalusOutput | None = None
        for attempt in range(_MAX_FORMAT_ATTEMPTS):
            turn_text = current_turn if attempt == 0 else DAEDALUS_FORMAT_CORRECTION
            prompt = builder.build(turn_text)
            try:
                result = router.complete(NodeRole.CODER, prompt, state=state)
            except ChainExhaustedError as exc:
                state.halt_reason = f"daedalus: {exc}"
                return state

            builder.append_turn("user", turn_text)
            builder.append_turn("assistant", result.text)

            output = parse_daedalus_output(result.text)
            if not output.malformed:
                break

        assert output is not None
        if output.is_objection:
            # A genuine <xeno-objection> halts immediately (PRD S10: an
            # underspecified task gets a blocking objection, not an invented
            # requirement — there is no Odysseus to re-plan to until Phase
            # 3). A malformed response that survived the one corrective
            # retry gets the same treatment: there is nothing safe to act on
            # either way.
            state.halt_reason = f"daedalus objection: {output.objection}"
            return state

        before: dict[Path, str] = {}
        written: list[Path] = []
        for block in output.files:
            # Escaping the worktree is checked via the resolved path (so
            # `../`-style traversal cannot hide behind a symlink), but the
            # UNRESOLVED path is what gets written and stored: `worktree`
            # itself may be reached through a symlink (e.g. macOS's
            # /tmp -> /private/tmp), and resolving only `target` while every
            # other path in this module stays unresolved would silently
            # break every later `relative_to(worktree)` call downstream
            # (gates.py, talos.py's touched-file scoping).
            target = worktree / block.path
            try:
                target.resolve().relative_to(worktree.resolve())
            except ValueError:
                state.halt_reason = (
                    f"daedalus attempted to write outside the worktree: {block.path}"
                )
                return state
            before[target] = target.read_text() if target.exists() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(block.content)
            written.append(target)

        keyring.mark_worktree_written(written, reason=f"daedalus call {call_index}")
        touched_files[:] = written

        diff_text = _compute_diff(worktree, written, before)
        diff_path = paths.workspace / f"diff_{call_index}.patch"
        diff_path.write_text(diff_text)
        state.diff_handle = Handle.for_file(
            diff_path,
            summary=f"daedalus call {call_index}: {', '.join(b.path for b in output.files)}",
        )

        return state

    return node
