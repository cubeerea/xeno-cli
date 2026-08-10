"""The researcher node (PRD S10, S13 Phase 3): repo orientation for the
planner, and per-task file selection for the coder/debugger.

Argus narrows context; it never gates correctness (PRD S10). A hallucinated
or empty file selection is handled gracefully here rather than halting the
run — Daedalus and Chiron already work fine with zero context handles (that
was Phase 1 and 2's whole starting point), just less efficiently.

Deliberately scoped without tree-sitter: the exit criterion is autonomous
multi-file completion, not a symbol table, and file-tree-plus-task-description
is enough signal for a light-tier model to pick relevant files without paying
to read their content — which would make Argus as expensive as the node it
exists to keep cheap.

Argus's JOB 3 (`xeno.adapters.discovery`) is the one exception to "narrows
context, never gates correctness": toolchain discovery genuinely does decide
which commands Talos's gates run — but never the pass/fail verdict itself
(PRD S8.2), and it is a one-time, cached, validated call, not part of this
module's per-task loop.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole
from xeno.graph.context import build_codebase_map
from xeno.graph.nodeops import complete_with_format_retry
from xeno.graph.plan import current_task, read_plan
from xeno.graph.prompts import (
    ARGUS_FORMAT_CORRECTION,
    ARGUS_L2_PREFIX,
    ARGUS_RESEARCH_PREFIX,
    ARGUS_SKELETON_PREFIX,
    ARGUS_SYSTEM,
    parse_argus_research_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: PRD S7.2: the rung that means "re-research because a patch already
#: failed" rather than "first look at this task."
_L2_RUNG = 2


def make_argus_nodes(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    publish_skeleton: Callable[[Handle], None],
) -> tuple[Callable[[AgentState], AgentState], Callable[[AgentState], AgentState]]:
    """Build Argus's two node functions, sharing ONE `PromptBuilder`.

    "One builder per node per run" (PRD T8, `PromptBuilder`'s docstring) is
    not optional here: the builder fingerprints one system text per
    `NodeRole` for the process lifetime and raises on a second one, and both
    of Argus's jobs route through `NodeRole.RESEARCHER` — that is what
    selects Argus's tier. `ARGUS_SYSTEM` covers both jobs; the current
    turn's prefix says which one this call is (same technique
    `xeno.graph.odysseus` uses for its plan-vs-replan framing).

    Returns `(skeleton_node, research_node)`. `publish_skeleton` hands the
    skeleton Handle to `xeno.graph.build`'s shared run context — Odysseus
    needs it, but by reference, not through `AgentState` (see
    `xeno.graph.odysseus`'s docstring for why it does not belong there).
    """
    builder = PromptBuilder(
        node=NodeRole.RESEARCHER,
        keyring=keyring,
        system_text=ARGUS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    skeleton_call_index = 0
    research_call_index = 0

    def skeleton_node(state: AgentState) -> AgentState:
        nonlocal skeleton_call_index
        skeleton_call_index += 1

        builder.set_codebase_map(
            build_codebase_map(worktree, max_content_bytes=0), require_fresh=False
        )
        current_turn = ARGUS_SKELETON_PREFIX
        prompt = builder.build(current_turn)
        try:
            result = router.complete(NodeRole.RESEARCHER, prompt, state=state)
        except ChainExhaustedError as exc:
            state.halt_reason = f"argus (skeleton): {exc}"
            return state

        builder.append_turn("user", current_turn)
        builder.append_turn("assistant", result.text)

        skeleton_path = paths.workspace / f"skeleton_{skeleton_call_index}.txt"
        skeleton_path.write_text(result.text.strip())
        publish_skeleton(
            Handle.for_file(skeleton_path, summary="Argus's repo skeleton for planning")
        )
        return state

    def research_node(state: AgentState) -> AgentState:
        nonlocal research_call_index
        research_call_index += 1
        # Deliberately NOT `state.iterations_this_task += 1`: CB-1's per-task
        # cap (PRD S7.3) bounds WRITE attempts (Daedalus, Chiron) — Argus
        # never writes, only looks, the same reason Talos's own evaluation
        # calls never increment this counter either.

        is_l2 = state.ladder_rung == _L2_RUNG
        assert state.plan is not None, "argus research only runs once a plan exists"
        plan = read_plan(state.plan)
        task = current_task(plan, state.task_cursor)

        builder.set_codebase_map(
            build_codebase_map(worktree, max_content_bytes=0), require_fresh=False
        )
        lines = [
            ARGUS_RESEARCH_PREFIX,
            f"Current task: {task.description}",
            f"Acceptance: {task.acceptance}",
        ]
        if is_l2:
            report = state.eval_report
            assert report is not None, "L2 only happens after a task has failed"
            lines.append(ARGUS_L2_PREFIX)
            if report.first_failure:
                lines.append(f"Prior failure detail: {report.first_failure}")
        current_turn = "\n".join(lines)

        try:
            output = complete_with_format_retry(
                router=router,
                builder=builder,
                node=NodeRole.RESEARCHER,
                state=state,
                current_turn=current_turn,
                correction=ARGUS_FORMAT_CORRECTION,
                parse=parse_argus_research_output,
            )
        except ChainExhaustedError as exc:
            state.halt_reason = f"argus (research): {exc}"
            return state

        if output.files:
            existing = {_rel_key(h.path, worktree): h for h in state.context_handles}
            for ref in output.files:
                target = worktree / ref.path
                try:
                    resolved = target.resolve()
                    resolved.relative_to(worktree.resolve())
                except ValueError:
                    continue  # hallucinated a path outside the worktree; skip, don't halt
                if not resolved.is_file():
                    continue  # hallucinated a path that does not exist; skip, don't halt
                existing[ref.path] = Handle.for_file(target, summary=ref.reason)
            state.context_handles = list(existing.values())

        return state

    return skeleton_node, research_node


def _rel_key(path: Path, worktree: Path) -> str:
    try:
        return path.relative_to(worktree).as_posix()
    except ValueError:
        return path.as_posix()
