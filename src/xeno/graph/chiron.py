"""The debugger node (PRD S10): diagnoses one specific failure and applies a
targeted patch, or declines when it cannot form a hypothesis.

Distinct from Daedalus even when bound to the same underlying model (PRD
S10): Chiron's context is failure-shaped (a diff, a critical error span,
what was already tried) rather than task-shaped, and merging the two
reintroduces the "rewrite the whole file every time" failure mode a targeted
patcher exists to avoid.
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
    CHIRON_FORMAT_CORRECTION,
    CHIRON_SYSTEM,
    parse_chiron_output,
)
from xeno.graph.testfiles import is_test_file
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router


def _build_current_turn(state: AgentState) -> str:
    report = state.eval_report
    assert report is not None, "chiron is only ever entered after a Talos failure"

    lines = [
        f"Task: {state.goal}",
        "",
        f"Evaluation failed: failed_command={report.failed_command!r}",
    ]
    if report.first_failure:
        lines.append(f"Critical failure detail: {report.first_failure}")
    lines.append(
        "Diagnose the specific cause and apply the smallest patch that fixes it. Do "
        "not rewrite files wholesale — patch only the file(s) that need to change."
    )
    return "\n".join(lines)


def make_chiron_node(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    touched_files: list[Path],
    report_declined: Callable[[bool, str], None],
) -> Callable[[AgentState], AgentState]:
    """Build the Chiron node.

    `report_declined` mirrors Talos's `intervention` callable in reverse:
    Chiron is the one deciding whether this call counts as a MODEL_AUTHORED
    intervention or a DECLINED one for CB-4 (PRD S7.3), and the caller in
    `xeno.graph.build` needs that decision after the node returns — a
    LangGraph node can only return `AgentState`, so this is how the extra bit
    gets out.
    """
    builder = PromptBuilder(
        node=NodeRole.DEBUGGER,
        keyring=keyring,
        system_text=CHIRON_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    call_index = 0

    def node(state: AgentState) -> AgentState:
        nonlocal call_index
        call_index += 1
        # CB-1's per-task cap (PRD S7.3) is a backstop above RUNG_BUDGETS,
        # not a Daedalus-specific counter — every attempt at the task,
        # including each L1 patch, must count against it, or a task stuck
        # oscillating through Chiron alone could never trip it.
        state.iterations_this_task += 1

        # PRD S9.6.5: refreshed immediately before build(), never stale.
        focus = [h.path for h in state.context_handles] or None
        builder.set_codebase_map(build_codebase_map(worktree, focus=focus), require_fresh=False)

        current_turn = _build_current_turn(state)

        try:
            output = complete_with_format_retry(
                router=router,
                builder=builder,
                node=NodeRole.DEBUGGER,
                state=state,
                current_turn=current_turn,
                correction=CHIRON_FORMAT_CORRECTION,
                parse=parse_chiron_output,
            )
        except ChainExhaustedError as exc:
            state.halt_reason = f"chiron: {exc}"
            return state

        if output.declined:
            report_declined(True, output.decline_reason)
            return state

        test_hits = [b.path for b in output.files if is_test_file(Path(b.path).as_posix())]
        if test_hits:
            # PRD S10 hard rule: a patch touching a test file is rejected
            # outright, not partially applied — refusing the write is the
            # enforcement, not a note for the reviewer to weigh later.
            report_declined(
                True, f"patch touched test file(s), refused: {', '.join(test_hits)}"
            )
            return state

        try:
            written, diff_text = write_file_blocks(worktree, output.files)
        except WorktreeEscape as exc:
            state.halt_reason = f"chiron attempted to write outside the worktree: {exc.path}"
            return state

        keyring.mark_worktree_written(written, reason=f"chiron call {call_index}")
        # Accumulate, not replace: Talos's lint/type scope must cover every
        # file touched across the whole task, including Daedalus's original
        # write, or a regression Chiron introduces in a file it does not
        # revisit could go unchecked on a later call.
        for path in written:
            if path not in touched_files:
                touched_files.append(path)

        diff_path = paths.workspace / f"chiron_diff_{call_index}.patch"
        diff_path.write_text(diff_text)
        state.diff_handle = Handle.for_file(
            diff_path,
            summary=f"chiron call {call_index}: {', '.join(b.path for b in output.files)}",
        )

        report_declined(False, "")
        return state

    return node
