"""The Phase 2 graph: Daedalus -> Talos -> Chiron, sandboxed, with a bounded
L0/L1 failure loop (PRD S7.2, S13 Phase 2).

L2 (re-research), L3 (rollback + rewrite), L4 (re-plan), and L5 (halt to
Cerberus) all name nodes or substrates — Argus, a checkpoint mechanism,
Odysseus, Cerberus — that do not exist until Phase 3/4 (PRD S13: "Debugger
(Chiron) node; ladder rungs L0 and L1" is the whole of Phase 2's ladder
scope). When Chiron's L1 budget is exhausted, the run halts and reports
rather than escalating further. This is the same kind of phase-scoped
substitution Phase 1 made when it stood a rewrite loop in for L1 before
Chiron existed — just one rung later now that Chiron is real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from xeno.adapters.base import LanguageAdapter
from xeno.core.breakers import (
    BreakerPanel,
    Intervention,
    check_destructive_action,
    record_diff,
)
from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState
from xeno.core.types import RUNG_BUDGETS
from xeno.graph.chiron import make_chiron_node
from xeno.graph.daedalus import make_daedalus_node
from xeno.graph.talos import make_talos_node
from xeno.prompt.keys import DEFAULT_IGNORES, CacheKeyring
from xeno.router.router import Router
from xeno.sandbox.pool import WarmPool

_L0_RUNG = 0
_L1_RUNG = 1

_RouteAfterTalos = Literal["retry_eval", "patch", "done", "halt"]
_RouteAfterWrite = Literal["evaluate", "halt"]


class _LoopContext:
    """Bookkeeping shared by reference between the write nodes, the Talos
    node, and the routing functions — the only things that need to agree on
    "what just happened" (PRD S7.3)."""

    def __init__(self, worktree: Path) -> None:
        self.intervention = Intervention.NONE
        self.chiron_declined = False
        self.chiron_decline_reason = ""
        #: For CB-6's illegal-removal check: what the worktree looked like
        #: after the previous write, and which paths were newly created
        #: SINCE THE START of this run (not merely since the last write).
        self.last_snapshot = _snapshot(worktree)
        self.created_this_run: set[str] = set()

    def set_chiron_result(self, declined: bool, reason: str) -> None:
        self.chiron_declined = declined
        self.chiron_decline_reason = reason


def build_graph(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    runlog: RunLog,
    pool: WarmPool,
    adapter: LanguageAdapter,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Compile the three-node graph."""
    touched_files: list[Path] = []
    ctx = _LoopContext(worktree)
    breaker_panel = BreakerPanel(config.limits)

    daedalus = make_daedalus_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        touched_files=touched_files,
    )
    chiron = make_chiron_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        touched_files=touched_files,
        report_declined=ctx.set_chiron_result,
    )
    talos = make_talos_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        touched_files=touched_files,
        pool=pool,
        adapter=adapter,
        breaker_panel=breaker_panel,
        runlog=runlog,
        intervention=lambda: ctx.intervention,
    )

    def _after_write(state: AgentState) -> None:
        """Bookkeeping that must happen after ANY write (Daedalus or
        Chiron) and before Talos ever sees it: diff history for CB-5, and
        the destructive-action guard for CB-6, which fires "regardless of
        test status" (PRD S7.3, S7.4) — i.e. it cannot wait for the next
        polled-breaker check inside `_advance_ladder`, which only runs after
        Talos has already evaluated.

        Callers must only invoke this when a write actually happened — a
        Chiron decline leaves `state.diff_handle` pointing at whatever the
        LAST real write produced, and re-running this against that same
        handle would record a spurious repeat in `diff_history` (a false
        CB-5 trip) for a call that changed nothing.
        """
        assert state.diff_handle is not None, "caller must only invoke this after a real write"
        record_diff(state, state.diff_handle.sha256)

        after = _snapshot(worktree)
        removed_now = ctx.last_snapshot - after
        created_now = after - ctx.last_snapshot
        illegal = sorted(removed_now - ctx.created_this_run)
        ctx.created_this_run |= created_now
        ctx.last_snapshot = after

        diff_text = state.diff_handle.path.read_text()
        verdict = check_destructive_action(
            diff_text,
            removed_files=illegal,
            created_this_run=sorted(ctx.created_this_run),
            limits=config.limits,
        )
        if verdict is not None:
            trip = breaker_panel.trip(state, verdict)
            runlog.event(EventKind.BREAKER_FIRED, code=trip.code.value, detail=trip.detail)

    def daedalus_step(state: AgentState) -> AgentState:
        runlog.event(EventKind.NODE_ENTER, node="daedalus")
        state = daedalus(state)
        if not state.halted:
            _after_write(state)
        if not state.halted:
            ctx.intervention = Intervention.MODEL_AUTHORED
        runlog.event(EventKind.NODE_EXIT, node="daedalus", halted=state.halted)
        return state

    def chiron_step(state: AgentState) -> AgentState:
        runlog.event(EventKind.NODE_ENTER, node="chiron")
        state = chiron(state)
        if not state.halted:
            if ctx.chiron_declined:
                ctx.intervention = Intervention.DECLINED
            else:
                _after_write(state)
                if not state.halted:
                    ctx.intervention = Intervention.MODEL_AUTHORED
        runlog.event(
            EventKind.NODE_EXIT,
            node="chiron",
            halted=state.halted,
            declined=ctx.chiron_declined,
            decline_reason=ctx.chiron_decline_reason,
        )
        return state

    def talos_step(state: AgentState) -> AgentState:
        # Every mutation the ladder decision needs (ladder_rung, rung_attempts,
        # halt_reason, breaker trips) happens HERE, inside a real node, and
        # never in a conditional-edge function. LangGraph reconstructs state
        # from each node's RETURN value between steps — a routing function
        # only decides an edge key, and mutating `state` inside one is a
        # no-op silently discarded on the next hop.
        runlog.event(EventKind.NODE_ENTER, node="talos")
        state = talos(state)
        passed = state.eval_report.passed if state.eval_report else None
        runlog.event(EventKind.NODE_EXIT, node="talos", passed=passed)
        if not passed:
            _advance_ladder(state)
        return state

    def _advance_ladder(state: AgentState) -> None:
        if state.halted:
            # CB-4 (PRD S7.3) already halted from inside talos() itself —
            # event-driven, not polled here (see make_talos_node's
            # docstring) — so there is nothing left for this poll to decide.
            # Checked first so a coincidentally-exhausted rung budget can
            # never overwrite CB-4's own halt_reason with a different one.
            return

        report = state.eval_report
        assert report is not None, "talos always sets eval_report before returning"

        breach = breaker_panel.check(state)
        if breach is not None:
            trip = breaker_panel.trip(state, breach)
            runlog.event(EventKind.BREAKER_FIRED, code=trip.code.value, detail=trip.detail)
            return

        if report.infrastructure_failure:
            # PRD S10: infra failures get one L0 retry, then straight to
            # ESCALATE — never a patch, since there is no code defect for
            # Chiron to fix. Phase 2 has no Cerberus, so ESCALATE == halt.
            if state.ladder_rung == _L0_RUNG and state.rung_attempts < RUNG_BUDGETS[_L0_RUNG]:
                state.rung_attempts += 1
                ctx.intervention = Intervention.NONE
                runlog.event(EventKind.LADDER_ADVANCE, rung="L0", reason="infra retry")
                return
            state.halt_reason = (
                f"talos infrastructure failure: {report.first_failure or 'see full_log_handle'}"
            )
            return

        if state.ladder_rung == _L0_RUNG:
            if state.rung_attempts < RUNG_BUDGETS[_L0_RUNG]:
                state.rung_attempts += 1
                ctx.intervention = Intervention.NONE
                runlog.event(EventKind.LADDER_ADVANCE, rung="L0")
                return
            state.ladder_rung = _L1_RUNG
            state.rung_attempts = 1
            runlog.event(EventKind.LADDER_ADVANCE, rung="L1")
            return

        if state.ladder_rung == _L1_RUNG:
            if state.rung_attempts < RUNG_BUDGETS[_L1_RUNG]:
                state.rung_attempts += 1
                runlog.event(EventKind.LADDER_ADVANCE, rung="L1", attempt=state.rung_attempts)
                return
            state.halt_reason = (
                f"failure signature {state.failure_signature} survived the L1 patch "
                f"budget ({RUNG_BUDGETS[_L1_RUNG]} attempts); Phase 2 has no Argus to "
                "hand off to for L2 (PRD S13)"
            )
            return

        # Unreachable: no code path in this module sets ladder_rung outside
        # {0, 1}. Guarded rather than assumed, so a future edit that adds a
        # third rung fails loudly here instead of looping silently.
        state.halt_reason = f"unhandled ladder rung {state.ladder_rung}"

    def route_after_write(state: AgentState) -> _RouteAfterWrite:
        """Pure read of state the write step already committed. No mutation."""
        return "halt" if state.halted else "evaluate"

    def route_after_talos(state: AgentState) -> _RouteAfterTalos:
        """Pure read of state `talos_step` already committed. No mutation."""
        if state.halted:
            return "halt"
        report = state.eval_report
        assert report is not None, "talos always sets eval_report before returning"
        if report.passed:
            return "done"
        return "retry_eval" if state.ladder_rung == _L0_RUNG else "patch"

    graph = StateGraph(AgentState)
    graph.add_node("daedalus", daedalus_step)
    graph.add_node("chiron", chiron_step)
    graph.add_node("talos", talos_step)
    graph.add_edge(START, "daedalus")
    graph.add_conditional_edges("daedalus", route_after_write, {"evaluate": "talos", "halt": END})
    graph.add_conditional_edges("chiron", route_after_write, {"evaluate": "talos", "halt": END})
    graph.add_conditional_edges(
        "talos",
        route_after_talos,
        {"retry_eval": "talos", "patch": "chiron", "done": END, "halt": END},
    )
    return graph.compile()


def run_graph(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    runlog: RunLog,
    state: AgentState,
    pool: WarmPool,
    adapter: LanguageAdapter,
) -> AgentState:
    """Compile and run the graph to completion, returning the final state."""
    compiled = build_graph(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        runlog=runlog,
        pool=pool,
        adapter=adapter,
    )
    # Bounded independently of CB-1: this is LangGraph's own step counter, a
    # backstop against a routing bug looping forever rather than a tuning
    # knob — CB-1's much lower per-task/per-run caps fire first in practice.
    result = compiled.invoke(state, config={"recursion_limit": 200})
    return result if isinstance(result, AgentState) else AgentState.model_validate(result)


def _snapshot(worktree: Path) -> set[str]:
    """Relative posix paths of every real file in the worktree, for CB-6's
    illegal-removal check. Cheap enough for the suite's small fixtures;
    revisit if a much larger worktree makes a full walk per write
    noticeable."""
    snapshot: set[str] = set()
    for path in worktree.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(worktree)
        if any(part in DEFAULT_IGNORES for part in rel.parts):
            continue
        snapshot.add(rel.as_posix())
    return snapshot
