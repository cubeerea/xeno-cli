"""The full eight-node graph (PRD S13 Phase 3-4): Argus (skeleton) ->
Odysseus (plan) -> [per task: Argus (research) -> Daedalus -> Talos ->
escalation ladder] -> Cerberus -> done or reject-and-return.

The escalation ladder (PRD S7.2) is monotonic within a task
(`AgentState.ladder_rung`'s docstring): L0 (re-run evaluation) -> L1 (Chiron
patches, budget 3) -> L2 (Argus re-researches, then Chiron patches again,
budget 2) -> L3 (roll back to the last checkpoint, Daedalus rewrites from
scratch, budget 2) -> L4 (Odysseus re-plans the stuck task, then a single
research-and-rewrite attempt) -> L5 (halt).

Cerberus is the sole human gate (PRD S8.1): every "halt" edge label in this
graph, and `route_after_talos`'s "done" label, target the Cerberus node
rather than `END` — "every node's failure mode routes to Cerberus, never to
the user." Cerberus itself resolves to APPROVE/ESCALATE (both terminal,
routed to `END`; the actual human interaction happens in `xeno.cli` after
`run_graph` returns) or REJECT_AND_RETURN (non-terminal, PRD S8.3: routes
back to Daedalus or Odysseus with written objections, budgeted separately
from the ladder via `AgentState.reject_count`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from xeno.adapters.base import LanguageAdapter
from xeno.core import vcs
from xeno.core.breakers import (
    BreakerPanel,
    Intervention,
    check_destructive_action,
    record_diff,
)
from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths, short_run_id, slugify
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState, Handle
from xeno.core.types import RUNG_BUDGETS, LadderRung, NodeRole, Verdict
from xeno.graph.argus import make_argus_nodes
from xeno.graph.cerberus import make_cerberus_node
from xeno.graph.checkpoints import checkpoint_step, rollback_step
from xeno.graph.chiron import make_chiron_node
from xeno.graph.daedalus import make_daedalus_node
from xeno.graph.odysseus import make_odysseus_node
from xeno.graph.talos import make_talos_node
from xeno.prompt.keys import DEFAULT_IGNORES, CacheKeyring
from xeno.router.router import Router
from xeno.sandbox.pool import WarmPool

_L0_RUNG = 0
_L1_RUNG = 1
_L2_RUNG = 2
_L3_RUNG = 3
_L4_RUNG = 4
_MAX_BUDGETED_RUNG = max(RUNG_BUDGETS)  # 4: L4 is the last rung with a budget; beyond it is L5/halt
_RUNG_LABELS = tuple(rung.value for rung in LadderRung)  # index n -> "L{n}"

_RouteAfterTalos = Literal[
    "retry_eval", "patch", "research_l2", "rollback_l3", "replan_l4", "next_task", "done", "halt"
]
_RouteAfterWrite = Literal["evaluate", "halt"]
_RouteAfterOdysseus = Literal["research", "halt"]
_RouteAfterArgusSkeleton = Literal["plan", "halt"]
_RouteAfterArgusResearch = Literal["daedalus", "chiron", "halt"]
_RouteAfterCerberus = Literal["reject_daedalus", "reject_odysseus", "human"]

_ROUTE_BY_RUNG: dict[int, _RouteAfterTalos] = {
    _L0_RUNG: "retry_eval",
    _L1_RUNG: "patch",
    _L2_RUNG: "research_l2",
    _L3_RUNG: "rollback_l3",
    _L4_RUNG: "replan_l4",
}


class _LoopContext:
    """Bookkeeping shared by reference between nodes and the routing
    functions — the only things that need to agree on "what just happened"
    (PRD S7.3) or "what did Argus find" (PRD S13 Phase 3)."""

    def __init__(self, worktree: Path) -> None:
        self.intervention = Intervention.NONE
        self.chiron_declined = False
        self.chiron_decline_reason = ""
        #: For CB-6's illegal-removal check: what the worktree looked like
        #: after the previous write, and which paths were newly created
        #: SINCE THE START of this run (not merely since the last write).
        self.last_snapshot = _snapshot(worktree)
        self.created_this_run: set[str] = set()
        #: Argus's most recent repo-skeleton Handle. Odysseus reads this by
        #: reference (`xeno.graph.odysseus`'s docstring explains why it is
        #: not an `AgentState` field): a one-shot planning aid, not
        #: something any other node ever needs again.
        self.skeleton_handle: Handle | None = None

    def set_chiron_result(self, declined: bool, reason: str) -> None:
        self.chiron_declined = declined
        self.chiron_decline_reason = reason

    def set_skeleton(self, handle: Handle) -> None:
        self.skeleton_handle = handle

    def get_skeleton(self) -> Handle | None:
        return self.skeleton_handle

    def resync_after_rollback(self, worktree: Path) -> None:
        """L3 (PRD S7.2) discards worktree state out from under this
        context's bookkeeping via `git reset --hard` + `git clean -fd` — a
        mutation the write nodes did not make. `_after_write`'s CB-6
        accounting (PRD S7.4) is otherwise keyed off whatever it last saw,
        so without this it keeps comparing against a pre-rollback snapshot
        for the rest of the task: harmless on its own (the illegal-removal
        check already treats anything upstream `created_this_run` as fair
        game to disappear), but it lets staleness accumulate across L3's
        multiple rollback attempts for no reason. Resetting here keeps the
        bookkeeping an honest reflection of the actual worktree."""
        self.last_snapshot = _snapshot(worktree)
        self.created_this_run &= self.last_snapshot


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
    goal: str,
    repo_root: Path,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Compile the eight-node graph."""
    touched_files: list[Path] = []
    ctx = _LoopContext(worktree)
    breaker_panel = BreakerPanel(config.limits)

    #: PRD S13 Phase 3's minimal git substrate: one commit of the run's
    #: starting state, so L3's first rollback (before any checkpoint exists
    #: yet) has a target — and also the squash boundary Cerberus/the CLI
    #: diff and squash against (PRD S8.4).
    initial_sha = vcs.init_repo(worktree)

    #: PRD S8.4: every run operates on its own dedicated branch. Created
    #: right after `init_repo` so the whole run, including any L3 rollback,
    #: happens on it rather than a detached/default one.
    branch = f"{config.git.branch_prefix}{slugify(goal)}-{short_run_id(paths.run_id)}"
    vcs.create_branch(worktree, branch)
    if config.git.open_pr:
        vcs.inherit_origin_remote(worktree, repo_root)

    argus_skeleton, argus_research = make_argus_nodes(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        publish_skeleton=ctx.set_skeleton,
    )
    odysseus = make_odysseus_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        skeleton=ctx.get_skeleton,
    )
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
    cerberus = make_cerberus_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        initial_sha=initial_sha,
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

    def _node_step(name: str, fn: Any) -> Any:
        """Wrap a node with the enter/exit logging every node gets — the
        deterministic steps (rollback) and every model-calling node share
        this, so it is factored out once rather than repeated per node."""

        def step(state: AgentState) -> AgentState:
            runlog.event(EventKind.NODE_ENTER, node=name)
            state = fn(state)
            runlog.event(EventKind.NODE_EXIT, node=name, halted=state.halted)
            return state

        return step

    argus_skeleton_step = _node_step("argus_skeleton", argus_skeleton)
    odysseus_step = _node_step("odysseus", odysseus)
    argus_research_step = _node_step("argus_research", argus_research)

    def cerberus_step(state: AgentState) -> AgentState:
        runlog.event(EventKind.NODE_ENTER, node="cerberus")
        state = cerberus(state)
        runlog.event(
            EventKind.VERDICT,
            verdict=state.review_verdict.value if state.review_verdict else None,
            reject_count=state.reject_count,
        )
        runlog.event(EventKind.NODE_EXIT, node="cerberus", halted=state.halted)
        return state

    def rollback_node(state: AgentState) -> AgentState:
        runlog.event(EventKind.NODE_ENTER, node="rollback")
        rollback_step(state, worktree, initial_sha=initial_sha)
        ctx.resync_after_rollback(worktree)
        runlog.event(EventKind.NODE_EXIT, node="rollback")
        return state

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
        if passed:
            checkpoint_step(state, worktree)
            runlog.event(
                EventKind.CHECKPOINT,
                task_index=state.checkpoints[-1].task_index,
                sha=state.checkpoints[-1].sha,
                task_cursor=state.task_cursor,
                task_count=state.task_count,
            )
        elif not state.halted:
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
            # ESCALATE — never a patch/re-research/rewrite, since there is
            # no CODE defect for any of Argus/Chiron/Daedalus/Odysseus to
            # act on. Phase 3 still has no Cerberus (PRD S13 Phase 4), so
            # ESCALATE == halt regardless of how many ladder rungs exist.
            if state.ladder_rung == _L0_RUNG and state.rung_attempts < RUNG_BUDGETS[_L0_RUNG]:
                state.rung_attempts += 1
                ctx.intervention = Intervention.NONE
                runlog.event(EventKind.LADDER_ADVANCE, rung="L0", reason="infra retry")
                return
            state.halt_reason = (
                f"talos infrastructure failure: {report.first_failure or 'see full_log_handle'}"
            )
            return

        rung = state.ladder_rung
        if state.rung_attempts < RUNG_BUDGETS[rung]:
            state.rung_attempts += 1
            if rung == _L0_RUNG:
                ctx.intervention = Intervention.NONE
            runlog.event(
                EventKind.LADDER_ADVANCE, rung=_RUNG_LABELS[rung], attempt=state.rung_attempts
            )
            return

        if rung >= _MAX_BUDGETED_RUNG:
            # Monotonic ladder (PRD, `AgentState.ladder_rung`'s docstring):
            # L4's one re-plan attempt is already spent by having reached
            # here at all. No re-descent through L0-L3 with fresh budgets —
            # that would make the breaker guarantee of bounded attempts
            # meaningless. Straight to L5.
            state.halt_reason = (
                f"failure signature {state.failure_signature} survived every ladder rung "
                f"through {_RUNG_LABELS[rung]} (PRD S7.2); halting at L5"
            )
            return

        state.ladder_rung = rung + 1
        state.rung_attempts = 1
        runlog.event(EventKind.LADDER_ADVANCE, rung=_RUNG_LABELS[state.ladder_rung])

    def route_after_write(state: AgentState) -> _RouteAfterWrite:
        """Pure read of state the write step already committed. No mutation."""
        return "halt" if state.halted else "evaluate"

    def route_after_odysseus(state: AgentState) -> _RouteAfterOdysseus:
        return "halt" if state.halted else "research"

    def route_after_argus_skeleton(state: AgentState) -> _RouteAfterArgusSkeleton:
        return "halt" if state.halted else "plan"

    def route_after_argus_research(state: AgentState) -> _RouteAfterArgusResearch:
        """L2 (PRD S7.2) is the only rung where Argus hands off to Chiron
        rather than Daedalus — a fresh task (rung 0) and an L4 re-plan
        (rung 4, `xeno.graph.odysseus` already turned it back into a
        from-scratch write) both want a full Daedalus write."""
        if state.halted:
            return "halt"
        return "chiron" if state.ladder_rung == _L2_RUNG else "daedalus"

    def route_after_talos(state: AgentState) -> _RouteAfterTalos:
        """Pure read of state `talos_step` already committed. No mutation."""
        if state.halted:
            return "halt"
        report = state.eval_report
        assert report is not None, "talos always sets eval_report before returning"
        if report.passed:
            return "next_task" if state.task_cursor < state.task_count else "done"
        return _ROUTE_BY_RUNG[state.ladder_rung]

    def route_after_cerberus(state: AgentState) -> _RouteAfterCerberus:
        """Cerberus's own node always resolves to one of the three verdicts
        before returning — unlike every other route function here, this does
        not check `state.halted` first, because the review IS the
        resolution of any halt, not a pass-through of one (PRD S8.3)."""
        if state.review_verdict is Verdict.REJECT_AND_RETURN:
            if state.reject_destination is NodeRole.CODER:
                return "reject_daedalus"
            return "reject_odysseus"
        return "human"

    graph = StateGraph(AgentState)
    graph.add_node("argus_skeleton", argus_skeleton_step)
    graph.add_node("odysseus", odysseus_step)
    graph.add_node("argus_research", argus_research_step)
    graph.add_node("daedalus", daedalus_step)
    graph.add_node("chiron", chiron_step)
    graph.add_node("rollback", rollback_node)
    graph.add_node("talos", talos_step)
    graph.add_node("cerberus", cerberus_step)

    graph.add_edge(START, "argus_skeleton")
    graph.add_conditional_edges(
        "argus_skeleton", route_after_argus_skeleton, {"plan": "odysseus", "halt": "cerberus"}
    )
    graph.add_conditional_edges(
        "odysseus", route_after_odysseus, {"research": "argus_research", "halt": "cerberus"}
    )
    graph.add_conditional_edges(
        "argus_research",
        route_after_argus_research,
        {"daedalus": "daedalus", "chiron": "chiron", "halt": "cerberus"},
    )
    graph.add_conditional_edges(
        "daedalus", route_after_write, {"evaluate": "talos", "halt": "cerberus"}
    )
    graph.add_conditional_edges(
        "chiron", route_after_write, {"evaluate": "talos", "halt": "cerberus"}
    )
    graph.add_edge("rollback", "daedalus")
    graph.add_conditional_edges(
        "talos",
        route_after_talos,
        {
            "retry_eval": "talos",
            "patch": "chiron",
            "research_l2": "argus_research",
            "rollback_l3": "rollback",
            "replan_l4": "odysseus",
            "next_task": "argus_research",
            "done": "cerberus",
            "halt": "cerberus",
        },
    )
    graph.add_conditional_edges(
        "cerberus",
        route_after_cerberus,
        {"reject_daedalus": "argus_research", "reject_odysseus": "odysseus", "human": END},
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
    repo_root: Path,
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
        goal=state.goal,
        repo_root=repo_root,
    )
    # Bounded independently of CB-1: this is LangGraph's own step counter, a
    # backstop against a routing bug looping forever rather than a tuning
    # knob — CB-1's much lower per-task/per-run caps fire first in practice.
    result = compiled.invoke(state, config={"recursion_limit": 1000})
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
