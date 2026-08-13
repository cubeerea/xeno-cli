"""The full graph (PRD S13 Phase 3-4).

Two nested loops. The outer one is the roadmap:

    Argus (skeleton) -> Odysseus (roadmap)
      -> per milestone: Lachesis (expand) -> [inner loop] -> Lachesis (verify)
      -> Cerberus -> done or reject-and-return

and the inner one is the plan Lachesis just expanded:

    per task: Argus (research) -> Daedalus -> Talos -> escalation ladder

The nesting exists because of WHEN things are knowable. Odysseus runs before
any code exists, so it can only write coarse milestones; Lachesis runs
immediately before each milestone, with the previous ones' code on disk, so
it can write tasks that name real modules and functions. It then closes the
milestone by writing that milestone's tests — the only node permitted to —
and Talos runs the full gate set including the test command for the first
time (`xeno.core.types.GateProfile`). Implementation tasks are gated on
everything EXCEPT tests, because a task cannot be checked against tests
describing code it has not written yet.

The escalation ladder (PRD S7.2) is monotonic within a task
(`AgentState.ladder_rung`'s docstring): L0 (re-run evaluation) -> L1 (Chiron
patches, budget 3) -> L2 (Argus re-researches, then Chiron patches again,
budget 2) -> L3 (roll back to the last checkpoint and rewrite from scratch,
budget 2) -> L4 (Lachesis re-expands the stuck task, then a single
research-and-rewrite attempt) -> L5 (halt). L4 re-expands rather than
re-plans because the task that got stuck is one Lachesis wrote; the roadmap
above it is very rarely what was wrong.

The ladder is about code that was written and failed. A milestone that could
not be turned into tasks at all never reaches it: Lachesis objects, and the
"replan" edge sends the objection back to Odysseus, who rewrites that
milestone and everything after it (`MAX_PLAN_OBJECTIONS` bounds the cycle,
and exhausting it halts to Cerberus like everything else). Together with a
reviewer rejection those are the only two ways Odysseus is re-entered, and
both arrive as another node's written objection over the milestones already
built.

Because Chiron may never touch a test file and Lachesis may write nothing
else, a failing verification gate cannot be resolved by weakening the check:
the ladder has no path to the test, so it is forced to fix the code.

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

from xeno.core import vcs
from xeno.core.breakers import (
    BreakerPanel,
    Intervention,
    check_destructive_action,
    record_diff,
)
from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths, run_branch_name
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState, EvalReport, Handle
from xeno.core.types import RUNG_BUDGETS, GateProfile, LadderRung, NodeRole, Verdict
from xeno.graph.argus import make_argus_nodes
from xeno.graph.cerberus import make_cerberus_node
from xeno.graph.checkpoints import CheckpointKind, checkpoint_step, rollback_step
from xeno.graph.chiron import make_chiron_node
from xeno.graph.daedalus import make_daedalus_node
from xeno.graph.lachesis import MAX_PLAN_OBJECTIONS, make_lachesis_nodes
from xeno.graph.law import ProjectLaw
from xeno.graph.odysseus import make_odysseus_node
from xeno.graph.talos import make_talos_node
from xeno.graph.toolchain import ToolchainSession
from xeno.prompt.keys import DEFAULT_IGNORES, CacheKeyring
from xeno.router.router import Router

_L0_RUNG = 0
_L1_RUNG = 1
_L2_RUNG = 2
_L3_RUNG = 3
_L4_RUNG = 4
_MAX_BUDGETED_RUNG = max(RUNG_BUDGETS)  # 4: L4 is the last rung with a budget; beyond it is L5/halt
_RUNG_LABELS = tuple(rung.value for rung in LadderRung)  # index n -> "L{n}"

#: How many times a write may be refused for touching a test file before the
#: run halts. A refusal loops straight back to the writer with the reason
#: attached, and nothing about it advances the ladder or a breaker's counters
#: — so without a bound of its own, a model that will not stop bundling tests
#: would loop forever.
MAX_WRITE_REFUSALS = 3

_RouteAfterTalos = Literal[
    "retry_eval",
    "patch",
    "research_l2",
    "rollback_l3",
    "reexpand_l4",
    "next_task",
    "verify",
    "next_milestone",
    "done",
    "halt",
]
_RouteAfterWrite = Literal["evaluate", "retry", "halt"]
_RouteAfterOdysseus = Literal["expand", "halt"]
_RouteAfterExpand = Literal["research", "replan", "halt"]
_RouteAfterRollback = Literal["daedalus", "verify"]
_RouteAfterArgusSkeleton = Literal["plan", "halt"]
_RouteAfterArgusResearch = Literal["daedalus", "chiron", "halt"]
_RouteAfterCerberus = Literal["reject_daedalus", "reject_odysseus", "human"]

_ROUTE_BY_RUNG: dict[int, _RouteAfterTalos] = {
    _L0_RUNG: "retry_eval",
    _L1_RUNG: "patch",
    _L2_RUNG: "research_l2",
    _L3_RUNG: "rollback_l3",
    _L4_RUNG: "reexpand_l4",
}


class _LoopContext:
    """Bookkeeping shared by reference between nodes and the routing
    functions — the only things that need to agree on "what just happened"
    (PRD S7.3) or "what did Argus find" (PRD S13 Phase 3)."""

    def __init__(self, worktree: Path) -> None:
        self.intervention = Intervention.NONE
        self.chiron_declined = False
        self.chiron_decline_reason = ""
        self.chiron_refusal = ""
        #: Daedalus bundled a test file into its response, so nothing was
        #: written and the call has to be retried rather than evaluated.
        self.daedalus_refused = False
        self.daedalus_refusal = ""
        self.daedalus_refusals = 0
        #: The same, for Lachesis writing outside the test tree.
        self.lachesis_refused = False
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
        # Only a REFUSED patch is worth replaying to Chiron. An ordinary
        # decline ("I cannot form a hypothesis") is Chiron's own reasoned
        # answer, and quoting it back would just invite it to repeat itself;
        # a refusal is the harness overruling a patch it actually wrote, and
        # is the one thing it otherwise has no way to find out about.
        self.chiron_refusal = reason if declined and reason.startswith("patch touched") else ""

    def take_refusal(self) -> str:
        """Read-and-clear: a refusal describes the immediately preceding
        attempt, so replaying it into a later, unrelated call would be
        misleading rather than helpful."""
        refusal, self.chiron_refusal = self.chiron_refusal, ""
        return refusal

    def set_daedalus_result(self, refused: bool, reason: str) -> None:
        self.daedalus_refused = refused
        if refused:
            self.daedalus_refusal = reason
            self.daedalus_refusals += 1

    def take_daedalus_refusal(self) -> str:
        refusal, self.daedalus_refusal = self.daedalus_refusal, ""
        return refusal

    def set_lachesis_result(self, refused: bool) -> None:
        self.lachesis_refused = refused

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
    session: ToolchainSession,
    goal: str,
    repo_root: Path,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Compile the eight-node graph."""
    touched_files: list[Path] = []
    #: Read from the real repo, not the worktree: memory.md is project
    #: state that outlives any single run, so it lives beside runs/ and
    #: worktrees/ rather than inside the disposable copy.
    law = ProjectLaw(repo_root=repo_root)
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
    branch = run_branch_name(config.git.branch_prefix, goal, paths.run_id)
    vcs.create_branch(worktree, branch)
    if config.git.open_pr:
        vcs.inherit_origin_remote(worktree, repo_root)

    argus_skeleton, argus_research = make_argus_nodes(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        law=law,
        publish_skeleton=ctx.set_skeleton,
    )
    odysseus = make_odysseus_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        law=law,
        skeleton=ctx.get_skeleton,
    )
    lachesis_expand, lachesis_verify = make_lachesis_nodes(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        law=law,
        touched_files=touched_files,
        toolchain_established=lambda: session.toolchain.established,
        report_refused=ctx.set_lachesis_result,
    )
    daedalus = make_daedalus_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        law=law,
        touched_files=touched_files,
        report_refused=ctx.set_daedalus_result,
        last_refusal=ctx.take_daedalus_refusal,
    )
    chiron = make_chiron_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        law=law,
        touched_files=touched_files,
        report_declined=ctx.set_chiron_result,
        last_refusal=ctx.take_refusal,
    )
    talos = make_talos_node(
        router=router,
        config=config,
        keyring=keyring,
        paths=paths,
        worktree=worktree,
        touched_files=touched_files,
        session=session,
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
        law=law,
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
            runlog.event(
                EventKind.NODE_EXIT,
                node=name,
                halted=state.halted,
                detail=_node_detail(name, state),
            )
            return state

        return step

    def _node_detail(name: str, state: AgentState) -> str:
        """A one-line summary of what a node actually produced.

        Lives here rather than inside each node because this is where the
        post-node state and `touched_files` are both in scope, so no node
        has to grow a reporting responsibility to make a run readable. Kept
        to facts already in state — a summary that could disagree with the
        run log would be worse than no summary.
        """
        if state.halted:
            return state.halt_reason or "halted"
        if name == "odysseus":
            return f"{state.milestone_count} milestone(s) mapped"
        if name == "lachesis_expand":
            if state.plan_objection:
                # Checked first: an objection produced no tasks, so the task
                # counts below still describe the PREVIOUS milestone and the
                # row would read as a successful expansion of this one.
                return (
                    f"objected to milestone {state.milestone_cursor + 1} "
                    f"({state.plan_objection_count}/{MAX_PLAN_OBJECTIONS}); back to odysseus"
                )
            remaining = state.task_count - state.milestone_task_start
            return f"milestone {state.milestone_cursor + 1}: {remaining} task(s)"
        if name in ("daedalus", "chiron", "lachesis_verify"):
            if not touched_files:
                return ""
            names = ", ".join(sorted(p.name for p in touched_files))
            return f"wrote {names}"
        if name == "argus_research":
            return f"{len(state.context_handles)} file(s) in context"
        return ""

    argus_skeleton_step = _node_step("argus_skeleton", argus_skeleton)
    odysseus_step = _node_step("odysseus", odysseus)
    argus_research_step = _node_step("argus_research", argus_research)
    lachesis_expand_step = _node_step("lachesis_expand", lachesis_expand)

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
        if not state.halted and not ctx.daedalus_refused:
            _after_write(state)
            if not state.halted:
                ctx.intervention = Intervention.MODEL_AUTHORED
        if ctx.daedalus_refused and ctx.daedalus_refusals >= MAX_WRITE_REFUSALS:
            state.halt_reason = (
                f"daedalus: {ctx.daedalus_refusals} responses in a row included a test file "
                f"(last: {ctx.daedalus_refusal or 'see log'}); tests are Lachesis's to write"
            )
        runlog.event(
            EventKind.NODE_EXIT,
            node="daedalus",
            halted=state.halted,
            declined=ctx.daedalus_refused,
            decline_reason=ctx.daedalus_refusal if ctx.daedalus_refused else "",
        )
        return state

    def lachesis_verify_step(state: AgentState) -> AgentState:
        runlog.event(EventKind.NODE_ENTER, node="lachesis_verify")
        state = lachesis_verify(state)
        if not state.halted and not ctx.lachesis_refused:
            _after_write(state)
            if not state.halted:
                # A milestone's tests are as much a model-authored change as
                # a patch is, and CB-4 has to see them that way: a failure
                # signature surviving a fresh test suite is exactly the kind
                # of no-progress loop it exists to stop.
                ctx.intervention = Intervention.MODEL_AUTHORED
        runlog.event(
            EventKind.NODE_EXIT,
            node="lachesis_verify",
            halted=state.halted,
            declined=ctx.lachesis_refused,
            detail=_node_detail("lachesis_verify", state) if not ctx.lachesis_refused else "",
        )
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
        report = state.eval_report
        passed = report.passed if report else None
        runlog.event(
            EventKind.NODE_EXIT,
            node="talos",
            passed=passed,
            failed_command=report.failed_command if report else "",
            detail=_talos_detail(report),
        )
        if passed:
            kind = checkpoint_step(state, worktree, router.ledger)
            runlog.event(
                EventKind.MILESTONE if kind is CheckpointKind.MILESTONE else EventKind.CHECKPOINT,
                # Not `kind=`: that is `RunLog.event`'s own first parameter.
                checkpoint_kind=kind.value,
                task_index=state.checkpoints[-1].task_index,
                sha=state.checkpoints[-1].sha,
                task_cursor=state.task_cursor,
                task_count=state.task_count,
                milestone_cursor=state.milestone_cursor,
                milestone_count=state.milestone_count,
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
            # act on. `halt_reason` routes to Cerberus, which reports the
            # decision deterministically rather than spending a review call.
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

    def route_after_daedalus(state: AgentState) -> _RouteAfterWrite:
        """A refused write must NOT fall through to an evaluation.

        Nothing was written, so the worktree still holds the previous task's
        green state — Talos would pass it and `checkpoint_step` would commit a
        task that implemented nothing. Looping back is the only safe edge, and
        `MAX_WRITE_REFUSALS` (enforced in `daedalus_step`) is what stops it
        being an infinite one.
        """
        if state.halted:
            return "halt"
        return "retry" if ctx.daedalus_refused else "evaluate"

    def route_after_lachesis_verify(state: AgentState) -> _RouteAfterWrite:
        """Same reasoning as `route_after_daedalus`, one level up: a refused
        verification wrote no tests, and evaluating anyway would pass the
        milestone on the strength of tests that do not exist."""
        if state.halted:
            return "halt"
        return "retry" if ctx.lachesis_refused else "evaluate"

    def route_after_odysseus(state: AgentState) -> _RouteAfterOdysseus:
        return "halt" if state.halted else "expand"

    def route_after_expand(state: AgentState) -> _RouteAfterExpand:
        """`halted` first, though the two cannot actually both be live:
        `_object_to_roadmap` leaves `plan_objection` unset on the branch that
        exhausts the budget, precisely so this does not have to encode which
        one wins."""
        if state.halted:
            return "halt"
        return "replan" if state.plan_objection else "research"

    def route_after_rollback(state: AgentState) -> _RouteAfterRollback:
        """L3 discards the current attempt; whoever authored it rewrites it.

        During a milestone's verification that is Lachesis, not Daedalus.
        Sending Daedalus instead would be actively wrong: the rollback has
        just deleted the milestone's tests, so the full gate run that follows
        would be evaluating source against a suite that no longer exists.
        """
        return "verify" if state.gate_profile is GateProfile.FULL else "daedalus"

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
            if state.gate_profile is GateProfile.IMPLEMENTATION:
                # Still inside a milestone. When its last task lands, the
                # milestone is built but nothing has RUN it yet — the test
                # command was held back from every one of those gate runs —
                # so the next stop is Lachesis writing the tests that do.
                return "next_task" if state.task_cursor < state.task_count else "verify"
            # A full gate run just passed, tests included. `checkpoint_step`
            # already advanced `milestone_cursor` past it.
            return "next_milestone" if state.milestone_cursor < state.milestone_count else "done"
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
    graph.add_node("lachesis_expand", lachesis_expand_step)
    graph.add_node("argus_research", argus_research_step)
    graph.add_node("daedalus", daedalus_step)
    graph.add_node("chiron", chiron_step)
    graph.add_node("rollback", rollback_node)
    graph.add_node("lachesis_verify", lachesis_verify_step)
    graph.add_node("talos", talos_step)
    graph.add_node("cerberus", cerberus_step)

    graph.add_edge(START, "argus_skeleton")
    graph.add_conditional_edges(
        "argus_skeleton", route_after_argus_skeleton, {"plan": "odysseus", "halt": "cerberus"}
    )
    graph.add_conditional_edges(
        "odysseus", route_after_odysseus, {"expand": "lachesis_expand", "halt": "cerberus"}
    )
    graph.add_conditional_edges(
        "lachesis_expand",
        route_after_expand,
        {"research": "argus_research", "replan": "odysseus", "halt": "cerberus"},
    )
    graph.add_conditional_edges(
        "argus_research",
        route_after_argus_research,
        {"daedalus": "daedalus", "chiron": "chiron", "halt": "cerberus"},
    )
    graph.add_conditional_edges(
        "daedalus",
        route_after_daedalus,
        {"evaluate": "talos", "retry": "daedalus", "halt": "cerberus"},
    )
    graph.add_conditional_edges(
        "chiron", route_after_write, {"evaluate": "talos", "halt": "cerberus"}
    )
    graph.add_conditional_edges(
        "lachesis_verify",
        route_after_lachesis_verify,
        {"evaluate": "talos", "retry": "lachesis_verify", "halt": "cerberus"},
    )
    graph.add_conditional_edges(
        "rollback", route_after_rollback, {"daedalus": "daedalus", "verify": "lachesis_verify"}
    )
    graph.add_conditional_edges(
        "talos",
        route_after_talos,
        {
            "retry_eval": "talos",
            "patch": "chiron",
            "research_l2": "argus_research",
            "rollback_l3": "rollback",
            "reexpand_l4": "lachesis_expand",
            "next_task": "argus_research",
            "verify": "lachesis_verify",
            "next_milestone": "lachesis_expand",
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
    session: ToolchainSession,
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
        session=session,
        goal=state.goal,
        repo_root=repo_root,
    )
    # Bounded independently of CB-1: this is LangGraph's own step counter, a
    # backstop against a routing bug looping forever rather than a tuning
    # knob — CB-1's much lower per-task/per-run caps fire first in practice.
    result = compiled.invoke(state, config={"recursion_limit": 1000})
    return result if isinstance(result, AgentState) else AgentState.model_validate(result)


def _talos_detail(report: EvalReport | None) -> str:
    """What the gates said, in one line. `first_failure` is already the
    triaged <=500-char critical span (PRD S10), so there is nothing further
    to summarize — only to shorten for a single terminal row."""
    if report is None:
        return ""
    if report.passed:
        return "all gates passed"
    if report.infrastructure_failure:
        return f"infrastructure failure in {report.failed_command or 'gates'}"
    failed = report.failed_command or "gates"
    excerpt = _first_meaningful_line(report.first_failure or "")
    return f"{failed} failed — {excerpt}" if excerpt else f"{failed} failed"


def _first_meaningful_line(text: str) -> str:
    """The first line a reader would actually want.

    Talos's triage is a model call, and models like to wrap an excerpt in a
    markdown fence — so the literal first line is often "```python", which
    tells a reader watching the live view nothing at all about why the gates
    failed.
    """
    for raw in text.strip().splitlines():
        line = raw.strip()
        if line and not line.startswith("```"):
            return line[:100]
    return ""


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
