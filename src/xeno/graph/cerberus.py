"""The reviewer node (PRD S8, S13 Phase 4): the sole human gate.

Cerberus is entered from exactly two situations, and behaves differently in
each (PRD S8.2, S8.3):

  - The run is already halted (an L5 ladder exhaustion, a circuit breaker
    trip, or any other node's own halt) — every decision here has already
    been made, so Cerberus reports it deterministically, with NO model call.
    "Cerberus does not judge; it reports" reads as a template job in this
    case, and skipping the call removes the one place a malformed response
    or `ChainExhaustedError` could turn an already-bad situation ambiguous.
  - A full task loop finished clean (E15: every task passed Talos's gates)
    — this is the genuine subjective judgment call (PRD S8.2) a green Talos
    is necessary but not sufficient for, and it is the only branch that
    spends a FLAGSHIP call.

Cerberus never writes source (PRD S10: "Writes GIT. No source — Cerberus
never edits code, only judges and commits") and never touches the worktree's
files directly — only `xeno.graph.build`'s Cerberus step and `xeno.cli`'s
post-graph human gate call into `xeno.core.vcs` for the actual git writes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xeno.core import vcs
from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole, Verdict
from xeno.graph.plan import Plan, PlanTask, read_plan, write_plan
from xeno.graph.prompts import (
    CERBERUS_FORMAT_CORRECTION,
    CERBERUS_SYSTEM,
    CerberusOutput,
    parse_cerberus_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: Same rationale as every other wire-format node's: one corrective nudge,
#: not an open loop.
_MAX_FORMAT_ATTEMPTS = 2


def _write_handle(paths: RunPaths, call_index: int, label: str, text: str) -> Handle:
    path = paths.workspace / f"cerberus_{label}_{call_index}.txt"
    path.write_text(text)
    return Handle.for_file(path, summary=f"cerberus {label}, call {call_index}")


def _deterministic_escalate_report(state: AgentState) -> str:
    """PRD S8.3 ESCALATE via L5/breaker: Cerberus does not judge here, it
    reports what the harness already decided — no synthesis needed, since
    every fact in this report already lives in `state`."""
    lines = [
        f"Goal: {state.goal}",
        f"Halt reason: {state.halt_reason}",
        f"Ladder rung reached: L{state.ladder_rung} (attempt {state.rung_attempts})",
        f"Task {state.task_cursor + 1} of {state.task_count}",
    ]
    if state.breaker_trips:
        trips = "; ".join(f"{t.code.value}: {t.detail}" for t in state.breaker_trips)
        lines.append(f"Circuit breakers fired: {trips}")
    if state.eval_report is not None:
        report = state.eval_report
        lines.append(
            f"Last evaluation: parse_ok={report.parse_ok} lint_errors={report.lint_errors} "
            f"type_errors={report.type_errors} "
            f"tests_failed={report.tests_failed}/{report.tests_run}"
        )
        if report.first_failure:
            lines.append(f"Last failure detail: {report.first_failure}")
    if state.checkpoints:
        last = state.checkpoints[-1]
        lines.append(
            f"Last green checkpoint: {last.sha} (task {last.task_index}, {last.message!r})"
        )
    else:
        lines.append("Last green checkpoint: none — no task in this run ever passed")
    lines.append(
        "Recommended action: inspect the preserved branch and worktree, "
        "then either redirect with a more specific goal or abandon this run."
    )
    return "\n".join(lines)


def make_cerberus_node(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    initial_sha: str,
) -> Callable[[AgentState], AgentState]:
    """Build the Cerberus node. `initial_sha` (from `vcs.init_repo`, PRD S13)
    is the base every review diff and eventual squash is computed against —
    the run's whole accumulated change, not any single write."""
    builder = PromptBuilder(
        node=NodeRole.REVIEWER,
        keyring=keyring,
        system_text=CERBERUS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    call_index = 0

    def node(state: AgentState) -> AgentState:
        nonlocal call_index
        call_index += 1

        diff_text = vcs.diff_since(worktree, initial_sha)
        diff_path = paths.workspace / f"review_diff_{call_index}.patch"
        diff_path.write_text(diff_text)
        state.review_diff_handle = Handle.for_file(
            diff_path, summary=f"cumulative run diff since {initial_sha[:12]}"
        )

        if state.halted:
            # E12/E13/any generalized halt — deterministic report, no
            # FLAGSHIP call. `halt_reason` is already set by whoever halted
            # the run; this branch only records the verdict and the report.
            report_text = _deterministic_escalate_report(state)
            state.cerberus_notes = _write_handle(paths, call_index, "report", report_text)
            state.review_verdict = Verdict.ESCALATE
            return state

        assert state.plan is not None, "E15 only happens once a plan exists"
        plan = read_plan(state.plan)
        current_turn = _build_current_turn(state, plan, diff_text)

        output: CerberusOutput | None = None
        for attempt in range(_MAX_FORMAT_ATTEMPTS):
            turn_text = current_turn if attempt == 0 else CERBERUS_FORMAT_CORRECTION
            prompt = builder.build(turn_text)
            try:
                result = router.complete(NodeRole.REVIEWER, prompt, state=state)
            except ChainExhaustedError as exc:
                # PRD S8.2 "Failure": the harness must never auto-approve in
                # the absence of a review. `cerberus_notes` stays unset — its
                # absence on an ESCALATE IS the UNREVIEWED signal.
                state.halt_reason = f"cerberus: {exc}"
                state.review_verdict = Verdict.ESCALATE
                return state

            builder.append_turn("user", turn_text)
            builder.append_turn("assistant", result.text)

            output = parse_cerberus_output(result.text)
            if not output.malformed:
                break

        assert output is not None
        return _apply_verdict(state, output, config, paths, call_index, plan)

    return node


def _build_current_turn(state: AgentState, plan: Plan, diff_text: str) -> str:
    lines = [
        f"Goal: {state.goal}",
        "",
        "Plan:",
        *(f"- {t.description} (acceptance: {t.acceptance})" for t in plan.tasks),
        "",
        f"Full accumulated diff for this run ({len(state.checkpoints)} checkpoint(s)):",
        diff_text if diff_text.strip() else "(no changes)",
    ]
    return "\n".join(lines)


def _apply_verdict(
    state: AgentState,
    output: CerberusOutput,
    config: XenoConfig,
    paths: RunPaths,
    call_index: int,
    plan: Plan,
) -> AgentState:
    if output.malformed:
        # No safe verdict to act on — never silently drop a completed run.
        state.cerberus_notes = _write_handle(
            paths,
            call_index,
            "report",
            "Cerberus's response could not be parsed after a corrective retry; "
            "escalating rather than guessing at a verdict.",
        )
        state.review_verdict = Verdict.ESCALATE
        state.halt_reason = "cerberus: malformed response, no verdict could be applied"
        return state

    if output.verdict is Verdict.APPROVE:
        assert output.commit_message is not None
        state.commit_message = output.commit_message
        state.cerberus_notes = _write_handle(
            paths, call_index, "notes", output.notes or "(no additional notes)"
        )
        state.review_verdict = Verdict.APPROVE
        return state

    if output.verdict is Verdict.REJECT_AND_RETURN:
        assert output.objections is not None
        assert output.destination is not None
        if state.reject_count >= config.limits.max_rejections_per_run:
            # The Nth rejection beyond budget (default budget 2, so the 3rd
            # attempt) converts to ESCALATE (PRD S8.3) — does NOT touch
            # `ladder_rung`, since this budget is separate from the
            # per-task escalation ladder.
            state.cerberus_notes = _write_handle(
                paths,
                call_index,
                "report",
                f"Rejection budget ({config.limits.max_rejections_per_run}) exhausted. "
                f"Final objections: {output.objections}",
            )
            state.review_verdict = Verdict.ESCALATE
            state.halt_reason = "cerberus: rejection budget exhausted"
            return state

        state.reject_count += 1
        state.reject_destination = output.destination
        state.cerberus_notes = _write_handle(paths, call_index, "objections", output.objections)
        state.review_verdict = Verdict.REJECT_AND_RETURN

        if output.destination is NodeRole.CODER:
            # E16 (PRD S8.3): append one task, built from the objection, so
            # the existing per-task loop (argus_research -> daedalus ->
            # talos -> checkpoint_step) picks it up transparently — no
            # change needed to any of that machinery.
            new_task = PlanTask(
                description=output.objections,
                acceptance="All Talos gates pass and Cerberus's objection above is resolved.",
            )
            updated = Plan(tasks=[*plan.tasks, new_task])
            plan_path = paths.workspace / f"plan_cerberus_{call_index}.json"
            write_plan(plan_path, updated)
            state.plan = Handle.for_file(
                plan_path,
                summary=f"plan (cerberus reject v{call_index}): {len(updated.tasks)} task(s)",
            )
            state.task_count = len(updated.tasks)
        return state

    assert output.verdict is Verdict.ESCALATE
    assert output.report is not None
    state.cerberus_notes = _write_handle(paths, call_index, "report", output.report)
    state.review_verdict = Verdict.ESCALATE
    state.halt_reason = "cerberus: escalated for human judgment"
    return state
