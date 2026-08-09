"""The evaluator node (PRD S10): deterministic gates, plus log triage only.

"THE GATES ARE DETERMINISTIC TOOLS, not model calls" (PRD S8.2) — parse,
lint, type, and test verdicts are decided entirely by `xeno.graph.gates`
before any model is involved. The one model call this node makes is used
only to compress a failure log to its <=500-char critical span; it never
participates in the pass/fail decision, and its failure is never fatal to
the run (falls back to a deterministic truncation).

PRD S13 Phase 2: gates run inside a sandboxed container acquired from the
warm pool — zero host execution of generated code. `xeno run` still operates
on a throwaway worktree under `.xeno/`, never the user's working tree; the
sandbox is a second, stricter isolation boundary on top of that, not a
replacement for it (the codebase map, diffs, and everything else Daedalus
and Chiron touch still live in the host-side worktree).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xeno.adapters.base import LanguageAdapter
from xeno.core.breakers import BreakerPanel, Intervention, failure_signature, observe_failure
from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState, EvalReport, Handle
from xeno.core.types import NodeRole
from xeno.graph.context import build_codebase_map
from xeno.graph.gates import run_gates
from xeno.graph.prompts import TALOS_TRIAGE_SYSTEM
from xeno.graph.testfiles import is_test_file
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router
from xeno.sandbox.pool import WarmPool

#: The triage call sees at most this much of the raw log — enough context for
#: a light model to find the relevant excerpt without paying to re-read a
#: multi-thousand-line pytest dump in full.
_TRIAGE_LOG_WINDOW = 4000


def make_talos_node(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    paths: RunPaths,
    worktree: Path,
    touched_files: list[Path],
    pool: WarmPool,
    adapter: LanguageAdapter,
    breaker_panel: BreakerPanel,
    runlog: RunLog,
    intervention: Callable[[], Intervention],
) -> Callable[[AgentState], AgentState]:
    """Build the Talos node.

    `intervention` is a callable rather than a fixed value because the same
    node instance is reused across the run's whole failure loop, and what
    counts as an intervention for CB-4 differs between an L0 retry (nothing
    changed) and a Chiron patch (PRD S7.3) — the caller in `xeno.graph.build`
    decides which, immediately before each call.

    `breaker_panel` is the SAME instance `xeno.graph.build` uses for every
    other breaker, passed in rather than constructed here: CB-4 is
    event-driven (it needs to know the signature AND the intervention type,
    both only available right here, right after `observe_failure`), but it
    must still go through the one shared panel that records trips onto
    state — a second, disconnected panel would let CB-4 update
    `signature_streak` without ever actually halting the run.
    """
    builder = PromptBuilder(
        node=NodeRole.EVALUATOR,
        keyring=keyring,
        system_text=TALOS_TRIAGE_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    call_index = 0
    #: Coverage DELTA (PRD S10 gate 5) is measured against the previous
    #: evaluation within this same run — there is no checkpoint-based
    #: baseline until Phase 3, so "previous Talos call" is the best available
    #: reference point. None until two green evaluations have happened.
    last_coverage: float | None = None

    def node(state: AgentState) -> AgentState:
        nonlocal call_index, last_coverage
        call_index += 1

        sandbox = pool.acquire()
        try:
            sandbox.sync_worktree(worktree, secrets=config.secrets)
            outcome = run_gates(sandbox, adapter, worktree, touched_files)
        finally:
            pool.release(sandbox)

        log_path = paths.workspace / f"talos_log_{call_index}.txt"
        log_path.write_text(outcome.log)
        log_handle = Handle.for_file(
            log_path, summary=f"talos evaluation {call_index}: {len(outcome.log)} bytes"
        )

        passed = (
            outcome.parse_ok
            and not outcome.infrastructure_failure
            and outcome.lint_errors == 0
            and outcome.type_errors == 0
            and outcome.tests_failed == 0
        )

        first_failure = "" if passed else _triage(outcome.log, state, builder, worktree, router)

        coverage_delta = None
        if outcome.coverage_percent is not None:
            if last_coverage is not None:
                coverage_delta = outcome.coverage_percent - last_coverage
            last_coverage = outcome.coverage_percent

        touched_rel = [str(p.relative_to(worktree)) for p in touched_files]
        report = EvalReport(
            parse_ok=outcome.parse_ok,
            lint_errors=outcome.lint_errors,
            type_errors=outcome.type_errors,
            tests_run=outcome.tests_run,
            tests_failed=outcome.tests_failed,
            coverage_delta=coverage_delta,
            first_failure=first_failure[:500],
            full_log_handle=log_handle,
            infrastructure_failure=outcome.infrastructure_failure,
            touched_test_files=[r for r in touched_rel if is_test_file(r)],
        )
        state.eval_report = report

        if not passed:
            signature = failure_signature(
                outcome.failing_test_ids,
                outcome.exception_type,
                outcome.failing_location,
                lint_signature=outcome.lint_signature,
                type_signature=outcome.type_signature,
            )
            verdict = observe_failure(state, signature, intervention=intervention())
            if verdict is not None:
                trip = breaker_panel.trip(state, verdict)
                runlog.event(EventKind.BREAKER_FIRED, code=trip.code.value, detail=trip.detail)

        return state

    return node


def _triage(
    log: str,
    state: AgentState,
    builder: PromptBuilder,
    worktree: Path,
    router: Router,
) -> str:
    """Best-effort log compression. Never blocks the run on failure."""
    window = log[-_TRIAGE_LOG_WINDOW:]
    try:
        focus = [h.path for h in state.context_handles] or None
        builder.set_codebase_map(build_codebase_map(worktree, focus=focus), require_fresh=False)
        prompt = builder.build(window)
        result = router.complete(NodeRole.EVALUATOR, prompt, state=state)
        builder.append_turn("user", window)
        builder.append_turn("assistant", result.text)
        return result.text.strip()
    except ChainExhaustedError:
        # The gates already produced the verdict; losing the triage call only
        # costs summarization quality, never correctness (PRD S8.2).
        return window[-500:]
