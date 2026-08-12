"""The evaluator node (PRD S10): deterministic gates, plus log triage only.

"THE GATES ARE DETERMINISTIC TOOLS, not model calls" (PRD S8.2) — the
pass/fail verdict is decided entirely by `xeno.graph.gates` running a
`DiscoveredToolchain`'s commands and reading their exit codes, before any
model is involved. The one model call this node makes is used only to
compress a failure log to its <=500-char critical span; it never
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
from xeno.graph.toolchain import ToolchainSession
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.delimit import as_data
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

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
    session: ToolchainSession,
    breaker_panel: BreakerPanel,
    runlog: RunLog,
    intervention: Callable[[], Intervention],
) -> Callable[[AgentState], AgentState]:
    """Build the Talos node.

    `session` supplies the sandbox pool and the toolchain, and is asked to
    re-derive both immediately before each evaluation. This node is the right
    place for that check because it is the only point where the toolchain is
    actually consumed: whatever Daedalus or Chiron just wrote is on disk now,
    so if that write established or changed the project's build declaration,
    this evaluation must gate on the new one rather than a snapshot taken
    before the run started (`xeno.graph.toolchain`).

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

    def node(state: AgentState) -> AgentState:
        nonlocal call_index
        call_index += 1

        # Before the gates, not after: a write that added a manifest (the
        # greenfield scaffold) or a dependency has to change what gets run
        # THIS evaluation, or the ladder spends its budget on a toolchain
        # that no longer describes the worktree.
        session.refresh_if_stale(state)

        sandbox = session.pool.acquire()
        try:
            sandbox.sync_worktree(worktree, secrets=config.secrets)
            outcome = run_gates(sandbox, session.toolchain, profile=state.gate_profile)
        finally:
            session.pool.release(sandbox)

        log_path = paths.workspace / f"talos_log_{call_index}.txt"
        log_path.write_text(outcome.log)
        log_handle = Handle.for_file(
            log_path, summary=f"talos evaluation {call_index}: {len(outcome.log)} bytes"
        )

        first_failure = (
            "" if outcome.passed else _triage(outcome.log, state, builder, worktree, router)
        )

        touched_rel = [str(p.relative_to(worktree)) for p in touched_files]
        report = EvalReport(
            passed=outcome.passed,
            failed_command=outcome.failed_command,
            first_failure=first_failure[:500],
            full_log_handle=log_handle,
            infrastructure_failure=outcome.infrastructure_failure,
            touched_test_files=[r for r in touched_rel if is_test_file(r)],
        )
        state.eval_report = report

        if not outcome.passed:
            signature = failure_signature(
                outcome.failed_command, outcome.exit_code or 0, outcome.log
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
    # PRD S11.4: gate output quotes the repository back at us — assertion
    # messages, source excerpts, filenames — so it is untrusted for the same
    # reason the files themselves are.
    turn_text = as_data(window, label="gate output")
    try:
        focus = [h.path for h in state.context_handles] or None
        builder.set_codebase_map(build_codebase_map(worktree, focus=focus), require_fresh=False)
        prompt = builder.build(turn_text)
        result = router.complete(NodeRole.EVALUATOR, prompt, state=state)
        builder.append_turn("user", turn_text)
        builder.append_turn("assistant", result.text)
        return result.text.strip()
    except ChainExhaustedError:
        # The gates already produced the verdict; losing the triage call only
        # costs summarization quality, never correctness (PRD S8.2).
        return window[-500:]
