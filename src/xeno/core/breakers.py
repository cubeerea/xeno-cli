"""Circuit breakers (PRD S7.3, S7.4).

Orthogonal to the escalation ladder: these fire regardless of rung and force an
immediate jump to L5 / Cerberus. Phase 0 ships CB-1, CB-3, and CB-4 — before
any loop exists to break — because the failure they prevent is the one that
kills naive harnesses: burning a whole budget re-attempting the same wrong fix.

CB-2, CB-5, and CB-6 land in Phase 2 alongside the first real cycle:

  * CB-2 (wall clock) is polled the same way as CB-1/CB-3.
  * CB-5 (diff thrash) is also polled, but only meaningful once a diff has
    actually been produced — `xeno.graph.build` appends each new diff's
    content hash to `state.diff_history` right after Daedalus or Chiron
    writes, before Talos ever runs, so by the time this fires the signal is
    already in state.
  * CB-6 (destructive action) is event-driven like CB-4, not polled: it must
    fire "regardless of test status" (PRD S7.3), i.e. before Talos, so
    `xeno.graph.build` calls `check_destructive_action` directly right after
    a write rather than waiting for the next poll.

CB-4 is the one that actually catches collapse, and its counting rule is
narrow enough to be worth restating (PRD S7.3):

  * the signature seen on a plan task's FIRST failure is the baseline and is
    not counted;
  * the streak increments only when a signature SURVIVES AN INTERVENTION
    unchanged — an L1/L2/L3 attempt that produced a model-authored change;
  * an L0 retry never increments, because nothing was changed;
  * a patch Chiron DECLINED to write never increments, because no intervention
    occurred (PRD S10, Chiron failure mode);
  * halt at 3.

Getting any of those wrong makes the breaker either fire on healthy runs or
never fire at all, so they are enforced here rather than left to call sites.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto

from xeno.core.config import Limits
from xeno.core.state import AgentState, BreakerTrip
from xeno.core.types import BreakerCode, Tier

#: PRD S7.3 CB-4: halt when a signature survives this many interventions.
SIGNATURE_STREAK_LIMIT = 3

#: PRD CB-5: how many recent diff hashes `state.diff_history` retains. See
#: the field's docstring in `xeno.core.state` for why this is a window, not
#: the full run.
DIFF_HISTORY_WINDOW = 10


class Intervention(StrEnum):
    """What, if anything, changed between two evaluations."""

    #: L0 retry, or the first evaluation of a task. Nothing was changed.
    NONE = auto()
    #: An L1/L2/L3 attempt that produced a model-authored change.
    MODEL_AUTHORED = auto()
    #: Chiron could not form a hypothesis and declined to patch. No
    #: intervention occurred, so this must not count against the streak.
    DECLINED = auto()


@dataclass(frozen=True, slots=True)
class BreakerVerdict:
    code: BreakerCode
    detail: str

    def to_trip(self) -> BreakerTrip:
        return BreakerTrip(code=self.code, detail=self.detail[:500])


#: How much of a failing command's output feeds the signature. Deliberately
#: coarser than the old adapter-specific structured signature (PRD S12
#: revised: a `DiscoveredToolchain`'s commands have no adapter-parsed rule
#: codes/test IDs to hash) — same defect vs. different defect is now judged
#: on "did the failing command's output tail change," not a precise finding
#: identity. A real, accepted precision loss for CB-4's no-progress detector.
_SIGNATURE_TAIL_CHARS = 500


def failure_signature(failed_command: str, exit_code: int, output_tail: str) -> str:
    """Normalized hash of (which command failed + its exit code + a tail of
    its output).

    Trimmed to the tail rather than the whole log so that noise earlier in
    the output (e.g. an install step's progress bar) does not dominate the
    comparison — the tail is where a test runner or linter's actual finding
    lives.
    """
    tail = output_tail.strip()[-_SIGNATURE_TAIL_CHARS:]
    payload = f"{failed_command.strip()}|{exit_code}|{tail}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CB-1: iteration cap
# ---------------------------------------------------------------------------


def check_iteration_cap(state: AgentState, limits: Limits) -> BreakerVerdict | None:
    if state.iterations_this_task >= limits.max_iterations_per_task:
        return BreakerVerdict(
            BreakerCode.CB1_ITERATION_CAP,
            f"plan task {state.task_cursor} hit {state.iterations_this_task} iterations "
            f"(cap {limits.max_iterations_per_task})",
        )
    if state.iteration_count >= limits.max_iterations_per_run:
        return BreakerVerdict(
            BreakerCode.CB1_ITERATION_CAP,
            f"run hit {state.iteration_count} iterations (cap {limits.max_iterations_per_run})",
        )
    return None


# ---------------------------------------------------------------------------
# CB-3: budget cap
# ---------------------------------------------------------------------------


def check_budget_cap(state: AgentState, limits: Limits) -> BreakerVerdict | None:
    """Halt on breach. Never downgrade a tier to keep going.

    Silently dropping to a weaker model would produce worse output AND hide the
    cost problem behind a run that appears to have succeeded — which is why
    PRD S9.5 forbids downward fallback and this breaker halts instead.
    """
    if state.usd_spent >= limits.max_usd_per_run:
        return BreakerVerdict(
            BreakerCode.CB3_BUDGET_CAP,
            f"spent ${state.usd_spent:.4f} of ${limits.max_usd_per_run:.2f} run budget",
        )
    for tier, cap in limits.max_tokens_per_tier.items():
        spent = state.tokens_spent.get(tier, 0)
        if spent >= cap:
            return BreakerVerdict(
                BreakerCode.CB3_BUDGET_CAP,
                f"tier '{tier}' used {spent} tokens (cap {cap})",
            )
    return None


def would_breach_budget(
    state: AgentState,
    limits: Limits,
    *,
    projected_usd: float,
    tier: Tier,
    projected_tokens: int,
) -> BreakerVerdict | None:
    """Pre-call check, so a single expensive flagship call cannot vault the
    ceiling and only be noticed afterwards."""
    if state.usd_spent + projected_usd > limits.max_usd_per_run:
        return BreakerVerdict(
            BreakerCode.CB3_BUDGET_CAP,
            f"projected ${state.usd_spent + projected_usd:.4f} would exceed the "
            f"${limits.max_usd_per_run:.2f} run budget",
        )
    cap = limits.max_tokens_per_tier.get(tier)
    if cap is not None and state.tokens_spent.get(tier, 0) + projected_tokens > cap:
        return BreakerVerdict(
            BreakerCode.CB3_BUDGET_CAP,
            f"projected tier '{tier}' usage would exceed its {cap}-token cap",
        )
    return None


# ---------------------------------------------------------------------------
# CB-2: wall-clock cap
# ---------------------------------------------------------------------------


def check_wall_clock_cap(state: AgentState, limits: Limits) -> BreakerVerdict | None:
    """PRD S7.3: 'No "unlimited" default.' Bounds the whole run in real time,
    independent of iteration or token counts — a run stuck retrying a slow
    sandbox or a slow model still has to stop eventually."""
    elapsed_minutes = state.elapsed_seconds / 60
    if elapsed_minutes >= limits.max_runtime_minutes:
        return BreakerVerdict(
            BreakerCode.CB2_WALL_CLOCK_CAP,
            f"run hit {elapsed_minutes:.1f} minutes (cap {limits.max_runtime_minutes:.1f})",
        )
    return None


# ---------------------------------------------------------------------------
# CB-5: diff thrash detector
# ---------------------------------------------------------------------------


def record_diff(state: AgentState, sha256: str) -> None:
    """Append a new diff's content hash to the rolling window (PRD CB-5).

    Call this right after Daedalus or Chiron sets `state.diff_handle` — the
    check below assumes the LATEST entry is always the diff just produced.
    """
    state.diff_history = [*state.diff_history[-(DIFF_HISTORY_WINDOW - 1) :], sha256]


def check_diff_thrash(state: AgentState, limits: Limits) -> BreakerVerdict | None:
    """PRD CB-5: the working diff returned to a byte-identical prior state —
    the agent is oscillating between two wrong answers, not converging.

    `limits` is accepted only to match the other POLLED_BREAKERS' call
    signature; this check does not use it.
    """
    if len(state.diff_history) < 2:
        return None
    latest = state.diff_history[-1]
    if latest in state.diff_history[:-1]:
        return BreakerVerdict(
            BreakerCode.CB5_DIFF_THRASH,
            f"diff signature {latest} repeated a prior state — oscillating, not converging",
        )
    return None


# ---------------------------------------------------------------------------
# CB-6: destructive action guard
# ---------------------------------------------------------------------------


def check_destructive_action(
    diff_text: str,
    *,
    removed_files: Sequence[str],
    created_this_run: Sequence[str],
    limits: Limits,
) -> BreakerVerdict | None:
    """PRD CB-6: escalates straight to L5 regardless of test status — the
    caller must invoke this right after a write, before Talos ever runs, not
    wait for the next polled-breaker check (PRD S7.3, S7.4).

    `removed_files` is whatever the caller determines disappeared from the
    worktree this call (PRD S13 Phase 2's wire format has no delete
    mechanism, so this is currently unreachable in practice — implemented
    now so a future write mechanism that CAN delete trips it automatically).
    """
    deleted_lines = sum(
        1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    if deleted_lines > limits.max_deleted_lines:
        return BreakerVerdict(
            BreakerCode.CB6_DESTRUCTIVE_ACTION,
            f"{deleted_lines} lines deleted (cap {limits.max_deleted_lines})",
        )
    illegal = [f for f in removed_files if f not in created_this_run]
    if illegal:
        return BreakerVerdict(
            BreakerCode.CB6_DESTRUCTIVE_ACTION,
            f"removed file(s) not created this run: {', '.join(illegal)}",
        )
    return None


# ---------------------------------------------------------------------------
# CB-4: no-progress detector
# ---------------------------------------------------------------------------


def observe_failure(
    state: AgentState,
    signature: str,
    *,
    intervention: Intervention,
) -> BreakerVerdict | None:
    """Record a failure and update signature_streak per the S7.3 counting rule.

    Mutates `state`. Returns a verdict when the streak reaches the limit.
    """
    previous = state.failure_signature

    if previous is None:
        # Baseline: the first failure of a plan task establishes the signature
        # and is explicitly not counted.
        state.failure_signature = signature
        state.signature_streak = 0
        return None

    if signature != previous:
        # A different failure is progress of a kind — the previous defect was
        # displaced, even if the new one is no easier.
        state.failure_signature = signature
        state.signature_streak = 0
        return None

    if intervention is not Intervention.MODEL_AUTHORED:
        # L0 retries and declined patches changed nothing, so an unchanged
        # signature says nothing about collapse.
        return None

    state.signature_streak += 1
    if state.signature_streak >= SIGNATURE_STREAK_LIMIT:
        return BreakerVerdict(
            BreakerCode.CB4_NO_PROGRESS,
            f"failure signature {signature} survived {state.signature_streak} "
            f"interventions unchanged on plan task {state.task_cursor}",
        )
    return None


def reset_for_new_task(state: AgentState) -> None:
    """Advancing to the next plan task resets the ladder and the CB-4 baseline
    (PRD S7.2). The signature must be cleared too, or the next task's first
    failure would be compared against the previous task's defect. Same for
    CB-5's diff history: a fresh task starts a fresh oscillation window."""
    state.ladder_rung = 0
    state.rung_attempts = 0
    state.failure_signature = None
    state.signature_streak = 0
    state.iterations_this_task = 0
    state.diff_history = []


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


#: Breakers evaluated on every polling check. CB-4 and CB-6 are event-driven
#: (`observe_failure`, `check_destructive_action`) rather than polled, because
#: each needs information a poll does not have — CB-4 the kind of
#: intervention that preceded the failure, CB-6 the diff that was just
#: written and the need to fire before Talos runs at all (PRD S7.3, S7.4).
POLLED_BREAKERS: tuple[Callable[[AgentState, Limits], BreakerVerdict | None], ...] = (
    check_iteration_cap,
    check_budget_cap,
    check_wall_clock_cap,
    check_diff_thrash,
)


class BreakerPanel:
    """Evaluates the polled breakers and records trips onto state."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits

    def check(self, state: AgentState) -> BreakerVerdict | None:
        for breaker in POLLED_BREAKERS:
            verdict = breaker(state, self.limits)
            if verdict is not None:
                return verdict
        return None

    def trip(self, state: AgentState, verdict: BreakerVerdict) -> BreakerTrip:
        """Record a fired breaker and halt the run.

        The destination is always Cerberus, never the user (PRD S8.1): a
        breaker firing is a reason to escalate to the gate, not to start
        prompting mid-run.
        """
        record = verdict.to_trip()
        state.breaker_trips = [*state.breaker_trips, record]
        state.halt_reason = f"{verdict.code.value}: {verdict.detail}"
        return record
