"""CB-1, CB-3, CB-4 (PRD S7.3).

CB-4's counting rule gets the most coverage here because it is the breaker the
PRD calls "the single most important" and because every one of its clauses is a
way to get the breaker wrong: count too eagerly and healthy runs halt, count
too late and collapse runs to the budget ceiling.
"""

from __future__ import annotations

import time

from xeno.core.breakers import (
    DIFF_HISTORY_WINDOW,
    SIGNATURE_STREAK_LIMIT,
    BreakerPanel,
    Intervention,
    check_budget_cap,
    check_destructive_action,
    check_diff_thrash,
    check_iteration_cap,
    check_wall_clock_cap,
    failure_signature,
    observe_failure,
    record_diff,
    reset_for_new_task,
    would_breach_budget,
)
from xeno.core.config import Limits
from xeno.core.state import AgentState
from xeno.core.types import BreakerCode, Tier


def make_state(**kwargs: object) -> AgentState:
    return AgentState(run_id="r", goal="g", **kwargs)  # type: ignore[arg-type]


def _seconds_ago(seconds: float) -> float:
    return time.time() - seconds


# ---- CB-1 ------------------------------------------------------------------


def test_cb1_fires_on_per_task_cap() -> None:
    limits = Limits(max_iterations_per_task=3)
    verdict = check_iteration_cap(make_state(iterations_this_task=3), limits)
    assert verdict is not None
    assert verdict.code is BreakerCode.CB1_ITERATION_CAP


def test_cb1_fires_on_per_run_cap_even_when_task_is_fresh() -> None:
    limits = Limits(max_iterations_per_task=12, max_iterations_per_run=20)
    state = make_state(iterations_this_task=0, iteration_count=20)
    verdict = check_iteration_cap(state, limits)
    assert verdict is not None
    assert "run hit" in verdict.detail


def test_cb1_silent_below_caps() -> None:
    state = make_state(iterations_this_task=1, iteration_count=1)
    assert check_iteration_cap(state, Limits()) is None


# ---- CB-3 ------------------------------------------------------------------


def test_cb3_fires_at_usd_ceiling() -> None:
    verdict = check_budget_cap(make_state(usd_spent=2.0), Limits(max_usd_per_run=2.0))
    assert verdict is not None
    assert verdict.code is BreakerCode.CB3_BUDGET_CAP


def test_cb3_fires_on_per_tier_token_cap() -> None:
    limits = Limits(max_tokens_per_tier={Tier.FLAGSHIP: 1000})
    state = make_state(tokens_spent={Tier.FLAGSHIP: 1000})
    verdict = check_budget_cap(state, limits)
    assert verdict is not None
    assert "flagship" in verdict.detail


def test_cb3_precheck_catches_a_call_that_would_vault_the_ceiling() -> None:
    """A flagship review of a full diff is the most expensive call in the graph.
    Noticing the breach only after paying for it defeats the cap."""
    state = make_state(usd_spent=1.90)
    verdict = would_breach_budget(
        state,
        Limits(max_usd_per_run=2.00),
        projected_usd=0.50,
        tier=Tier.FLAGSHIP,
        projected_tokens=10_000,
    )
    assert verdict is not None
    assert "projected" in verdict.detail


def test_cb3_precheck_allows_a_call_that_fits() -> None:
    state = make_state(usd_spent=0.10)
    assert (
        would_breach_budget(
            state,
            Limits(max_usd_per_run=2.00),
            projected_usd=0.05,
            tier=Tier.LIGHT,
            projected_tokens=100,
        )
        is None
    )


# ---- CB-2: wall-clock cap ---------------------------------------------------


def test_cb2_fires_past_the_runtime_cap() -> None:
    limits = Limits(max_runtime_minutes=1.0)
    state = make_state(wall_clock_start=_seconds_ago(90))
    verdict = check_wall_clock_cap(state, limits)
    assert verdict is not None
    assert verdict.code is BreakerCode.CB2_WALL_CLOCK_CAP


def test_cb2_silent_below_the_runtime_cap() -> None:
    limits = Limits(max_runtime_minutes=45.0)
    state = make_state(wall_clock_start=_seconds_ago(5))
    assert check_wall_clock_cap(state, limits) is None


# ---- CB-5: diff thrash -------------------------------------------------------


def test_cb5_silent_on_a_single_diff() -> None:
    state = make_state()
    record_diff(state, "hash-a")
    assert check_diff_thrash(state, Limits()) is None


def test_cb5_silent_when_every_diff_is_distinct() -> None:
    state = make_state()
    for h in ("hash-a", "hash-b", "hash-c"):
        record_diff(state, h)
    assert check_diff_thrash(state, Limits()) is None


def test_cb5_fires_when_a_diff_repeats_a_prior_state() -> None:
    state = make_state()
    record_diff(state, "hash-a")
    record_diff(state, "hash-b")
    record_diff(state, "hash-a")  # back to the first state
    verdict = check_diff_thrash(state, Limits())
    assert verdict is not None
    assert verdict.code is BreakerCode.CB5_DIFF_THRASH


def test_cb5_does_not_fire_on_consecutive_identical_calls_within_the_window() -> None:
    """A repeat only matters against an EARLIER entry — the check compares the
    latest hash against everything before it, not against itself."""
    state = make_state()
    record_diff(state, "hash-a")
    assert check_diff_thrash(state, Limits()) is None


def test_diff_history_window_is_capped() -> None:
    state = make_state()
    for i in range(DIFF_HISTORY_WINDOW + 5):
        record_diff(state, f"hash-{i}")
    assert len(state.diff_history) == DIFF_HISTORY_WINDOW
    assert state.diff_history[-1] == f"hash-{DIFF_HISTORY_WINDOW + 4}"


# ---- CB-6: destructive action guard -----------------------------------------


def test_cb6_fires_past_the_deleted_lines_cap() -> None:
    diff = "\n".join(["--- a/x.py", "+++ b/x.py", *[f"-line {i}" for i in range(5)]])
    verdict = check_destructive_action(
        diff, removed_files=(), created_this_run=(), limits=Limits(max_deleted_lines=3)
    )
    assert verdict is not None
    assert verdict.code is BreakerCode.CB6_DESTRUCTIVE_ACTION


def test_cb6_diff_markers_are_not_counted_as_deletions() -> None:
    """The `---` file-header line also starts with '-', but it is not a
    content deletion — if it were wrongly counted alongside the one real
    deletion below, 2 > the cap of 1 would fire."""
    diff = "--- a/x.py\n+++ b/x.py\n-one real deletion\n"
    verdict = check_destructive_action(
        diff, removed_files=(), created_this_run=(), limits=Limits(max_deleted_lines=1)
    )
    assert verdict is None


def test_cb6_silent_below_the_deleted_lines_cap() -> None:
    diff = "--- a/x.py\n+++ b/x.py\n-one line\n+one replacement\n"
    verdict = check_destructive_action(
        diff, removed_files=(), created_this_run=(), limits=Limits(max_deleted_lines=200)
    )
    assert verdict is None


def test_cb6_fires_on_removing_a_file_not_created_this_run() -> None:
    verdict = check_destructive_action(
        "", removed_files=["legacy.py"], created_this_run=(), limits=Limits()
    )
    assert verdict is not None
    assert verdict.code is BreakerCode.CB6_DESTRUCTIVE_ACTION
    assert "legacy.py" in verdict.detail


def test_cb6_silent_removing_a_file_created_this_run() -> None:
    verdict = check_destructive_action(
        "", removed_files=["scratch.py"], created_this_run=["scratch.py"], limits=Limits()
    )
    assert verdict is None


# ---- CB-4: the counting rule ----------------------------------------------


SIG_A = "aaaa1111"
SIG_B = "bbbb2222"


def test_first_failure_is_the_baseline_and_is_not_counted() -> None:
    state = make_state()
    assert observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED) is None
    assert state.failure_signature == SIG_A
    assert state.signature_streak == 0


def test_streak_increments_only_when_a_signature_survives_an_intervention() -> None:
    state = make_state()
    observe_failure(state, SIG_A, intervention=Intervention.NONE)  # baseline
    observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED)
    assert state.signature_streak == 1
    observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED)
    assert state.signature_streak == 2


def test_l0_retry_never_increments_the_streak() -> None:
    """An L0 retry changes nothing, so an unchanged signature says nothing
    about collapse (PRD S7.3)."""
    state = make_state()
    observe_failure(state, SIG_A, intervention=Intervention.NONE)  # baseline
    for _ in range(5):
        observe_failure(state, SIG_A, intervention=Intervention.NONE)
    assert state.signature_streak == 0


def test_declined_patch_never_increments_the_streak() -> None:
    """Chiron declining to patch means no intervention occurred (PRD S10)."""
    state = make_state()
    observe_failure(state, SIG_A, intervention=Intervention.NONE)
    observe_failure(state, SIG_A, intervention=Intervention.DECLINED)
    observe_failure(state, SIG_A, intervention=Intervention.DECLINED)
    assert state.signature_streak == 0


def test_a_changed_signature_resets_the_streak() -> None:
    state = make_state()
    observe_failure(state, SIG_A, intervention=Intervention.NONE)
    observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED)
    assert state.signature_streak == 1
    observe_failure(state, SIG_B, intervention=Intervention.MODEL_AUTHORED)
    assert state.signature_streak == 0
    assert state.failure_signature == SIG_B


def test_cb4_halts_at_three_surviving_interventions() -> None:
    state = make_state()
    observe_failure(state, SIG_A, intervention=Intervention.NONE)  # baseline, uncounted
    verdicts = [
        observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED)
        for _ in range(SIGNATURE_STREAK_LIMIT)
    ]
    assert verdicts[0] is None and verdicts[1] is None
    assert verdicts[2] is not None
    assert verdicts[2].code is BreakerCode.CB4_NO_PROGRESS


def test_new_plan_task_clears_the_previous_task_signature() -> None:
    """Otherwise the next task's first failure is compared against the previous
    task's defect and the baseline clause silently stops applying."""
    state = make_state()
    observe_failure(state, SIG_A, intervention=Intervention.NONE)
    observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED)
    assert state.signature_streak == 1
    record_diff(state, "hash-a")

    reset_for_new_task(state)
    assert state.failure_signature is None
    assert state.signature_streak == 0
    assert state.ladder_rung == 0
    assert state.iterations_this_task == 0
    assert state.diff_history == []  # CB-5's window is also task-scoped

    observe_failure(state, SIG_A, intervention=Intervention.MODEL_AUTHORED)
    assert state.signature_streak == 0  # baseline again, not a continuation


# ---- signature normalization ----------------------------------------------


def test_signature_ignores_test_ordering_and_duplicates() -> None:
    """Parallel test runners reorder failures; that is noise, not progress."""
    a = failure_signature(["test_b", "test_a"], "AssertionError", "app/x.py:12")
    b = failure_signature(["test_a", "test_b", "test_a"], "AssertionError", "app/x.py:12")
    assert a == b


def test_signature_distinguishes_different_failures() -> None:
    a = failure_signature(["test_a"], "AssertionError", "app/x.py:12")
    b = failure_signature(["test_a"], "TypeError", "app/x.py:12")
    assert a != b


def test_signature_distinguishes_different_lint_only_failures() -> None:
    """Regression: before lint_signature/type_signature existed, every
    lint-only failure hashed identically since the test-shaped fields
    (failing_test_ids, exception_type, failing_location) are all empty when
    only the lint gate is red — CB-4 could not tell one lint defect from
    another."""
    a = failure_signature([], "", "", lint_signature="E501:cli.py:38")
    b = failure_signature([], "", "", lint_signature="F401:cli.py:5")
    assert a != b


def test_signature_distinguishes_different_type_only_failures() -> None:
    a = failure_signature([], "", "", type_signature="arg-type:mod.py:10")
    b = failure_signature([], "", "", type_signature="assignment:mod.py:22")
    assert a != b


def test_signature_same_lint_finding_is_the_same_signature() -> None:
    a = failure_signature([], "", "", lint_signature="E501:cli.py:38")
    b = failure_signature([], "", "", lint_signature="E501:cli.py:38")
    assert a == b


# ---- panel -----------------------------------------------------------------


def test_panel_trip_halts_the_run_and_records_the_breaker() -> None:
    panel = BreakerPanel(Limits(max_iterations_per_task=1))
    state = make_state(iterations_this_task=5)
    verdict = panel.check(state)
    assert verdict is not None

    trip = panel.trip(state, verdict)
    assert state.halted
    assert trip.code is BreakerCode.CB1_ITERATION_CAP
    assert state.halt_reason is not None
    assert state.halt_reason.startswith("CB-1")
    assert len(state.breaker_trips) == 1
