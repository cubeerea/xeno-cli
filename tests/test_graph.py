"""Phase 3: the full six-node graph (PRD S13) — Argus (skeleton) -> Odysseus
(plan) -> per task [Argus (research) -> Daedalus -> Talos -> the L0-L5
escalation ladder] -> next task or done.

Uses the same fake-provider technique as test_router.py: what is under test
is graph/ladder/breaker/checkpoint wiring, not wire formats, so the network
boundary is faked and everything above it (Router, PromptBuilder,
CacheKeyring, breakers, `xeno.core.vcs`) runs for real — including a real
git repo per test, the same philosophy test_sandbox.py uses for Docker.

Most tests fake Talos's gates via monkeypatch AND use a fake sandbox pool,
because what is under test is the ladder/breaker/checkpoint routing, not the
sandbox or the gate tools themselves — those get their own live-Docker
coverage in `test_happy_path_passes_with_real_sandbox_and_gates`, the one
test that exercises the real warm pool end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import docker
import pytest

from xeno.adapters.base import LanguageAdapter
from xeno.adapters.python import PythonAdapter
from xeno.core.config import Limits, ProviderSpec, SandboxConfig, XenoConfig
from xeno.core.ledger import CostLedger
from xeno.core.paths import RunPaths
from xeno.core.runlog import NullRunLog
from xeno.core.state import AgentState
from xeno.core.types import DEFAULT_NODE_TIERS, NodeRole, Tier
from xeno.core.usage import Usage
from xeno.graph.build import run_graph
from xeno.graph.gates import GateOutcome
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers.base import CompletionResult, Provider
from xeno.router.router import Router
from xeno.sandbox.pool import WarmPool

DAEDALUS_OK = '<xeno-file path="pkg/mod.py">x = 1\n</xeno-file>'
CHIRON_PATCH_OK = '<xeno-file path="pkg/mod.py">x = 2\n</xeno-file>'
PLANNER_SINGLE_TASK = (
    '<xeno-task acceptance="pkg/mod.py exists and gates pass">write pkg/mod.py</xeno-task>'
)
PLANNER_TWO_TASKS = (
    '<xeno-task acceptance="pkg/a.py exists and gates pass">write pkg/a.py</xeno-task>'
    '<xeno-task acceptance="pkg/b.py exists and gates pass">write pkg/b.py</xeno-task>'
)
ARGUS_SKELETON_TEXT = "A small Python package with no existing modules."
ARGUS_NO_FILES = "<xeno-no-files>nothing to reference yet</xeno-no-files>"


def _mod_write(value: int) -> str:
    """A `pkg/mod.py` write block with distinct-per-call content — several
    ladder tests need every attempt's diff to differ, or CB-5 (diff thrash)
    trips on a repeated hash before the rung under test ever exhausts."""
    return f'<xeno-file path="pkg/mod.py">x = {value}\n</xeno-file>'


class ScriptedProvider(Provider):
    """Returns a canned response per node, distinguished by which node the
    prompt was built for — not by model name, so the fake stays correct if
    the fixture config's model strings ever change.

    Argus's two jobs (skeleton vs. per-task research) share `NodeRole.
    RESEARCHER` (PRD T8 forces one system text per node — see
    `xeno.graph.argus`'s docstring), so they are told apart by the current
    turn's job-selector prefix instead, exactly like the real router's
    caller distinguishes them.
    """

    def __init__(
        self,
        name: str,
        spec: ProviderSpec,
        *,
        planner_text: str | Callable[[], str] = PLANNER_SINGLE_TASK,
        skeleton_text: str = ARGUS_SKELETON_TEXT,
        research_text: str | Callable[[], str] = ARGUS_NO_FILES,
        daedalus_text: str | Callable[[], str] = DAEDALUS_OK,
        triage_text: str = "triaged",
        chiron_text: str | Callable[[], str] = CHIRON_PATCH_OK,
    ) -> None:
        super().__init__(name, spec)
        self.planner_text = planner_text
        self.skeleton_text = skeleton_text
        self.research_text = research_text
        self.daedalus_text = daedalus_text
        self.triage_text = triage_text
        self.chiron_text = chiron_text
        self.cache_capable = True
        self.calls: list[NodeRole] = []

    def complete(self, prompt, model, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.calls.append(prompt.node)
        if prompt.node is NodeRole.CODER:
            text = self.daedalus_text() if callable(self.daedalus_text) else self.daedalus_text
        elif prompt.node is NodeRole.DEBUGGER:
            text = self.chiron_text() if callable(self.chiron_text) else self.chiron_text
        elif prompt.node is NodeRole.PLANNER:
            text = self.planner_text() if callable(self.planner_text) else self.planner_text
        elif prompt.node is NodeRole.RESEARCHER:
            if prompt.current_turn.startswith("JOB 1"):
                text = self.skeleton_text
            else:
                text = self.research_text() if callable(self.research_text) else self.research_text
        else:
            text = self.triage_text
        return CompletionResult(
            text=text,
            model=model.model,
            usage=Usage(input_tokens=500, output_tokens=50),
            latency_ms=5.0,
        )

    def health_check(self) -> tuple[bool, str]:
        return True, "fake"


class _FakeSandbox:
    """Stands in for `xeno.sandbox.pool.Sandbox` when `run_gates` itself is
    monkeypatched — nothing ever calls `.exec` on this, but Talos always
    calls `.sync_worktree` unconditionally before running gates."""

    def sync_worktree(self, worktree: Path, *, secrets: object = None) -> None:
        pass


class FakePool:
    """Stands in for `xeno.sandbox.pool.WarmPool` with no Docker dependency,
    for every test that monkeypatches `run_gates` and so never actually
    touches a container."""

    def acquire(self) -> _FakeSandbox:
        return _FakeSandbox()

    def release(self, sandbox: _FakeSandbox) -> None:
        pass


def pass_outcome(*, coverage_percent: float | None = None) -> GateOutcome:
    return GateOutcome(
        parse_ok=True,
        lint_errors=0,
        type_errors=0,
        tests_run=3,
        tests_failed=0,
        failing_test_ids=(),
        exception_type="",
        failing_location="",
        infrastructure_failure=False,
        coverage_percent=coverage_percent,
        log="all green",
    )


def fail_outcome(*, signature: str = "A") -> GateOutcome:
    return GateOutcome(
        parse_ok=True,
        lint_errors=0,
        type_errors=0,
        tests_run=3,
        tests_failed=1,
        failing_test_ids=(f"tests/test_x.py::test_{signature}",),
        exception_type="AssertionError",
        failing_location=f"tests/test_x.py:{signature}",
        infrastructure_failure=False,
        log=f"FAILED tests/test_x.py::test_{signature} - AssertionError: nope",
    )


@pytest.fixture
def graph_config(providers: dict[str, ProviderSpec]) -> XenoConfig:
    from xeno.core.config import ModelSpec, NodeSpec

    return XenoConfig(
        providers=providers,
        tiers={
            Tier.FLAGSHIP: (ModelSpec(provider="ollama", model="big"),),
            Tier.MEDIUM: (ModelSpec(provider="ollama", model="qwen2.5-coder:14b"),),
            Tier.LIGHT: (ModelSpec(provider="ollama", model="qwen2.5-coder:7b"),),
        },
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
        limits=Limits(),
    )


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "worktree"
    wt.mkdir()
    return wt


@pytest.fixture
def run_paths(tmp_path: Path) -> RunPaths:
    return RunPaths(repo_root=tmp_path, run_id="t").ensure()


def _run(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    *,
    goal: str = "write pkg/mod.py",
    planner_text: str | Callable[[], str] = PLANNER_SINGLE_TASK,
    skeleton_text: str = ARGUS_SKELETON_TEXT,
    research_text: str | Callable[[], str] = ARGUS_NO_FILES,
    daedalus_text: str | Callable[[], str] = DAEDALUS_OK,
    chiron_text: str | Callable[[], str] = CHIRON_PATCH_OK,
    pool: object | None = None,
    adapter: LanguageAdapter | None = None,
) -> tuple[AgentState, ScriptedProvider]:
    ledger = CostLedger(run_id="t")
    router = Router(graph_config, ledger=ledger)
    fake = ScriptedProvider(
        "ollama",
        graph_config.providers["ollama"],
        planner_text=planner_text,
        skeleton_text=skeleton_text,
        research_text=research_text,
        daedalus_text=daedalus_text,
        chiron_text=chiron_text,
    )
    router._providers["ollama"] = fake
    keyring = CacheKeyring(run_id="t", worktree_root=worktree)
    state = AgentState(run_id="t", goal=goal)
    final = run_graph(
        router=router,
        config=graph_config,
        keyring=keyring,
        paths=run_paths,
        worktree=worktree,
        runlog=NullRunLog(),
        state=state,
        pool=pool or FakePool(),  # type: ignore[arg-type]
        adapter=adapter or PythonAdapter(),
    )
    return final, fake


def _script_gates(monkeypatch: pytest.MonkeyPatch, outcomes: list[GateOutcome]) -> None:
    queue = list(outcomes)

    def fake_run_gates(sandbox, adapter, worktree, touched_files):  # type: ignore[no-untyped-def]
        return queue.pop(0) if queue else outcomes[-1]

    monkeypatch.setattr("xeno.graph.talos.run_gates", fake_run_gates)


# ---- happy path: real gates, real sandbox, no monkeypatching ---------------


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
    except Exception:
        return False
    return True


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not reachable"
)


@pytest.fixture
def real_pool(tmp_path: Path) -> Iterator[WarmPool]:
    client = docker.from_env()
    adapter = PythonAdapter()
    pool = WarmPool(
        client,
        adapter,
        SandboxConfig(warm_pool_size=1, memory="512m", cpus=1.0),
        workspace_root=tmp_path / "sandboxes",
        network_disabled=True,
    )
    pool.ensure_image()
    pool.fill()
    try:
        yield pool
    finally:
        pool.shutdown()


@requires_docker
def test_happy_path_passes_with_real_sandbox_and_gates(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths, real_pool: WarmPool
) -> None:
    final, fake = _run(graph_config, worktree, run_paths, pool=real_pool)

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert not final.halted
    assert final.ladder_rung == 0
    assert final.rung_attempts == 0
    assert (worktree / "pkg" / "mod.py").read_text() == "x = 1"
    assert final.task_count == 1
    assert final.task_cursor == 1
    assert len(final.checkpoints) == 1
    # Skeleton, plan, per-task research, then the one write — no failure, so
    # no triage call and no Chiron call was ever made.
    assert fake.calls == [
        NodeRole.RESEARCHER,
        NodeRole.PLANNER,
        NodeRole.RESEARCHER,
        NodeRole.CODER,
    ]


def test_daedalus_objection_halts_without_writing(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths
) -> None:
    final, _ = _run(
        graph_config,
        worktree,
        run_paths,
        daedalus_text=(
            "<xeno-objection>the task does not say which function to write</xeno-objection>"
        ),
    )
    assert final.halted
    assert final.halt_reason is not None
    assert "objection" in final.halt_reason
    assert not (worktree / "pkg").exists()


def test_odysseus_objection_halts_before_any_task_starts(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths
) -> None:
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        planner_text=(
            "<xeno-objection>the goal requires a payment gateway this repository "
            "has no trace of and none was specified</xeno-objection>"
        ),
    )
    assert final.halted
    assert final.halt_reason is not None
    assert "odysseus objection" in final.halt_reason
    assert final.task_count == 0
    assert not (worktree / "pkg").exists()
    # The skeleton call happens before Odysseus's own; nothing downstream of
    # the objection (research, Daedalus) is ever reached.
    assert fake.calls == [NodeRole.RESEARCHER, NodeRole.PLANNER]


# ---- the ladder loop: gates monkeypatched to script failures ---------------


def test_l0_retry_recovers_without_a_chiron_patch(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(monkeypatch, [fail_outcome(), pass_outcome()])
    final, fake = _run(graph_config, worktree, run_paths)

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert final.ladder_rung == 0  # never escalated past L0
    assert final.rung_attempts == 0  # reset by checkpoint_step on task completion
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 0  # never escalated to Chiron
    assert fake.calls.count(NodeRole.EVALUATOR) == 1  # one triage call, for the one failure


def test_l0_never_increments_the_no_progress_streak(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(monkeypatch, [fail_outcome(), pass_outcome()])
    final, _ = _run(graph_config, worktree, run_paths)
    assert final.signature_streak == 0


def test_l1_chiron_patch_engages_after_l0_is_exhausted_and_recovers(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(
        monkeypatch,
        [fail_outcome(), fail_outcome(), pass_outcome()],  # initial, L0 retry, post-patch
    )
    final, fake = _run(graph_config, worktree, run_paths)

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert final.ladder_rung == 0  # reset by checkpoint_step on task completion
    assert final.rung_attempts == 0
    assert fake.calls.count(NodeRole.CODER) == 1  # Daedalus writes exactly once
    assert fake.calls.count(NodeRole.DEBUGGER) == 1  # one Chiron patch
    assert (worktree / "pkg" / "mod.py").read_text() == "x = 2"  # Chiron's patch landed
    # First failure is baseline (uncounted); the L0 retry is a NONE
    # intervention (uncounted); recovery means the streak never engages.
    assert final.signature_streak == 0


def test_chiron_decline_advances_the_rung_without_counting_as_progress(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 4 evaluations: initial, L0 retry, post-decline (worktree unchanged, so
    # scripted as another failure), post-real-patch (scripted as the pass).
    _script_gates(
        monkeypatch, [fail_outcome(), fail_outcome(), fail_outcome(), pass_outcome()]
    )
    chiron_responses = iter(
        ["<xeno-decline>no hypothesis fits this failure</xeno-decline>", CHIRON_PATCH_OK]
    )
    final, fake = _run(
        graph_config, worktree, run_paths, chiron_text=lambda: next(chiron_responses)
    )

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert (worktree / "pkg" / "mod.py").read_text() == "x = 2"  # the second, real patch landed
    assert final.ladder_rung == 0  # reset by checkpoint_step on task completion
    assert final.rung_attempts == 0
    assert final.signature_streak == 0  # a decline is not an intervention (PRD S10)
    assert fake.calls.count(NodeRole.DEBUGGER) == 2


def test_chiron_refuses_to_patch_a_test_file(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chiron refuses on every attempt (fixed veto-triggering text). A
    refused write is a decline, not a MODEL_AUTHORED intervention (PRD
    S7.3), so it never counts toward CB-4 no matter how many times it
    repeats — and, now that Argus exists (PRD S13 Phase 3), a stubbornly
    refusing Chiron no longer halts the run outright the way it did in
    Phase 2: L1 exhausts (3 declines) and escalates to L2 (2 more declines,
    with Argus re-researching in between) rather than stopping. The final
    evaluation is scripted to pass regardless, to isolate "declines don't
    get stuck or falsely trip a breaker" from L3's own rollback-and-rewrite
    behavior, which gets its own dedicated test.
    """
    _script_gates(monkeypatch, [fail_outcome()] * 6 + [pass_outcome()])
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        chiron_text='<xeno-file path="tests/test_mod.py">assert True\n</xeno-file>',
    )
    assert not (worktree / "tests").exists()  # every refused write, nothing ever landed
    assert final.eval_report is not None
    assert final.eval_report.passed
    assert final.ladder_rung == 0  # reset by checkpoint_step on completion
    assert final.signature_streak == 0  # declines never count toward CB-4
    assert fake.calls.count(NodeRole.DEBUGGER) == 5  # 3 at L1 + 2 at L2, every one refused
    assert len(final.breaker_trips) == 0


def test_l1_patch_budget_halts_when_the_signature_never_changes(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RUNG_BUDGETS[1] == 3 and CB-4's SIGNATURE_STREAK_LIMIT == 3 land on
    the same attempt here — CB-4 fires first (it is checked inside talos(),
    before the ladder's own rung-exhaustion branch ever runs), so the halt
    is attributed to CB-4, not "L1 patch budget", and Phase 3's L2/L3/L4
    rungs are never reached at all.

    Each patch writes DIFFERENT content (a counter) rather than reusing the
    fixed CHIRON_PATCH_OK text: Chiron "fixing" nothing and reproducing an
    IDENTICAL diff on consecutive attempts would trip CB-5 (diff thrash)
    first, which is correct behavior for that scenario but not what this
    test is isolating — CB-4's own counting rule, independent of CB-5.
    """
    _script_gates(monkeypatch, [fail_outcome()] * 6)
    counter = iter(range(1, 10))

    def varying_chiron_patch() -> str:
        return f'<xeno-file path="pkg/mod.py">x = {next(counter)}\n</xeno-file>'

    final, fake = _run(graph_config, worktree, run_paths, chiron_text=varying_chiron_patch)

    assert final.halted
    assert final.halt_reason is not None
    assert final.halt_reason.startswith("CB-4")
    assert final.ladder_rung == 1
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 3  # three L1 attempts, then CB-4 halts
    assert final.signature_streak == 3
    assert len(final.breaker_trips) == 1
    assert final.breaker_trips[0].code.value == "CB-4"


def test_a_changed_signature_resets_the_streak(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(
        monkeypatch,
        [
            fail_outcome(signature="A"),
            fail_outcome(signature="A"),  # L0 retry, same signature
            fail_outcome(signature="B"),  # post-patch: a DIFFERENT defect
            pass_outcome(),  # second patch fixes it
        ],
    )
    final, _ = _run(graph_config, worktree, run_paths)
    assert final.eval_report is not None
    assert final.eval_report.passed
    assert final.signature_streak == 0  # the B->pass transition reset it


def test_cb1_iteration_cap_halts_a_persistently_failing_task(
    providers: dict[str, ProviderSpec],
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xeno.core.config import ModelSpec, NodeSpec

    config = XenoConfig(
        providers=providers,
        tiers={
            Tier.FLAGSHIP: (ModelSpec(provider="ollama", model="big"),),
            Tier.MEDIUM: (ModelSpec(provider="ollama", model="qwen2.5-coder:14b"),),
            Tier.LIGHT: (ModelSpec(provider="ollama", model="qwen2.5-coder:7b"),),
        },
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
        limits=Limits(max_iterations_per_task=1),
    )
    # A different signature on every attempt so CB-4 never wins the race —
    # this test is specifically about CB-1, not CB-4.
    _script_gates(monkeypatch, [fail_outcome(signature=str(i)) for i in range(6)])
    final, fake = _run(config, worktree, run_paths)

    assert final.halted
    assert final.halt_reason is not None
    assert final.halt_reason.startswith("CB-1")
    assert len(final.breaker_trips) == 1
    # CB-1 fires the moment Daedalus's single write brings
    # iterations_this_task to the cap of 1 - Chiron never even gets a turn.
    # Argus's research calls do NOT count against this cap (xeno.graph.argus:
    # it never writes, so it is not a "write attempt" the way Daedalus and
    # Chiron are) — only CODER/DEBUGGER calls move this counter.
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 0


def test_cb1_per_task_cap_also_counts_chiron_attempts(
    providers: dict[str, ProviderSpec],
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Daedalus runs exactly once per task before any ladder
    escalation (Chiron does the L1 looping), so a per-task cap that only
    Daedalus incremented would never fire for a task stuck oscillating
    through L1 alone. Cap set to 3 so it can only be reached via Daedalus's
    one call PLUS two Chiron attempts, not by either alone — and not by
    Argus's research calls, which deliberately do not count (see
    test_cb1_iteration_cap_halts_a_persistently_failing_task)."""
    from xeno.core.config import ModelSpec, NodeSpec

    config = XenoConfig(
        providers=providers,
        tiers={
            Tier.FLAGSHIP: (ModelSpec(provider="ollama", model="big"),),
            Tier.MEDIUM: (ModelSpec(provider="ollama", model="qwen2.5-coder:14b"),),
            Tier.LIGHT: (ModelSpec(provider="ollama", model="qwen2.5-coder:7b"),),
        },
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
        limits=Limits(max_iterations_per_task=3),
    )
    _script_gates(monkeypatch, [fail_outcome(signature=str(i)) for i in range(6)])
    final, fake = _run(config, worktree, run_paths)

    assert final.halted
    assert final.halt_reason is not None
    assert final.halt_reason.startswith("CB-1")
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 2  # 1 (daedalus) + 2 (chiron) == cap of 3


def test_infrastructure_failure_gets_one_l0_retry_then_halts_without_a_patch(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infra = GateOutcome(
        parse_ok=False,
        lint_errors=0,
        type_errors=0,
        tests_run=0,
        tests_failed=0,
        failing_test_ids=(),
        exception_type="",
        failing_location="",
        infrastructure_failure=True,
        log="INFRASTRUCTURE: ruff not found",
    )
    _script_gates(monkeypatch, [infra, infra])
    final, fake = _run(graph_config, worktree, run_paths)

    assert final.halted
    assert final.halt_reason is not None
    assert "infrastructure failure" in final.halt_reason
    # Never a patch: Chiron cannot fix a missing tool (PRD S10), and neither
    # can Argus re-researching or Odysseus re-planning — an infra fault
    # always halts after one L0 retry, regardless of how many ladder rungs
    # exist above L1.
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 0


# ---- Phase 3: L2/L3/L4, checkpoints, and multi-task plans ------------------


def test_l2_re_research_engages_after_l1_is_exhausted_and_recovers(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Distinct signatures throughout so CB-4 (streak limit 3) never fires —
    # this isolates rung-budget exhaustion, not CB-4 (see
    # test_l1_patch_budget_halts_when_the_signature_never_changes for that).
    _script_gates(
        monkeypatch, [fail_outcome(signature=str(i)) for i in range(5)] + [pass_outcome()]
    )
    counter = iter(range(2, 20))

    def varying_chiron_patch() -> str:
        return f'<xeno-file path="pkg/mod.py">x = {next(counter)}\n</xeno-file>'

    final, fake = _run(graph_config, worktree, run_paths, chiron_text=varying_chiron_patch)

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert final.ladder_rung == 0  # reset by checkpoint_step on completion
    assert fake.calls.count(NodeRole.CODER) == 1  # Daedalus writes exactly once (no L3 needed)
    assert fake.calls.count(NodeRole.DEBUGGER) == 4  # L1's 3 attempts + L2's 1
    # skeleton (once) + initial per-task research (rung 0) + L2 re-research
    assert fake.calls.count(NodeRole.RESEARCHER) == 3
    assert len(final.checkpoints) == 1


def test_l3_rollback_and_rewrite_engages_after_l2_is_exhausted(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(
        monkeypatch, [fail_outcome(signature=str(i)) for i in range(7)] + [pass_outcome()]
    )
    chiron_counter = iter(range(2, 20))
    daedalus_responses = iter(
        [DAEDALUS_OK, '<xeno-file path="pkg/mod.py">x = 99\n</xeno-file>']
    )

    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        daedalus_text=lambda: next(daedalus_responses),
        chiron_text=lambda: _mod_write(next(chiron_counter)),
    )

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert final.ladder_rung == 0
    assert fake.calls.count(NodeRole.CODER) == 2  # initial write + L3's rewrite
    assert fake.calls.count(NodeRole.DEBUGGER) == 5  # L1's 3 + L2's 2, never called again
    assert fake.calls.count(NodeRole.RESEARCHER) == 4  # skeleton + initial + L2's 2 rounds
    # L3's rewrite is what's actually committed, not a layering of both
    # writes — proves the rollback (to the run's initial, pre-task commit,
    # since no checkpoint exists yet for this still-incomplete task)
    # actually cleared the worktree before Daedalus wrote again.
    assert (worktree / "pkg" / "mod.py").read_text() == "x = 99"
    assert len(final.checkpoints) == 1
    assert len(final.checkpoints[0].sha) == 40


def test_l4_replan_engages_after_l3_is_exhausted_and_halts_at_l5_if_it_still_fails(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Also proves the ladder is monotonic (`AgentState.ladder_rung`'s
    docstring): once L4's single re-plan attempt fails too, the run halts
    straight to L5 rather than re-descending through L0-L3 with fresh
    budgets, which would make the breaker's bounded-attempts guarantee
    meaningless."""
    _script_gates(monkeypatch, [fail_outcome(signature=str(i)) for i in range(10)])
    daedalus_counter = iter(range(1, 20))
    chiron_counter = iter(range(100, 120))

    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        daedalus_text=lambda: _mod_write(next(daedalus_counter)),
        chiron_text=lambda: _mod_write(next(chiron_counter)),
    )

    assert final.halted
    assert final.halt_reason is not None
    assert "L4" in final.halt_reason
    assert final.ladder_rung == 4
    assert fake.calls.count(NodeRole.CODER) == 4  # initial + 2 L3 rounds + 1 L4 rewrite
    assert fake.calls.count(NodeRole.DEBUGGER) == 5  # L1's 3 + L2's 2
    assert fake.calls.count(NodeRole.PLANNER) == 2  # initial plan + the L4 re-plan
    assert fake.calls.count(NodeRole.RESEARCHER) == 5  # skeleton + initial + 2xL2 + L4
    assert final.checkpoints == []  # the task never passed, so nothing was ever committed


def test_two_task_plan_checkpoints_each_task_and_advances_the_cursor(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(monkeypatch, [pass_outcome(), pass_outcome()])
    daedalus_responses = iter(
        [
            '<xeno-file path="pkg/a.py">a = 1\n</xeno-file>',
            '<xeno-file path="pkg/b.py">b = 1\n</xeno-file>',
        ]
    )
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        planner_text=PLANNER_TWO_TASKS,
        daedalus_text=lambda: next(daedalus_responses),
    )

    assert not final.halted
    assert final.task_count == 2
    assert final.task_cursor == 2
    assert len(final.checkpoints) == 2
    assert [cp.task_index for cp in final.checkpoints] == [0, 1]
    assert (worktree / "pkg" / "a.py").read_text() == "a = 1"
    assert (worktree / "pkg" / "b.py").read_text() == "b = 1"
    assert fake.calls.count(NodeRole.PLANNER) == 1  # one plan, never re-planned
    assert fake.calls.count(NodeRole.CODER) == 2
    assert fake.calls.count(NodeRole.DEBUGGER) == 0  # both tasks pass on the first try
    # skeleton (once) + one research call per task
    assert fake.calls.count(NodeRole.RESEARCHER) == 3
