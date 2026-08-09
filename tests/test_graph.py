"""Phase 2: the Daedalus -> Talos -> Chiron loop (PRD S13).

Uses the same fake-provider technique as test_router.py: what is under test
is graph/ladder/breaker wiring, not wire formats, so the network boundary is
faked and everything above it (Router, PromptBuilder, CacheKeyring,
breakers) runs for real.

Most tests fake Talos's gates via monkeypatch AND use a fake sandbox pool,
because what is under test is the ladder/breaker routing, not the sandbox or
the gate tools themselves — those get their own live-Docker coverage in
`test_happy_path_passes_with_real_sandbox_and_gates`, the one test that
exercises the real warm pool end to end.
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


class ScriptedProvider(Provider):
    """Returns a canned response per node, distinguished by which node the
    prompt was built for — not by model name, so the fake stays correct if
    the fixture config's model strings ever change."""

    def __init__(
        self,
        name: str,
        spec: ProviderSpec,
        *,
        daedalus_text: str = DAEDALUS_OK,
        triage_text: str = "triaged",
        chiron_text: str | Callable[[], str] = CHIRON_PATCH_OK,
    ) -> None:
        super().__init__(name, spec)
        self.daedalus_text = daedalus_text
        self.triage_text = triage_text
        self.chiron_text = chiron_text
        self.cache_capable = True
        self.calls: list[NodeRole] = []

    def complete(self, prompt, model, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.calls.append(prompt.node)
        if prompt.node is NodeRole.CODER:
            text = self.daedalus_text
        elif prompt.node is NodeRole.DEBUGGER:
            text = self.chiron_text() if callable(self.chiron_text) else self.chiron_text
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
    daedalus_text: str = DAEDALUS_OK,
    chiron_text: str | Callable[[], str] = CHIRON_PATCH_OK,
    pool: object | None = None,
    adapter: LanguageAdapter | None = None,
) -> tuple[AgentState, ScriptedProvider]:
    ledger = CostLedger(run_id="t")
    router = Router(graph_config, ledger=ledger)
    fake = ScriptedProvider(
        "ollama",
        graph_config.providers["ollama"],
        daedalus_text=daedalus_text,
        chiron_text=chiron_text,
    )
    router._providers["ollama"] = fake
    keyring = CacheKeyring(run_id="t", worktree_root=worktree)
    state = AgentState(run_id="t", goal="write pkg/mod.py")
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
    # No failure, so no triage call and no Chiron call was ever made.
    assert fake.calls == [NodeRole.CODER]


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
    assert list(worktree.iterdir()) == []


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
    assert final.rung_attempts == 1
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
    assert final.ladder_rung == 1
    assert final.rung_attempts == 1
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
    assert final.ladder_rung == 1
    assert final.rung_attempts == 2  # first patch declined, second recovers
    assert final.signature_streak == 0  # a decline is not an intervention (PRD S10)
    assert fake.calls.count(NodeRole.DEBUGGER) == 2


def test_chiron_refuses_to_patch_a_test_file(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Chiron refuses on every attempt (fixed veto-triggering text), so this
    # exhausts the whole L1 budget without ever counting toward CB-4 (a
    # refused write is a decline, not a MODEL_AUTHORED intervention).
    _script_gates(monkeypatch, [fail_outcome()] * 6)
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        chiron_text='<xeno-file path="tests/test_mod.py">assert True\n</xeno-file>',
    )
    assert not (worktree / "tests").exists()  # every refused write, nothing ever landed
    assert final.halted
    assert final.halt_reason is not None
    assert "L1 patch" in final.halt_reason
    assert final.ladder_rung == 1
    assert final.rung_attempts == 3  # all three L1 attempts refused
    assert final.signature_streak == 0  # declines never count toward CB-4
    assert fake.calls.count(NodeRole.DEBUGGER) == 3


def test_l1_patch_budget_halts_when_the_signature_never_changes(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RUNG_BUDGETS[1] == 3 and CB-4's SIGNATURE_STREAK_LIMIT == 3 land on
    the same attempt here — CB-4 fires first (it is checked inside talos(),
    before the ladder's own rung-exhaustion branch ever runs), so the halt
    is attributed to CB-4, not "L1 patch budget".

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
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 0


def test_cb1_per_task_cap_also_counts_chiron_attempts(
    providers: dict[str, ProviderSpec],
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Daedalus runs exactly once per task in Phase 2 (Chiron
    does all the looping), so a per-task cap that only Daedalus incremented
    would never fire for a task stuck oscillating through L1 alone. Cap set
    to 3 so it can only be reached via Daedalus's one call PLUS two Chiron
    attempts, not by either alone."""
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
    # Never a patch: Chiron cannot fix a missing tool (PRD S10).
    assert fake.calls.count(NodeRole.CODER) == 1
    assert fake.calls.count(NodeRole.DEBUGGER) == 0
