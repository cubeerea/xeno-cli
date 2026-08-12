"""The full graph (PRD S13): two nested loops.

Outer, once per run: Argus (skeleton) -> Odysseus (roadmap) -> per milestone
[Lachesis (expand) -> ... -> Lachesis (verify)] -> Cerberus.
Inner, once per task: Argus (research) -> Daedalus -> Talos -> the L0-L5
escalation ladder -> next task, or the milestone's verification.

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

from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain, GenericAdapter
from xeno.core.config import Limits, ProviderSpec, SandboxConfig, XenoConfig
from xeno.core.ledger import CostLedger
from xeno.core.paths import RunPaths
from xeno.core.runlog import NullRunLog
from xeno.core.state import AgentState
from xeno.core.types import DEFAULT_NODE_TIERS, Breakpoint, NodeRole, Tier, Verdict
from xeno.core.usage import Usage
from xeno.graph.build import MAX_WRITE_REFUSALS, run_graph
from xeno.graph.gates import GateOutcome
from xeno.graph.lachesis import MAX_PLAN_OBJECTIONS
from xeno.graph.plan import read_roadmap
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers.base import CompletionResult, Provider, ProviderError
from xeno.router.router import Router
from xeno.sandbox.pool import WarmPool, _ensure_image

#: What `GenericAdapter.image()`'s DOCKERFILE already provides (ruff, mypy,
#: pytest) — mirrors the old hardcoded PythonAdapter gate chain minus the
#: dedicated parse step, since `ruff check`/`mypy` already surface syntax
#: errors and the new gate chain has no per-file targeting to hand a parse
#: step (PRD S12 revised: commands are discovered/static, not touched-files-
#: scoped).
_TEST_TOOLCHAIN = DiscoveredToolchain(
    install=None,
    required=(
        DiscoveredCommand(name="lint", argv=("python3", "-m", "ruff", "check", ".")),
        DiscoveredCommand(
            name="typecheck",
            argv=("python3", "-m", "mypy", "--no-error-summary", "--hide-error-context", "."),
        ),
        DiscoveredCommand(
            name="test",
            argv=("python3", "-m", "pytest", "-q", "--tb=short"),
            #: What `xeno.adapters.discovery._classify` would derive for this
            #: command, set explicitly since the fixture bypasses discovery.
            #: Without it the gate profiles collapse into one and the
            #: implementation runs would try to gate on tests that do not
            #: exist yet.
            is_test=True,
        ),
    ),
    advisory=(),
    fingerprint="test-fixture",
)

DAEDALUS_OK = '<xeno-file path="pkg/mod.py">x = 1\n</xeno-file>'
CHIRON_PATCH_OK = '<xeno-file path="pkg/mod.py">x = 2\n</xeno-file>'

#: Odysseus's output is a roadmap now — milestones, no acceptance criteria.
ROADMAP_ONE = '<xeno-milestone outcome="pkg/mod.py works">build the module</xeno-milestone>'
ROADMAP_TWO = (
    '<xeno-milestone outcome="pkg/a.py works">build a</xeno-milestone>'
    '<xeno-milestone outcome="pkg/b.py works">build b</xeno-milestone>'
)
ROADMAP_EXTRA = (
    '<xeno-milestone outcome="pkg/mod2.py works">build the second thing</xeno-milestone>'
)
#: What Odysseus sends back after Lachesis objects: the same target, described
#: in terms the specifier said it could actually decompose.
ROADMAP_REPLANNED = (
    '<xeno-milestone outcome="pkg/mod.py works">build the module from scratch</xeno-milestone>'
)
#: Lachesis refusing a milestone on the merits — the escape hatch, not the
#: parser's "found no usable tags" fallback, which routes elsewhere.
LACHESIS_OBJECTION = (
    "<xeno-objection>this milestone says to extend the parser and no parser "
    "exists in the repository</xeno-objection>"
)

#: Lachesis JOB 1 turns one milestone into tasks; JOB 2 writes its tests.
EXPAND_SINGLE_TASK = (
    '<xeno-task acceptance="pkg/mod.py exists and gates pass">write pkg/mod.py</xeno-task>'
)
EXPAND_TWO_TASKS = (
    '<xeno-task acceptance="pkg/a.py exists and gates pass">write pkg/a.py</xeno-task>'
    '<xeno-task acceptance="pkg/b.py exists and gates pass">write pkg/b.py</xeno-task>'
)
LACHESIS_TESTS = (
    '<xeno-file path="tests/test_mod.py">def test_mod():\n    assert True\n</xeno-file>'
)
ARGUS_SKELETON_TEXT = "A small Python package with no existing modules."
ARGUS_NO_FILES = "<xeno-no-files>nothing to reference yet</xeno-no-files>"
CERBERUS_APPROVE = (
    "<xeno-verdict>approve</xeno-verdict>\n"
    "<xeno-commit-message>feat: test change</xeno-commit-message>"
)
CERBERUS_REJECT_DAEDALUS = (
    "<xeno-verdict>reject_and_return</xeno-verdict>\n"
    "<xeno-destination>daedalus</xeno-destination>\n"
    "<xeno-objections>fix the bug</xeno-objections>"
)
CERBERUS_REJECT_ODYSSEUS = (
    "<xeno-verdict>reject_and_return</xeno-verdict>\n"
    "<xeno-destination>odysseus</xeno-destination>\n"
    "<xeno-objections>the plan misses a step</xeno-objections>"
)


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
        planner_text: str | Callable[[], str] = ROADMAP_ONE,
        expand_text: str | Callable[[], str] = EXPAND_SINGLE_TASK,
        verify_text: str | Callable[[], str] = LACHESIS_TESTS,
        skeleton_text: str = ARGUS_SKELETON_TEXT,
        research_text: str | Callable[[], str] = ARGUS_NO_FILES,
        daedalus_text: str | Callable[[], str] = DAEDALUS_OK,
        triage_text: str = "triaged",
        chiron_text: str | Callable[[], str] = CHIRON_PATCH_OK,
        reviewer_text: str | Callable[[], str] = CERBERUS_APPROVE,
        fail_reviewer: bool = False,
    ) -> None:
        super().__init__(name, spec)
        self.planner_text = planner_text
        self.expand_text = expand_text
        self.verify_text = verify_text
        self.skeleton_text = skeleton_text
        self.research_text = research_text
        self.daedalus_text = daedalus_text
        self.triage_text = triage_text
        self.chiron_text = chiron_text
        self.reviewer_text = reviewer_text
        #: Forces every REVIEWER call to raise a non-retryable ProviderError,
        #: exhausting the (single-entry) chain — the UNREVIEWED path (PRD
        #: S8.2 "Failure"): Cerberus's own model call failed outright.
        self.fail_reviewer = fail_reviewer
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
        elif prompt.node is NodeRole.SPECIFIER:
            # Lachesis's two jobs share one system text (PRD T8), so they are
            # told apart by the current turn's selector — exactly as Argus's
            # three are, and as the real router's caller does.
            source = (
                self.expand_text if prompt.current_turn.startswith("JOB 1") else self.verify_text
            )
            text = source() if callable(source) else source
        elif prompt.node is NodeRole.RESEARCHER:
            if prompt.current_turn.startswith("JOB 1"):
                text = self.skeleton_text
            else:
                text = self.research_text() if callable(self.research_text) else self.research_text
        elif prompt.node is NodeRole.REVIEWER:
            if self.fail_reviewer:
                # `retryable=True` so the router walks the chain rather than
                # re-raising immediately — with this fixture's single-entry
                # FLAGSHIP chain, walking it IS exhausting it, which is what
                # raises the `ChainExhaustedError` `cerberus.py` catches.
                raise ProviderError("simulated flagship outage", provider=self.name, retryable=True)
            text = self.reviewer_text() if callable(self.reviewer_text) else self.reviewer_text
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


class FakeSession:
    """Stands in for `xeno.graph.toolchain.ToolchainSession`.

    The toolchain is fixed and `refresh_if_stale` is a no-op: mid-run
    re-discovery has its own tests (`tests/test_toolchain.py`), and wiring a
    live one in here would make every ladder test depend on a model call it
    has nothing to say about.
    """

    def __init__(
        self,
        *,
        pool: object | None = None,
        toolchain: DiscoveredToolchain = _TEST_TOOLCHAIN,
    ) -> None:
        self.pool = pool or FakePool()
        self.toolchain = toolchain
        self.adapter = GenericAdapter(toolchain)
        self.refresh_calls = 0

    def refresh_if_stale(self, state: object) -> bool:
        self.refresh_calls += 1
        return False


def pass_outcome() -> GateOutcome:
    return GateOutcome(
        passed=True,
        failed_command="",
        exit_code=0,
        infrastructure_failure=False,
        log="all green",
    )


def fail_outcome(*, signature: str = "A") -> GateOutcome:
    return GateOutcome(
        passed=False,
        failed_command="test",
        exit_code=1,
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
    planner_text: str | Callable[[], str] = ROADMAP_ONE,
    expand_text: str | Callable[[], str] = EXPAND_SINGLE_TASK,
    verify_text: str | Callable[[], str] = LACHESIS_TESTS,
    skeleton_text: str = ARGUS_SKELETON_TEXT,
    research_text: str | Callable[[], str] = ARGUS_NO_FILES,
    daedalus_text: str | Callable[[], str] = DAEDALUS_OK,
    chiron_text: str | Callable[[], str] = CHIRON_PATCH_OK,
    reviewer_text: str | Callable[[], str] = CERBERUS_APPROVE,
    fail_reviewer: bool = False,
    session: object | None = None,
) -> tuple[AgentState, ScriptedProvider]:
    ledger = CostLedger(run_id="t")
    router = Router(graph_config, ledger=ledger)
    fake = ScriptedProvider(
        "ollama",
        graph_config.providers["ollama"],
        planner_text=planner_text,
        expand_text=expand_text,
        verify_text=verify_text,
        skeleton_text=skeleton_text,
        research_text=research_text,
        daedalus_text=daedalus_text,
        chiron_text=chiron_text,
        reviewer_text=reviewer_text,
        fail_reviewer=fail_reviewer,
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
        session=session or FakeSession(),  # type: ignore[arg-type]
        repo_root=run_paths.repo_root,
    )
    return final, fake


def _script_gates(monkeypatch: pytest.MonkeyPatch, outcomes: list[GateOutcome]) -> None:
    queue = list(outcomes)

    def fake_run_gates(sandbox, toolchain, **kwargs):  # type: ignore[no-untyped-def]
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
    adapter = GenericAdapter(_TEST_TOOLCHAIN)
    pool = WarmPool(
        client,
        adapter,
        SandboxConfig(warm_pool_size=1, memory="512m", cpus=1.0),
        workspace_root=tmp_path / "sandboxes",
        network_disabled=True,
    )
    _ensure_image(client, adapter.image(), adapter.dockerfile())
    pool.fill()
    try:
        yield pool
    finally:
        pool.shutdown()


@requires_docker
def test_happy_path_passes_with_real_sandbox_and_gates(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths, real_pool: WarmPool
) -> None:
    # Deliberately NO test fixture on disk. `pytest` exits 5 ("no tests
    # collected") on an empty suite, which `run_gates` correctly reads as a
    # failed required command — so this run can only go green if the gate
    # profiles do what they claim: the implementation task gates on
    # lint+typecheck with the test command held back, and pytest runs for the
    # first time on the verification pass, against the test Lachesis itself
    # wrote. Adding a placeholder here would hide exactly the behaviour under
    # test.
    final, fake = _run(graph_config, worktree, run_paths, session=FakeSession(pool=real_pool))

    assert final.eval_report is not None
    assert final.eval_report.passed
    assert not final.halted
    assert final.ladder_rung == 0
    assert final.rung_attempts == 0
    assert (worktree / "pkg" / "mod.py").read_text() == "x = 1"
    assert (worktree / "tests" / "test_mod.py").is_file(), "Lachesis wrote the milestone's tests"
    assert final.task_count == 1
    assert final.task_cursor == 1
    assert final.milestone_cursor == 1
    # Two: the task's own green, then the milestone's, which is the first
    # evaluation in the run where the test command actually ran.
    assert len(final.checkpoints) == 2
    assert final.review_verdict is Verdict.APPROVE
    # Skeleton, roadmap, expand, per-task research, the one write, the
    # milestone's tests, then Cerberus's review — no failure, so no triage
    # call and no Chiron call was ever made.
    assert fake.calls == [
        NodeRole.RESEARCHER,
        NodeRole.PLANNER,
        NodeRole.SPECIFIER,
        NodeRole.RESEARCHER,
        NodeRole.CODER,
        NodeRole.SPECIFIER,
        NodeRole.REVIEWER,
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
        chiron_text='<xeno-file path="tests/test_chiron.py">assert True\n</xeno-file>',
    )
    # A distinct path from the one Lachesis legitimately writes, so this
    # asserts Chiron's write never landed rather than that no test exists.
    assert not (worktree / "tests" / "test_chiron.py").exists()
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
        passed=False,
        failed_command="lint",
        exit_code=None,
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
    # The task's own green, then the milestone's once its tests pass.
    assert len(final.checkpoints) == 2


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
    assert len(final.checkpoints) == 2  # the task's, then the milestone's
    assert len(final.checkpoints[0].sha) == 40


def test_l4_reexpand_engages_after_l3_is_exhausted_and_halts_at_l5_if_it_still_fails(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Also proves the ladder is monotonic (`AgentState.ladder_rung`'s
    docstring): once L4's single re-expansion attempt fails too, the run halts
    straight to L5 rather than re-descending through L0-L3 with fresh
    budgets, which would make the breaker's bounded-attempts guarantee
    meaningless.

    L4 goes to Lachesis, not Odysseus: the task that got stuck is one
    Lachesis wrote, and re-deriving the whole roadmap over a single
    unachievable task would throw away milestones that are already built and
    tested.
    """
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
    assert fake.calls.count(NodeRole.PLANNER) == 1, "the roadmap is never re-derived by the ladder"
    assert fake.calls.count(NodeRole.SPECIFIER) == 2  # initial expansion + the L4 re-expansion
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
        expand_text=EXPAND_TWO_TASKS,
        daedalus_text=lambda: next(daedalus_responses),
    )

    assert not final.halted
    assert final.task_count == 2
    assert final.task_cursor == 2
    # One per task, plus the milestone's own once its tests pass.
    assert len(final.checkpoints) == 3
    assert [cp.task_index for cp in final.checkpoints] == [0, 1, 2]
    assert (worktree / "pkg" / "a.py").read_text() == "a = 1"
    assert (worktree / "pkg" / "b.py").read_text() == "b = 1"
    assert fake.calls.count(NodeRole.PLANNER) == 1  # one roadmap, never revised
    assert fake.calls.count(NodeRole.SPECIFIER) == 2  # one expansion + one verification
    assert fake.calls.count(NodeRole.CODER) == 2
    assert fake.calls.count(NodeRole.DEBUGGER) == 0  # both tasks pass on the first try
    # skeleton (once) + one research call per task
    assert fake.calls.count(NodeRole.RESEARCHER) == 3


# ---- Phase 4: Cerberus, the human gate (PRD S8, S13) -----------------------


def test_e15_approve_reviews_the_diff_and_reaches_a_terminal_approve(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(monkeypatch, [pass_outcome()])
    final, fake = _run(graph_config, worktree, run_paths)

    assert not final.halted
    assert final.review_verdict is Verdict.APPROVE
    assert final.commit_message == "feat: test change"
    assert final.cerberus_notes is not None
    assert final.review_diff_handle is not None
    assert "x = 1" in final.review_diff_handle.read_text()
    assert fake.calls.count(NodeRole.REVIEWER) == 1


def test_l5_halt_reaches_cerberus_and_escalates_with_no_model_call(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E12 (PRD S8.3): Cerberus does not judge an already-halted run, it
    reports — deterministically, spending no FLAGSHIP call."""
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
    assert final.ladder_rung == 4
    assert final.review_verdict is Verdict.ESCALATE
    assert final.cerberus_notes is not None
    assert "L4" in final.cerberus_notes.read_text()
    assert final.review_diff_handle is not None
    assert fake.calls.count(NodeRole.REVIEWER) == 0  # deterministic report, no model call


def test_breaker_trip_reaches_cerberus_and_escalates_with_no_model_call(
    providers: dict[str, ProviderSpec],
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E13 (PRD S8.3): a circuit breaker trip is also handed to Cerberus,
    not straight to the caller — same deterministic, no-model-call report."""
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
    _script_gates(monkeypatch, [fail_outcome(signature=str(i)) for i in range(6)])
    final, fake = _run(config, worktree, run_paths)

    assert final.halted
    assert final.halt_reason is not None
    assert final.halt_reason.startswith("CB-1")
    assert final.review_verdict is Verdict.ESCALATE
    assert final.cerberus_notes is not None
    assert "CB-1" in final.cerberus_notes.read_text()
    assert fake.calls.count(NodeRole.REVIEWER) == 0


def test_e16_reject_to_daedalus_appends_a_task_and_the_run_still_completes(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(monkeypatch, [pass_outcome(), pass_outcome()])
    reviewer_responses = iter([CERBERUS_REJECT_DAEDALUS, CERBERUS_APPROVE])
    final, fake = _run(
        graph_config, worktree, run_paths, reviewer_text=lambda: next(reviewer_responses)
    )

    assert not final.halted
    assert final.review_verdict is Verdict.APPROVE
    assert final.reject_count == 1
    assert final.reject_destination is None  # cleared once Daedalus consumed it
    assert final.task_count == 2  # the original task plus Cerberus's appended one
    # The task's, the milestone's, then the appended fix task's.
    assert len(final.checkpoints) == 3
    assert final.task_cursor == 2, "the appended fix is a task and advances the cursor"
    assert fake.calls.count(NodeRole.REVIEWER) == 2
    assert fake.calls.count(NodeRole.CODER) == 2  # original write + the fix
    assert fake.calls.count(NodeRole.PLANNER) == 1  # never routed to Odysseus


def test_e17_reject_to_odysseus_extends_the_roadmap_and_the_run_still_completes(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PLANNER rejection now operates at roadmap level.

    Every milestone already passed its own tests by the time Cerberus sees the
    run, so the objection can only be that the roadmap was incomplete — the
    answer is another milestone, which Lachesis then expands like any other.
    """
    _script_gates(monkeypatch, [pass_outcome()])
    reviewer_responses = iter([CERBERUS_REJECT_ODYSSEUS, CERBERUS_APPROVE])
    # Odysseus keeps the milestones already built and appends to them, so the
    # rejection response carries only the milestone that was missing.
    planner_responses = iter([ROADMAP_ONE, ROADMAP_EXTRA])
    expand_responses = iter(
        [
            EXPAND_SINGLE_TASK,
            '<xeno-task acceptance="pkg/mod2.py exists and gates pass">'
            "write pkg/mod2.py</xeno-task>",
        ]
    )
    daedalus_responses = iter([DAEDALUS_OK, '<xeno-file path="pkg/mod2.py">y = 1\n</xeno-file>'])
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        reviewer_text=lambda: next(reviewer_responses),
        planner_text=lambda: next(planner_responses),
        expand_text=lambda: next(expand_responses),
        daedalus_text=lambda: next(daedalus_responses),
    )

    assert not final.halted
    assert final.review_verdict is Verdict.APPROVE
    assert final.reject_count == 1
    assert final.reject_destination is None  # cleared once Odysseus consumed it
    assert final.milestone_count == 2  # the completed milestone kept, one appended
    assert final.milestone_cursor == 2
    assert final.task_count == 2  # completed task kept, the new milestone's appended
    assert fake.calls.count(NodeRole.REVIEWER) == 2
    assert fake.calls.count(NodeRole.PLANNER) == 2  # initial roadmap + the E17 extension
    assert fake.calls.count(NodeRole.SPECIFIER) == 4  # expand + verify, per milestone
    assert (worktree / "pkg" / "mod2.py").read_text() == "y = 1"


def test_a_lachesis_objection_sends_the_milestone_back_to_odysseus(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A milestone that cannot be expanded is a defect in the roadmap, and
    Odysseus is the only node that may rewrite one.

    This is the run that used to die: the objection was the whole halt reason
    and nobody acted on it. Odysseus had never been told, even though it wrote
    the milestone and could have replaced it.
    """
    _script_gates(monkeypatch, [pass_outcome()])
    planner_turns: list[str] = []

    class _Recording(ScriptedProvider):
        def complete(self, prompt, model, **kwargs):  # type: ignore[no-untyped-def]
            if prompt.node is NodeRole.PLANNER:
                planner_turns.append(prompt.current_turn)
            return super().complete(prompt, model, **kwargs)

    planner_responses = iter([ROADMAP_ONE, ROADMAP_REPLANNED])
    expand_responses = iter([LACHESIS_OBJECTION, EXPAND_SINGLE_TASK])
    ledger = CostLedger(run_id="t")
    router = Router(graph_config, ledger=ledger)
    fake = _Recording(
        "ollama",
        graph_config.providers["ollama"],
        planner_text=lambda: next(planner_responses),
        expand_text=lambda: next(expand_responses),
    )
    router._providers["ollama"] = fake
    final = run_graph(
        router=router,
        config=graph_config,
        keyring=CacheKeyring(run_id="t", worktree_root=worktree),
        paths=run_paths,
        worktree=worktree,
        runlog=NullRunLog(),
        state=AgentState(run_id="t", goal="write pkg/mod.py"),
        session=FakeSession(),  # type: ignore[arg-type]
        repo_root=run_paths.repo_root,
    )

    assert not final.halted
    assert final.review_verdict is Verdict.APPROVE
    assert final.plan_objection is None, "cleared once Odysseus consumed it"
    assert final.plan_objection_count == 0, "the re-planned milestone expanded, so the budget reset"
    assert fake.calls.count(NodeRole.PLANNER) == 2  # the roadmap, then the re-plan

    # The re-plan is a repair, not a re-roll: Odysseus is told which milestone
    # was refused and why, or the second call is the same prompt as the first.
    assert "no parser exists in the repository" in planner_turns[1]
    assert "milestone 1 of 1" in planner_turns[1]
    assert "build the module" in planner_turns[1], "the refused milestone is quoted back"

    # And its answer replaced the milestone rather than being appended beside it.
    assert final.roadmap is not None
    assert [m.description for m in read_roadmap(final.roadmap).milestones] == [
        "build the module from scratch"
    ]
    assert final.milestone_count == 1
    assert (worktree / "pkg" / "mod.py").exists()


def test_the_objection_budget_halts_to_cerberus_once_odysseus_cannot_fix_it(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cycle is bounded like every other one in this graph, and what the
    bound buys is a diagnosis: three objections mean Odysseus re-planned with
    the objections in hand and the milestone stayed unbuildable, so the fault
    is upstream of both nodes."""
    _script_gates(monkeypatch, [pass_outcome()])
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        expand_text=LACHESIS_OBJECTION,
    )

    assert final.halted
    assert final.plan_objection_count == MAX_PLAN_OBJECTIONS
    assert final.plan_objection is None, "an exhausted budget routes to Cerberus, not Odysseus"
    reason = final.halt_reason or ""
    assert "no parser exists in the repository" in reason
    assert "still not expandable" in reason

    # Odysseus was re-entered for every objection except the last, which had
    # no budget left to spend on it.
    assert fake.calls.count(NodeRole.PLANNER) == MAX_PLAN_OBJECTIONS
    assert fake.calls.count(NodeRole.SPECIFIER) == MAX_PLAN_OBJECTIONS
    # A halt is Cerberus's to resolve, and nothing was ever built to review.
    assert final.review_verdict is Verdict.ESCALATE
    assert not (worktree / "pkg").exists()


def test_reject_budget_exhaustion_converts_the_third_rejection_to_escalate(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD S8.3: `max_rejections_per_run` (default 2, `Limits()`'s default)
    — the rejection BEYOND budget converts to ESCALATE without touching
    `ladder_rung`, a budget entirely separate from the escalation ladder."""
    _script_gates(monkeypatch, [pass_outcome(), pass_outcome(), pass_outcome()])
    reviewer_responses = iter(
        [CERBERUS_REJECT_DAEDALUS, CERBERUS_REJECT_DAEDALUS, CERBERUS_REJECT_DAEDALUS]
    )
    final, fake = _run(
        graph_config, worktree, run_paths, reviewer_text=lambda: next(reviewer_responses)
    )

    assert final.halted
    assert final.review_verdict is Verdict.ESCALATE
    assert final.reject_count == 2  # Limits().max_rejections_per_run
    assert final.ladder_rung == 0  # the reject budget never touches the ladder
    assert final.task_count == 3  # original + 2 appended-then-rejected-again tasks
    # The first task's, the milestone's, then one per appended fix task.
    assert len(final.checkpoints) == 4
    assert fake.calls.count(NodeRole.REVIEWER) == 3


def test_cerberus_chain_exhaustion_escalates_as_unreviewed(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD S8.2 'Failure': the harness must never auto-approve in the
    absence of a review. A `None` `cerberus_notes` on an ESCALATE IS the
    UNREVIEWED signal."""
    _script_gates(monkeypatch, [pass_outcome()])
    final, fake = _run(graph_config, worktree, run_paths, fail_reviewer=True)

    assert final.halted
    assert final.review_verdict is Verdict.ESCALATE
    assert final.cerberus_notes is None
    assert final.review_diff_handle is not None  # still computed before the call was attempted
    assert fake.calls.count(NodeRole.REVIEWER) == 1  # the attempt was made and failed


# ---- test-file guardrail (PRD S10) -----------------------------------------


def test_passing_tasks_accumulate_touched_test_files_onto_the_run(tmp_path: Path) -> None:
    """`EvalReport` is per-task and is overwritten by the next task, but
    Cerberus reviews the whole run at the end — so the flag has to be carried
    up to run level as each task checkpoints."""
    from xeno.core import vcs
    from xeno.core.ledger import CostLedger
    from xeno.core.state import EvalReport, Handle
    from xeno.graph.checkpoints import checkpoint_step
    from xeno.graph.plan import Plan, PlanTask, write_plan

    plan_path = tmp_path / "plan.json"
    write_plan(
        plan_path,
        Plan(
            tasks=[
                PlanTask(description="a", acceptance="x"),
                PlanTask(description="b", acceptance="y"),
            ]
        ),
    )
    vcs.init_repo(tmp_path)

    state = AgentState(run_id="r", goal="g")
    state.plan = Handle.for_file(plan_path, summary="plan")
    ledger = CostLedger(run_id="r")

    state.eval_report = EvalReport(passed=True, touched_test_files=["tests/test_a.py"])
    checkpoint_step(state, tmp_path, ledger)
    state.eval_report = EvalReport(passed=True, touched_test_files=["tests/test_b.py"])
    checkpoint_step(state, tmp_path, ledger)

    assert state.touched_test_files == ["tests/test_a.py", "tests/test_b.py"]


def test_cerberus_is_told_when_the_run_modified_test_files() -> None:
    """Chiron is refused outright for touching a test file, but Daedalus
    writes tests legitimately — telling the two apart is Cerberus's call, so
    the fact must actually reach its prompt."""
    from xeno.graph.cerberus import _build_current_turn
    from xeno.graph.plan import Plan, PlanTask

    state = AgentState(run_id="r", goal="add rate limiting")
    plan = Plan(tasks=[PlanTask(description="a", acceptance="x")])

    clean = _build_current_turn(state, plan, "diff")
    assert "modified test file(s)" not in clean

    state.touched_test_files = ["tests/test_limits.py"]
    flagged = _build_current_turn(state, plan, "diff")
    assert "modified test file(s): tests/test_limits.py" in flagged
    assert "weakened to make the gates pass" in flagged


# ---- the plan reaches the nodes that act on it ----------------------------


def _plan_handle(tmp_path: Path, description: str, acceptance: str) -> object:
    from xeno.core.state import Handle
    from xeno.graph.plan import Plan, PlanTask, write_plan

    path = tmp_path / "plan.json"
    write_plan(path, Plan(tasks=[PlanTask(description=description, acceptance=acceptance)]))
    return Handle.for_file(path, summary="plan")


def test_daedalus_is_told_the_current_task_not_just_the_goal(tmp_path: Path) -> None:
    """`DAEDALUS_SYSTEM` instructs this node to implement "the current plan
    task". Until the task was actually supplied, that pointed at nothing and
    every task in the plan was implemented as the whole goal — on a
    greenfield run, the scaffold task produced the finished feature and the
    toolchain the rest of the plan depended on was never created.
    """
    from xeno.graph.daedalus import _build_current_turn

    state = AgentState(run_id="t", goal="build a calculator library")
    state.plan = _plan_handle(  # type: ignore[assignment]
        tmp_path, "Create pyproject.toml and a placeholder test", "pytest exits 0"
    )
    state.task_count = 1

    turn = _build_current_turn(state, "")

    assert "Create pyproject.toml and a placeholder test" in turn
    assert "pytest exits 0" in turn
    assert "build a calculator library" in turn, "the goal is still the context"


def test_chiron_is_told_the_current_task_too(tmp_path: Path) -> None:
    """The gates that just failed were evaluating the CURRENT task, so a
    repair has to target that rather than the plan's destination."""
    from xeno.core.state import EvalReport
    from xeno.graph.chiron import _build_current_turn

    state = AgentState(run_id="t", goal="build a calculator library")
    state.plan = _plan_handle(  # type: ignore[assignment]
        tmp_path, "Create pyproject.toml and a placeholder test", "pytest exits 0"
    )
    state.eval_report = EvalReport(passed=False, failed_command="test", first_failure="boom")

    turn = _build_current_turn(state, "")

    assert "Create pyproject.toml and a placeholder test" in turn
    assert "pytest exits 0" in turn


def test_chiron_is_told_why_its_last_patch_was_refused(tmp_path: Path) -> None:
    """A refusal the offender cannot see is indistinguishable from the patch
    silently failing — which is how one refused block burns a whole rung."""
    from xeno.core.state import EvalReport
    from xeno.graph.chiron import _build_current_turn

    state = AgentState(run_id="t", goal="g")
    state.eval_report = EvalReport(passed=False, failed_command="lint", first_failure="W292")

    turn = _build_current_turn(state, "patch touched test file(s), refused: tests/test_x.py")

    assert "refused" in turn
    assert "tests/test_x.py" in turn
    assert "NOTHING was written" in turn


def test_write_nodes_fall_back_to_the_goal_when_no_plan_exists() -> None:
    """Neither write node should crash a run over a missing plan."""
    from xeno.graph.daedalus import _build_current_turn

    state = AgentState(run_id="t", goal="do the thing")
    assert "do the thing" in _build_current_turn(state, "")


# ---- greenfield planning --------------------------------------------------


EXPANSION_IGNORES_SCAFFOLD = '<xeno-task acceptance="mod exists">add pkg/x.py</xeno-task>'


def _greenfield_plan(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths, established: bool
) -> list[str]:
    from xeno.core.ledger import CostLedger as _Ledger
    from xeno.core.state import Handle
    from xeno.graph.lachesis import make_lachesis_nodes
    from xeno.graph.plan import Roadmap, read_plan, write_roadmap

    router = Router(graph_config, ledger=_Ledger(run_id="t"))
    router._providers["ollama"] = ScriptedProvider(
        "ollama", graph_config.providers["ollama"], expand_text=EXPANSION_IGNORES_SCAFFOLD
    )
    expand, _verify = make_lachesis_nodes(
        router=router,
        config=graph_config,
        keyring=CacheKeyring(run_id="t", worktree_root=worktree),
        paths=run_paths,
        worktree=worktree,
        touched_files=[],
        toolchain_established=lambda: established,
        report_refused=lambda _refused: None,
    )

    roadmap_path = run_paths.workspace / "roadmap.json"
    write_roadmap(roadmap_path, Roadmap(milestones=[_milestone("build the thing", "it works")]))

    state = AgentState(run_id="t", goal="build a thing")
    state.roadmap = Handle.for_file(roadmap_path, summary="roadmap")
    state.milestone_count = 1

    state = expand(state)
    assert state.plan is not None
    return [t.description for t in read_plan(state.plan).tasks]


def _milestone(description: str, outcome: str) -> object:
    from xeno.graph.plan import Milestone

    return Milestone(description=description, outcome=outcome)


def test_a_greenfield_expansion_always_starts_by_establishing_the_toolchain(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths
) -> None:
    """The scaffold task is prepended by the harness, not requested from a
    model. A greenfield plan that does not begin by creating a manifest
    cannot gate a single one of its own tasks — every later task fails on
    "no toolchain" until the ladder gives up — and a real model was observed
    skipping it despite being told not to.
    """
    tasks = _greenfield_plan(graph_config, worktree, run_paths, established=False)

    assert len(tasks) == 2
    assert "Establish this project's toolchain" in tasks[0]
    assert tasks[1] == "add pkg/x.py", "the expansion's own tasks still follow"


def test_the_scaffold_task_no_longer_asks_for_a_placeholder_test(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths
) -> None:
    """It used to, purely to dodge `pytest` exiting 5 on an empty suite. That
    reason is gone — the test command is held back until Lachesis has written
    tests — and the requirement is now actively wrong, since Daedalus is
    refused any write that touches a test file."""
    scaffold = _greenfield_plan(graph_config, worktree, run_paths, established=False)[0]
    assert "placeholder test" not in scaffold


def test_an_established_toolchain_gets_no_scaffold_task(
    graph_config: XenoConfig, worktree: Path, run_paths: RunPaths
) -> None:
    tasks = _greenfield_plan(graph_config, worktree, run_paths, established=True)
    assert tasks == ["add pkg/x.py"]


# ---- the milestone loop ---------------------------------------------------


def test_a_two_milestone_roadmap_expands_and_verifies_each_one_in_turn(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the split: the second milestone is expanded AFTER
    the first one's code is on disk, so its tasks can name real code instead
    of predicting it."""
    _script_gates(monkeypatch, [pass_outcome()])
    expand_responses = iter(
        [
            '<xeno-task acceptance="pkg/a.py exists">write pkg/a.py</xeno-task>',
            '<xeno-task acceptance="pkg/b.py exists">write pkg/b.py</xeno-task>',
        ]
    )
    verify_responses = iter(
        [
            '<xeno-file path="tests/test_a.py">def test_a():\n    assert True\n</xeno-file>',
            '<xeno-file path="tests/test_b.py">def test_b():\n    assert True\n</xeno-file>',
        ]
    )
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
        planner_text=ROADMAP_TWO,
        expand_text=lambda: next(expand_responses),
        verify_text=lambda: next(verify_responses),
        daedalus_text=lambda: next(daedalus_responses),
    )

    assert not final.halted
    assert final.milestone_count == 2
    assert final.milestone_cursor == 2
    assert final.task_count == 2, "the plan grew one milestone at a time"
    assert (worktree / "tests" / "test_a.py").is_file()
    assert (worktree / "tests" / "test_b.py").is_file()
    # 2 tasks + 2 milestones.
    assert len(final.checkpoints) == 4
    assert fake.calls.count(NodeRole.PLANNER) == 1  # one roadmap for the whole run
    assert fake.calls.count(NodeRole.SPECIFIER) == 4  # expand + verify, twice


def test_the_second_expansion_sees_the_first_milestones_code(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not merely that it runs later — that the codebase map it is handed
    actually contains what the previous milestone built. This is the entire
    reason expansion was split out of planning."""
    _script_gates(monkeypatch, [pass_outcome()])
    turns: list[str] = []

    class _Recording(ScriptedProvider):
        def complete(self, prompt, model, **kwargs):  # type: ignore[no-untyped-def]
            if prompt.node is NodeRole.SPECIFIER and prompt.current_turn.startswith("JOB 1"):
                block = prompt.block(Breakpoint.CODEBASE_MAP)
                turns.append(block.text if block else "")
            return super().complete(prompt, model, **kwargs)

    ledger = CostLedger(run_id="t")
    router = Router(graph_config, ledger=ledger)
    router._providers["ollama"] = _Recording(
        "ollama",
        graph_config.providers["ollama"],
        planner_text=ROADMAP_TWO,
        daedalus_text='<xeno-file path="pkg/first.py">FIRST = 1\n</xeno-file>',
    )
    run_graph(
        router=router,
        config=graph_config,
        keyring=CacheKeyring(run_id="t", worktree_root=worktree),
        paths=run_paths,
        worktree=worktree,
        runlog=NullRunLog(),
        state=AgentState(run_id="t", goal="build a thing"),
        session=FakeSession(),  # type: ignore[arg-type]
        repo_root=run_paths.repo_root,
    )

    assert len(turns) == 2
    assert "pkg/first.py" not in turns[0], "nothing existed when the first expansion ran"
    assert "pkg/first.py" in turns[1], "the second expansion can name what was actually built"


# ---- test authorship is exclusive to Lachesis -----------------------------


def test_daedalus_bundling_a_test_file_is_refused_and_retried(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused WHOLE, then retried — never evaluated in between.

    Falling through to Talos after a refused write would be actively wrong:
    nothing was written, so the worktree still holds the previous task's green
    state and the gates would checkpoint a task that implemented nothing.
    """
    _script_gates(monkeypatch, [pass_outcome()])
    daedalus_responses = iter(
        [
            '<xeno-file path="pkg/mod.py">x = 1\n</xeno-file>'
            '<xeno-file path="tests/test_mine.py">assert True\n</xeno-file>',
            DAEDALUS_OK,
        ]
    )
    final, fake = _run(
        graph_config, worktree, run_paths, daedalus_text=lambda: next(daedalus_responses)
    )

    assert not final.halted
    assert not (worktree / "tests" / "test_mine.py").exists()
    assert (worktree / "pkg" / "mod.py").read_text() == "x = 1", "the retry's write landed"
    assert fake.calls.count(NodeRole.CODER) == 2  # the refused attempt, then the clean one


def test_daedalus_refusing_forever_halts_rather_than_looping(
    graph_config: XenoConfig,
    worktree: Path,
    run_paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_gates(monkeypatch, [pass_outcome()])
    final, fake = _run(
        graph_config,
        worktree,
        run_paths,
        daedalus_text='<xeno-file path="tests/test_mine.py">assert True\n</xeno-file>',
    )

    assert final.halted
    assert "tests are Lachesis's to write" in (final.halt_reason or "")
    assert fake.calls.count(NodeRole.CODER) == MAX_WRITE_REFUSALS
    assert final.review_verdict is Verdict.ESCALATE, "a halt still reaches the human gate"
