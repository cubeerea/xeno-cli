"""Routing, fallback chains, and cost attribution (PRD S9.5, S15.1).

Uses a fake provider rather than mocked HTTP: what is under test is routing
policy — which model gets picked, what happens when one fails, how tokens are
attributed — not wire formats. Those are covered in test_providers.py.
"""

from __future__ import annotations

import pytest

from xeno.core.config import Limits, ModelSpec, NodeSpec, ProviderSpec, XenoConfig
from xeno.core.ledger import CostLedger
from xeno.core.runlog import EventKind, NullRunLog
from xeno.core.state import AgentState
from xeno.core.types import DEFAULT_NODE_TIERS, Breakpoint, NodeRole, ProviderFamily, Tier
from xeno.core.usage import Usage
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers.base import CompletionResult, Provider, ProviderError
from xeno.router.router import BudgetExceededError, ChainExhaustedError, Router


class FakeProvider(Provider):
    """Scripted provider: each call pops the next outcome."""

    def __init__(self, name: str, spec: ProviderSpec) -> None:
        super().__init__(name, spec)
        self.outcomes: list[CompletionResult | ProviderError] = []
        self.calls: list[str] = []
        self.cache_capable = True

    def complete(self, prompt, model, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.calls.append(model.ref)
        outcome = self.outcomes.pop(0) if self.outcomes else _ok(model.model)
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome

    def health_check(self) -> tuple[bool, str]:
        return True, "fake"


def _ok(model: str, *, input_tokens: int = 1000, cached: int = 0) -> CompletionResult:
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=100,
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )
    return CompletionResult(
        text="ok",
        model=model,
        usage=usage,
        latency_ms=12.5,
        by_breakpoint={Breakpoint.SYSTEM: _stats(cached, input_tokens - cached)},
    )


def _stats(hit: int, miss: int):  # type: ignore[no-untyped-def]
    from xeno.core.state import BreakpointStats

    return BreakpointStats(hit_tokens=hit, miss_tokens=miss)


@pytest.fixture
def router_and_fakes(config: XenoConfig) -> tuple[Router, dict[str, FakeProvider]]:
    ledger = CostLedger(run_id="t")
    router = Router(config, ledger=ledger)
    fakes = {
        name: FakeProvider(name, spec) for name, spec in config.providers.items()
    }
    router._providers.update(fakes)
    return router, fakes


@pytest.fixture
def prompt(keyring: CacheKeyring):  # type: ignore[no-untyped-def]
    return PromptBuilder(node=NodeRole.CODER, keyring=keyring, system_text="sys").build("go")


# ---- chain walking ---------------------------------------------------------


def test_primary_entry_is_used_when_it_succeeds(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    router, fakes = router_and_fakes
    result = router.complete(NodeRole.CODER, prompt)
    assert result.model.ref == "ollama/qwen2.5-coder:14b"
    assert fakes["openrouter"].calls == []


def test_retryable_failure_walks_to_the_next_chain_entry(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    router, fakes = router_and_fakes
    fakes["ollama"].outcomes = [ProviderError("OOM", provider="ollama", retryable=True)]
    result = router.complete(NodeRole.CODER, prompt)
    assert result.model.ref == "openrouter/glm-5.2"
    assert result.escalated


def test_non_retryable_failure_raises_instead_of_walking(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    """An auth failure is a config defect. The next provider would fail the
    same way for a different reason and the real cause would be obscured."""
    router, fakes = router_and_fakes
    fakes["ollama"].outcomes = [
        ProviderError("HTTP 401", provider="ollama", retryable=False)
    ]
    with pytest.raises(ProviderError, match="401"):
        router.complete(NodeRole.CODER, prompt)
    assert fakes["openrouter"].calls == []


def test_exhausted_chain_halts_rather_than_downgrading(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    """PRD S9.5: downward fallback is forbidden. A flagship chain with nothing
    left must not quietly serve the light model."""
    router, fakes = router_and_fakes
    fakes["openrouter"].outcomes = [
        ProviderError("429", provider="openrouter", retryable=True)
    ]
    with pytest.raises(ChainExhaustedError) as exc:
        router.complete(NodeRole.REVIEWER, prompt)
    assert "Downward fallback is forbidden" in str(exc.value)
    assert fakes["ollama"].calls == []


# ---- tier attribution (M1.1) ----------------------------------------------


def test_upward_fallback_is_billed_as_flagship_not_medium(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    """M1.1 counts by the model ACTUALLY BILLED, so an escalation must count
    against the metric rather than hide behind the node's declared tier."""
    router, fakes = router_and_fakes
    fakes["ollama"].outcomes = [ProviderError("down", provider="ollama", retryable=True)]
    result = router.complete(NodeRole.CODER, prompt)

    assert result.record.declared_tier is Tier.MEDIUM
    assert result.record.billed_tier is Tier.FLAGSHIP
    assert router.ledger.tokens_by_billed_tier["flagship"] > 0
    # The failed medium attempt is still recorded — it just consumed no tokens.
    assert router.ledger.tokens_by_billed_tier.get("medium", 0) == 0


def test_escalation_is_surfaced_in_the_ledger(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    router, fakes = router_and_fakes
    fakes["ollama"].outcomes = [ProviderError("down", provider="ollama", retryable=True)]
    router.complete(NodeRole.CODER, prompt)
    escalations = router.ledger.escalations
    assert len(escalations) == 1
    assert escalations[0]["declared_tier"] == "medium"
    assert escalations[0]["billed_tier"] == "flagship"


def test_local_calls_are_priced_at_zero_not_unpriced(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    """Local models have no price field, but that must not make the run total a
    lower bound — they genuinely cost nothing."""
    router, _ = router_and_fakes
    router.complete(NodeRole.EVALUATOR, prompt)
    assert router.ledger.usd_spent == 0.0
    assert not router.ledger.has_unpriced_calls


# ---- state and budget ------------------------------------------------------


def test_state_accumulates_tokens_by_model_and_tier(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    router, _ = router_and_fakes
    state = AgentState(run_id="t", goal="g")
    router.complete(NodeRole.CODER, prompt, state=state)
    assert state.tokens_by_model["ollama/qwen2.5-coder:14b"] == 1100
    assert state.tokens_spent[Tier.MEDIUM] == 1100
    assert state.iteration_count == 1


def test_budget_precheck_blocks_the_call_before_it_is_made(config: XenoConfig, prompt) -> None:  # type: ignore[no-untyped-def]
    ledger = CostLedger(run_id="t")
    router = Router(config, ledger=ledger)
    fake = FakeProvider("openrouter", config.providers["openrouter"])
    router._providers["openrouter"] = fake

    state = AgentState(run_id="t", goal="g", usd_spent=1.999)
    with pytest.raises(BudgetExceededError):
        router.complete(
            NodeRole.REVIEWER, prompt, state=state, limits=Limits(max_usd_per_run=2.00)
        )
    assert fake.calls == []  # never dispatched


def test_failed_calls_are_still_recorded_in_the_ledger(router_and_fakes, prompt) -> None:  # type: ignore[no-untyped-def]
    """A run that burned money on timeouts before succeeding should say so."""
    router, fakes = router_and_fakes
    fakes["ollama"].outcomes = [ProviderError("timeout", provider="ollama", retryable=True)]
    router.complete(NodeRole.CODER, prompt)
    assert len(router.ledger.calls) == 2
    assert [c.ok for c in router.ledger.calls] == [False, True]


# ---- secret scanning at the boundary --------------------------------------


def test_outbound_context_is_scanned_before_dispatch(config: XenoConfig, keyring) -> None:  # type: ignore[no-untyped-def]
    ledger = CostLedger(run_id="t")
    router = Router(config, ledger=ledger)
    fake = FakeProvider("ollama", config.providers["ollama"])
    router._providers["ollama"] = fake

    captured: list[str] = []
    original = fake.complete

    def capture(prompt, model, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(prompt.current_turn)
        return original(prompt, model, **kwargs)

    fake.complete = capture  # type: ignore[method-assign]

    builder = PromptBuilder(node=NodeRole.EVALUATOR, keyring=keyring, system_text="sys")
    leaked = "here is the key: ghp_" + "a" * 36
    router.complete(NodeRole.EVALUATOR, builder.build(leaked))

    assert "ghp_" not in captured[0]
    assert "[REDACTED:github_token" in captured[0]


def test_downward_fallback_config_is_rejected_at_load() -> None:
    """A flagship chain falling back to the light tier's model is a downgrade
    wearing a flagship label."""
    from xeno.core.config import ConfigError

    weak = ModelSpec(provider="ollama", model="qwen2.5-coder:7b")
    with pytest.raises(ConfigError, match="downward fallback is forbidden"):
        XenoConfig(
            providers={
                "ollama": ProviderSpec(
                    family=ProviderFamily.OLLAMA, base_url="http://localhost:11434"
                )
            },
            tiers={
                Tier.FLAGSHIP: (
                    ModelSpec(provider="ollama", model="big"),
                    weak,
                ),
                Tier.MEDIUM: (ModelSpec(provider="ollama", model="mid"),),
                Tier.LIGHT: (weak,),
            },
            nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
            allow_unpriced_models=True,
        )


def test_chain_primary_is_never_reported_as_an_escalation(config: XenoConfig, prompt) -> None:  # type: ignore[no-untyped-def]
    """Binding a flagship-class model at position 0 of a lower tier is a
    deliberate config choice, not a fallback. Reporting it as an escalation
    floods the ledger on any config where one model serves several tiers.
    """
    ledger = CostLedger(run_id="t")
    router = Router(config, ledger=ledger)
    router._providers["ollama"] = FakeProvider("ollama", config.providers["ollama"])
    # ollama/qwen2.5-coder:7b heads the light chain; give it a second home in
    # flagship so its billed tier outranks the researcher's declared tier.
    router._billed_tiers["ollama/qwen2.5-coder:7b"] = Tier.FLAGSHIP

    result = router.complete(NodeRole.RESEARCHER, prompt)
    assert result.record.billed_tier is Tier.FLAGSHIP  # attribution still honest
    assert not result.escalated
    assert ledger.escalations == []


# ---- incomplete answers ----------------------------------------------------


class _RecordingLog(NullRunLog):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def event(self, kind, **payload):  # type: ignore[no-untyped-def]
        record = super().event(kind, **payload)
        self.events.append(record)
        return record


def _incomplete_events(log: _RecordingLog) -> list[dict]:
    return [e for e in log.events if e["kind"] == EventKind.MODEL_INCOMPLETE.value]


def _router_with_log(config: XenoConfig, outcome: CompletionResult):  # type: ignore[no-untyped-def]
    log = _RecordingLog()
    router = Router(config, ledger=CostLedger(run_id="t"), runlog=log)
    fakes = {name: FakeProvider(name, spec) for name, spec in config.providers.items()}
    fakes["ollama"].outcomes = [outcome]
    router._providers.update(fakes)
    return router, log


def test_a_prompt_the_backend_truncated_is_reported_as_such(config, prompt) -> None:  # type: ignore[no-untyped-def]
    """Distinct from a short ANSWER: the model was shown less than it was
    sent, so it answered a different question. Reporting that as `truncated`
    would send a reader hunting for a format problem that is not there."""
    outcome = _ok("qwen2.5-coder:14b")
    outcome.prompt_truncated = True
    router, log = _router_with_log(config, outcome)

    router.complete(NodeRole.CODER, prompt)

    events = _incomplete_events(log)
    assert len(events) == 1
    assert events[0]["reason"] == "prompt_truncated"


def test_a_truncated_prompt_outranks_a_truncated_answer_in_the_report(config, prompt) -> None:  # type: ignore[no-untyped-def]
    """Both flags can be set at once — a cut-off prompt often produces a
    rambling answer that then hits the output cap. The prompt is the cause."""
    outcome = _ok("qwen2.5-coder:14b")
    outcome.prompt_truncated = True
    outcome.finish_reason = "length"
    router, log = _router_with_log(config, outcome)

    router.complete(NodeRole.CODER, prompt)

    assert _incomplete_events(log)[0]["reason"] == "prompt_truncated"


def test_a_complete_answer_reports_nothing(config, prompt) -> None:  # type: ignore[no-untyped-def]
    router, log = _router_with_log(config, _ok("qwen2.5-coder:14b"))
    router.complete(NodeRole.CODER, prompt)
    assert _incomplete_events(log) == []
