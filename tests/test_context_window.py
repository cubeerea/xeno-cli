"""The pre-send context-window guard (`Provider.assert_context_fits`).

Ollama has refused over-long prompts since `num_ctx` derivation landed. Every
OTHER provider was sending them blind: `approx_tokens` existed and was used
only to project cost, never compared against a limit. That mattered more once
the default medium and light tiers could be hosted models, because the failure
is silent by construction — the call succeeds, a plausible answer comes back,
the parser accepts it, and nothing distinguishes it from a call that saw the
whole prompt.
"""

from __future__ import annotations

from typing import Any

import pytest

from xeno.core.config import ModelSpec, ProviderSpec
from xeno.core.types import Breakpoint, NodeRole, ProviderFamily
from xeno.prompt.assembly import AssembledPrompt, Block, CacheTTL
from xeno.router.providers.base import MIN_OUTPUT_HEADROOM, ProviderError
from xeno.router.providers.openai_compat import OpenAICompatProvider, OpenRouterProvider

REMOTE = ProviderSpec(
    base_url="https://example.invalid/api/v1",
    family=ProviderFamily.AGGREGATOR,
    api_key_env="TEST_KEY",
)
MODEL = ModelSpec(provider="openrouter", model="vendor/big-model")


def _prompt(chars: int) -> AssembledPrompt:
    return AssembledPrompt(
        node=NodeRole.CODER,
        blocks=(
            Block(breakpoint=Breakpoint.SYSTEM, text="S" * chars, ttl=CacheTTL.LONG),
            Block(breakpoint=Breakpoint.CURRENT_TURN, text="go", ttl=CacheTTL.NONE),
        ),
        history=(),
    )


class _FakeModels:
    """Stands in for GET /models. Records how often it was actually called."""

    def __init__(self, payload: Any, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.calls = 0

    def get(self, path: str) -> Any:
        assert path == "/models"
        self.calls += 1
        return self

    @property
    def status_code(self) -> int:
        return self.status

    def json(self) -> Any:
        return self.payload

    @property
    def text(self) -> str:
        return "fake"


def _wired(payload: Any, *, status: int = 200) -> tuple[OpenAICompatProvider, _FakeModels]:
    provider = OpenAICompatProvider("openrouter", REMOTE)
    listing = _FakeModels(payload, status=status)
    provider._client = listing  # type: ignore[assignment]
    return provider, listing


_OPENROUTER_LISTING = {"data": [{"id": "vendor/big-model", "context_length": 8192}]}


# ---- discovery ------------------------------------------------------------


def test_a_window_is_discovered_from_the_models_listing() -> None:
    provider, _ = _wired(_OPENROUTER_LISTING)
    assert provider.context_limit("vendor/big-model") == 8192


def test_the_listing_is_fetched_once_per_process() -> None:
    """One request of a few hundred kilobytes. Per call would add it to every
    node in the graph."""
    provider, listing = _wired(_OPENROUTER_LISTING)
    for _ in range(4):
        provider.context_limit("vendor/big-model")
    assert listing.calls == 1


def test_a_provider_that_reports_no_windows_is_not_re_asked() -> None:
    """'Asked, and it would not say' must stay distinguishable from 'not asked
    yet', or a provider that reports nothing gets re-fetched forever."""
    provider, listing = _wired({"data": [{"id": "vendor/big-model"}]})
    for _ in range(3):
        assert provider.context_limit("vendor/big-model") is None
    assert listing.calls == 1


@pytest.mark.parametrize("key", ["context_length", "max_model_len", "context_window"])
def test_the_three_spellings_are_all_accepted(key: str) -> None:
    """Same wire protocol, different self-description: OpenRouter says
    context_length, vLLM says max_model_len."""
    provider, _ = _wired({"data": [{"id": "m", key: 4096}]})
    assert provider.context_limit("m") == 4096


def test_an_unreachable_listing_does_not_take_the_run_down() -> None:
    """Not knowing the limit is a reason to skip the check, not to refuse."""
    provider, _ = _wired(None, status=503)
    assert provider.context_limit("vendor/big-model") is None
    provider.assert_context_fits(_prompt(400_000), MODEL, max_tokens=4096)


def test_a_configured_max_context_beats_discovery() -> None:
    """`max_context` exists to force a SMALLER window than the model allows;
    a discovered ceiling overriding it would undo the only reason to set it."""
    spec = REMOTE.model_copy(update={"max_context": 2048})
    provider = OpenAICompatProvider("openrouter", spec)
    provider._client = _FakeModels(_OPENROUTER_LISTING)  # type: ignore[assignment]
    assert provider.context_limit("vendor/big-model") == 2048


# ---- the guard ------------------------------------------------------------


def test_a_prompt_over_the_window_is_refused_before_it_is_sent() -> None:
    provider, _ = _wired(_OPENROUTER_LISTING)
    oversized = _prompt(60_000)
    with pytest.raises(ProviderError) as excinfo:
        provider.assert_context_fits(oversized, MODEL, max_tokens=4096)

    message = str(excinfo.value)
    assert "8192" in message, "the ceiling, copy-pasteable into max_context"
    assert str(oversized.approx_tokens) in message, "what the call actually needed"
    assert "system prompt" in message, "what silently goes missing, not just that it does"


def test_a_prompt_that_fits_is_left_alone() -> None:
    provider, _ = _wired(_OPENROUTER_LISTING)
    provider.assert_context_fits(_prompt(4_000), MODEL, max_tokens=512)


def test_generation_room_counts_against_the_window() -> None:
    """A prompt that fits with nothing left to answer in is not a call worth
    making: every parser in xeno.graph.prompts reads a fragment as a format
    failure, so the run pays twice and learns nothing."""
    provider, _ = _wired({"data": [{"id": "m", "context_length": 1000}]})
    model = ModelSpec(provider="openrouter", model="m")
    prompt = _prompt(3_600)

    # The prompt alone fits the 1000-token window with room to spare...
    assert prompt.approx_tokens < 1000
    # ...and is still refused, because there would be nothing left to answer in.
    with pytest.raises(ProviderError):
        provider.assert_context_fits(prompt, model, max_tokens=8)
    assert prompt.approx_tokens + MIN_OUTPUT_HEADROOM > 1000


def test_history_counts_toward_the_prompt(  # the whole reason this guard exists
) -> None:
    """Accumulated history is the part that grows without bound, so a guard
    that measured blocks alone would miss the only thing that overflows."""
    provider, _ = _wired({"data": [{"id": "m", "context_length": 4096}]})
    model = ModelSpec(provider="openrouter", model="m")
    from xeno.prompt.assembly import Turn

    small_blocks = AssembledPrompt(
        node=NodeRole.CODER,
        blocks=(
            Block(breakpoint=Breakpoint.SYSTEM, text="S" * 40, ttl=CacheTTL.LONG),
            Block(breakpoint=Breakpoint.CURRENT_TURN, text="go", ttl=CacheTTL.NONE),
        ),
        history=tuple(Turn(role="assistant", content="F" * 4_000) for _ in range(4)),
    )
    with pytest.raises(ProviderError, match="4096"):
        provider.assert_context_fits(small_blocks, model, max_tokens=256)


def test_the_refusal_is_retryable_so_the_chain_can_escalate() -> None:
    """An overflow is one of the few errors the NEXT chain entry genuinely may
    not have — escalation entries are usually the larger-window models."""
    provider, _ = _wired(_OPENROUTER_LISTING)
    with pytest.raises(ProviderError) as excinfo:
        provider.assert_context_fits(_prompt(60_000), MODEL, max_tokens=4096)
    assert excinfo.value.retryable is True


def test_openrouter_inherits_the_guard() -> None:
    """The subclass overrides payload and headers, not the window logic."""
    provider = OpenRouterProvider("openrouter", REMOTE)
    provider._client = _FakeModels(_OPENROUTER_LISTING)  # type: ignore[assignment]
    assert provider.context_limit("vendor/big-model") == 8192
    with pytest.raises(ProviderError):
        provider.assert_context_fits(_prompt(60_000), MODEL, max_tokens=4096)
