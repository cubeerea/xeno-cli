"""Provider wire formats and usage normalization (PRD S9.6.3).

The usage-parsing tests matter more than they look. `prompt_tokens` is a TOTAL
that already includes cached tokens; treating it as fresh-only would inflate
M1.4 and understate cost at the same time, and both errors point in the
flattering direction.
"""

from __future__ import annotations

import pytest

from xeno.core.config import ModelSpec, ProviderSpec
from xeno.core.types import Breakpoint, NodeRole, ProviderFamily
from xeno.core.usage import Usage
from xeno.prompt.assembly import AssembledPrompt, Block, CacheTTL, Turn
from xeno.router.pricing import price_call, uncached_price
from xeno.router.providers import build_provider
from xeno.router.providers.base import attribute_cache_to_breakpoints
from xeno.router.providers.ollama import OllamaProvider, parse_ollama_usage
from xeno.router.providers.openai_compat import (
    OpenAICompatProvider,
    OpenRouterProvider,
    parse_openai_usage,
)

OLLAMA = ProviderSpec(family=ProviderFamily.OLLAMA, base_url="http://localhost:11434")
OPENROUTER = ProviderSpec(
    family=ProviderFamily.AGGREGATOR, base_url="https://openrouter.ai/api/v1"
)


def make_prompt(*, with_map: bool = True, with_history: bool = True) -> AssembledPrompt:
    blocks = [Block(breakpoint=Breakpoint.SYSTEM, text="SYSTEM", ttl=CacheTTL.LONG)]
    if with_map:
        blocks.append(Block(breakpoint=Breakpoint.CODEBASE_MAP, text="MAP", ttl=CacheTTL.LONG))
    if with_history:
        blocks.append(
            Block(breakpoint=Breakpoint.ACCUMULATED_HISTORY, text="", ttl=CacheTTL.SHORT)
        )
    blocks.append(Block(breakpoint=Breakpoint.CURRENT_TURN, text="NOW", ttl=CacheTTL.NONE))
    history = (Turn(role="user", content="earlier"),) if with_history else ()
    return AssembledPrompt(node=NodeRole.CODER, blocks=tuple(blocks), history=history)


# ---- registry --------------------------------------------------------------


def test_registry_maps_families_to_clients() -> None:
    assert isinstance(build_provider("ollama", OLLAMA), OllamaProvider)
    assert isinstance(build_provider("openrouter", OPENROUTER), OpenRouterProvider)
    openai_spec = ProviderSpec(family=ProviderFamily.OPENAI, base_url="https://api.openai.com/v1")
    assert isinstance(build_provider("openai", openai_spec), OpenAICompatProvider)


def test_anthropic_family_fails_loudly_rather_than_silently() -> None:
    """A half-written client that mis-reports usage would corrupt M1.4, which
    is worse than a clear 'not wired yet'."""
    from xeno.router.providers import UnsupportedProviderError

    spec = ProviderSpec(family=ProviderFamily.ANTHROPIC, base_url="https://api.anthropic.com")
    with pytest.raises(UnsupportedProviderError, match="no client in Phase 0"):
        build_provider("anthropic", spec)


# ---- message construction --------------------------------------------------


def test_static_layers_lead_and_the_current_turn_is_last() -> None:
    provider = OpenAICompatProvider("openai", OPENROUTER)
    messages = provider.build_messages(make_prompt())
    assert messages[0]["role"] == "system"
    assert "SYSTEM" in messages[0]["content"] and "MAP" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "NOW"}
    assert messages[1] == {"role": "user", "content": "earlier"}


def test_ollama_matches_the_same_ordering() -> None:
    messages = OllamaProvider("ollama", OLLAMA).build_messages(make_prompt())
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "NOW"


def test_cache_markers_are_withheld_until_the_probe_succeeds() -> None:
    """Emitting markers speculatively risks paying a write premium for a cache
    that is never read (OQ-11)."""
    provider = OpenRouterProvider("openrouter", OPENROUTER)
    assert provider.cache_capable is None
    content = provider.build_messages(make_prompt())[0]["content"]
    assert isinstance(content, str)  # plain string: no markers


def test_cache_markers_are_emitted_once_the_probe_succeeds() -> None:
    provider = OpenRouterProvider("openrouter", OPENROUTER)
    provider.cache_capable = True
    content = provider.build_messages(make_prompt())[0]["content"]
    assert isinstance(content, list)
    assert all(part["cache_control"]["ttl"] == "1h" for part in content)


def test_openrouter_always_requests_detailed_usage() -> None:
    """Without it the response carries no cached-token figure at all, which
    would silently zero M1.4 rather than fail loudly."""
    provider = OpenRouterProvider("openrouter", OPENROUTER)
    model = ModelSpec(provider="openrouter", model="glm-5.2")
    payload = provider.build_payload(make_prompt(), model, max_tokens=100, temperature=0.0)
    assert payload["usage"] == {"include": True}


def test_ollama_keeps_the_model_resident() -> None:
    """Reloading weights between ladder iterations would dominate the latency
    this tier exists to keep low."""
    provider = OllamaProvider("ollama", OLLAMA)
    captured: dict[str, object] = {}
    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {"message": {"content": "x"}}

    provider._post = fake_post  # type: ignore[method-assign]
    provider.complete(
        make_prompt(), ModelSpec(provider="ollama", model="m"), max_tokens=10
    )
    assert captured["keep_alive"] == "30m"
    assert captured["stream"] is False


# ---- usage normalization ---------------------------------------------------


def test_openai_prompt_tokens_is_a_total_that_includes_cached() -> None:
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
    )
    assert usage.input_tokens == 1000
    assert usage.cache_read_tokens == 800
    assert usage.fresh_input_tokens == 200


def test_usage_tolerates_a_missing_details_block() -> None:
    usage = parse_openai_usage({"prompt_tokens": 100, "completion_tokens": 10})
    assert usage.cache_read_tokens == 0
    assert usage.fresh_input_tokens == 100


def test_usage_clamps_a_provider_overreporting_cached_tokens() -> None:
    usage = parse_openai_usage(
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 500}}
    )
    assert usage.cache_read_tokens == 100
    assert usage.fresh_input_tokens == 0


def test_inconsistent_usage_is_rejected_rather_than_absorbed() -> None:
    with pytest.raises(ValueError, match="misreporting"):
        Usage(input_tokens=100, cache_read_tokens=80, cache_write_tokens=80)


def test_ollama_reports_no_cached_tokens_by_design() -> None:
    """Ollama bills nothing, and inventing a figure would corrupt M1.4 with
    numbers from a provider that has no billing at all."""
    usage = parse_ollama_usage({"prompt_eval_count": 900, "eval_count": 100})
    assert usage.input_tokens == 900
    assert usage.cache_read_tokens == 0


# ---- breakpoint attribution ------------------------------------------------


def test_cached_tokens_fill_layers_in_prefix_order() -> None:
    """Caching consumes a prefix, so the split is not arbitrary — it follows
    the order a prefix cache would have matched."""
    prompt = make_prompt()
    usage = Usage(input_tokens=10_000, output_tokens=10, cache_read_tokens=2)
    stats = attribute_cache_to_breakpoints(prompt, usage)
    assert stats[Breakpoint.SYSTEM].hit_tokens == 1
    assert stats[Breakpoint.CODEBASE_MAP].hit_tokens == 1
    assert Breakpoint.CURRENT_TURN not in stats


def test_uncached_call_is_all_miss() -> None:
    stats = attribute_cache_to_breakpoints(
        make_prompt(), Usage(input_tokens=1000, output_tokens=10)
    )
    assert all(s.hit_tokens == 0 for s in stats.values())
    assert stats[Breakpoint.SYSTEM].miss_tokens > 0


# ---- pricing ---------------------------------------------------------------


PRICED = ModelSpec(
    provider="openrouter", model="glm-5.2", usd_per_1m_input=1.40, usd_per_1m_output=2.20
)


def test_local_calls_cost_zero_not_unknown() -> None:
    local = ModelSpec(provider="ollama", model="m")
    assert price_call(Usage(input_tokens=10_000, output_tokens=1000), local, OLLAMA) == 0.0


def test_unpriced_remote_call_returns_none_not_zero() -> None:
    """None is not zero, and callers must not treat it as such — it means CB-3
    cannot enforce the budget."""
    unpriced = ModelSpec(provider="openrouter", model="mystery")
    assert price_call(Usage(input_tokens=1000), unpriced, OPENROUTER) is None


def test_aggregator_cache_pricing_claims_no_discount_by_default() -> None:
    """UNVERIFIED per OQ-11. A projection must never claim a saving that was
    not measured, so cached tokens cost full price until proven otherwise."""
    usage = Usage(input_tokens=1000, output_tokens=0, cache_read_tokens=900)
    assert price_call(usage, PRICED, OPENROUTER) == pytest.approx(
        uncached_price(usage, PRICED, OPENROUTER)
    )


def test_explicit_per_model_multipliers_override_the_family_default() -> None:
    """The escape hatch for when OQ-11 finally gets a real answer."""
    verified = ModelSpec(
        provider="openrouter",
        model="glm-5.2",
        usd_per_1m_input=1.40,
        usd_per_1m_output=2.20,
        cache_read_multiplier=0.10,
    )
    usage = Usage(input_tokens=1000, output_tokens=0, cache_read_tokens=1000)
    assert price_call(usage, verified, OPENROUTER) == pytest.approx(
        1000 * 1.40 / 1_000_000 * 0.10
    )


def test_uncached_equivalent_ignores_the_discount() -> None:
    """Reported next to the real figure so the marginal USD impact of caching
    is visible on its own rather than folded into M1.2 (PRD S15.1)."""
    usage = Usage(input_tokens=1000, output_tokens=100, cache_read_tokens=900)
    assert uncached_price(usage, PRICED, OPENROUTER) == pytest.approx(
        1000 * 1.40 / 1_000_000 + 100 * 2.20 / 1_000_000
    )
