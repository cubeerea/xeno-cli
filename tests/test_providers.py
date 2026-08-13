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
from xeno.router.providers.base import ProviderError, attribute_cache_to_breakpoints
from xeno.router.providers.ollama import OllamaProvider, parse_ollama_usage
from xeno.router.providers.openai_compat import (
    OpenAICompatProvider,
    OpenRouterProvider,
    _extract_message,
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
    def fake_post(path: str, payload: dict[str, object]) -> tuple[dict[str, object], float]:
        captured.update(payload)
        return {"message": {"content": "x"}}, 1.0

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


def test_prefix_cache_support_defaults_to_false_so_the_warning_can_fire() -> None:
    """PRD S9.6.3's local-backend warning was previously unreachable: the
    capability was duck-typed and its one implementation returned a literal
    True. Defaulting to False on the base class means a new local provider
    trips the warning unless it explicitly claims the capability."""
    from xeno.router.providers.base import Provider

    assert Provider.supports_prefix_cache(object()) is False  # type: ignore[arg-type]
    assert OllamaProvider("ollama", OLLAMA).supports_prefix_cache() is True


# ---- reasoning models -----------------------------------------------------
#
# A run halted because a node's response "contained no tags". Reconstructing
# it from prompt-size arithmetic showed 2,473 billed output tokens against
# roughly 636 tokens of visible text: the rest was reasoning, dropped at this
# boundary and therefore invisible to every later diagnosis.


def test_reasoning_is_read_from_whichever_field_the_provider_uses() -> None:
    for field in ("reasoning", "reasoning_content"):
        body = {"choices": [{"message": {"content": "answer", "field": "x"}}]}
        body["choices"][0]["message"][field] = "let me think"
        text, reasoning, _ = _extract_message(body, provider="p")
        assert (text, reasoning) == ("answer", "let me think"), field


def test_openrouter_structured_reasoning_details_are_read_too() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": "answer",
                    "reasoning_details": [{"text": "step one "}, {"summary": "step two"}],
                }
            }
        ]
    }
    _, reasoning, _ = _extract_message(body, provider="p")
    assert reasoning == "step one step two"


def test_a_finish_reason_of_length_is_carried_out_of_the_provider() -> None:
    """Truncation used to be undetectable at every layer, so a response cut
    off mid-block was reported as a model that ignored the format — and one
    cut off just AFTER a block parsed cleanly into a silently short result."""
    body = {"choices": [{"finish_reason": "length", "message": {"content": "half an ans"}}]}
    _, _, finish = _extract_message(body, provider="p")
    assert finish == "length"


def test_an_empty_content_field_is_not_mistaken_for_a_short_answer() -> None:
    """`str(None or "")` returns "" with no error and no event, which is how a
    response that never left the reasoning channel became indistinguishable
    from prose."""
    body = {"choices": [{"message": {"content": None, "reasoning": "all of it"}}]}
    text, reasoning, _ = _extract_message(body, provider="p")
    assert text == ""
    assert reasoning == "all of it"


def test_reasoning_tokens_are_counted_as_the_output_subset_they_are() -> None:
    usage = parse_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 2473,
            "completion_tokens_details": {"reasoning_tokens": 1837},
        }
    )
    assert usage.output_tokens == 2473
    assert usage.reasoning_tokens == 1837


def test_a_provider_claiming_more_reasoning_than_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceed total output"):
        Usage(output_tokens=10, reasoning_tokens=11)


# ---- Ollama context window and residency -----------------------------------
#
# Ollama serves its own default window regardless of what the model supports
# and silently discards the front of anything longer — which by ASSEMBLY_ORDER
# is the system prompt. These pin the derivation that stops that happening,
# and the release that stops the weights outliving the run.


class _FakeDaemon:
    """Routes by path, so /api/show and /api/chat can answer differently."""

    def __init__(self, *, context_length: int | None = 32768) -> None:
        self.context_length = context_length
        self.chats: list[dict] = []
        self.shows: list[dict] = []
        self.unloads: list[dict] = []

    def post(self, path: str, payload: dict) -> tuple[dict, float]:
        if path == "/api/show":
            self.shows.append(payload)
            if self.context_length is None:
                return {}, 1.0
            return {"model_info": {"qwen2.context_length": self.context_length}}, 1.0
        if payload.get("keep_alive") == 0:
            self.unloads.append(payload)
            return {}, 1.0
        self.chats.append(payload)
        return {"message": {"content": "ok"}, "prompt_eval_count": 500, "eval_count": 5}, 1.0


def _wired(spec: ProviderSpec = OLLAMA, **kwargs) -> tuple[OllamaProvider, _FakeDaemon]:
    provider = OllamaProvider("ollama", spec)
    daemon = _FakeDaemon(**kwargs)
    provider._post = daemon.post  # type: ignore[method-assign]
    return provider, daemon


def _big_prompt(chars: int) -> AssembledPrompt:
    return AssembledPrompt(
        node=NodeRole.SPECIFIER,
        blocks=(
            Block(breakpoint=Breakpoint.SYSTEM, text="S" * chars, ttl=CacheTTL.LONG),
            Block(breakpoint=Breakpoint.CURRENT_TURN, text="go", ttl=CacheTTL.NONE),
        ),
        history=(),
    )


_MODEL = ModelSpec(provider="ollama", model="m")


def test_num_ctx_is_sent_and_sized_to_the_prompt() -> None:
    """Without this the daemon serves its own default — 4096 on the ones this
    was built against — no matter what the model supports."""
    provider, daemon = _wired()
    # ~5000 prompt tokens at 4 chars/token, plus 2000 of generation room.
    provider.complete(_big_prompt(20_000), _MODEL, max_tokens=2000)

    assert daemon.chats[0]["options"]["num_ctx"] == 8192, "smallest bucket that fits"


def test_the_window_is_clamped_to_what_the_model_reports() -> None:
    provider, daemon = _wired(context_length=8192)
    provider.complete(_big_prompt(20_000), _MODEL, max_tokens=2000)

    assert daemon.chats[0]["options"]["num_ctx"] == 8192
    assert daemon.shows, "the ceiling is discovered, not assumed"


def test_the_window_never_steps_back_down() -> None:
    """Shrinking would reload the weights to reclaim memory already spent,
    and the next large prompt would reload them straight back."""
    provider, daemon = _wired()
    provider.complete(_big_prompt(40_000), _MODEL, max_tokens=2000)
    provider.complete(_big_prompt(100), _MODEL, max_tokens=10)

    first, second = (c["options"]["num_ctx"] for c in daemon.chats)
    assert first == 16384
    assert second == first, "a small follow-up must not trigger a reload"


def test_the_discovered_ceiling_is_asked_for_once() -> None:
    provider, daemon = _wired()
    for _ in range(3):
        provider.complete(_big_prompt(100), _MODEL, max_tokens=10)

    assert len(daemon.shows) == 1


def test_a_prompt_too_large_for_the_model_is_refused_not_truncated() -> None:
    """The whole point: Ollama would take this call, discard the front of the
    prompt, and answer confidently from what was left."""
    provider, daemon = _wired(context_length=4096)

    with pytest.raises(ProviderError) as excinfo:
        provider.complete(_big_prompt(20_000), _MODEL, max_tokens=1000)

    message = str(excinfo.value)
    assert "4096" in message, "the ceiling the reader has to work around"
    assert str(_big_prompt(20_000).approx_tokens) in message, "what the call actually needed"
    assert "system prompt" in message
    # Retryable so the ROUTER walks the tier chain. Retrying the same model
    # would indeed be pointless, but the next chain entry is a different
    # model — and an escalation entry is typically the larger-window one, so
    # an overflow is exactly the error the chain exists to route around.
    assert excinfo.value.retryable is True, "the next chain entry may have a bigger window"
    assert not daemon.chats, "nothing was sent"


def test_a_daemon_that_will_not_report_a_ceiling_still_runs() -> None:
    """Not knowing the limit is a reason to skip the clamp, not to refuse."""
    provider, daemon = _wired(context_length=None)
    provider.complete(_big_prompt(20_000), _MODEL, max_tokens=2000)

    assert daemon.chats[0]["options"]["num_ctx"] == 8192


def test_keep_alive_comes_from_the_provider_spec() -> None:
    spec = ProviderSpec(family=ProviderFamily.OLLAMA, base_url="http://x", keep_alive="5m")
    provider, daemon = _wired(spec)
    provider.complete(make_prompt(), _MODEL, max_tokens=10)

    assert daemon.chats[0]["keep_alive"] == "5m"


def test_close_unloads_every_model_the_run_loaded() -> None:
    """A timer cannot know a run ended, which is why the weights outlive it."""
    provider, daemon = _wired()
    provider.complete(make_prompt(), _MODEL, max_tokens=10)
    provider.complete(make_prompt(), ModelSpec(provider="ollama", model="other"), max_tokens=10)
    provider.close()

    assert [u["model"] for u in daemon.unloads] == ["m", "other"]
    assert all(u["keep_alive"] == 0 for u in daemon.unloads)


def test_close_leaves_the_model_warm_when_asked_to() -> None:
    spec = ProviderSpec(family=ProviderFamily.OLLAMA, base_url="http://x", release_on_exit=False)
    provider, daemon = _wired(spec)
    provider.complete(make_prompt(), _MODEL, max_tokens=10)
    provider.close()

    assert daemon.unloads == []


def test_close_survives_a_daemon_that_has_already_gone_away() -> None:
    """Teardown must not take a run's exit path down: the memory is reclaimed
    by the daemon's own timer anyway."""
    provider, _ = _wired()
    provider.complete(make_prompt(), _MODEL, max_tokens=10)

    def explode(path: str, payload: dict) -> tuple[dict, float]:
        raise ProviderError("daemon gone", provider="ollama", retryable=False)

    provider._post = explode  # type: ignore[method-assign]
    provider.close()  # must not raise


def test_an_evaluated_prompt_far_short_of_the_estimate_is_flagged() -> None:
    """`prompt_eval_count` was always in the response and only ever used for
    accounting. Comparing it against what was sent is what makes a window
    overflow observable instead of silent."""
    provider, _ = _wired()
    result = provider.complete(_big_prompt(20_000), _MODEL, max_tokens=100)

    # The fake reports 500 evaluated against a ~5000-token prompt.
    assert result.prompt_truncated is True


def test_a_normal_call_is_not_flagged_as_truncated() -> None:
    provider, _ = _wired()
    result = provider.complete(_big_prompt(2000), _MODEL, max_tokens=100)

    # ~500 estimated, 500 evaluated: estimator drift, not truncation.
    assert result.prompt_truncated is False
