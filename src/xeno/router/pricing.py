"""USD pricing for a model call (PRD S9.6.3, S15.1).

Token accounting itself lives in `xeno.core.usage`; this module only turns a
`Usage` into dollars, which requires knowing the provider family's cache
multipliers.
"""

from __future__ import annotations

from xeno.core.config import ModelSpec, ProviderSpec
from xeno.core.types import ProviderFamily
from xeno.core.usage import Usage

#: Per-family cache pricing defaults.
#:
#: ANTHROPIC: verified — 1.25x write, 0.10x read (PRD S9.6.3, ref [16]).
#: OPENAI: automatic prefix matching, so there is no write premium; the read
#:   discount is provider- and model-specific and is left conservative.
#: AGGREGATOR: UNVERIFIED (OQ-11). Defaults to no discount, so a projection
#:   never claims a saving that was not measured. The startup capability probe
#:   corrects this per provider when it can demonstrate caching works.
#: OLLAMA: local, unbilled. KV-cache reuse buys latency, not dollars.
FAMILY_CACHE_MULTIPLIERS: dict[ProviderFamily, tuple[float, float]] = {
    ProviderFamily.ANTHROPIC: (1.25, 0.10),
    ProviderFamily.OPENAI: (1.00, 0.50),
    ProviderFamily.AGGREGATOR: (1.00, 1.00),
    ProviderFamily.OLLAMA: (1.00, 1.00),
}


def cache_multipliers(model: ModelSpec, provider: ProviderSpec) -> tuple[float, float]:
    """(write, read) multipliers against the base input rate.

    An explicit per-model override in xeno.yaml always wins — that is the
    escape hatch for when OQ-11 finally gets a real answer.
    """
    default_write, default_read = FAMILY_CACHE_MULTIPLIERS[provider.family]
    fields_set = model.model_fields_set
    write = (
        model.cache_write_multiplier
        if "cache_write_multiplier" in fields_set
        else default_write
    )
    read = model.cache_read_multiplier if "cache_read_multiplier" in fields_set else default_read
    return write, read


def price_call(usage: Usage, model: ModelSpec, provider: ProviderSpec) -> float | None:
    """USD for one call. None when the model has no price.

    None is not zero, and callers must not treat it as such: an unpriced remote
    model means CB-3 cannot enforce max_usd_per_run, which is exactly why
    config rejects one unless `allow_unpriced_models` is set.
    """
    if provider.is_local:
        return 0.0
    if not model.priced:
        return None

    assert model.usd_per_1m_input is not None
    assert model.usd_per_1m_output is not None
    in_rate = model.usd_per_1m_input / 1_000_000
    out_rate = model.usd_per_1m_output / 1_000_000
    write_mult, read_mult = cache_multipliers(model, provider)

    return (
        usage.fresh_input_tokens * in_rate
        + usage.cache_write_tokens * in_rate * write_mult
        + usage.cache_read_tokens * in_rate * read_mult
        + usage.output_tokens * out_rate
    )


def uncached_price(usage: Usage, model: ModelSpec, provider: ProviderSpec) -> float | None:
    """What the same call would have cost with caching off.

    Reported next to the actual figure so the marginal USD impact of S9.6 is
    visible on its own rather than folded into M1.2 (PRD S15.1).
    """
    if provider.is_local:
        return 0.0
    if not model.priced:
        return None
    assert model.usd_per_1m_input is not None
    assert model.usd_per_1m_output is not None
    return (
        usage.input_tokens * model.usd_per_1m_input / 1_000_000
        + usage.output_tokens * model.usd_per_1m_output / 1_000_000
    )
