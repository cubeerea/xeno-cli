"""Startup cache-capability probe (PRD S9.6.3, OQ-11).

Caching support and pricing for open-weight models served through aggregators
were not established during the PRD's research. Rather than assume parity with
Anthropic or OpenAI, the router sends a duplicate prefix at startup and looks
at what comes back.

The probe reports capability only on POSITIVE evidence: a provider-reported
cached-token count above zero on the second call. Latency is recorded as
secondary evidence but never used to declare success on its own — a warm route,
a quiet moment on a shared endpoint, or a shorter completion all produce a
faster second call without any cache involved, and a false positive here would
put fabricated savings straight into cost.json.

A provider that fails the probe is not broken. It just has cache-dependent cost
projections disabled, which is the honest state of knowledge until OQ-11 is
answered from the provider's own documentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from xeno.core.config import ModelSpec, XenoConfig
from xeno.core.types import Breakpoint, NodeRole, ProviderFamily, Tier
from xeno.prompt.assembly import AssembledPrompt, Block, CacheTTL
from xeno.router.providers import Provider, ProviderError

#: The probe prefix must clear OpenAI's 1,024-token minimum cacheable prefix
#: (PRD S9.6.3, ref [17]) or a capable provider would fail for the wrong
#: reason. ~4 chars/token, with headroom.
_PROBE_PREFIX_CHARS = 6000

_PROBE_QUESTION = "Reply with exactly the word: ok"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    provider: str
    model: str
    cache_capable: bool
    evidence: str
    error: str | None = None


def _probe_prefix() -> str:
    """Deterministic filler.

    Must be byte-identical between the two calls — random filler would
    guarantee a cache miss and make every provider look incapable. Must also be
    stable across processes so a probe result is reproducible when replaying a
    run.
    """
    unit = (
        "Xeno CLI cache capability probe. This block is deterministic filler used "
        "to establish a cacheable prompt prefix. It carries no instructions. "
    )
    return (unit * ((_PROBE_PREFIX_CHARS // len(unit)) + 1))[:_PROBE_PREFIX_CHARS]


def _probe_prompt() -> AssembledPrompt:
    prefix = _probe_prefix()
    return AssembledPrompt(
        node=NodeRole.EVALUATOR,
        blocks=(
            Block(
                breakpoint=Breakpoint.SYSTEM,
                text=prefix,
                cache_key="probe",
                ttl=CacheTTL.LONG,
            ),
            Block(breakpoint=Breakpoint.CURRENT_TURN, text=_PROBE_QUESTION, ttl=CacheTTL.NONE),
        ),
        history=(),
    )


def probe_provider(provider: Provider, model: ModelSpec) -> ProbeResult:
    """Send the same prefix twice and inspect the second response's usage."""
    if provider.family is ProviderFamily.OLLAMA:
        # Local inference is unbilled; "caching" is KV reuse and buys latency,
        # not dollars (PRD S9.6.3). Excluded from M1.4 by construction.
        provider.cache_capable = False
        return ProbeResult(
            provider=provider.name,
            model=model.model,
            cache_capable=False,
            evidence="local backend: KV-cache reuse gives latency, not billing discounts",
        )

    prompt = _probe_prompt()
    try:
        # The first call's RESULT is irrelevant — it exists to populate the
        # provider's cache so the second call has something to hit.
        provider.complete(prompt, model, max_tokens=8, temperature=0.0)
        # A brief pause: some providers populate the cache asynchronously and
        # a back-to-back second call can miss a cache that does in fact work.
        time.sleep(1.0)
        second = provider.complete(prompt, model, max_tokens=8, temperature=0.0)
    except ProviderError as exc:
        provider.cache_capable = False
        return ProbeResult(
            provider=provider.name,
            model=model.model,
            cache_capable=False,
            evidence="probe call failed; cache-dependent projections disabled",
            error=str(exc),
        )

    cached = second.usage.cache_read_tokens
    capable = cached > 0
    provider.cache_capable = capable

    if capable:
        evidence = (
            f"second call reported {cached} cached input tokens of "
            f"{second.usage.input_tokens}"
        )
    else:
        evidence = (
            "second call reported no cached tokens; cache-dependent cost "
            "projections disabled for this provider (OQ-11)"
        )

    return ProbeResult(
        provider=provider.name,
        model=model.model,
        cache_capable=capable,
        evidence=evidence,
    )


def probe_targets(config: XenoConfig) -> list[tuple[str, ModelSpec]]:
    """One representative model per aggregator provider in use.

    Only aggregators are probed. Anthropic's mechanics are documented and
    verified, OpenAI's are automatic, and local backends do not bill — the
    unknown is the aggregator path, so that is what gets measured.
    """
    seen: set[str] = set()
    targets: list[tuple[str, ModelSpec]] = []
    for tier in (Tier.FLAGSHIP, Tier.MEDIUM, Tier.LIGHT):
        for entry in config.tiers.get(tier, ()):
            spec = config.providers[entry.provider]
            if spec.family is not ProviderFamily.AGGREGATOR:
                continue
            if entry.provider in seen:
                continue
            seen.add(entry.provider)
            targets.append((entry.provider, entry))
    return targets
