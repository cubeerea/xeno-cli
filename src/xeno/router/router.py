"""Tiered model routing with fallback chains (PRD S9.5, T4).

Release v1 routing is STATIC: each node declares a tier in xeno.yaml and the
router satisfies it with the chain for that tier. No classifier. Dynamic
complexity-based promotion is deferred to v1.1 because a misrouting classifier
is a silent quality regression that is very hard to debug.

Two rules shape everything here:

  DOWNWARD FALLBACK IS FORBIDDEN. An exhausted chain halts the run. Quietly
  serving a weaker model would produce worse output and hide the cost problem
  behind an apparently successful run — the same reasoning behind CB-3.

  UPWARD FALLBACK IS PERMITTED but must be declared, is logged, and is
  surfaced in the cost ledger as a tier escalation. M1.1 counts tokens by the
  MODEL ACTUALLY BILLED, so an escalation correctly counts against the metric
  instead of hiding behind the node's declared tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from xeno.core.breakers import BreakerVerdict, would_breach_budget
from xeno.core.config import Limits, ModelSpec, XenoConfig
from xeno.core.ledger import CallRecord, CostLedger
from xeno.core.runlog import EventKind, NullRunLog, RunLog
from xeno.core.state import AgentState
from xeno.core.types import TIER_RANK, Breakpoint, NodeRole, Tier
from xeno.core.usage import Usage
from xeno.prompt.assembly import AssembledPrompt
from xeno.router.pricing import price_call, uncached_price
from xeno.router.probe import ProbeResult, probe_provider, probe_targets
from xeno.router.providers import Provider, ProviderError, build_provider
from xeno.security.outbound import sanitize
from xeno.security.scanner import SecretScanner


class BudgetExceededError(Exception):
    """CB-3 fired. The run halts; no tier is downgraded to continue."""

    def __init__(self, verdict: BreakerVerdict) -> None:
        super().__init__(verdict.detail)
        self.verdict = verdict


class ChainExhaustedError(Exception):
    """Every entry in a tier's chain failed. The run halts (PRD S9.5)."""

    def __init__(self, tier: Tier, attempts: list[str]) -> None:
        super().__init__(
            f"tier '{tier}' chain exhausted after {len(attempts)} attempt(s): "
            + "; ".join(attempts)
            + ". Downward fallback is forbidden, so the run halts here."
        )
        self.tier = tier
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class RouterResult:
    text: str
    record: CallRecord
    model: ModelSpec
    provider_name: str
    escalated: bool


class Router:
    """Resolves a node to a model and executes the call."""

    def __init__(
        self,
        config: XenoConfig,
        *,
        ledger: CostLedger,
        runlog: RunLog | None = None,
        scanner: SecretScanner | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.runlog = runlog or NullRunLog()
        self.scanner = scanner or SecretScanner(
            entropy_threshold=config.secrets.entropy_threshold
        )
        self.scan_enabled = config.secrets.scan_outbound_context
        self._providers: dict[str, Provider] = {}
        self._billed_tiers = self._index_billed_tiers()

    # ---- providers ------------------------------------------------------

    def provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = build_provider(name, self.config.providers[name])
        return self._providers[name]

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()

    def _index_billed_tiers(self) -> dict[str, Tier]:
        """Map each model to the HIGHEST tier that serves it.

        This is what makes upward fallback visible to M1.1: when medium's chain
        falls through to the same model flagship uses, the tokens are billed as
        flagship because that is the model that was actually billed, regardless
        of which node asked for it.
        """
        billed: dict[str, Tier] = {}
        for tier, chain in self.config.tiers.items():
            for entry in chain:
                current = billed.get(entry.ref)
                if current is None or TIER_RANK[tier] > TIER_RANK[current]:
                    billed[entry.ref] = tier
        return billed

    def billed_tier(self, entry: ModelSpec) -> Tier:
        return self._billed_tiers.get(entry.ref, Tier.LIGHT)

    # ---- probe ----------------------------------------------------------

    def probe_caching(self) -> list[ProbeResult]:
        """Run the startup capability probe for aggregator providers."""
        results: list[ProbeResult] = []
        if not self.config.caching.enabled or not self.config.caching.probe_aggregators_at_startup:
            return results
        for name, entry in probe_targets(self.config):
            result = probe_provider(self.provider(name), entry)
            results.append(result)
            self.runlog.event(
                EventKind.CACHE_PROBE,
                provider=result.provider,
                model=result.model,
                cache_capable=result.cache_capable,
                evidence=result.evidence,
                error=result.error,
            )
        return results

    def warn_local_backends_without_prefix_cache(self) -> list[str]:
        """PRD S9.6.3: prefer backends with prefix-cache support and warn when
        a configured local backend lacks it."""
        warnings: list[str] = []
        for name, spec in self.config.providers.items():
            if not spec.is_local:
                continue
            provider = self.provider(name)
            supports = getattr(provider, "supports_prefix_cache", None)
            if supports is not None and not supports():
                warnings.append(
                    f"local backend '{name}' has no prefix-cache reuse; repeated "
                    f"Daedalus/Chiron/Talos calls will re-process the whole prefix"
                )
        return warnings

    # ---- the call -------------------------------------------------------

    def complete(
        self,
        node: NodeRole,
        prompt: AssembledPrompt,
        *,
        state: AgentState | None = None,
        limits: Limits | None = None,
    ) -> RouterResult:
        """Execute one node's model call, walking the tier chain on failure."""
        declared_tier = self.config.tier_for(node)
        node_spec = self.config.nodes[node]
        chain = self.config.chain_for(declared_tier)
        effective_limits = limits or self.config.limits

        sent_prompt = prompt
        if self.scan_enabled:
            sanitized = sanitize(prompt, self.scanner)
            sent_prompt = sanitized.prompt
            if sanitized.findings:
                self.runlog.event(
                    EventKind.SECRET_REDACTED,
                    node=node.value,
                    count=len(sanitized.findings),
                    breakpoints=sorted(b.value for b in sanitized.breakpoints_hit),
                    labels=sorted({f.label for f in sanitized.findings}),
                )

        attempts: list[str] = []
        for index, entry in enumerate(chain):
            provider = self.provider(entry.provider)
            provider_spec = self.config.provider_for(entry)
            billed = self.billed_tier(entry)
            # An escalation is something the router FELL BACK to. Position 0 is
            # the tier's declared primary: binding a flagship-class model there
            # is a deliberate config choice, not a fallback, and reporting it as
            # one would flood the ledger on any config where a single model
            # serves several tiers. The billed_tier attribution below is
            # unaffected — M1.1 still counts those tokens as flagship, which is
            # the honest answer to a different question.
            escalated = index > 0 and TIER_RANK[billed] > TIER_RANK[declared_tier]

            if state is not None:
                breach = self._precheck_budget(
                    state, effective_limits, sent_prompt, entry, node_spec.max_tokens, billed
                )
                if breach is not None:
                    raise BudgetExceededError(breach)

            if index > 0:
                self.runlog.event(
                    EventKind.FALLBACK,
                    node=node.value,
                    tier=declared_tier.value,
                    to_model=entry.ref,
                    position=index,
                    previous_error=attempts[-1] if attempts else None,
                )
            if escalated:
                # Declared explicitly in config, logged here, and surfaced in
                # cost.json's tier_escalations (PRD S9.5).
                self.runlog.event(
                    EventKind.TIER_ESCALATION,
                    node=node.value,
                    declared_tier=declared_tier.value,
                    billed_tier=billed.value,
                    model=entry.ref,
                    declared_in_config=entry.escalation,
                )

            started = time.perf_counter()
            try:
                result = provider.complete(
                    sent_prompt,
                    entry,
                    max_tokens=node_spec.max_tokens,
                    temperature=node_spec.temperature,
                )
            except ProviderError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                attempts.append(f"{entry.ref}: {exc}")
                self.runlog.event(
                    EventKind.MODEL_ERROR,
                    node=node.value,
                    model=entry.ref,
                    error=str(exc),
                    retryable=exc.retryable,
                    latency_ms=round(latency_ms, 2),
                )
                self.ledger.record(
                    CallRecord(
                        ts=time.time(),
                        node=node.value,
                        declared_tier=declared_tier,
                        billed_tier=billed,
                        provider=entry.provider,
                        model=entry.model,
                        escalation=escalated,
                        usage=Usage(),
                        by_breakpoint={},
                        usd=0.0,
                        usd_uncached=0.0,
                        latency_ms=latency_ms,
                        cache_capable=bool(provider.cache_capable),
                        ok=False,
                        error=str(exc),
                    )
                )
                if not exc.retryable:
                    # Auth failures and malformed requests are config defects.
                    # The next provider would fail the same way for a different
                    # reason, so walking the chain would just obscure the cause.
                    raise
                continue

            usd = price_call(result.usage, entry, provider_spec)
            record = self.ledger.record(
                CallRecord(
                    ts=time.time(),
                    node=node.value,
                    declared_tier=declared_tier,
                    billed_tier=billed,
                    provider=entry.provider,
                    model=entry.model,
                    escalation=escalated,
                    usage=result.usage,
                    by_breakpoint=result.by_breakpoint,
                    usd=usd,
                    usd_uncached=uncached_price(result.usage, entry, provider_spec),
                    latency_ms=result.latency_ms,
                    cache_capable=bool(provider.cache_capable),
                    ok=True,
                    prefix_signature=sent_prompt.prefix_signature(),
                )
            )

            if state is not None:
                self._apply_to_state(state, record, billed)

            self.runlog.event(
                EventKind.MODEL_CALL,
                node=node.value,
                declared_tier=declared_tier.value,
                billed_tier=billed.value,
                model=entry.ref,
                input_tokens=result.usage.input_tokens,
                cached_tokens=result.usage.cache_read_tokens,
                output_tokens=result.usage.output_tokens,
                usd=usd,
                latency_ms=round(result.latency_ms, 2),
                prefix_signature=record.prefix_signature,
            )
            return RouterResult(
                text=result.text,
                record=record,
                model=entry,
                provider_name=entry.provider,
                escalated=escalated,
            )

        raise ChainExhaustedError(declared_tier, attempts)

    # ---- helpers --------------------------------------------------------

    def _precheck_budget(
        self,
        state: AgentState,
        limits: Limits,
        prompt: AssembledPrompt,
        entry: ModelSpec,
        max_tokens: int,
        billed: Tier,
    ) -> BreakerVerdict | None:
        """Project this call's cost before making it.

        Deliberately pessimistic: it assumes zero cache hits and a full-length
        completion. A flagship review of a whole diff is the most expensive
        call in the graph, and discovering it blew the ceiling only afterwards
        defeats the point of having one.
        """
        provider_spec = self.config.provider_for(entry)
        estimated_input = sum(
            b.approx_tokens for b in prompt.blocks if b.text
        ) + sum(len(t.content) // 4 for t in prompt.history)
        projected = Usage(input_tokens=estimated_input, output_tokens=max_tokens)
        projected_usd = price_call(projected, entry, provider_spec) or 0.0
        return would_breach_budget(
            state,
            limits,
            projected_usd=projected_usd,
            tier=billed,
            projected_tokens=projected.total_tokens,
        )

    def _apply_to_state(self, state: AgentState, record: CallRecord, billed: Tier) -> None:
        tokens = record.usage.total_tokens
        state.tokens_spent = {
            **state.tokens_spent,
            billed: state.tokens_spent.get(billed, 0) + tokens,
        }
        state.tokens_by_model = {
            **state.tokens_by_model,
            record.model_ref: state.tokens_by_model.get(record.model_ref, 0) + tokens,
        }
        state.usd_spent = state.usd_spent + (record.usd or 0.0)

        stats = state.cache_stats_for(record.node)
        for breakpoint, entry in record.by_breakpoint.items():
            stats.record(
                Breakpoint(breakpoint),
                hit=entry.hit_tokens,
                miss=entry.miss_tokens,
                write=entry.write_tokens,
            )
        state.cache_stats = {**state.cache_stats, record.node: stats}
        state.iteration_count = state.iteration_count + 1
