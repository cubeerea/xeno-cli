"""Per-run cost ledger -> cost.json (PRD S15.1, S9.6.4).

This is the instrument for M1.1-M1.4, which means its job is to make those
metrics falsifiable rather than flattering. Three details carry that weight:

  * Tokens are attributed to the model ACTUALLY BILLED, not the tier the node
    declared. An upward fallback from medium to a flagship model must count
    against M1.1 (PRD S9.5), and it only does if the ledger records what was
    billed.
  * Cached, cache-write, and fresh input tokens are tracked separately per
    breakpoint from day one. A ledger that cannot tell them apart cannot report
    M1.4 at all (PRD S9.6.4).
  * An unpriced call makes the run total a LOWER BOUND, and cost.json says so
    explicitly instead of quietly reporting a smaller number.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xeno.core.state import BreakpointStats
from xeno.core.types import Breakpoint, Tier
from xeno.core.usage import Usage


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model call, priced and attributed."""

    ts: float
    node: str
    declared_tier: Tier
    #: The tier the billed model natively serves. Differs from declared_tier
    #: exactly when an upward fallback fired — which is what M1.1 must catch.
    billed_tier: Tier
    provider: str
    model: str
    escalation: bool
    usage: Usage
    by_breakpoint: dict[Breakpoint, BreakpointStats]
    usd: float | None
    usd_uncached: float | None
    latency_ms: float
    cache_capable: bool
    ok: bool = True
    error: str | None = None
    prefix_signature: str | None = None

    @property
    def model_ref(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "node": self.node,
            "declared_tier": self.declared_tier.value,
            "billed_tier": self.billed_tier.value,
            "provider": self.provider,
            "model": self.model,
            "escalation": self.escalation,
            "input_tokens": self.usage.input_tokens,
            "fresh_input_tokens": self.usage.fresh_input_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_write_tokens": self.usage.cache_write_tokens,
            "output_tokens": self.usage.output_tokens,
            "by_breakpoint": {
                bp.value: stats.model_dump() for bp, stats in self.by_breakpoint.items()
            },
            "usd": self.usd,
            "usd_uncached": self.usd_uncached,
            "latency_ms": self.latency_ms,
            "cache_capable": self.cache_capable,
            "ok": self.ok,
            "error": self.error,
            "prefix_signature": self.prefix_signature,
        }


@dataclass
class CostLedger:
    """Accumulates call records and renders cost.json."""

    run_id: str
    started_at: float = field(default_factory=time.time)
    calls: list[CallRecord] = field(default_factory=list)
    completed_tasks_usd: list[float] = field(default_factory=list)
    caching_enabled: bool = True
    #: Running total at the previous task boundary, so `complete_task` can
    #: attribute spend to one task without every caller tracking it.
    _usd_at_last_task: float = 0.0

    def record(self, call: CallRecord) -> CallRecord:
        self.calls.append(call)
        return call

    def mark_task_completed(self, usd: float) -> None:
        """Record USD attributable to one completed plan task, for M1.3."""
        self.completed_tasks_usd.append(usd)

    def complete_task(self) -> float:
        """Close the current task's cost window and attribute it to M1.3.

        Everything billed since the previous task boundary belongs to the task
        that just passed — including its failed attempts and ladder climbs,
        which is the point: M1.3 measures what a completed task actually
        COSTS, not what its final successful attempt cost.
        """
        spent = self.usd_spent
        attributable = spent - self._usd_at_last_task
        self._usd_at_last_task = spent
        self.mark_task_completed(attributable)
        return attributable

    # ---- aggregates -----------------------------------------------------

    @property
    def usd_spent(self) -> float:
        """Sum of priced calls. See `has_unpriced_calls` before trusting it as
        a total — CB-3 must not enforce a ceiling against a partial figure."""
        return sum(c.usd for c in self.calls if c.usd is not None)

    @property
    def usd_uncached_equivalent(self) -> float:
        return sum(c.usd_uncached for c in self.calls if c.usd_uncached is not None)

    @property
    def has_unpriced_calls(self) -> bool:
        return any(c.usd is None for c in self.calls)

    @property
    def tokens_by_model(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.calls:
            totals[call.model_ref] = totals.get(call.model_ref, 0) + call.usage.total_tokens
        return totals

    @property
    def tokens_by_billed_tier(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.calls:
            key = call.billed_tier.value
            totals[key] = totals.get(key, 0) + call.usage.total_tokens
        return totals

    @property
    def usd_by_provider(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for call in self.calls:
            if call.usd is None:
                continue
            totals[call.provider] = totals.get(call.provider, 0.0) + call.usd
        return totals

    @property
    def escalations(self) -> list[dict[str, Any]]:
        """Upward fallbacks, surfaced in the ledger as tier escalations
        (PRD S9.5) rather than buried in the run log."""
        return [
            {
                "ts": c.ts,
                "node": c.node,
                "declared_tier": c.declared_tier.value,
                "billed_tier": c.billed_tier.value,
                "model": c.model_ref,
                "tokens": c.usage.total_tokens,
            }
            for c in self.calls
            if c.escalation
        ]

    def cache_stats_by_breakpoint(self) -> dict[str, BreakpointStats]:
        """Aggregated across calls whose provider passed the capability probe.

        Providers that failed the probe are excluded rather than counted as
        100% miss: their real behaviour is unknown (OQ-11), and folding an
        unknown into the denominator would make M1.4 look worse than measured
        instead of honestly narrower.
        """
        totals: dict[str, BreakpointStats] = {}
        for call in self.calls:
            if not call.cache_capable:
                continue
            for bp, stats in call.by_breakpoint.items():
                agg = totals.setdefault(bp.value, BreakpointStats())
                agg.hit_tokens += stats.hit_tokens
                agg.miss_tokens += stats.miss_tokens
                agg.write_tokens += stats.write_tokens
        return totals

    # ---- metrics --------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        by_tier = self.tokens_by_billed_tier
        total_tokens = sum(by_tier.values())
        flagship_tokens = by_tier.get(Tier.FLAGSHIP.value, 0)

        cache_totals = self.cache_stats_by_breakpoint()
        eligible = sum(s.eligible_tokens for s in cache_totals.values())
        hits = sum(s.hit_tokens for s in cache_totals.values())

        completed = self.completed_tasks_usd

        return {
            # M1.1: >= 70% of billed tokens served by non-flagship MODELS.
            "M1.1_non_flagship_token_share": (
                (total_tokens - flagship_tokens) / total_tokens if total_tokens else None
            ),
            "M1.1_target": 0.70,
            # M1.2 needs the all-flagship baseline run; the ledger supplies the
            # treatment arm only. The suite runner pairs them.
            "M1.2_usd_this_run": self.usd_spent,
            # M1.3: median USD per COMPLETED task.
            "M1.3_median_usd_per_completed_task": (
                statistics.median(completed) if completed else None
            ),
            "M1.3_target": 0.50,
            "M1.3_completed_tasks": len(completed),
            # M1.4: >= 40% of eligible input tokens served from cache.
            "M1.4_cache_hit_rate": (hits / eligible if eligible else None),
            "M1.4_target": 0.40,
            "M1.4_eligible_tokens": eligible,
            "M1.4_excluded_calls_not_cache_capable": sum(
                1 for c in self.calls if not c.cache_capable
            ),
            "usd_saved_by_caching": round(self.usd_uncached_equivalent - self.usd_spent, 6),
        }

    # ---- output ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        # Each of these walks every call in the run, and two are re-walked
        # inside metrics() — bound to locals so one serialization costs one
        # pass apiece rather than two or three.
        usd_spent = self.usd_spent
        uncached = self.usd_uncached_equivalent
        cache_by_breakpoint = self.cache_stats_by_breakpoint()
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "wall_clock_seconds": round(time.time() - self.started_at, 3),
            "caching_enabled": self.caching_enabled,
            "totals": {
                "calls": len(self.calls),
                "failed_calls": sum(1 for c in self.calls if not c.ok),
                "usd": round(usd_spent, 6),
                "usd_is_lower_bound": self.has_unpriced_calls,
                "usd_uncached_equivalent": round(uncached, 6),
                "tokens": sum(c.usage.total_tokens for c in self.calls),
            },
            "tokens_by_model": self.tokens_by_model,
            "tokens_by_billed_tier": self.tokens_by_billed_tier,
            "usd_by_provider": {k: round(v, 6) for k, v in self.usd_by_provider.items()},
            "cache_by_breakpoint": {
                bp: {**stats.model_dump(), "hit_rate": stats.hit_rate}
                for bp, stats in cache_by_breakpoint.items()
            },
            "tier_escalations": self.escalations,
            "latency_ms": self.latency_summary(),
            "metrics": self.metrics(),
            "calls": [c.to_dict() for c in self.calls],
        }

    def latency_summary(self) -> dict[str, Any]:
        by_tier: dict[str, list[float]] = {}
        for call in self.calls:
            by_tier.setdefault(call.billed_tier.value, []).append(call.latency_ms)
        return {
            tier: {
                "n": len(values),
                "p50": round(statistics.median(values), 2),
                "max": round(max(values), 2),
            }
            for tier, values in sorted(by_tier.items())
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return path
