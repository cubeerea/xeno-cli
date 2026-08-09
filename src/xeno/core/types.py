"""Core enumerations shared across the harness.

Kept free of imports from the rest of the package so every other module can
depend on it without cycles.
"""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    """Capability class a node declares and the router satisfies (PRD S9.1)."""

    FLAGSHIP = "flagship"
    MEDIUM = "medium"
    LIGHT = "light"


# Ordering used to decide whether a fallback is upward (permitted, logged) or
# downward (forbidden — PRD S9.5). Higher value == more capable.
TIER_RANK: dict[Tier, int] = {Tier.LIGHT: 0, Tier.MEDIUM: 1, Tier.FLAGSHIP: 2}


class NodeRole(StrEnum):
    """The six nodes of the graph. Role is the primary name (PRD S1.1)."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    EVALUATOR = "evaluator"
    DEBUGGER = "debugger"
    REVIEWER = "reviewer"


CALLSIGNS: dict[NodeRole, str] = {
    NodeRole.PLANNER: "Odysseus",
    NodeRole.RESEARCHER: "Argus",
    NodeRole.CODER: "Daedalus",
    NodeRole.EVALUATOR: "Talos",
    NodeRole.DEBUGGER: "Chiron",
    NodeRole.REVIEWER: "Cerberus",
}

DEFAULT_NODE_TIERS: dict[NodeRole, Tier] = {
    NodeRole.PLANNER: Tier.FLAGSHIP,
    NodeRole.RESEARCHER: Tier.LIGHT,
    NodeRole.CODER: Tier.MEDIUM,
    NodeRole.EVALUATOR: Tier.LIGHT,
    NodeRole.DEBUGGER: Tier.MEDIUM,
    NodeRole.REVIEWER: Tier.FLAGSHIP,
}


class Breakpoint(StrEnum):
    """The four prompt layers, in assembly order (PRD S9.6.2, T8).

    Declaration order is assembly order and is load-bearing: `sorted()` over
    these values must never be used to order a prompt. Use ASSEMBLY_ORDER.
    """

    SYSTEM = "system"
    CODEBASE_MAP = "codebase_map"
    ACCUMULATED_HISTORY = "accumulated_history"
    CURRENT_TURN = "current_turn"


#: Static-first assembly order. CURRENT_TURN is always last and never cached.
ASSEMBLY_ORDER: tuple[Breakpoint, ...] = (
    Breakpoint.SYSTEM,
    Breakpoint.CODEBASE_MAP,
    Breakpoint.ACCUMULATED_HISTORY,
    Breakpoint.CURRENT_TURN,
)

#: Layers eligible for caching. CURRENT_TURN is excluded by definition — it is
#: different on every call, so a breakpoint on it would only cost write price.
CACHEABLE_BREAKPOINTS: tuple[Breakpoint, ...] = (
    Breakpoint.SYSTEM,
    Breakpoint.CODEBASE_MAP,
    Breakpoint.ACCUMULATED_HISTORY,
)


class ProviderFamily(StrEnum):
    """How a provider handles prompt caching (PRD S9.6.3).

    This drives cache_control injection, not the wire format: OpenRouter speaks
    the OpenAI wire protocol but its caching behaviour is unverified (OQ-11),
    so it gets its own family and is probed at startup.
    """

    ANTHROPIC = "anthropic"  # explicit cache_control markers; 1.25x write, 0.10x read
    OPENAI = "openai"  # automatic prefix matching; 1024-token minimum prefix
    OLLAMA = "ollama"  # local; KV-cache reuse, latency benefit only, no billing
    AGGREGATOR = "aggregator"  # OpenRouter et al; caching UNVERIFIED, see OQ-11


class Verdict(StrEnum):
    """Reviewer verdicts (PRD S8.3).

    APPROVE and ESCALATE are terminal and close the S8.1 invariant window.
    REJECT_AND_RETURN is non-terminal: it returns work into the graph.
    """

    APPROVE = "approve"
    REJECT_AND_RETURN = "reject_and_return"
    ESCALATE = "escalate"


TERMINAL_VERDICTS: frozenset[Verdict] = frozenset({Verdict.APPROVE, Verdict.ESCALATE})


class LadderRung(StrEnum):
    """The bounded escalation ladder (PRD S7.2). Values match ladder_rung ints."""

    L0_RETRY = "L0"
    L1_PATCH = "L1"
    L2_RE_RESEARCH = "L2"
    L3_ROLLBACK_REWRITE = "L3"
    L4_RE_PLAN = "L4"
    L5_HALT = "L5"


#: Per-rung attempt budgets (PRD S7.2). L5 is terminal and has no budget.
RUNG_BUDGETS: dict[int, int] = {0: 1, 1: 3, 2: 2, 3: 2, 4: 1}

MAX_RUNG = 5


class BreakerCode(StrEnum):
    """Circuit breakers (PRD S7.3). CB-2/5/6 land in Phase 2."""

    CB1_ITERATION_CAP = "CB-1"
    CB2_WALL_CLOCK_CAP = "CB-2"
    CB3_BUDGET_CAP = "CB-3"
    CB4_NO_PROGRESS = "CB-4"
    CB5_DIFF_THRASH = "CB-5"
    CB6_DESTRUCTIVE_ACTION = "CB-6"


#: Breakers implemented in Phase 0 (PRD S7.4).
PHASE_0_BREAKERS: frozenset[BreakerCode] = frozenset(
    {BreakerCode.CB1_ITERATION_CAP, BreakerCode.CB3_BUDGET_CAP, BreakerCode.CB4_NO_PROGRESS}
)
