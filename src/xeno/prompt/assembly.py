"""Static-first prompt assembly (PRD T8, S9.6.2).

    [ SYSTEM PROMPT + TOOL SCHEMAS ]   breakpoint 1 - changes only on upgrade
    [ PROJECT LAW ]                    breakpoint 2 - spec, roadmap, memory
    [ CODEBASE MAP / CONTEXT HANDLES ] breakpoint 3 - changes only on a write
    [ ACCUMULATED HISTORY ]            breakpoint 4 - append-only within a run
    [ CURRENT TURN ]                   never cached

T8 calls this a construction rule rather than an optimization, and the
distinction is the whole point: caching only pays off if nothing upstream of a
breakpoint ever changes. A single node that quietly interpolates a timestamp
into its system prompt costs the entire run its breakpoint-1 hits, and does so
invisibly — the calls still succeed, they just cost full price.

So the invariants are enforced here rather than documented:

  * ordering is produced by the builder, not chosen by the caller;
  * history is append-only, with the growing prefix verified against what was
    sent last time;
  * a node's system text is fingerprinted on first use and any later drift
    within the process raises.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from xeno.core.types import ASSEMBLY_ORDER, Breakpoint, NodeRole
from xeno.prompt.keys import CacheKeyring


class CacheTTL(StrEnum):
    """Requested cache lifetime (PRD S9.6.3).

    SYSTEM and CODEBASE MAP request LONG because a run may last
    max_runtime_minutes (default 45) and a 5-minute cache would cold-start
    across any multi-minute gap, such as a long Talos test run. ACCUMULATED
    HISTORY stays on SHORT since it refreshes every turn regardless.
    """

    NONE = "none"
    SHORT = "5m"
    LONG = "1h"


DEFAULT_TTLS: dict[Breakpoint, CacheTTL] = {
    Breakpoint.SYSTEM: CacheTTL.LONG,
    Breakpoint.PROJECT_LAW: CacheTTL.LONG,
    Breakpoint.CODEBASE_MAP: CacheTTL.LONG,
    Breakpoint.ACCUMULATED_HISTORY: CacheTTL.SHORT,
    Breakpoint.CURRENT_TURN: CacheTTL.NONE,
}


class PromptAssemblyError(RuntimeError):
    """A violation of the static-first construction rules."""


@dataclass(frozen=True, slots=True)
class Block:
    """One prompt layer, tagged with its breakpoint and caching intent."""

    breakpoint: Breakpoint
    text: str
    cache_key: str | None = None
    ttl: CacheTTL = CacheTTL.NONE

    @property
    def cacheable(self) -> bool:
        return self.ttl is not CacheTTL.NONE and bool(self.text)

    @property
    def approx_tokens(self) -> int:
        """Cheap 4-chars-per-token estimate.

        Used only for the OpenAI minimum-cacheable-prefix check (PRD S9.6.3)
        and for pre-call projections. Billing always uses provider-reported
        usage, never this.
        """
        return max(1, len(self.text) // 4)


@dataclass(frozen=True, slots=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    """A prompt in static-first order, ready for a provider to translate."""

    node: NodeRole
    blocks: tuple[Block, ...]
    history: tuple[Turn, ...]

    def __post_init__(self) -> None:
        order = [b.breakpoint for b in self.blocks]
        expected = [bp for bp in ASSEMBLY_ORDER if bp in set(order)]
        if order != expected:
            raise PromptAssemblyError(
                f"blocks are out of static-first order: {order} (expected {expected}). "
                f"Dynamic content before static content defeats every breakpoint after it."
            )

    def block(self, breakpoint: Breakpoint) -> Block | None:
        return next((b for b in self.blocks if b.breakpoint is breakpoint), None)

    #: The system region, in assembly order. Everything ahead of history that
    #: carries text: a provider concatenates these into one system message.
    _STATIC = (Breakpoint.SYSTEM, Breakpoint.PROJECT_LAW, Breakpoint.CODEBASE_MAP)

    @property
    def static_blocks(self) -> tuple[Block, ...]:
        """SYSTEM + PROJECT LAW + CODEBASE MAP: the provider-agnostic system
        region. Order is taken from `blocks`, which `__post_init__` has already
        verified against ASSEMBLY_ORDER — never from the tuple above."""
        return tuple(b for b in self.blocks if b.breakpoint in self._STATIC)

    @property
    def current_turn(self) -> str:
        block = self.block(Breakpoint.CURRENT_TURN)
        return block.text if block else ""

    @property
    def approx_tokens(self) -> int:
        """Cheap size estimate for the whole prompt, blocks plus history.

        Same 4-chars-per-token rule as `Block.approx_tokens` and the same
        caveat: never used for billing, only for the two decisions that must
        be made BEFORE a call exists to measure — projecting its cost against
        the budget breaker, and sizing the context window a local backend is
        asked to allocate for it.
        """
        return sum(b.approx_tokens for b in self.blocks if b.text) + sum(
            len(t.content) // 4 for t in self.history
        )

    def prefix_signature(self) -> str:
        """Digest of everything ahead of the current turn.

        Two calls sharing this signature share a cacheable prefix. Logged per
        call so a run's JSONL shows directly whether the prefix held or drifted
        — which is the difference between M1.4 being measured and guessed.
        """
        hasher = hashlib.sha256()
        for block in self.blocks:
            if block.breakpoint is Breakpoint.CURRENT_TURN:
                continue
            hasher.update(block.breakpoint.value.encode())
            hasher.update(b"\0")
            hasher.update(block.text.encode())
            hasher.update(b"\0")
        for turn in self.history:
            hasher.update(turn.role.encode())
            hasher.update(b"\0")
            hasher.update(turn.content.encode())
            hasher.update(b"\0")
        return hasher.hexdigest()


#: Per-process record of each node's system text. Drift within a run means a
#: node is interpolating something dynamic into breakpoint 1, which silently
#: destroys its cache hit rate.
_SYSTEM_FINGERPRINTS: dict[NodeRole, str] = {}


def reset_system_fingerprints() -> None:
    """Test hook. Not called in normal operation."""
    _SYSTEM_FINGERPRINTS.clear()


@dataclass
class PromptBuilder:
    """Assembles one node's prompt. One builder per node per run.

    The builder owns ordering so callers cannot get it wrong: there is no API
    for placing the current turn anywhere but last.
    """

    node: NodeRole
    keyring: CacheKeyring
    system_text: str
    caching_enabled: bool = True
    _law_text: str = field(default="", init=False)
    _codebase_text: str = field(default="", init=False)
    _history: list[Turn] = field(default_factory=list, init=False)
    _prefix_log: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._check_system_stability()

    def _check_system_stability(self) -> None:
        digest = hashlib.sha256(self.system_text.encode()).hexdigest()
        seen = _SYSTEM_FINGERPRINTS.get(self.node)
        if seen is None:
            _SYSTEM_FINGERPRINTS[self.node] = digest
        elif seen != digest:
            raise PromptAssemblyError(
                f"{self.node.value}'s system prompt changed within this process "
                f"({seen[:12]} -> {digest[:12]}). Breakpoint 1 must be byte-identical on "
                f"every call to a node (PRD T8) — something dynamic (a timestamp, a "
                f"run_id, a task description) has leaked into the system layer."
            )

    def set_project_law(self, text: str) -> None:
        """Install the PROJECT LAW block (spec, roadmap, memory).

        No freshness assertion, unlike `set_codebase_map`: law's cache key is
        derived from its own text (`CacheKeyring.law_key`), so a stale render
        cannot be served under a current key. There is nothing for a caller to
        get wrong and therefore nothing to assert.
        """
        self._law_text = text

    def set_codebase_map(self, text: str, *, require_fresh: bool = True) -> None:
        """Install the codebase map / context handle block.

        `require_fresh` asserts the worktree has not changed since the map was
        built. Callers pass False only when deliberately rebuilding the map
        from a just-mutated worktree — that is, when they are the ones
        refreshing it (PRD S9.6.5).
        """
        if require_fresh:
            self.keyring.assert_codebase_fresh()
        self._codebase_text = text

    def append_turn(self, role: Literal["user", "assistant"], content: str) -> None:
        """Append to accumulated history. There is deliberately no insert or
        replace: a mutation anywhere but the tail invalidates the whole
        breakpoint-3 prefix (PRD S9.6.1)."""
        self._history.append(Turn(role=role, content=content))
        self.keyring.advance_history(self.node)

    def build(self, current_turn: str) -> AssembledPrompt:
        """Emit the prompt. Verifies the history prefix only grew."""
        self._assert_history_append_only()

        blocks: list[Block] = [
            Block(
                breakpoint=Breakpoint.SYSTEM,
                text=self.system_text,
                cache_key=self.keyring.system_key(self.node, self.system_text),
                ttl=self._ttl(Breakpoint.SYSTEM),
            )
        ]
        if self._law_text:
            blocks.append(
                Block(
                    breakpoint=Breakpoint.PROJECT_LAW,
                    text=self._law_text,
                    cache_key=self.keyring.law_key(self._law_text),
                    ttl=self._ttl(Breakpoint.PROJECT_LAW),
                )
            )
        if self._codebase_text:
            blocks.append(
                Block(
                    breakpoint=Breakpoint.CODEBASE_MAP,
                    text=self._codebase_text,
                    cache_key=self.keyring.codebase_key(),
                    ttl=self._ttl(Breakpoint.CODEBASE_MAP),
                )
            )
        if self._history:
            blocks.append(
                Block(
                    breakpoint=Breakpoint.ACCUMULATED_HISTORY,
                    text="",  # carried structurally in `history`, not inlined
                    cache_key=self.keyring.history_key(self.node),
                    ttl=self._ttl(Breakpoint.ACCUMULATED_HISTORY),
                )
            )
        blocks.append(
            Block(breakpoint=Breakpoint.CURRENT_TURN, text=current_turn, ttl=CacheTTL.NONE)
        )
        return AssembledPrompt(node=self.node, blocks=tuple(blocks), history=tuple(self._history))

    def _ttl(self, breakpoint: Breakpoint) -> CacheTTL:
        if not self.caching_enabled:
            return CacheTTL.NONE
        return DEFAULT_TTLS[breakpoint]

    def _assert_history_append_only(self) -> None:
        """Recompute the digest after each successive turn and confirm the
        recorded sequence is a prefix of the current one."""
        digests = []
        hasher = hashlib.sha256()
        for turn in self._history:
            hasher.update(turn.role.encode())
            hasher.update(b"\0")
            hasher.update(turn.content.encode())
            hasher.update(b"\0")
            digests.append(hasher.hexdigest())
        if self._prefix_log != digests[: len(self._prefix_log)]:
            raise PromptAssemblyError(
                f"{self.node.value}'s accumulated history was edited in place. History is "
                f"append-only (PRD S9.6.1); rewriting an earlier turn invalidates the "
                f"entire breakpoint-3 prefix."
            )
        self._prefix_log = digests
