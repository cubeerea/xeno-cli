"""Provider interface (PRD S9.5, S13 Phase 0).

A provider owns exactly two things: translating an `AssembledPrompt` into its
wire format — including whatever cache markers its family requires — and
normalizing the response's token accounting into `Usage`.

Everything else (tier selection, fallback, budget enforcement, logging) belongs
to the router, so adding a provider never means touching routing policy.

Calls are synchronous. Phase 0's graph is sequential and the exit criterion
exercises tiers one at a time; introducing async here would buy nothing and
would have to be threaded through every node. Revisit if parallel node
execution ever lands.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from xeno.core.config import ModelSpec, ProviderSpec
from xeno.core.state import BreakpointStats
from xeno.core.types import Breakpoint, ProviderFamily
from xeno.core.usage import Usage
from xeno.prompt.assembly import AssembledPrompt


class ProviderError(Exception):
    """A failed model call.

    `retryable` decides whether the router walks to the next entry in the tier
    chain or halts the run. Timeouts, rate limits, 5xx, and local OOM are
    chain-advancing; an auth failure or a malformed request is not, because the
    next provider would fail the same way for a different reason and the real
    problem is the config.
    """

    def __init__(self, message: str, *, provider: str, retryable: bool) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


@dataclass(slots=True)
class CompletionResult:
    text: str
    model: str
    usage: Usage
    latency_ms: float
    #: Per-breakpoint cached/fresh attribution. Providers report cache totals,
    #: not a per-block split, so this is attributed by the router from the
    #: prompt's own block sizes (see `attribute_cache_to_breakpoints`).
    by_breakpoint: dict[Breakpoint, BreakpointStats] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    """Base class for an inference endpoint."""

    family: ProviderFamily

    def __init__(self, name: str, spec: ProviderSpec) -> None:
        self.name = name
        self.spec = spec
        self.family = spec.family
        #: Set by the startup capability probe (PRD S9.6.3). None means
        #: "not yet probed"; False disables cache-dependent cost projections
        #: for this provider rather than assuming parity (OQ-11).
        self.cache_capable: bool | None = None
        self._client: httpx.Client | None = None

    # ---- lifecycle ------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.spec.base_url,
                timeout=self.spec.timeout_seconds,
                headers=self.default_headers(),
            )
        return self._client

    def default_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        key = self.spec.api_key()
        if key:
            headers["authorization"] = f"Bearer {key}"
        return headers

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- interface ------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        prompt: AssembledPrompt,
        model: ModelSpec,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Run one completion. Raises ProviderError on failure."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """(reachable, detail). Never raises; used by `xeno models test`."""

    # ---- shared helpers -------------------------------------------------

    def supports_explicit_cache_markers(self) -> bool:
        """Whether this provider wants cache_control breakpoints emitted.

        Aggregators only get markers once the probe has demonstrated they do
        something. Emitting them speculatively risks paying a write premium
        for a cache that is never read (PRD S9.6.3, OQ-11).
        """
        if self.family is ProviderFamily.ANTHROPIC:
            return True
        if self.family is ProviderFamily.AGGREGATOR:
            return self.cache_capable is True
        return False

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self.client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"timeout after {self.spec.timeout_seconds}s",
                provider=self.name,
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"transport error: {exc}", provider=self.name, retryable=True
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            raise ProviderError(
                f"HTTP {response.status_code}: {response.text[:400]}",
                provider=self.name,
                retryable=_is_retryable_status(response.status_code),
            )
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"non-JSON response: {response.text[:200]}", provider=self.name, retryable=True
            ) from exc
        body["_latency_ms"] = elapsed_ms
        return body


def _is_retryable_status(status: int) -> bool:
    # 408 timeout, 409 conflict, 429 rate limit, 5xx server-side: try the next
    # entry in the chain. 401/403 (auth) and 400/404/422 (request shape) are
    # config defects and would recur identically on every provider.
    return status in {408, 409, 429} or status >= 500


def attribute_cache_to_breakpoints(
    prompt: AssembledPrompt,
    usage: Usage,
) -> dict[Breakpoint, BreakpointStats]:
    """Split provider-reported cache totals across the prompt's breakpoints.

    Providers report one cached-token figure for the whole request, not a
    per-block split, so a per-breakpoint number cannot be measured directly.
    Caching consumes a PREFIX, though, so the attribution is not arbitrary:
    cached tokens fill the layers in assembly order until exhausted, which is
    exactly the order a prefix cache would have matched them.

    The result is an estimate and cost.json labels it as one — but it is an
    estimate of the split, not of the total, and the total is what M1.4 is
    measured against.
    """
    stats: dict[Breakpoint, BreakpointStats] = {}
    remaining_read = usage.cache_read_tokens
    remaining_write = usage.cache_write_tokens

    for block in prompt.blocks:
        if block.breakpoint is Breakpoint.CURRENT_TURN:
            continue
        size = block.approx_tokens if block.text else _history_tokens(prompt, block.breakpoint)
        if size <= 0:
            continue
        entry = BreakpointStats()
        take_read = min(remaining_read, size)
        entry.hit_tokens = take_read
        remaining_read -= take_read

        take_write = min(remaining_write, size - take_read)
        entry.write_tokens = take_write
        remaining_write -= take_write

        entry.miss_tokens = max(0, size - take_read - take_write)
        stats[block.breakpoint] = entry

    return stats


def _history_tokens(prompt: AssembledPrompt, breakpoint: Breakpoint) -> int:
    if breakpoint is not Breakpoint.ACCUMULATED_HISTORY:
        return 0
    return max(1, sum(len(t.content) for t in prompt.history) // 4)
