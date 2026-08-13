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

#: Generation room that must fit alongside the prompt for a call to be worth
#: making at all. A node cut off below this returns a fragment, and every
#: parser in `xeno.graph.prompts` reads a fragment as a format failure — so the
#: run pays for the call, pays again for the corrective retry, and still has
#: nothing to act on.
MIN_OUTPUT_HEADROOM = 1024


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
    #: The provider's own stop reason, verbatim. "length" means the response
    #: was cut off by max_tokens — a state the harness previously could not
    #: observe at all, so a response truncated mid-block was reported as a
    #: model that ignored the format.
    finish_reason: str | None = None
    #: A reasoning model's scratchpad, when the provider returns it OUTSIDE
    #: `content`. Carried for diagnosis and NEVER parsed as an answer: a
    #: scratchpad routinely contains draft tags the model then revised, and
    #: acting on those would execute a plan it had already discarded.
    reasoning: str = ""
    #: The provider evaluated materially fewer prompt tokens than were sent,
    #: i.e. it discarded part of the prompt before reading it. Distinct from
    #: `truncated`, which is about the ANSWER being cut short: this one says
    #: the model never saw the whole QUESTION. Only a backend that reports
    #: what it actually evaluated can set it, which today means Ollama.
    prompt_truncated: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def silent(self) -> bool:
        """Output tokens were billed but the text came back empty — the answer
        never left the reasoning channel."""
        return not self.text.strip() and self.usage.output_tokens > 0


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

    # ---- context window -------------------------------------------------

    def context_limit(self, model: str) -> int | None:
        """The model's context window in tokens, or `None` when unknown.

        `None` is a real answer, not a failure: a backend that will not say
        what a model's window is gives no grounds to refuse the call, so the
        guard treats not-knowing as "do not clamp". Subclasses override to
        discover it (Ollama from `/api/show`, OpenRouter from `/models`);
        the base answer is whatever the user pinned in config.

        `ProviderSpec.max_context` wins over discovery wherever it is set,
        because its documented purpose is to force a SMALLER window than the
        model allows — a discovered ceiling that overrode it would silently
        undo the only reason anyone sets it.
        """
        del model
        return self.spec.max_context

    def assert_context_fits(
        self, prompt: AssembledPrompt, model: ModelSpec, *, max_tokens: int
    ) -> None:
        """Refuse a prompt that cannot fit the model's window, before sending.

        Every provider in this package will otherwise accept an over-long
        prompt and answer from whatever survived — Ollama discards the front
        (which is the system prompt), and hosted APIs differ by vendor and by
        day. The common thread is that the call SUCCEEDS: a plausible answer
        comes back, the parser accepts it, and nothing in the log distinguishes
        it from a call that saw the whole prompt. Ollama has checked this since
        `num_ctx` derivation landed; the rest of the providers were sending
        blind, which matters more now that the default medium and light tiers
        can be hosted models.

        Raised as retryable so the router walks the tier chain: a window
        overflow is one of the few errors the NEXT entry genuinely may not
        have, since escalation entries are usually larger-window models. If
        every entry overflows, `ChainExhaustedError` reports all of them.
        """
        ceiling = self.context_limit(model.model)
        if ceiling is None:
            return
        needed = prompt.approx_tokens + MIN_OUTPUT_HEADROOM
        if needed <= ceiling:
            return
        raise ProviderError(
            f"{model.model} serves a {ceiling}-token context window, but this call "
            f"needs ~{prompt.approx_tokens} for the prompt and at least "
            f"{MIN_OUTPUT_HEADROOM} to answer in. Sending it would have the provider "
            f"drop the front of the prompt and answer from the rest — and per "
            f"ASSEMBLY_ORDER the front is the system prompt, so the symptom is a model "
            f"that appears not to know the output format it was never shown. Route this "
            f"node to a larger-window model, or lower its max_tokens.",
            provider=self.name,
            retryable=True,
        )

    # ---- shared helpers -------------------------------------------------

    def supports_prefix_cache(self) -> bool:
        """Whether this backend reuses a KV cache across calls (PRD S9.6.3).

        Defaults to False so the router's local-backend warning fires for any
        provider that has not explicitly claimed the capability — the
        conservative direction, since a wrong True silently costs full-prefix
        reprocessing on every Daedalus/Chiron/Talos call.
        """
        return False

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

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Returns `(body, latency_ms)`.

        Latency travels alongside the body rather than inside it: injecting a
        private `_latency_ms` key mutated the provider's own response, lied
        about the declared `dict[str, Any]` return type whenever an endpoint
        answered with a JSON array, and left every caller reaching for a key
        that no provider actually sends.
        """
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
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"non-JSON response: {response.text[:200]}", provider=self.name, retryable=True
            ) from exc
        if not isinstance(body, dict):
            raise ProviderError(
                f"expected a JSON object, got {type(body).__name__}",
                provider=self.name,
                retryable=True,
            )
        return body, elapsed_ms


def plain_messages(prompt: AssembledPrompt) -> list[dict[str, Any]]:
    """Static-first messages with no cache markers: the system region, then
    history, then the current turn (PRD S9.6.2).

    Shared because it is the ONLY shape Ollama has and the shape every
    OpenAI-compatible provider falls back to when explicit cache markers are
    not warranted — the two were byte-for-byte the same logic. Order comes
    from the assembled prompt, which already enforces it; this must not
    reorder.
    """
    messages: list[dict[str, Any]] = []
    static = [b.text for b in prompt.static_blocks if b.text]
    if static:
        messages.append({"role": "system", "content": "\n\n".join(static)})
    messages.extend({"role": t.role, "content": t.content} for t in prompt.history)
    messages.append({"role": "user", "content": prompt.current_turn})
    return messages


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
