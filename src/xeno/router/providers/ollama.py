"""Ollama provider (PRD S9.3, S9.6.3 local inference).

Local inference is unbilled, so "caching" here is not a cost lever at all — it
is KV-cache reuse across calls that share an identical prefix, and the payoff
is time-to-first-token on repeated Daedalus/Chiron/Talos calls within a run.
That still depends on static-first assembly holding: llama.cpp reuses the KV
cache only for a matching prefix, so a system prompt that drifts between calls
loses the benefit exactly as an API cache would lose its discount.

Usage is reported with zero cached tokens by design. Ollama does not expose
prefix-reuse counts, and inventing a number here would corrupt M1.4 — a metric
about billed tokens — with figures from a provider that bills nothing. Latency
is where the local benefit shows up, and latency is recorded.

Two things this provider must do that a remote one does not, both because a
local daemon serves memory rather than a metered API:

* **Ask for a context window.** Ollama serves its own default — 4096 on the
  daemons this was built against — no matter what the model supports, and
  SILENTLY DISCARDS the front of anything longer. Per `ASSEMBLY_ORDER` the
  front is the system prompt, so the symptom is a model that appears not to
  know the output format it was never shown. The window is derived per call
  and bucketed; nobody is asked to configure it.
* **Say when the run is over.** `keep_alive` is a timer, and a timer cannot
  know a run ended, so weights outlive the run by design. `close()` unloads
  them — the difference between a machine that is slow during a run and one
  that is still slow half an hour later.
"""

from __future__ import annotations

import contextlib
from typing import Any

from xeno.core.config import ModelSpec, ProviderSpec
from xeno.core.usage import Usage
from xeno.prompt.assembly import AssembledPrompt
from xeno.router.providers.base import (
    CompletionResult,
    Provider,
    ProviderError,
    attribute_cache_to_breakpoints,
    plain_messages,
)

#: Windows this provider will ask for, smallest first. Bucketed rather than
#: sized to each prompt because Ollama RELOADS the model whenever `num_ctx`
#: changes, and a reload between ladder iterations costs far more than the
#: memory an oversized window holds. Powers of two so a growing codebase map
#: steps the window a handful of times per run rather than on every call.
_CONTEXT_BUCKETS = (4096, 8192, 16384, 32768, 65536, 131072)

#: How far `prompt_eval_count` may fall below the estimate before it is read
#: as truncation rather than estimator drift. The estimate is a flat
#: 4-chars-per-token rule, wrong by perhaps a fifth either way; a window
#: truncation is off by a multiple, not a margin.
_TRUNCATION_RATIO = 0.6


class OllamaProvider(Provider):
    """Talks to a local Ollama daemon over /api/chat."""

    chat_path = "/api/chat"
    show_path = "/api/show"

    def __init__(self, name: str, spec: ProviderSpec) -> None:
        super().__init__(name, spec)
        #: Discovered context ceilings, including `None` for "the daemon would
        #: not say" — cached so an older daemon missing the field is asked
        #: once per process rather than once per call.
        self._context_limits: dict[str, int | None] = {}
        #: Highest window asked for per model, so it never steps back DOWN.
        #: Shrinking would reload the weights to reclaim memory already spent,
        #: and the next large prompt would reload them straight back.
        self._window_floor: dict[str, int] = {}
        #: Models this process actually caused to be loaded, so teardown
        #: releases those and only those.
        self._resident: set[str] = set()

    def default_headers(self) -> dict[str, str]:
        # No auth: local daemon. Provider API keys never leave the host process
        # and are never sent to a local endpoint either (PRD S11.3).
        return {"content-type": "application/json"}

    def build_messages(self, prompt: AssembledPrompt) -> list[dict[str, Any]]:
        return plain_messages(prompt)

    def complete(
        self,
        prompt: AssembledPrompt,
        model: ModelSpec,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResult:
        # Also checked by the router before dispatch. Repeated here because
        # this provider is the one that will definitely truncate rather than
        # error, and the check is a dict lookup — cheap enough that "whoever
        # calls complete() is protected" beats saving it.
        self.assert_context_fits(prompt, model, max_tokens=max_tokens)
        estimated_prompt = prompt.approx_tokens
        payload = {
            "model": model.model,
            "messages": self.build_messages(prompt),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self._window(model.model, estimated_prompt, max_tokens),
            },
            # Hold the model resident between calls. Reloading weights between
            # ladder iterations would dominate the latency this tier exists to
            # keep low. Released in `close()` rather than left to expire.
            "keep_alive": self.spec.keep_alive,
        }
        body, latency_ms = self._post(self.chat_path, payload)
        self._resident.add(model.model)

        message = body.get("message") or {}
        text = str(message.get("content") or "")
        if not text and body.get("error"):
            raise ProviderError(str(body["error"]), provider=self.name, retryable=True)

        usage = parse_ollama_usage(body)
        return CompletionResult(
            text=text,
            model=str(body.get("model", model.model)),
            usage=usage,
            latency_ms=latency_ms,
            by_breakpoint=attribute_cache_to_breakpoints(prompt, usage),
            # Local models think too: Ollama reports "length" here when
            # num_predict was hit, and puts a reasoning model's scratchpad in
            # `thinking` rather than `content`.
            finish_reason=str(body.get("done_reason") or "") or None,
            reasoning=str(message.get("thinking") or ""),
            prompt_truncated=_looks_truncated(usage.input_tokens, estimated_prompt),
        )

    def _window(self, model: str, prompt_tokens: int, max_tokens: int) -> int:
        """The context window to ask for, in tokens.

        Derived rather than configured: the caller knows the prompt it just
        assembled and the daemon knows what the model was built with, so
        neither fact is the user's to supply (`ProviderSpec.max_context`
        exists to force a SMALLER window, not to make one mandatory).
        """
        # The overflow REFUSAL now lives in `Provider.assert_context_fits`,
        # which the router calls for every provider before dispatch — this
        # method is left with the job only it can do, which is choosing the
        # window to actually ask for.
        ceiling = self.context_limit(model)
        needed = prompt_tokens + max_tokens
        window = next((b for b in _CONTEXT_BUCKETS if b >= needed), _CONTEXT_BUCKETS[-1])
        if ceiling is not None:
            window = min(window, ceiling)
        window = max(window, self._window_floor.get(model, 0))
        self._window_floor[model] = window
        return window

    def context_limit(self, model: str) -> int | None:
        """What window the model was built with, per the daemon, or `None`.

        `None` is a real answer and is cached as one: an older daemon that
        does not report `model_info` should be asked once, not once per call,
        and the caller treats not-knowing as "do not clamp" rather than as an
        error — a missing ceiling is no reason to refuse to run.
        """
        if self.spec.max_context is not None:
            return self.spec.max_context
        if model in self._context_limits:
            return self._context_limits[model]

        limit: int | None = None
        try:
            body, _ = self._post(self.show_path, {"model": model})
            info = body.get("model_info") or {}
            # Keyed by architecture (`qwen2.general...`, `llama.context_length`),
            # so the suffix is the only stable part to match on.
            for key, value in info.items():
                if str(key).endswith(".context_length"):
                    limit = int(value)
                    break
        except (ProviderError, ValueError, TypeError):
            limit = None

        self._context_limits[model] = limit
        return limit

    def close(self) -> None:
        """Release every model this process loaded, then close the socket.

        The daemon cannot know a run ended — `keep_alive` is a timer, so
        without this the weights sit in memory long after the last call, which
        is why a machine feels slow AFTER a run rather than during one.
        `keep_alive: 0` unloads immediately.

        Best-effort by construction: a failure here costs memory the daemon's
        own timer will reclaim anyway, and taking a run's exit path down over
        it would be a far worse trade.
        """
        if self.spec.release_on_exit:
            for model in sorted(self._resident):
                with contextlib.suppress(Exception):
                    self._post(self.chat_path, {"model": model, "messages": [], "keep_alive": 0})
        self._resident.clear()
        super().close()

    def health_check(self) -> tuple[bool, str]:
        try:
            response = self.client.get("/api/tags")
        except Exception as exc:
            return False, f"daemon unreachable at {self.spec.base_url}: {exc}"
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"
        try:
            names = [m["name"] for m in response.json().get("models", [])]
        except (ValueError, KeyError, TypeError):
            return True, "reachable (model list unparsable)"
        return True, f"{len(names)} model(s) pulled"

    def supports_prefix_cache(self) -> bool:
        """Whether this backend reuses a KV cache across calls.

        PRD S9.6.3 says the router prefers backends with prefix-cache support
        and warns when a configured local backend lacks it. Ollama (llama.cpp)
        reuses the prompt cache for a matching prefix on a resident model, so
        this is True whenever `keep_alive` is in effect.
        """
        return True


def _looks_truncated(evaluated: int, estimated: int) -> bool:
    """Did the daemon read materially less of the prompt than we sent?

    `prompt_eval_count` is Ollama's own report of what it actually evaluated,
    and it has always been in the response — parsed into `Usage.input_tokens`
    and then used only for accounting. Comparing it against what was sent is
    what turns a window overflow from a silent answer into an observable
    event, and it costs nothing because both numbers are already in hand.

    With `num_ctx` derived correctly this should never fire. That is the
    point: it is the check on the derivation, not a substitute for it.
    """
    if evaluated <= 0 or estimated <= 0:
        return False
    return evaluated < estimated * _TRUNCATION_RATIO


def parse_ollama_usage(raw: dict[str, Any]) -> Usage:
    """Ollama reports evaluation counts, not billing tokens.

    Cached counts stay at zero deliberately — see the module docstring.
    """
    return Usage(
        input_tokens=int(raw.get("prompt_eval_count") or 0),
        output_tokens=int(raw.get("eval_count") or 0),
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
