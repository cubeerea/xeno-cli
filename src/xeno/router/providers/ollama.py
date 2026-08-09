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
"""

from __future__ import annotations

from typing import Any

from xeno.core.config import ModelSpec
from xeno.core.usage import Usage
from xeno.prompt.assembly import AssembledPrompt
from xeno.router.providers.base import (
    CompletionResult,
    Provider,
    ProviderError,
    attribute_cache_to_breakpoints,
)


class OllamaProvider(Provider):
    """Talks to a local Ollama daemon over /api/chat."""

    chat_path = "/api/chat"

    def default_headers(self) -> dict[str, str]:
        # No auth: local daemon. Provider API keys never leave the host process
        # and are never sent to a local endpoint either (PRD S11.3).
        return {"content-type": "application/json"}

    def build_messages(self, prompt: AssembledPrompt) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        static = [b.text for b in prompt.static_blocks if b.text]
        if static:
            messages.append({"role": "system", "content": "\n\n".join(static)})
        messages.extend({"role": t.role, "content": t.content} for t in prompt.history)
        messages.append({"role": "user", "content": prompt.current_turn})
        return messages

    def complete(
        self,
        prompt: AssembledPrompt,
        model: ModelSpec,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResult:
        payload = {
            "model": model.model,
            "messages": self.build_messages(prompt),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            # Hold the model resident between calls. Reloading weights between
            # ladder iterations would dominate the latency this tier exists to
            # keep low.
            "keep_alive": "30m",
        }
        body = self._post(self.chat_path, payload)

        message = body.get("message") or {}
        text = str(message.get("content") or "")
        if not text and body.get("error"):
            raise ProviderError(str(body["error"]), provider=self.name, retryable=True)

        usage = parse_ollama_usage(body)
        return CompletionResult(
            text=text,
            model=str(body.get("model", model.model)),
            usage=usage,
            latency_ms=float(body.get("_latency_ms", 0.0)),
            by_breakpoint=attribute_cache_to_breakpoints(prompt, usage),
            raw=body,
        )

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

    def installed_models(self) -> list[str]:
        try:
            response = self.client.get("/api/tags")
            return [str(m["name"]) for m in response.json().get("models", [])]
        except Exception:
            return []

    def supports_prefix_cache(self) -> bool:
        """Whether this backend reuses a KV cache across calls.

        PRD S9.6.3 says the router prefers backends with prefix-cache support
        and warns when a configured local backend lacks it. Ollama (llama.cpp)
        reuses the prompt cache for a matching prefix on a resident model, so
        this is True whenever `keep_alive` is in effect.
        """
        return True


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
