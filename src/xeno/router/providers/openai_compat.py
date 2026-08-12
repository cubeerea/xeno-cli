"""OpenAI-compatible chat-completions provider, plus the OpenRouter variant.

One wire format covers OpenAI, Moonshot, vLLM, and OpenRouter. Caching
behaviour does NOT follow the wire format, though, which is why family is a
separate axis from protocol: OpenAI does automatic prefix matching with a
1,024-token floor, while OpenRouter's support is unverified per underlying
model (PRD S9.6.3, OQ-11) and is therefore gated behind the startup probe.
"""

from __future__ import annotations

from typing import Any

from xeno.core.config import ModelSpec, ProviderSpec
from xeno.core.types import ProviderFamily
from xeno.core.usage import Usage
from xeno.prompt.assembly import AssembledPrompt, CacheTTL
from xeno.router.providers.base import (
    CompletionResult,
    Provider,
    ProviderError,
    attribute_cache_to_breakpoints,
    plain_messages,
)


class OpenAICompatProvider(Provider):
    """Chat-completions over the OpenAI protocol."""

    chat_path = "/chat/completions"

    def build_messages(self, prompt: AssembledPrompt) -> list[dict[str, Any]]:
        """Static-first: system region, then history, then the current turn.

        The system region concatenates SYSTEM and CODEBASE MAP in that order
        and nothing else ever goes in front of them (PRD S9.6.2). Order here is
        taken from the assembled prompt, which already enforces it — this
        method must not reorder.
        """
        messages: list[dict[str, Any]] = []

        static = [b for b in prompt.static_blocks if b.text]
        if static:
            if self.supports_explicit_cache_markers():
                content: list[dict[str, Any]] = []
                for block in static:
                    part: dict[str, Any] = {"type": "text", "text": block.text}
                    if block.cacheable:
                        part["cache_control"] = _cache_control(block.ttl)
                    content.append(part)
                messages.append({"role": "system", "content": content})
            else:
                return plain_messages(prompt)

        messages.extend({"role": t.role, "content": t.content} for t in prompt.history)
        messages.append({"role": "user", "content": prompt.current_turn})
        return messages

    def build_payload(
        self,
        prompt: AssembledPrompt,
        model: ModelSpec,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "model": model.model,
            "messages": self.build_messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

    def complete(
        self,
        prompt: AssembledPrompt,
        model: ModelSpec,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResult:
        body, latency_ms = self._post(
            self.chat_path,
            self.build_payload(prompt, model, max_tokens=max_tokens, temperature=temperature),
        )
        text, reasoning, finish_reason = _extract_message(body, provider=self.name)
        usage = parse_openai_usage(body.get("usage") or {})
        return CompletionResult(
            text=text,
            model=body.get("model", model.model),
            usage=usage,
            latency_ms=latency_ms,
            by_breakpoint=attribute_cache_to_breakpoints(prompt, usage),
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def health_check(self) -> tuple[bool, str]:
        if not self.spec.key_available():
            env = self.spec.api_key_env or "<none>"
            return False, f"no API key: set {env}"
        try:
            response = self.client.get("/models")
        except Exception as exc:
            return False, f"unreachable: {type(exc).__name__}: {exc}"
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text[:160]}"
        return True, "reachable"

class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter. Same protocol, different caching story.

    Detailed usage is opt-in on this API, and without it the response carries
    no cached-token figure at all — which would silently zero out M1.4 rather
    than fail loudly. So `usage.include` is always requested.
    """

    def __init__(self, name: str, spec: ProviderSpec) -> None:
        super().__init__(name, spec)
        self.family = ProviderFamily.AGGREGATOR

    def default_headers(self) -> dict[str, str]:
        headers = super().default_headers()
        # OpenRouter attributes traffic by these; harmless elsewhere.
        headers["HTTP-Referer"] = "https://github.com/hanksha/xeno-cli"
        headers["X-Title"] = "Xeno CLI"
        return headers

    def build_payload(
        self,
        prompt: AssembledPrompt,
        model: ModelSpec,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        payload = super().build_payload(
            prompt, model, max_tokens=max_tokens, temperature=temperature
        )
        payload["usage"] = {"include": True}
        return payload


def _cache_control(ttl: CacheTTL) -> dict[str, str]:
    """Anthropic-style marker, which OpenRouter passes through for models that
    support it. TTL follows S9.6.3: the long-lived static layers ask for 1h so
    they survive a multi-minute Talos test run mid-prompt."""
    marker = {"type": "ephemeral"}
    if ttl is CacheTTL.LONG:
        marker["ttl"] = "1h"
    return marker


def parse_openai_usage(raw: dict[str, Any]) -> Usage:
    """Normalize OpenAI-shaped usage.

    `prompt_tokens` is the TOTAL and already includes cached tokens, so the
    cached figure is subtracted downstream rather than added — inverting this
    would inflate M1.4 and understate cost at the same time.
    """
    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)

    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or raw.get("cached_tokens") or 0)

    # Anthropic-backed models proxied through an aggregator may surface the
    # cache-write figure under its native name.
    written = int(
        details.get("cache_creation_tokens")
        or raw.get("cache_creation_input_tokens")
        or 0
    )

    completion_details = raw.get("completion_tokens_details") or {}
    reasoning = min(int(completion_details.get("reasoning_tokens") or 0), completion_tokens)

    cached = min(cached, prompt_tokens)
    written = min(written, prompt_tokens - cached)
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=written,
        reasoning_tokens=reasoning,
    )


def _text_of(value: Any) -> str:
    """OpenAI-shaped content: a bare string, or a list of typed parts."""
    if isinstance(value, list):  # multipart response
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    return str(value or "")


def _reasoning_of(message: dict[str, Any]) -> str:
    """A reasoning model's scratchpad, which arrives OUTSIDE `content` under a
    field name that varies: OpenRouter normalizes to `reasoning` plus a
    structured `reasoning_details`, while several upstreams send
    `reasoning_content`.

    Read because dropping it is invisible in the worst way — the tokens are
    billed as output, so the run log shows a large response while the parser
    is handed a small one, and nothing explains the gap.
    """
    for key in ("reasoning", "reasoning_content"):
        text = _text_of(message.get(key))
        if text:
            return text
    details = message.get("reasoning_details")
    if isinstance(details, list):
        return "".join(
            str(entry.get("text") or entry.get("summary") or "")
            for entry in details
            if isinstance(entry, dict)
        )
    return ""


def _extract_message(body: dict[str, Any], *, provider: str) -> tuple[str, str, str | None]:
    choices = body.get("choices") or []
    if not choices:
        error = body.get("error")
        raise ProviderError(
            f"no choices in response{f': {error}' if error else ''}",
            provider=provider,
            retryable=True,
        )
    choice = choices[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason") or choice.get("native_finish_reason")
    return _text_of(message.get("content")), _reasoning_of(message), finish


__all__ = [
    "OpenAICompatProvider",
    "OpenRouterProvider",
    "parse_openai_usage",
]
