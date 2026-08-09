"""Provider registry.

Release-v1 Phase 0 wires the two providers the reference config actually uses:
Ollama for local tiers and OpenRouter for the flagship path. Everything else
resolves through the generic OpenAI-compatible client, which already covers
Moonshot, vLLM, and OpenAI itself.

Anthropic's native Messages API is deliberately absent rather than stubbed.
Its explicit cache_control mechanics are the reference the caching layer was
designed against (PRD S9.6.3), so the *shape* is supported throughout — but a
half-written client that silently mis-reports usage would corrupt M1.4, which
is worse than a clear "not wired yet".
"""

from __future__ import annotations

from xeno.core.config import ProviderSpec
from xeno.core.types import ProviderFamily
from xeno.router.providers.base import (
    CompletionResult,
    Provider,
    ProviderError,
    attribute_cache_to_breakpoints,
)
from xeno.router.providers.ollama import OllamaProvider
from xeno.router.providers.openai_compat import OpenAICompatProvider, OpenRouterProvider


class UnsupportedProviderError(Exception):
    """Config names a provider family with no client wired."""


def build_provider(name: str, spec: ProviderSpec) -> Provider:
    """Instantiate the client for a configured provider."""
    match spec.family:
        case ProviderFamily.OLLAMA:
            return OllamaProvider(name, spec)
        case ProviderFamily.AGGREGATOR:
            return OpenRouterProvider(name, spec)
        case ProviderFamily.OPENAI:
            return OpenAICompatProvider(name, spec)
        case ProviderFamily.ANTHROPIC:
            raise UnsupportedProviderError(
                f"provider '{name}' declares family 'anthropic', which has no client in "
                f"Phase 0. Use an OpenAI-compatible endpoint, or route Anthropic models "
                f"through an aggregator."
            )
    raise UnsupportedProviderError(f"provider '{name}': unknown family {spec.family!r}")


__all__ = [
    "CompletionResult",
    "OllamaProvider",
    "OpenAICompatProvider",
    "OpenRouterProvider",
    "Provider",
    "ProviderError",
    "UnsupportedProviderError",
    "attribute_cache_to_breakpoints",
    "build_provider",
]
