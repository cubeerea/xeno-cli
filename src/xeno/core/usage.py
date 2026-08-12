"""Normalized token accounting (PRD S9.6.3, S15.1).

Providers report cache usage in incompatible shapes: Anthropic returns cached
tokens as separate fields alongside a fresh-only `input_tokens`, while
OpenAI-compatible endpoints report one `prompt_tokens` total with a cached
subset nested inside it. Both are normalized to `Usage` at the provider
boundary, so nothing downstream has to know which family it is looking at.

`input_tokens` is always the TOTAL, cached tokens included. Getting this
backwards would inflate M1.4 and understate cost at the same time.

This lives in `core` rather than `router` because the cost ledger is a core
concern — routing depends on accounting, not the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one model call."""

    input_tokens: int = 0  # TOTAL input, including cached
    output_tokens: int = 0
    cache_read_tokens: int = 0  # subset of input served from cache (discounted)
    cache_write_tokens: int = 0  # subset of input billed at the write premium
    #: Subset of `output_tokens` the provider spent on hidden reasoning. Billed
    #: as output and never present in the response text, so a node can appear
    #: to have written 2,473 tokens while sending back 636. Zero when the
    #: provider does not report it, which is NOT the same as "did not reason".
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(
                f"reasoning tokens ({self.reasoning_tokens}) exceed total output "
                f"({self.output_tokens}); the provider adapter is misreporting"
            )
        if self.cache_read_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError(
                f"cached tokens ({self.cache_read_tokens} read + "
                f"{self.cache_write_tokens} write) exceed total input "
                f"({self.input_tokens}); the provider adapter is misreporting"
            )

    @property
    def fresh_input_tokens(self) -> int:
        return self.input_tokens - self.cache_read_tokens - self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
