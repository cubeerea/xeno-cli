"""Prompt construction: static-first assembly, cache keys, DATA delimiting.

Every node prompt is built through this package. That is the enforcement point
for T8 — a node that assembles its own prompt string bypasses the caching
design and the injection invariants at the same time.
"""

from xeno.prompt.assembly import (
    AssembledPrompt,
    Block,
    CacheTTL,
    PromptAssemblyError,
    PromptBuilder,
    Turn,
)
from xeno.prompt.delimit import as_data
from xeno.prompt.keys import CacheKeyring, StaleCodebaseMapError

__all__ = [
    "AssembledPrompt",
    "Block",
    "CacheKeyring",
    "CacheTTL",
    "PromptAssemblyError",
    "PromptBuilder",
    "StaleCodebaseMapError",
    "Turn",
    "as_data",
]
