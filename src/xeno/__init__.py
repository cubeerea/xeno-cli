"""Xeno CLI - a terminal-native multi-agent coding harness.

The agent layer is "The Mortal Forge": seven nodes, each with a mythic callsign.
Roles are primary; callsigns are shorthand (PRD S1.1).
"""

__version__ = "0.1.0.dev0"

# The version participates in the SYSTEM cache key (PRD S9.6.5), so bumping it
# deliberately invalidates every cached system prompt. That is the intent: a
# system prompt served from a prior version's cache is a correctness bug.
XENO_VERSION = __version__

__all__ = ["XENO_VERSION", "__version__"]
