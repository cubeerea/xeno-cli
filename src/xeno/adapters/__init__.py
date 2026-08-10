"""The adapter layer (PRD S12, revised): the boundary between Talos and a
repo's toolchain. One `GenericAdapter`, backed by a `DiscoveredToolchain`
that `xeno.adapters.discovery` proposes once per repo (model-assisted,
cached) and validates before it is ever executed."""

from __future__ import annotations

from xeno.adapters.discovery import DiscoveryError, discover_toolchain
from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain, GenericAdapter

__all__ = [
    "DiscoveredCommand",
    "DiscoveredToolchain",
    "DiscoveryError",
    "GenericAdapter",
    "discover_toolchain",
]
