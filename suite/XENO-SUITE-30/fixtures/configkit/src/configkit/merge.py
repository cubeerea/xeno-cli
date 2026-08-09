"""Recursive merging helpers and dotted-path utilities for nested mappings.

The merge rules are intentionally simple:

* two mappings are merged key by key, recursing into nested mappings;
* any other value in ``override`` replaces the value in ``base``;
* lists are treated as ordinary values, so they are replaced wholesale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new mapping with ``override`` layered on top of ``base``.

    Neither argument is mutated. Nested mappings are merged recursively; every
    other value (including lists) is replaced by the value from ``override``.
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def merge_all(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge ``layers`` left to right, later layers winning."""
    result: dict[str, Any] = {}
    for layer in layers:
        result = deep_merge(result, layer)
    return result


def has_path(data: Mapping[str, Any], parts: Sequence[str]) -> bool:
    """Return ``True`` when the nested key path ``parts`` exists in ``data``."""
    node: Any = data
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def get_path(data: Mapping[str, Any], parts: Sequence[str], default: Any = None) -> Any:
    """Return the value stored at the nested key path ``parts``.

    ``default`` is returned when any component of the path is missing or when an
    intermediate value is not a mapping.
    """
    node: Any = data
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_path(data: dict[str, Any], parts: Sequence[str], value: Any) -> None:
    """Store ``value`` at the nested key path ``parts``, creating dicts as needed.

    Intermediate values that are not dictionaries are overwritten.
    """
    if not parts:
        raise ValueError("path must contain at least one component")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if isinstance(child, dict):
            nxt: dict[str, Any] = child
        else:
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
