"""Turning environment variables into a nested configuration layer.

A variable named ``APP__DB__PORT`` with prefix ``"APP"`` becomes
``{"db": {"port": "5432"}}``: the prefix is stripped, the remainder is split on
the double-underscore separator, and each component is lower-cased.

Values are returned exactly as the environment supplied them, i.e. as strings.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from configkit.merge import set_path

SEPARATOR = "__"
"""The token separating path components inside an environment variable name."""


def split_var_name(name: str, prefix: str) -> list[str] | None:
    """Split ``name`` into lower-cased path components, or ``None`` if unprefixed."""
    head = prefix if prefix.endswith(SEPARATOR) else prefix + SEPARATOR
    if not name.startswith(head):
        return None
    remainder = name[len(head) :]
    parts = [part.lower() for part in remainder.split(SEPARATOR) if part]
    return parts or None


def env_overrides(prefix: str, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build a nested override layer from every ``prefix``-scoped variable.

    ``environ`` defaults to :data:`os.environ`. Variables are processed in sorted
    name order so the result is deterministic, and every value stays a string.
    """
    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    source: Mapping[str, str] = os.environ if environ is None else environ
    result: dict[str, Any] = {}
    for name in sorted(source):
        parts = split_var_name(name, prefix)
        if parts is None:
            continue
        set_path(result, parts, source[name])
    return result
