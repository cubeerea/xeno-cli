"""ZS-10 acceptance spec: ``deep_merge(..., list_strategy=...)``.

``deep_merge`` must accept a keyword-only ``list_strategy`` of ``"replace"``
(today's behaviour, and still the default), ``"append"`` (base items followed by
override items) or ``"unique"`` (append, then drop duplicates keeping the
first-seen order). The strategy applies at every depth, and any other value
raises ``ValueError``.
"""

from __future__ import annotations

from typing import Any

import pytest

from configkit.merge import deep_merge


def test_replace_is_still_the_default() -> None:
    base: dict[str, Any] = {"hosts": ["a", "b"]}
    override: dict[str, Any] = {"hosts": ["c"]}
    assert deep_merge(base, override) == deep_merge(base, override, list_strategy="replace")


def test_replace_strategy_top_level() -> None:
    merged = deep_merge({"hosts": ["a", "b"]}, {"hosts": ["c"]}, list_strategy="replace")
    assert merged["hosts"] == ["c"]


def test_replace_strategy_nested() -> None:
    base: dict[str, Any] = {"db": {"pool": {"hosts": ["a", "b"]}}}
    override: dict[str, Any] = {"db": {"pool": {"hosts": ["c"]}}}
    merged = deep_merge(base, override, list_strategy="replace")
    assert merged["db"]["pool"]["hosts"] == ["c"]


def test_append_strategy_top_level() -> None:
    merged = deep_merge({"hosts": ["a", "b"]}, {"hosts": ["b", "c"]}, list_strategy="append")
    assert merged["hosts"] == ["a", "b", "b", "c"]


def test_append_strategy_nested() -> None:
    base: dict[str, Any] = {"db": {"pool": {"hosts": ["a"]}}}
    override: dict[str, Any] = {"db": {"pool": {"hosts": ["b", "a"]}}}
    merged = deep_merge(base, override, list_strategy="append")
    assert merged["db"]["pool"]["hosts"] == ["a", "b", "a"]


def test_append_uses_the_override_list_when_the_key_is_new() -> None:
    merged = deep_merge({"other": 1}, {"hosts": ["a"]}, list_strategy="append")
    assert merged["hosts"] == ["a"]


def test_unique_strategy_top_level_preserves_first_seen_order() -> None:
    merged = deep_merge({"hosts": ["a", "b"]}, {"hosts": ["b", "c", "a"]}, list_strategy="unique")
    assert merged["hosts"] == ["a", "b", "c"]


def test_unique_strategy_nested() -> None:
    base: dict[str, Any] = {"db": {"pool": {"hosts": ["x", "y", "x"]}}}
    override: dict[str, Any] = {"db": {"pool": {"hosts": ["y", "z"]}}}
    merged = deep_merge(base, override, list_strategy="unique")
    assert merged["db"]["pool"]["hosts"] == ["x", "y", "z"]


def test_strategies_do_not_mutate_the_input_lists() -> None:
    base: dict[str, Any] = {"hosts": ["a"]}
    override: dict[str, Any] = {"hosts": ["b"]}
    deep_merge(base, override, list_strategy="append")
    assert base["hosts"] == ["a"]
    assert override["hosts"] == ["b"]


def test_non_list_values_are_unaffected_by_the_strategy() -> None:
    merged = deep_merge({"db": {"port": 1}}, {"db": {"port": 2}}, list_strategy="unique")
    assert merged["db"]["port"] == 2


def test_unknown_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError):
        deep_merge({"hosts": ["a"]}, {"hosts": ["b"]}, list_strategy="concat")
