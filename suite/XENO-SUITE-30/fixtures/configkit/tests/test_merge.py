"""Baseline behaviour of the merge helpers."""

from __future__ import annotations

from typing import Any

import pytest

from configkit.merge import deep_merge, get_path, has_path, merge_all, set_path


def test_deep_merge_recurses_into_nested_mappings() -> None:
    base: dict[str, Any] = {"db": {"host": "localhost", "port": "5432"}, "debug": "0"}
    override: dict[str, Any] = {"db": {"port": "6543"}}
    merged = deep_merge(base, override)
    assert merged == {"db": {"host": "localhost", "port": "6543"}, "debug": "0"}


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base: dict[str, Any] = {"db": {"host": "localhost"}}
    override: dict[str, Any] = {"db": {"host": "remote"}}
    deep_merge(base, override)
    assert base == {"db": {"host": "localhost"}}
    assert override == {"db": {"host": "remote"}}


def test_deep_merge_replaces_lists_at_top_level() -> None:
    merged = deep_merge({"hosts": ["a", "b"]}, {"hosts": ["c"]})
    assert merged["hosts"] == ["c"]


def test_deep_merge_replaces_lists_when_nested() -> None:
    merged = deep_merge({"db": {"hosts": ["a", "b"]}}, {"db": {"hosts": ["c"]}})
    assert merged["db"]["hosts"] == ["c"]


def test_merge_all_applies_layers_left_to_right() -> None:
    merged = merge_all([{"a": 1, "b": 1}, {"b": 2, "c": 2}, {"c": 3}])
    assert merged == {"a": 1, "b": 2, "c": 3}


def test_path_helpers_read_and_write_nested_values() -> None:
    data: dict[str, Any] = {}
    set_path(data, ["db", "pool", "size"], "10")
    assert data == {"db": {"pool": {"size": "10"}}}
    assert has_path(data, ["db", "pool", "size"]) is True
    assert has_path(data, ["db", "pool", "timeout"]) is False
    assert get_path(data, ["db", "pool", "size"]) == "10"
    assert get_path(data, ["db", "missing", "size"], "fallback") == "fallback"


def test_set_path_rejects_an_empty_path() -> None:
    with pytest.raises(ValueError):
        set_path({}, [], "x")
