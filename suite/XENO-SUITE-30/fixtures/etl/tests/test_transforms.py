from __future__ import annotations

import pytest

from etl.sources import Row
from etl.transforms import (
    TransformError,
    add_constant,
    apply_all,
    cast_fields,
    filter_rows,
    rename_fields,
)


def test_rename_fields_maps_known_names() -> None:
    transform = rename_fields({"old": "new"})
    assert transform({"old": 1, "kept": 2}) == {"new": 1, "kept": 2}


def test_rename_fields_leaves_row_untouched() -> None:
    row: Row = {"old": 1}
    rename_fields({"old": "new"})(row)
    assert row == {"old": 1}


def test_cast_fields_applies_casters() -> None:
    transform = cast_fields({"age": int, "score": float})
    assert transform({"age": "41", "score": "1.5"}) == {"age": 41, "score": 1.5}


def test_cast_fields_skips_missing_fields() -> None:
    assert cast_fields({"age": int})({"name": "ada"}) == {"name": "ada"}


def test_cast_fields_raises_transform_error() -> None:
    with pytest.raises(TransformError):
        cast_fields({"age": int})({"age": "not-a-number"})


def test_filter_rows_keeps_matching() -> None:
    transform = filter_rows(lambda row: row["keep"] is True)
    assert transform({"keep": True}) == {"keep": True}


def test_filter_rows_drops_by_returning_none() -> None:
    transform = filter_rows(lambda row: row["keep"] is True)
    assert transform({"keep": False}) is None


def test_add_constant_sets_field() -> None:
    assert add_constant("src", "batch")({"id": 1}) == {"id": 1, "src": "batch"}


def test_add_constant_overwrites_existing_field() -> None:
    assert add_constant("src", "batch")({"src": "old"}) == {"src": "batch"}


def test_apply_all_chains_in_order() -> None:
    chain = [rename_fields({"a": "b"}), cast_fields({"b": int}), add_constant("ok", True)]
    assert apply_all(chain, {"a": "7"}) == {"b": 7, "ok": True}


def test_apply_all_short_circuits_on_drop() -> None:
    def explode(row: Row) -> Row | None:
        raise AssertionError("must not run after a drop")

    chain = [filter_rows(lambda row: False), explode]
    assert apply_all(chain, {"a": 1}) is None


def test_apply_all_with_no_transforms_is_identity() -> None:
    assert apply_all([], {"a": 1}) == {"a": 1}
