"""ZS-09 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from etl.pipeline import Pipeline
from etl.sinks import MemorySink
from etl.sources import ListSource
from etl.transforms import drop_nulls


def test_drop_nulls_drops_row_with_null_named_field() -> None:
    assert drop_nulls(["email"])({"id": 1, "email": None}) is None


def test_drop_nulls_drops_row_with_missing_named_field() -> None:
    assert drop_nulls(["email"])({"id": 1}) is None


def test_drop_nulls_keeps_complete_row_unchanged() -> None:
    assert drop_nulls(["id", "email"])({"id": 1, "email": "a@b.c"}) == {
        "id": 1,
        "email": "a@b.c",
    }


def test_drop_nulls_ignores_nulls_in_unnamed_fields() -> None:
    assert drop_nulls(["id"])({"id": 1, "email": None}) == {"id": 1, "email": None}


def test_drop_nulls_without_fields_checks_every_field() -> None:
    assert drop_nulls()({"a": 1, "b": None}) is None
    assert drop_nulls()({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_drop_nulls_without_fields_keeps_empty_row() -> None:
    assert drop_nulls()({}) == {}


def test_drop_nulls_does_not_treat_falsy_values_as_null() -> None:
    row = {"count": 0, "name": "", "flag": False, "items": []}
    assert drop_nulls()(row) == row
    assert drop_nulls(["count", "name", "flag", "items"])(row) == row


def test_drop_nulls_accepts_a_tuple_of_fields() -> None:
    assert drop_nulls(("email",))({"id": 1, "email": None}) is None


def test_drop_nulls_in_a_pipeline_counts_skipped_rows() -> None:
    sink = MemorySink()
    stats = Pipeline(
        ListSource(
            [
                {"id": 1, "email": "a@b.c"},
                {"id": 2, "email": None},
                {"id": 3},
            ]
        ),
        [drop_nulls(["email"])],
        sink,
    ).run()
    assert stats.rows_in == 3
    assert stats.rows_out == 1
    assert stats.rows_skipped == 2
    assert sink.rows == [{"id": 1, "email": "a@b.c"}]
