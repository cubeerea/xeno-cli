"""ZS-24 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

import pytest

from etl import ValidationFailed as ExportedValidationFailed
from etl import ValidationStage as ExportedValidationStage
from etl.config import build_pipeline
from etl.pipeline import Pipeline
from etl.sinks import MemorySink
from etl.sources import ListSource
from etl.transforms import add_constant
from etl.validation.rules import Violation, in_range, of_type, required
from etl.validation.stage import ValidationFailed, ValidationStage

ROWS = [
    {"id": 1, "email": "a@b.c", "age": 30},
    {"id": 2, "email": None, "age": 41},
    {"id": 3, "email": "d@e.f", "age": 900},
]


def test_required_flags_missing_and_null_fields() -> None:
    assert required("email")({"id": 1}) is not None
    assert required("email")({"email": None}) is not None
    assert required("email")({"email": "a@b.c"}) is None


def test_of_type_flags_wrong_types() -> None:
    assert of_type("age", int)({"age": "30"}) is not None
    assert of_type("age", int)({"age": 30}) is None


def test_in_range_bounds_are_inclusive() -> None:
    rule = in_range("age", 0, 120)
    assert rule({"age": 0}) is None
    assert rule({"age": 120}) is None
    assert rule({"age": -1}) is not None
    assert rule({"age": 121}) is not None


def test_violation_carries_field_and_message() -> None:
    violation = required("email")({"id": 1})
    assert isinstance(violation, Violation)
    assert violation.field == "email"
    assert isinstance(violation.message, str)
    assert violation.message


def test_strict_mode_raises_validation_failed_through_pipeline() -> None:
    stage = ValidationStage([required("email")], mode="strict")
    with pytest.raises(ValidationFailed):
        Pipeline(ListSource(ROWS), [], MemorySink(), validation=stage).run()


def test_quarantine_mode_drops_invalid_rows_and_counts_them() -> None:
    stage = ValidationStage([required("email"), in_range("age", 0, 120)], mode="quarantine")
    sink = MemorySink()
    stats = Pipeline(ListSource(ROWS), [], sink, validation=stage).run()
    assert stats.rows_in == 3
    assert stats.rows_out == 1
    assert stats.rows_invalid == 2
    assert [row["id"] for row in sink.rows] == [1]


def test_quarantine_mode_records_the_violations() -> None:
    stage = ValidationStage([required("email")], mode="quarantine")
    Pipeline(ListSource(ROWS), [], MemorySink(), validation=stage).run()
    assert len(stage.violations) == 1
    assert stage.violations[0].field == "email"


def test_valid_rows_report_zero_invalid() -> None:
    stage = ValidationStage([required("id")], mode="strict")
    stats = Pipeline(ListSource(ROWS), [], MemorySink(), validation=stage).run()
    assert stats.rows_invalid == 0
    assert stats.as_dict()["rows_invalid"] == 0


def test_validation_runs_before_transforms() -> None:
    stage = ValidationStage([required("email")], mode="quarantine")
    sink = MemorySink()
    Pipeline(ListSource(ROWS), [add_constant("checked", True)], sink, validation=stage).run()
    assert all(row["checked"] is True for row in sink.rows)
    assert len(sink.rows) == 2


def test_build_pipeline_accepts_a_validation_block() -> None:
    pipeline = build_pipeline(
        {
            "source": {"type": "list", "rows": ROWS},
            "validation": {
                "mode": "quarantine",
                "rules": [
                    {"type": "required", "field": "email"},
                    {"type": "in_range", "field": "age", "lo": 0, "hi": 120},
                ],
            },
            "sink": {"type": "memory"},
        }
    )
    stats = pipeline.run()
    assert stats.rows_out == 1
    assert stats.rows_invalid == 2


def test_build_pipeline_strict_validation_block_raises() -> None:
    pipeline = build_pipeline(
        {
            "source": {"type": "list", "rows": ROWS},
            "validation": {"mode": "strict", "rules": [{"type": "required", "field": "email"}]},
        }
    )
    with pytest.raises(ValidationFailed):
        pipeline.run()


def test_validation_symbols_are_exported_from_etl() -> None:
    assert ExportedValidationStage is ValidationStage
    assert ExportedValidationFailed is ValidationFailed
