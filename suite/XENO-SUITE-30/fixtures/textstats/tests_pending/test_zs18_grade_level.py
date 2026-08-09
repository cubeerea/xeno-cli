"""ZS-18 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

import re

import pytest

from textstats.readability import flesch_kincaid_grade
from textstats.report import render_text_report

FOX = "The quick brown fox jumps over the lazy dog."

PARAGRAPH = (
    "The river carried a small boat past the quiet village. "
    "Farmers cut the barley in the fields above the water. "
    "Children walked the path and counted the boats. "
    "Night came before the party reached the port."
)

GRADE_LINE_RE = re.compile(r"^Grade level:\s*(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE)


def test_grade_is_rounded_to_one_decimal_place() -> None:
    grade = flesch_kincaid_grade(FOX)
    assert isinstance(grade, float)
    assert grade == pytest.approx(2.3)
    assert round(grade, 1) == grade


def test_grade_of_a_simple_paragraph_sits_in_the_expected_band() -> None:
    grade = flesch_kincaid_grade(PARAGRAPH)
    assert 2.0 <= grade <= 6.0
    assert flesch_kincaid_grade(PARAGRAPH) > flesch_kincaid_grade("The cat sat on the mat.")


def test_report_gains_a_grade_level_line() -> None:
    match = GRADE_LINE_RE.search(render_text_report(FOX))
    assert match is not None
    assert float(match.group(1)) == pytest.approx(2.3)


def test_report_grade_line_tracks_the_function() -> None:
    for text in (FOX, PARAGRAPH, "The cat sat on the mat."):
        match = GRADE_LINE_RE.search(render_text_report(text))
        assert match is not None, f"no 'Grade level:' line for {text!r}"
        assert float(match.group(1)) == pytest.approx(flesch_kincaid_grade(text), abs=0.05)


def test_existing_report_lines_survive() -> None:
    report = render_text_report(FOX)
    assert report.splitlines()[0] == "Text statistics"
    assert "Words: 9" in report
    assert "Reading ease: 94.30" in report
    assert GRADE_LINE_RE.search(report) is not None
