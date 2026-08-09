from __future__ import annotations

from textstats.report import (
    LABEL_READING_EASE,
    LABEL_SENTENCES,
    LABEL_TOP_WORDS,
    LABEL_UNIQUE,
    LABEL_WORDS,
    SEPARATOR,
    TITLE,
    format_top_words,
    render_text_report,
)

SAMPLE = "The cat sat on the mat. The mat was flat."


def _line_for(report: str, label: str) -> str:
    prefix = f"{label}: "
    matches = [line for line in report.splitlines() if line.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {label!r} line"
    return matches[0][len(prefix) :]


def test_report_starts_with_title_and_underline() -> None:
    lines = render_text_report(SAMPLE).splitlines()
    assert lines[0] == TITLE
    assert lines[1] == SEPARATOR


def test_report_carries_the_core_counts() -> None:
    report = render_text_report(SAMPLE)
    assert _line_for(report, LABEL_WORDS) == "10"
    assert _line_for(report, LABEL_UNIQUE) == "7"
    assert _line_for(report, LABEL_SENTENCES) == "2"


def test_report_formats_reading_ease_to_two_places() -> None:
    assert _line_for(render_text_report(SAMPLE), LABEL_READING_EASE) == "117.16"


def test_report_lists_top_words() -> None:
    assert _line_for(render_text_report(SAMPLE), LABEL_TOP_WORDS) == "the (3), mat (2), cat (1)"


def test_format_top_words_handles_empty_text() -> None:
    assert format_top_words("") == "(none)"
    assert format_top_words("a a b", 1) == "a (2)"
