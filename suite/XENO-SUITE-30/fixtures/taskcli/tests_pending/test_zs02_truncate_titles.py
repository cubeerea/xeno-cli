"""ZS-02 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

import pytest

from taskcli.formatting import TITLE_WIDTH, render_table, truncate
from taskcli.models import Task


def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("hello", 10) == "hello"
    assert truncate("hello", 5) == "hello"


def test_truncate_appends_ellipsis_and_respects_width() -> None:
    assert truncate("abcdefghij", 6) == "abc..."
    assert len(truncate("abcdefghij", 6)) == 6


def test_truncate_rejects_widths_below_four() -> None:
    with pytest.raises(ValueError):
        truncate("abcdefghij", 3)


def test_title_width_is_forty() -> None:
    assert TITLE_WIDTH == 40


def test_render_table_truncates_long_titles() -> None:
    long_title = "z" * 80
    table = render_table([Task(id=1, title=long_title)])
    assert long_title not in table
    assert "..." in table
    assert ("z" * 37 + "...") in table
